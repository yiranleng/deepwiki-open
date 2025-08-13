import os
import logging
from fastapi import FastAPI, HTTPException, Query, Request, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from typing import List, Optional, Dict, Any, Literal
import json
from datetime import datetime
from pydantic import BaseModel, Field
import google.generativeai as genai
import asyncio

# 配置日志记录
from api.logging_config import setup_logging

setup_logging()
logger = logging.getLogger(__name__)


# 初始化FastAPI应用
app = FastAPI(
    title="流式API",
    description="用于流式聊天完成的API"
)

# 配置CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 允许所有来源
    allow_credentials=True,
    allow_methods=["*"],  # 允许所有方法
    allow_headers=["*"],  # 允许所有头部
)

# 辅助函数：获取adalflow默认根路径
def get_adalflow_default_root_path():
    """
    获取 adalflow 默认根路径。

    Returns:
        str: adalflow 的默认根路径
    """
    return os.path.expanduser(os.path.join("~", ".adalflow"))

# --- Pydantic模型 ---
class WikiPage(BaseModel):
    """
    Wiki页面的数据模型。
    """
    id: str
    title: str
    content: str
    filePaths: List[str]
    importance: str # 理想情况下应该是 Literal['high', 'medium', 'low']
    relatedPages: List[str]

class ProcessedProjectEntry(BaseModel):
    """
    已处理项目条目的数据模型。
    """
    id: str  # 文件名
    owner: str
    repo: str
    name: str  # owner/repo
    repo_type: str # 从type重命名为repo_type以便与现有模型区分
    submittedAt: int # 时间戳
    language: str # 从文件名提取

class RepoInfo(BaseModel):
    """
    仓库信息的数据模型。
    """
    owner: str
    repo: str
    type: str
    token: Optional[str] = None
    localPath: Optional[str] = None
    repoUrl: Optional[str] = None


class WikiSection(BaseModel):
    """
    Wiki章节的数据模型。
    """
    id: str
    title: str
    pages: List[str]
    subsections: Optional[List[str]] = None


class WikiStructureModel(BaseModel):
    """
    整体Wiki结构的数据模型。
    """
    id: str
    title: str
    description: str
    pages: List[WikiPage]
    sections: Optional[List[WikiSection]] = None
    rootSections: Optional[List[str]] = None

class WikiCacheData(BaseModel):
    """
    存储在Wiki缓存中的数据模型。
    """
    wiki_structure: WikiStructureModel
    generated_pages: Dict[str, WikiPage]
    repo_url: Optional[str] = None  # 兼容旧缓存
    repo: Optional[RepoInfo] = None
    provider: Optional[str] = None
    model: Optional[str] = None

class WikiCacheRequest(BaseModel):
    """
    保存Wiki缓存时请求体的数据模型。
    """
    repo: RepoInfo
    language: str
    wiki_structure: WikiStructureModel
    generated_pages: Dict[str, WikiPage]
    provider: str
    model: str

class WikiExportRequest(BaseModel):
    """
    请求Wiki导出的数据模型。
    """
    repo_url: str = Field(..., description="仓库的URL")
    pages: List[WikiPage] = Field(..., description="要导出的Wiki页面列表")
    format: Literal["markdown", "json"] = Field(..., description="导出格式（markdown或json）")

# --- 模型配置模型 ---
class Model(BaseModel):
    """
    LLM模型配置的数据模型
    """
    id: str = Field(..., description="模型标识符")
    name: str = Field(..., description="模型的显示名称")

class Provider(BaseModel):
    """
    LLM提供商配置的数据模型
    """
    id: str = Field(..., description="提供商标识符")
    name: str = Field(..., description="提供商的显示名称")
    models: List[Model] = Field(..., description="此提供商可用的模型列表")
    supportsCustomModel: Optional[bool] = Field(False, description="此提供商是否支持自定义模型")

class ModelConfig(BaseModel):
    """
    整个模型配置的数据模型
    """
    providers: List[Provider] = Field(..., description="可用模型提供商列表")
    defaultProvider: str = Field(..., description="默认提供商的ID")

class AuthorizationConfig(BaseModel):
    """
    授权配置的数据模型
    """
    code: str = Field(..., description="授权代码")

from api.config import configs, WIKI_AUTH_MODE, WIKI_AUTH_CODE

@app.get("/lang/config")
async def get_lang_config():
    """
    获取语言配置。
    
    Returns:
        dict: 语言配置信息
    """
    return configs["lang_config"]

@app.get("/auth/status")
async def get_auth_status():
    """
    检查 wiki 是否需要身份验证。
    
    Returns:
        dict: 包含身份验证要求的字典
    """
    return {"auth_required": WIKI_AUTH_MODE}

@app.post("/auth/validate")
async def validate_auth_code(request: AuthorizationConfig):
    """
    验证授权代码。
    
    Args:
        request: 包含授权代码的请求
        
    Returns:
        dict: 包含验证结果的字典
    """
    return {"success": WIKI_AUTH_CODE == request.code}

@app.get("/models/config", response_model=ModelConfig)
async def get_model_config():
    """
    获取可用的模型提供商及其模型。

    此端点返回可用模型提供商及其可在整个应用程序中使用的
    相应模型的配置。

    Returns:
        ModelConfig: 包含提供商及其模型的配置对象
    """
    try:
        logger.info("获取模型配置")

        # 从配置文件创建提供商
        providers = []
        default_provider = configs.get("default_provider", "google")

        # 基于config.py添加提供商配置
        for provider_id, provider_config in configs["providers"].items():
            models = []
            # 从配置添加模型
            for model_id in provider_config["models"].keys():
                # 如果可能的话获取更用户友好的显示名称
                models.append(Model(id=model_id, name=model_id))

            # 添加提供商及其模型
            providers.append(
                Provider(
                    id=provider_id,
                    name=f"{provider_id.capitalize()}",
                    supportsCustomModel=provider_config.get("supportsCustomModel", False),
                    models=models
                )
            )

        # 创建并返回完整配置
        config = ModelConfig(
            providers=providers,
            defaultProvider=default_provider
        )
        return config

    except Exception as e:
        logger.error(f"创建模型配置时出错: {str(e)}")
        # 出错时返回一些默认配置
        return ModelConfig(
            providers=[
                Provider(
                    id="google",
                    name="Google",
                    supportsCustomModel=True,
                    models=[
                        Model(id="gemini-2.0-flash", name="Gemini 2.0 Flash")
                    ]
                )
            ],
            defaultProvider="google"
        )

# wuxiaoxu test 生成wiki入口
@app.post("/export/wiki")
async def export_wiki(request: WikiExportRequest):
    """
    将 wiki 内容导出为 Markdown 或 JSON。

    Args:
        request: 包含 wiki 页面和格式的导出请求

    Returns:
        请求格式的可下载文件
    """
    try:
        logger.info(f"以 {request.format} 格式导出 {request.repo_url} 的wiki")

        # 从URL提取仓库名称用于文件名
        repo_parts = request.repo_url.rstrip('/').split('/')
        repo_name = repo_parts[-1] if len(repo_parts) > 0 else "wiki"

        # 获取当前时间戳用于文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        if request.format == "markdown":
            # 生成Markdown内容
            content = generate_markdown_export(request.repo_url, request.pages)
            filename = f"{repo_name}_wiki_{timestamp}.md"
            media_type = "text/markdown"
        else:  # JSON格式
            # 生成JSON内容
            content = generate_json_export(request.repo_url, request.pages)
            filename = f"{repo_name}_wiki_{timestamp}.json"
            media_type = "application/json"

        # 创建带有适当头部用于文件下载的响应
        response = Response(
            content=content,
            media_type=media_type,
            headers={
                "Content-Disposition": f"attachment; filename={filename}"
            }
        )

        return response

    except Exception as e:
        error_msg = f"导出wiki时出错: {str(e)}"
        logger.error(error_msg)
        raise HTTPException(status_code=500, detail=error_msg)

@app.get("/local_repo/structure")
async def get_local_repo_structure(path: str = Query(None, description="Path to local repository")):
    """
    返回本地仓库的文件树和 README 内容。
    """
    if not path:
        return JSONResponse(
            status_code=400,
            content={"error": "未提供路径。请提供'path'查询参数。"}
        )

    if not os.path.isdir(path):
        return JSONResponse(
            status_code=404,
            content={"error": f"目录未找到: {path}"}
        )

    try:
        logger.info(f"处理本地仓库: {path}")
        file_tree_lines = []
        readme_content = ""

        for root, dirs, files in os.walk(path):
            # 排除隐藏目录/文件和虚拟环境
            dirs[:] = [d for d in dirs if not d.startswith('.') and d != '__pycache__' and d != 'node_modules' and d != '.venv']
            for file in files:
                if file.startswith('.') or file == '__init__.py' or file == '.DS_Store':
                    continue
                rel_dir = os.path.relpath(root, path)
                rel_file = os.path.join(rel_dir, file) if rel_dir != '.' else file
                file_tree_lines.append(rel_file)
                # 查找README.md（不区分大小写）
                if file.lower() == 'readme.md' and not readme_content:
                    try:
                        with open(os.path.join(root, file), 'r', encoding='utf-8') as f:
                            readme_content = f.read()
                    except Exception as e:
                        logger.warning(f"无法读取README.md: {str(e)}")
                        readme_content = ""

        file_tree_str = '\n'.join(sorted(file_tree_lines))
        return {"file_tree": file_tree_str, "readme": readme_content}
    except Exception as e:
        logger.error(f"处理本地仓库时出错: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"error": f"处理本地仓库时出错: {str(e)}"}
        )

def generate_markdown_export(repo_url: str, pages: List[WikiPage]) -> str:
    """
    生成 wiki 页面的 Markdown 导出。

    Args:
        repo_url: 仓库 URL
        pages: wiki 页面列表

    Returns:
        Markdown 内容字符串
    """
    # Start with metadata
    markdown = f"# Wiki Documentation for {repo_url}\n\n"
    markdown += f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"

    # Add table of contents
    markdown += "## Table of Contents\n\n"
    for page in pages:
        markdown += f"- [{page.title}](#{page.id})\n"
    markdown += "\n"

    # Add each page
    for page in pages:
        markdown += f"<a id='{page.id}'></a>\n\n"
        markdown += f"## {page.title}\n\n"



        # Add related pages
        if page.relatedPages and len(page.relatedPages) > 0:
            markdown += "### Related Pages\n\n"
            related_titles = []
            for related_id in page.relatedPages:
                # Find the title of the related page
                related_page = next((p for p in pages if p.id == related_id), None)
                if related_page:
                    related_titles.append(f"[{related_page.title}](#{related_id})")

            if related_titles:
                markdown += "Related topics: " + ", ".join(related_titles) + "\n\n"

        # Add page content
        markdown += f"{page.content}\n\n"
        markdown += "---\n\n"

    return markdown

def generate_json_export(repo_url: str, pages: List[WikiPage]) -> str:
    """
    生成 wiki 页面的 JSON 导出。

    Args:
        repo_url: 仓库 URL
        pages: wiki 页面列表

    Returns:
        JSON 内容字符串
    """
    # 创建包含元数据和页面的字典
    export_data = {
        "metadata": {
            "repository": repo_url,
            "generated_at": datetime.now().isoformat(),
            "page_count": len(pages)
        },
        "pages": [page.model_dump() for page in pages]
    }

    # 转换为格式化的JSON字符串
    return json.dumps(export_data, indent=2)

# 导入简化的聊天实现
from api.simple_chat import chat_completions_stream
from api.websocket_wiki import handle_websocket_chat

# 将chat_completions_stream端点添加到主应用
app.add_api_route("/chat/completions/stream", chat_completions_stream, methods=["POST"])

# 添加WebSocket端点
app.add_websocket_route("/ws/chat", handle_websocket_chat)

# --- Wiki缓存辅助函数 ---

WIKI_CACHE_DIR = os.path.join(get_adalflow_default_root_path(), "wikicache")
os.makedirs(WIKI_CACHE_DIR, exist_ok=True)

def get_wiki_cache_path(owner: str, repo: str, repo_type: str, language: str) -> str:
    """
    为给定的 wiki 缓存生成文件路径。
    
    Args:
        owner: 仓库所有者
        repo: 仓库名称
        repo_type: 仓库类型
        language: 语言代码
        
    Returns:
        str: 缓存文件路径
    """
    filename = f"deepwiki_cache_{repo_type}_{owner}_{repo}_{language}.json"
    return os.path.join(WIKI_CACHE_DIR, filename)

async def read_wiki_cache(owner: str, repo: str, repo_type: str, language: str) -> Optional[WikiCacheData]:
    """
    从文件系统读取 wiki 缓存数据。
    
    Args:
        owner: 仓库所有者
        repo: 仓库名称
        repo_type: 仓库类型
        language: 语言代码
        
    Returns:
        Optional[WikiCacheData]: 缓存数据，如果不存在则返回None
    """
    cache_path = get_wiki_cache_path(owner, repo, repo_type, language)
    if os.path.exists(cache_path):
        try:
            with open(cache_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return WikiCacheData(**data)
        except Exception as e:
            logger.error(f"从 {cache_path} 读取wiki缓存时出错: {e}")
            return None
    return None

async def save_wiki_cache(data: WikiCacheRequest) -> bool:
    """
    将 wiki 缓存数据保存到文件系统。
    
    Args:
        data: Wiki缓存请求数据
        
    Returns:
        bool: 保存成功返回True，否则返回False
    """
    cache_path = get_wiki_cache_path(data.repo.owner, data.repo.repo, data.repo.type, data.language)
    logger.info(f"尝试保存wiki缓存。路径: {cache_path}")
    try:
        payload = WikiCacheData(
            wiki_structure=data.wiki_structure,
            generated_pages=data.generated_pages,
            repo=data.repo,
            provider=data.provider,
            model=data.model
        )
        # 记录要缓存的数据大小用于调试（避免记录大型完整内容）
        try:
            payload_json = payload.model_dump_json()
            payload_size = len(payload_json.encode('utf-8'))
            logger.info(f"已准备缓存的有效载荷。大小: {payload_size} 字节。")
        except Exception as ser_e:
            logger.warning(f"无法序列化有效载荷进行大小记录: {ser_e}")


        logger.info(f"写入缓存文件到: {cache_path}")
        with open(cache_path, 'w', encoding='utf-8') as f:
            json.dump(payload.model_dump(), f, indent=2)
        logger.info(f"Wiki缓存已成功保存到 {cache_path}")
        return True
    except IOError as e:
        logger.error(f"保存wiki缓存到 {cache_path} 时发生IOError: {e.strerror} (errno: {e.errno})", exc_info=True)
        return False
    except Exception as e:
        logger.error(f"保存wiki缓存到 {cache_path} 时发生意外错误: {e}", exc_info=True)
        return False

# --- Wiki缓存API端点 ---

@app.get("/api/wiki_cache", response_model=Optional[WikiCacheData])
async def get_cached_wiki(
    owner: str = Query(..., description="仓库所有者"),
    repo: str = Query(..., description="仓库名称"),
    repo_type: str = Query(..., description="仓库类型（例如，github、gitlab）"),
    language: str = Query(..., description="Wiki内容的语言")
):
    """
    检索仓库的缓存 wiki 数据（结构和生成的页面）。
    
    Args:
        owner: 仓库所有者
        repo: 仓库名称
        repo_type: 仓库类型
        language: Wiki内容的语言
        
    Returns:
        Optional[WikiCacheData]: 缓存的wiki数据，如果不存在则返回None
    """
    # 语言验证
    supported_langs = configs["lang_config"]["supported_languages"]
    if not supported_langs.__contains__(language):
        language = configs["lang_config"]["default"]

    logger.info(f"尝试检索 {owner}/{repo} ({repo_type}) 的wiki缓存，语言: {language}")
    cached_data = await read_wiki_cache(owner, repo, repo_type, language)
    if cached_data:
        return cached_data
    else:
        # 如果未找到则返回200状态码和空内容，因为前端期望这种行为
        # 或者，如果首选，可以抛出 HTTPException(status_code=404, detail="Wiki cache not found")
        logger.info(f"未找到 {owner}/{repo} ({repo_type}) 的wiki缓存，语言: {language}")
        return None

@app.post("/api/wiki_cache")
async def store_wiki_cache(request_data: WikiCacheRequest):
    """
    将生成的 wiki 数据（结构和页面）存储到服务器端缓存。
    
    Args:
        request_data: Wiki缓存请求数据
        
    Returns:
        dict: 包含成功消息的字典
        
    Raises:
        HTTPException: 保存失败时抛出500错误
    """
    # 语言验证
    supported_langs = configs["lang_config"]["supported_languages"]

    if not supported_langs.__contains__(request_data.language):
        request_data.language = configs["lang_config"]["default"]

    logger.info(f"尝试保存 {request_data.repo.owner}/{request_data.repo.repo} ({request_data.repo.type}) 的wiki缓存，语言: {request_data.language}")
    success = await save_wiki_cache(request_data)
    if success:
        return {"message": "Wiki缓存保存成功"}
    else:
        raise HTTPException(status_code=500, detail="保存wiki缓存失败")

@app.delete("/api/wiki_cache")
async def delete_wiki_cache(
    owner: str = Query(..., description="仓库所有者"),
    repo: str = Query(..., description="仓库名称"),
    repo_type: str = Query(..., description="仓库类型（例如，github、gitlab）"),
    language: str = Query(..., description="Wiki内容的语言"),
    authorization_code: Optional[str] = Query(None, description="授权代码")
):
    """
    从文件系统删除特定的 wiki 缓存。
    
    Args:
        owner: 仓库所有者
        repo: 仓库名称
        repo_type: 仓库类型
        language: Wiki内容的语言
        authorization_code: 授权代码（可选）
        
    Returns:
        dict: 包含成功消息的字典
        
    Raises:
        HTTPException: 各种错误情况下的相应错误
    """
    # 语言验证
    supported_langs = configs["lang_config"]["supported_languages"]
    if not supported_langs.__contains__(language):
        raise HTTPException(status_code=400, detail="不支持的语言")

    if WIKI_AUTH_MODE:
        logger.info("检查授权代码")
        if WIKI_AUTH_CODE != authorization_code:
            raise HTTPException(status_code=401, detail="授权代码无效")

    logger.info(f"尝试删除 {owner}/{repo} ({repo_type}) 的wiki缓存，语言: {language}")
    cache_path = get_wiki_cache_path(owner, repo, repo_type, language)

    if os.path.exists(cache_path):
        try:
            os.remove(cache_path)
            logger.info(f"成功删除wiki缓存: {cache_path}")
            return {"message": f"已成功删除 {owner}/{repo} ({language}) 的wiki缓存"}
        except Exception as e:
            logger.error(f"删除wiki缓存 {cache_path} 时出错: {e}")
            raise HTTPException(status_code=500, detail=f"删除wiki缓存失败: {str(e)}")
    else:
        logger.warning(f"未找到wiki缓存，无法删除: {cache_path}")
        raise HTTPException(status_code=404, detail="未找到wiki缓存")

@app.get("/health")
async def health_check():
    """
    Docker 和监控的健康检查端点。
    
    Returns:
        dict: 包含健康状态信息的字典
    """
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "service": "deepwiki-api"
    }

@app.get("/")
async def root():
    """
    根端点，用于检查 API 是否运行并动态列出可用端点。
    
    Returns:
        dict: 包含API信息和端点列表的字典
    """
    # 从FastAPI应用动态收集路由
    endpoints = {}
    for route in app.routes:
        if hasattr(route, "methods") and hasattr(route, "path"):
            # 跳过文档和静态路由
            if route.path in ["/openapi.json", "/docs", "/redoc", "/favicon.ico"]:
                continue
            # 按第一个路径段分组端点
            path_parts = route.path.strip("/").split("/")
            group = path_parts[0].capitalize() if path_parts[0] else "Root"
            method_list = list(route.methods - {"HEAD", "OPTIONS"})
            for method in method_list:
                endpoints.setdefault(group, []).append(f"{method} {route.path}")

    # 可选地，对端点进行排序以提高可读性
    for group in endpoints:
        endpoints[group].sort()

    return {
        "message": "欢迎使用流式API",
        "version": "1.0.0",
        "endpoints": endpoints
    }

# --- 已处理项目端点 --- （新端点）
@app.get("/api/processed_projects", response_model=List[ProcessedProjectEntry])
async def get_processed_projects():
    """
    列出 wiki 缓存目录中找到的所有已处理项目。
    项目通过名为 deepwiki_cache_{repo_type}_{owner}_{repo}_{language}.json 的文件识别。
    
    Returns:
        List[ProcessedProjectEntry]: 已处理项目条目列表
    """
    project_entries: List[ProcessedProjectEntry] = []
    # WIKI_CACHE_DIR 已在文件中全局定义

    try:
        if not os.path.exists(WIKI_CACHE_DIR):
            logger.info(f"缓存目录 {WIKI_CACHE_DIR} 未找到。返回空列表。")
            return []

        logger.info(f"扫描项目缓存文件: {WIKI_CACHE_DIR}")
        filenames = await asyncio.to_thread(os.listdir, WIKI_CACHE_DIR) # 使用asyncio.to_thread进行os.listdir

        for filename in filenames:
            if filename.startswith("deepwiki_cache_") and filename.endswith(".json"):
                file_path = os.path.join(WIKI_CACHE_DIR, filename)
                try:
                    stats = await asyncio.to_thread(os.stat, file_path) # 使用asyncio.to_thread进行os.stat
                    parts = filename.replace("deepwiki_cache_", "").replace(".json", "").split('_')

                    # 期望格式：repo_type_owner_repo_language
                    # 示例：deepwiki_cache_github_AsyncFuncAI_deepwiki-open_en.json
                    # parts = [github, AsyncFuncAI, deepwiki-open, en]
                    if len(parts) >= 4:
                        repo_type = parts[0]
                        owner = parts[1]
                        language = parts[-1] # 语言是最后一部分
                        repo = "_".join(parts[2:-1]) # 仓库名可以包含下划线

                        project_entries.append(
                            ProcessedProjectEntry(
                                id=filename,
                                owner=owner,
                                repo=repo,
                                name=f"{owner}/{repo}",
                                repo_type=repo_type,
                                submittedAt=int(stats.st_mtime * 1000), # 转换为毫秒
                                language=language
                            )
                        )
                    else:
                        logger.warning(f"无法从文件名解析项目详情: {filename}")
                except Exception as e:
                    logger.error(f"处理文件 {file_path} 时出错: {e}")
                    continue # 出错时跳过此文件

        # 按最近时间排序
        project_entries.sort(key=lambda p: p.submittedAt, reverse=True)
        logger.info(f"找到 {len(project_entries)} 个已处理项目条目。")
        return project_entries

    except Exception as e:
        logger.error(f"从 {WIKI_CACHE_DIR} 列出已处理项目时出错: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="从服务器缓存列出已处理项目失败。")
