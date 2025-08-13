import adalflow as adal
from adalflow.core.types import Document, List
from adalflow.components.data_process import TextSplitter, ToEmbeddings
import os
import subprocess
import json
import tiktoken
import logging
import base64
import re
import glob
from adalflow.utils import get_adalflow_default_root_path
from adalflow.core.db import LocalDB
from api.config import configs, DEFAULT_EXCLUDED_DIRS, DEFAULT_EXCLUDED_FILES
from api.ollama_patch import OllamaDocumentProcessor
from urllib.parse import urlparse, urlunparse, quote
import requests
from requests.exceptions import RequestException

from api.tools.embedder import get_embedder

# 配置日志记录
logger = logging.getLogger(__name__)

# OpenAI嵌入模型的最大token限制
MAX_EMBEDDING_TOKENS = 8192

def count_tokens(text: str, is_ollama_embedder: bool = None) -> int:
    """
    使用 tiktoken 计算文本字符串中的 token 数量。

    Args:
        text (str): 需要计算 token 数量的文本。
        is_ollama_embedder (bool, optional): 是否使用 Ollama 嵌入模型。
                                           如果为 None，将从配置中确定。

    Returns:
        int: 文本中的 token 数量。
    """
    try:
        # 如果未指定，则确定是否使用Ollama嵌入器
        if is_ollama_embedder is None:
            from api.config import is_ollama_embedder as check_ollama
            is_ollama_embedder = check_ollama()

        if is_ollama_embedder:
            encoding = tiktoken.get_encoding("cl100k_base")
        else:
            encoding = tiktoken.encoding_for_model("text-embedding-3-small")

        return len(encoding.encode(text))
    except Exception as e:
        # 如果tiktoken失败，则回退到简单近似
        logger.warning(f"使用tiktoken计算token时出错: {e}")
        # 粗略近似：每个token 4个字符
        return len(text) // 4

def download_repo(repo_url: str, local_path: str, type: str = "github", access_token: str = None) -> str:
    """
    下载 Git 仓库（GitHub、GitLab 或 Bitbucket）到指定的本地路径。

    Args:
        repo_url (str): 要克隆的 Git 仓库 URL。
        local_path (str): 仓库将被克隆到的本地目录。
        type (str): 仓库类型，支持 "github"、"gitlab"、"bitbucket"，默认为 "github"。
        access_token (str, optional): 私有仓库的访问令牌。

    Returns:
        str: git 命令的输出消息。
    """
    try:
        # 检查Git是否已安装
        logger.info(f"准备将仓库克隆到 {local_path}")
        subprocess.run(
            ["git", "--version"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        # 检查仓库是否已存在
        if os.path.exists(local_path) and os.listdir(local_path):
            # 目录存在且不为空
            logger.warning(f"仓库已存在于 {local_path}。使用现有仓库。")
            return f"使用现有仓库 {local_path}"

        # 确保本地路径存在
        os.makedirs(local_path, exist_ok=True)

        # 如果提供了访问令牌，则准备克隆URL
        clone_url = repo_url
        if access_token:
            parsed = urlparse(repo_url)
            # 确定仓库类型并相应地格式化URL
            if type == "github":
                # 格式: https://{token}@{domain}/owner/repo.git
                # 适用于github.com和企业GitHub域名
                clone_url = urlunparse((parsed.scheme, f"{access_token}@{parsed.netloc}", parsed.path, '', '', ''))
            elif type == "gitlab":
                # 格式: https://oauth2:{token}@gitlab.com/owner/repo.git
                clone_url = urlunparse((parsed.scheme, f"oauth2:{access_token}@{parsed.netloc}", parsed.path, '', '', ''))
            elif type == "bitbucket":
                # 格式: https://x-token-auth:{token}@bitbucket.org/owner/repo.git
                clone_url = urlunparse((parsed.scheme, f"x-token-auth:{access_token}@{parsed.netloc}", parsed.path, '', '', ''))

            logger.info("使用访问令牌进行身份验证")

        # 克隆仓库
        logger.info(f"从 {repo_url} 克隆仓库到 {local_path}")
        # 我们在日志中使用repo_url以避免在日志中暴露令牌
        result = subprocess.run(
            ["git", "clone", "--depth=1", "--single-branch", clone_url, local_path],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        logger.info("仓库克隆成功")
        return result.stdout.decode("utf-8")

    except subprocess.CalledProcessError as e:
        error_msg = e.stderr.decode('utf-8')
        # 清理错误消息以移除任何令牌
        if access_token and access_token in error_msg:
            error_msg = error_msg.replace(access_token, "***TOKEN***")
        raise ValueError(f"克隆过程中出错: {error_msg}")
    except Exception as e:
        raise ValueError(f"发生意外错误: {str(e)}")

# 别名用于向后兼容
download_github_repo = download_repo

def read_all_documents(path: str, is_ollama_embedder: bool = None, excluded_dirs: List[str] = None, excluded_files: List[str] = None,
                      included_dirs: List[str] = None, included_files: List[str] = None):
    """
    递归读取目录及其子目录中的所有文档。

    Args:
        path (str): 根目录路径。
        is_ollama_embedder (bool, optional): 是否使用 Ollama 嵌入模型进行 token 计数。
                                           如果为 None，将从配置中确定。
        excluded_dirs (List[str], optional): 要排除的目录列表。
                                           如果提供，将覆盖默认配置。
        excluded_files (List[str], optional): 要排除的文件模式列表。
                                            如果提供，将覆盖默认配置。
        included_dirs (List[str], optional): 要包含的目录列表。
                                           当提供时，只处理这些目录中的文件。
        included_files (List[str], optional): 要包含的文件模式列表。
                                            当提供时，只处理匹配这些模式的文件。

    Returns:
        list: 包含元数据的 Document 对象列表。
    """
    documents = []
    # 文件扩展名，优先考虑代码文件
    code_extensions = [".py", ".js", ".ts", ".java", ".cpp", ".c", ".h", ".hpp", ".go", ".rs",
                       ".jsx", ".tsx", ".html", ".css", ".php", ".swift", ".cs"]
    doc_extensions = [".md", ".txt", ".rst", ".json", ".yaml", ".yml"]

    # 确定过滤模式：包含或排除
    use_inclusion_mode = (included_dirs is not None and len(included_dirs) > 0) or (included_files is not None and len(included_files) > 0)

    if use_inclusion_mode:
        # 包含模式：仅处理指定的目录和文件
        final_included_dirs = set(included_dirs) if included_dirs else set()
        final_included_files = set(included_files) if included_files else set()

        logger.info(f"使用包含模式")
        logger.info(f"包含目录: {list(final_included_dirs)}")
        logger.info(f"包含文件: {list(final_included_files)}")

        # 转换为列表进行处理
        included_dirs = list(final_included_dirs)
        included_files = list(final_included_files)
        excluded_dirs = []
        excluded_files = []
    else:
        # 排除模式：使用默认排除加上任何额外的排除
        final_excluded_dirs = set(DEFAULT_EXCLUDED_DIRS)
        final_excluded_files = set(DEFAULT_EXCLUDED_FILES)

        # 从配置添加任何额外的排除目录
        if "file_filters" in configs and "excluded_dirs" in configs["file_filters"]:
            final_excluded_dirs.update(configs["file_filters"]["excluded_dirs"])

        # 从配置添加任何额外的排除文件
        if "file_filters" in configs and "excluded_files" in configs["file_filters"]:
            final_excluded_files.update(configs["file_filters"]["excluded_files"])

        # 添加任何显式提供的排除目录和文件
        if excluded_dirs is not None:
            final_excluded_dirs.update(excluded_dirs)

        if excluded_files is not None:
            final_excluded_files.update(excluded_files)

        # 转换回列表以进行兼容性
        excluded_dirs = list(final_excluded_dirs)
        excluded_files = list(final_excluded_files)
        included_dirs = []
        included_files = []

        logger.info(f"使用排除模式")
        logger.info(f"排除目录: {excluded_dirs}")
        logger.info(f"排除文件: {excluded_files}")

    logger.info(f"从 {path} 读取文档")

    def should_process_file(file_path: str, use_inclusion: bool, included_dirs: List[str], included_files: List[str],
                           excluded_dirs: List[str], excluded_files: List[str]) -> bool:
        """
        根据包含/排除规则确定是否应该处理文件。

        Args:
            file_path (str): 要检查的文件路径
            use_inclusion (bool): 是否使用包含模式
            included_dirs (List[str]): 要包含的目录列表
            included_files (List[str]): 要包含的文件列表
            excluded_dirs (List[str]): 要排除的目录列表
            excluded_files (List[str]): 要排除的文件列表

        Returns:
            bool: 如果应该处理文件则返回 True，否则返回 False
        """
        file_path_parts = os.path.normpath(file_path).split(os.sep)
        file_name = os.path.basename(file_path)

        if use_inclusion:
            # 包含模式：文件必须在包含目录中或匹配包含文件模式
            is_included = False

            # 检查文件是否在包含目录中
            if included_dirs:
                for included in included_dirs:
                    clean_included = included.strip("./").rstrip("/")
                    if clean_included in file_path_parts:
                        is_included = True
                        break

            # 检查文件是否匹配包含文件模式
            if not is_included and included_files:
                for included_file in included_files:
                    if file_name == included_file or file_name.endswith(included_file):
                        is_included = True
                        break

            # 如果未指定包含规则，则允许该类别中的所有文件
            if not included_dirs and not included_files:
                is_included = True
            elif not included_dirs and included_files:
                # 仅指定文件模式，允许所有目录
                pass  # is_included 已根据文件模式设置
            elif included_dirs and not included_files:
                # 仅指定目录模式，允许包含目录中的所有文件
                pass  # is_included 已根据目录模式设置

            return is_included
        else:
            # 排除模式：文件不能在排除目录中或匹配排除文件
            is_excluded = False

            # 检查文件是否在排除目录中
            for excluded in excluded_dirs:
                clean_excluded = excluded.strip("./").rstrip("/")
                if clean_excluded in file_path_parts:
                    is_excluded = True
                    break

            # 检查文件是否匹配排除文件模式
            if not is_excluded:
                for excluded_file in excluded_files:
                    if file_name == excluded_file:
                        is_excluded = True
                        break

            return not is_excluded

    # 首先处理代码文件
    for ext in code_extensions:
        files = glob.glob(f"{path}/**/*{ext}", recursive=True)
        for file_path in files:
            # 检查文件是否应根据包含/排除规则处理
            if not should_process_file(file_path, use_inclusion_mode, included_dirs, included_files, excluded_dirs, excluded_files):
                continue

            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
                    relative_path = os.path.relpath(file_path, path)

                    # 确定这是否是实现文件
                    is_implementation = (
                        not relative_path.startswith("test_")
                        and not relative_path.startswith("app_")
                        and "test" not in relative_path.lower()
                    )

                    # 检查token数量
                    token_count = count_tokens(content, is_ollama_embedder)
                    if token_count > MAX_EMBEDDING_TOKENS * 10:
                        logger.warning(f"跳过大型文件 {relative_path}: token数量 ({token_count}) 超过限制")
                        continue

                    doc = Document(
                        text=content,
                        meta_data={
                            "file_path": relative_path,
                            "type": ext[1:],
                            "is_code": True,
                            "is_implementation": is_implementation,
                            "title": relative_path,
                            "token_count": token_count,
                        },
                    )
                    documents.append(doc)
            except Exception as e:
                logger.error(f"读取 {file_path} 时出错: {e}")

    # 然后处理文档文件
    for ext in doc_extensions:
        files = glob.glob(f"{path}/**/*{ext}", recursive=True)
        for file_path in files:
            # 检查文件是否应根据包含/排除规则处理
            if not should_process_file(file_path, use_inclusion_mode, included_dirs, included_files, excluded_dirs, excluded_files):
                continue

            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
                    relative_path = os.path.relpath(file_path, path)

                    # 检查token数量
                    token_count = count_tokens(content, is_ollama_embedder)
                    if token_count > MAX_EMBEDDING_TOKENS:
                        logger.warning(f"跳过大型文件 {relative_path}: token数量 ({token_count}) 超过限制")
                        continue

                    doc = Document(
                        text=content,
                        meta_data={
                            "file_path": relative_path,
                            "type": ext[1:],
                            "is_code": False,
                            "is_implementation": False,
                            "title": relative_path,
                            "token_count": token_count,
                        },
                    )
                    documents.append(doc)
            except Exception as e:
                logger.error(f"读取 {file_path} 时出错: {e}")

    logger.info(f"找到 {len(documents)} 个文档")
    return documents

def prepare_data_pipeline(is_ollama_embedder: bool = None):
    """
    创建并返回数据转换管道。

    Args:
        is_ollama_embedder (bool, optional): 是否使用 Ollama 进行嵌入。
                                           如果为 None，将从配置中确定。

    Returns:
        adal.Sequential: 数据转换管道
    """
    from api.config import get_embedder_config, is_ollama_embedder as check_ollama

    # 如果未指定，则确定是否使用Ollama嵌入器
    if is_ollama_embedder is None:
        is_ollama_embedder = check_ollama()

    splitter = TextSplitter(**configs["text_splitter"])
    embedder_config = get_embedder_config()

    embedder = get_embedder()

    if is_ollama_embedder:
        # 使用Ollama文档处理器进行单文档处理
        embedder_transformer = OllamaDocumentProcessor(embedder=embedder)
    else:
        # 使用批处理处理其他嵌入器
        batch_size = embedder_config.get("batch_size", 500)
        embedder_transformer = ToEmbeddings(
            embedder=embedder, batch_size=batch_size
        )

    data_transformer = adal.Sequential(
        splitter, embedder_transformer
    )  # sequential将拆分器和嵌入器串联在一起
    return data_transformer

# wuxiaoxu test

def transform_documents_and_save_to_db(
    documents: List[Document], db_path: str, is_ollama_embedder: bool = None
) -> LocalDB:
    """
    转换文档列表并将其保存到本地数据库。

    Args:
        documents (list): Document 对象列表。
        db_path (str): 本地数据库文件路径。
        is_ollama_embedder (bool, optional): 是否使用 Ollama 进行嵌入。
                                           如果为 None，将从配置中确定。
    
    Returns:
        LocalDB: 包含转换后数据的本地数据库对象。
    """
    # 获取数据转换器
    data_transformer = prepare_data_pipeline(is_ollama_embedder)  # 构建数据处理流水线（如清洗、分段、嵌入），根据参数或配置选择嵌入器
    db = LocalDB()  # 新建 LocalDB 实例，用来载入原始文档并保存转换结果。from adalflow.core.db import LocalDB

    db.register_transformer(transformer=data_transformer, key="split_and_embed")  # 注册变换器并用 key 标识（可并存多种流水线）
    db.load(documents)  # 把传入的 Document 列表载入到本地 DB（可能做校验、去重、索引）
    db.transform(key="split_and_embed")  # 执行名为 "split_and_embed" 的变换：分段并生成 embedding，结果写回 DB
    os.makedirs(os.path.dirname(db_path), exist_ok=True)  # 确保保存路径的父目录存在（注意 dirname 为空的情况）
    db.save_state(filepath=db_path)  # 将 LocalDB 状态序列化并保存到磁盘（建议采用原子写入以防半写入）
    return db  # 返回已包含转换结果且已保存到磁盘的 LocalDB 实例

def get_github_file_content(repo_url: str, file_path: str, access_token: str = None) -> str:
    """
    使用 GitHub API 从 GitHub 仓库中检索文件内容。
    支持公共 GitHub (github.com) 和 GitHub Enterprise (自定义域名)。
    
    Args:
        repo_url (str): GitHub 仓库的 URL
                       (例如："https://github.com/username/repo" 或 "https://github.company.com/username/repo")
        file_path (str): 仓库内文件的路径 (例如："src/main.py")
        access_token (str, optional): 私有仓库的 GitHub 个人访问令牌

    Returns:
        str: 文件内容字符串

    Raises:
        ValueError: 如果无法获取文件或 URL 不是有效的 GitHub URL
    """
    try:
        # 解析仓库URL以支持github.com和enterprise GitHub
        parsed_url = urlparse(repo_url)
        if not parsed_url.scheme or not parsed_url.netloc:
            raise ValueError("不是有效的GitHub仓库URL")

        # 检查URL是否为GitHub-like结构
        path_parts = parsed_url.path.strip('/').split('/')
        if len(path_parts) < 2:
            raise ValueError("无效的GitHub URL格式 - 预期格式: https://domain/owner/repo")

        owner = path_parts[-2]
        repo = path_parts[-1].replace(".git", "")

        # 确定API基础URL
        if parsed_url.netloc == "github.com":
            # 公共GitHub
            api_base = "https://api.github.com"
        else:
            # GitHub Enterprise - API通常在https://domain/api/v3/
            api_base = f"{parsed_url.scheme}://{parsed_url.netloc}/api/v3"
        
        # 使用GitHub API获取文件内容
        # 获取文件内容的API端点是：/repos/{owner}/{repo}/contents/{path}
        api_url = f"{api_base}/repos/{owner}/{repo}/contents/{file_path}"

        # 从GitHub API获取文件内容
        headers = {}
        if access_token:
            headers["Authorization"] = f"token {access_token}"
        logger.info(f"从GitHub API获取文件内容: {api_url}")
        try:
            response = requests.get(api_url, headers=headers)
            response.raise_for_status()
        except RequestException as e:
            raise ValueError(f"获取文件内容失败: {e}")
        try:
            content_data = response.json()
        except json.JSONDecodeError:
            raise ValueError("GitHub API响应无效")

        # 检查是否收到错误响应
        if "message" in content_data and "documentation_url" in content_data:
            raise ValueError(f"GitHub API错误: {content_data['message']}")

        # GitHub API返回文件内容作为base64编码字符串
        if "content" in content_data and "encoding" in content_data:
            if content_data["encoding"] == "base64":
                # 内容可能被拆分为行，所以先连接它们
                content_base64 = content_data["content"].replace("\n", "")
                content = base64.b64decode(content_base64).decode("utf-8")
                return content
            else:
                raise ValueError(f"意外编码: {content_data['encoding']}")
        else:
            raise ValueError("文件内容未在GitHub API响应中找到")

    except Exception as e:
        raise ValueError(f"获取文件内容失败: {str(e)}")

def get_gitlab_file_content(repo_url: str, file_path: str, access_token: str = None) -> str:
    """
    从 GitLab 仓库（云端或自托管）中检索文件内容。

    Args:
        repo_url (str): GitLab 仓库 URL (例如："https://gitlab.com/username/repo" 或 "http://localhost/group/project")
        file_path (str): 仓库内文件路径 (例如："src/main.py")
        access_token (str, optional): GitLab 个人访问令牌

    Returns:
        str: 文件内容

    Raises:
        ValueError: 如果任何操作失败
    """
    try:
        # 解析和验证URL
        parsed_url = urlparse(repo_url)
        if not parsed_url.scheme or not parsed_url.netloc:
            raise ValueError("不是有效的GitLab仓库URL")

        gitlab_domain = f"{parsed_url.scheme}://{parsed_url.netloc}"
        if parsed_url.port not in (None, 80, 443):
            gitlab_domain += f":{parsed_url.port}"
        path_parts = parsed_url.path.strip("/").split("/")
        if len(path_parts) < 2:
            raise ValueError("无效的GitLab URL格式 - 预期格式: https://gitlab.domain.com/group/project")

        # 构建项目路径并进行API编码
        project_path = "/".join(path_parts).replace(".git", "")
        encoded_project_path = quote(project_path, safe='')

        # 对文件路径进行编码
        encoded_file_path = quote(file_path, safe='')

        # 尝试从项目信息获取默认分支
        default_branch = None
        try:
            project_info_url = f"{gitlab_domain}/api/v4/projects/{encoded_project_path}"
            project_headers = {}
            if access_token:
                project_headers["PRIVATE-TOKEN"] = access_token
            
            project_response = requests.get(project_info_url, headers=project_headers)
            if project_response.status_code == 200:
                project_data = project_response.json()
                default_branch = project_data.get('default_branch', 'main')
                logger.info(f"找到默认分支: {default_branch}")
            else:
                logger.warning(f"无法获取项目信息，使用'main'作为默认分支")
                default_branch = 'main'
        except Exception as e:
            logger.warning(f"获取项目信息失败: {e}，使用'main'作为默认分支")
            default_branch = 'main'

        api_url = f"{gitlab_domain}/api/v4/projects/{encoded_project_path}/repository/files/{encoded_file_path}/raw?ref={default_branch}"
        # 从GitLab API获取文件内容
        headers = {}
        if access_token:
            headers["PRIVATE-TOKEN"] = access_token
        logger.info(f"从GitLab API获取文件内容: {api_url}")
        try:
            response = requests.get(api_url, headers=headers)
            response.raise_for_status()
            content = response.text
        except RequestException as e:
            raise ValueError(f"获取文件内容失败: {e}")

        # 检查GitLab错误响应（JSON而不是原始文件）
        if content.startswith("{") and '"message":' in content:
            try:
                error_data = json.loads(content)
                if "message" in error_data:
                    raise ValueError(f"GitLab API错误: {error_data['message']}")
            except json.JSONDecodeError:
                pass

        return content

    except Exception as e:
        raise ValueError(f"获取文件内容失败: {str(e)}")

def get_bitbucket_file_content(repo_url: str, file_path: str, access_token: str = None) -> str:
    """
    使用 Bitbucket API 从 Bitbucket 仓库中检索文件内容。

    Args:
        repo_url (str): Bitbucket 仓库的 URL (例如："https://bitbucket.org/username/repo")
        file_path (str): 仓库内文件的路径 (例如："src/main.py")
        access_token (str, optional): 私有仓库的 Bitbucket 个人访问令牌

    Returns:
        str: 文件内容字符串
    """
    try:
        # 从Bitbucket URL提取所有者和仓库名称以创建唯一标识符
        if not (repo_url.startswith("https://bitbucket.org/") or repo_url.startswith("http://bitbucket.org/")):
            raise ValueError("不是有效的Bitbucket仓库URL")

        parts = repo_url.rstrip('/').split('/')
        if len(parts) < 5:
            raise ValueError("无效的Bitbucket URL格式")

        owner = parts[-2]
        repo = parts[-1].replace(".git", "")

        # 尝试从仓库信息获取默认分支
        default_branch = None
        try:
            repo_info_url = f"https://api.bitbucket.org/2.0/repositories/{owner}/{repo}"
            repo_headers = {}
            if access_token:
                repo_headers["Authorization"] = f"Bearer {access_token}"
            
            repo_response = requests.get(repo_info_url, headers=repo_headers)
            if repo_response.status_code == 200:
                repo_data = repo_response.json()
                default_branch = repo_data.get('mainbranch', {}).get('name', 'main')
                logger.info(f"找到默认分支: {default_branch}")
            else:
                logger.warning(f"无法获取仓库信息，使用'main'作为默认分支")
                default_branch = 'main'
        except Exception as e:
            logger.warning(f"获取仓库信息失败: {e}，使用'main'作为默认分支")
            default_branch = 'main'

        # 使用Bitbucket API获取文件内容
        # 获取文件内容的API端点是：/2.0/repositories/{owner}/{repo}/src/{branch}/{path}
        api_url = f"https://api.bitbucket.org/2.0/repositories/{owner}/{repo}/src/{default_branch}/{file_path}"

        # 从Bitbucket API获取文件内容
        headers = {}
        if access_token:
            headers["Authorization"] = f"Bearer {access_token}"
        logger.info(f"从Bitbucket API获取文件内容: {api_url}")
        try:
            response = requests.get(api_url, headers=headers)
            if response.status_code == 200:
                content = response.text
            elif response.status_code == 404:
                raise ValueError("文件未在Bitbucket上找到。请检查文件路径和仓库。")
            elif response.status_code == 401:
                raise ValueError("Bitbucket访问未授权。请检查您的访问令牌。")
            elif response.status_code == 403:
                raise ValueError("Bitbucket访问被拒绝。您可能没有权限访问此文件。")
            elif response.status_code == 500:
                raise ValueError("Bitbucket内部服务器错误。请稍后再试。")
            else:
                response.raise_for_status()
                content = response.text
            return content
        except RequestException as e:
            raise ValueError(f"获取文件内容失败: {e}")

    except Exception as e:
        raise ValueError(f"获取文件内容失败: {str(e)}")


def get_file_content(repo_url: str, file_path: str, type: str = "github", access_token: str = None) -> str:
    """
    从 Git 仓库（GitHub、GitLab 或 Bitbucket）中检索文件内容。

    Args:
        repo_url (str): 仓库的 URL
        file_path (str): 仓库内文件的路径
        type (str): 仓库类型，支持 "github"、"gitlab"、"bitbucket"，默认为 "github"
        access_token (str, optional): 私有仓库的访问令牌

    Returns:
        str: 文件内容字符串

    Raises:
        ValueError: 如果无法获取文件或 URL 无效
    """
    if type == "github":
        return get_github_file_content(repo_url, file_path, access_token)
    elif type == "gitlab":
        return get_gitlab_file_content(repo_url, file_path, access_token)
    elif type == "bitbucket":
        return get_bitbucket_file_content(repo_url, file_path, access_token)
    else:
        raise ValueError("不支持的仓库URL。仅支持GitHub和GitLab。")

class DatabaseManager:
    """
    管理 LocalDB 实例的创建、加载、转换和持久化。
    """

    def __init__(self):
        self.db = None
        self.repo_url_or_path = None
        self.repo_paths = None

    def prepare_database(self, repo_url_or_path: str, type: str = "github", access_token: str = None, is_ollama_embedder: bool = None,
                       excluded_dirs: List[str] = None, excluded_files: List[str] = None,
                       included_dirs: List[str] = None, included_files: List[str] = None) -> List[Document]:
        """
        从仓库创建新数据库。

        Args:
            repo_url_or_path (str): 仓库的 URL 或本地路径
            type (str): 仓库类型，支持 "github"、"gitlab"、"bitbucket"，默认为 "github"
            access_token (str, optional): 私有仓库的访问令牌
            is_ollama_embedder (bool, optional): 是否使用 Ollama 进行嵌入。
                                               如果为 None，将从配置中确定。
            excluded_dirs (List[str], optional): 要排除的目录列表
            excluded_files (List[str], optional): 要排除的文件模式列表
            included_dirs (List[str], optional): 要包含的目录列表
            included_files (List[str], optional): 要包含的文件模式列表

        Returns:
            List[Document]: Document 对象列表
        """
        # wuxiaoxu test

        self.reset_database()
        self._create_repo(repo_url_or_path, type, access_token)
        return self.prepare_db_index(is_ollama_embedder=is_ollama_embedder, excluded_dirs=excluded_dirs, excluded_files=excluded_files,
                                   included_dirs=included_dirs, included_files=included_files)

    def reset_database(self):
        """
        将数据库重置为初始状态。
        """
        self.db = None
        self.repo_url_or_path = None
        self.repo_paths = None

    def _extract_repo_name_from_url(self, repo_url_or_path: str, repo_type: str) -> str:
        """
        从 URL 中提取仓库名称以创建唯一标识符。

        Args:
            repo_url_or_path (str): 仓库的 URL 或路径
            repo_type (str): 仓库类型

        Returns:
            str: 提取的仓库名称
        """
        # 从URL中提取所有者和仓库名称以创建唯一标识符
        url_parts = repo_url_or_path.rstrip('/').split('/')

        if repo_type in ["github", "gitlab", "bitbucket"] and len(url_parts) >= 5:
            # GitHub URL格式：https://github.com/owner/repo
            # GitLab URL格式：https://gitlab.com/owner/repo 或 https://gitlab.com/group/subgroup/repo
            # Bitbucket URL格式：https://bitbucket.org/owner/repo
            owner = url_parts[-2]
            repo = url_parts[-1].replace(".git", "")
            repo_name = f"{owner}_{repo}"
        else:
            repo_name = url_parts[-1].replace(".git", "")
        return repo_name





# wuixaoxu test  入口

    def _create_repo(self, repo_url_or_path: str, repo_type: str = "github", access_token: str = None) -> None:
        """
        下载并准备所有路径。
        路径：
        ~/.adalflow/repos/{owner}_{repo_name} (对于URL，本地路径将相同)
        ~/.adalflow/databases/{owner}_{repo_name}.pkl

        Args:
            repo_url_or_path (str): 仓库的 URL 或本地路径
            repo_type (str): 仓库类型，默认为 "github"
            access_token (str, optional): 私有仓库的访问令牌
        """
        logger.info(f"准备仓库存储 {repo_url_or_path}...")

        try:
            root_path = get_adalflow_default_root_path()

            os.makedirs(root_path, exist_ok=True)
            # url
            if repo_url_or_path.startswith("https://") or repo_url_or_path.startswith("http://"):
                # 从URL提取仓库名称
                repo_name = self._extract_repo_name_from_url(repo_url_or_path, repo_type)
                logger.info(f"提取仓库名称: {repo_name}")

                save_repo_dir = os.path.join(root_path, "repos", repo_name)

                # 检查仓库目录是否已存在且不为空
                if not (os.path.exists(save_repo_dir) and os.listdir(save_repo_dir)):
                    # 仅当仓库不存在或为空时才下载
                    download_repo(repo_url_or_path, save_repo_dir, repo_type, access_token)
                else:
                    logger.info(f"仓库已存在于 {save_repo_dir}。使用现有仓库。")
            else:  # 本地路径
                repo_name = os.path.basename(repo_url_or_path)
                save_repo_dir = repo_url_or_path

            save_db_file = os.path.join(root_path, "databases", f"{repo_name}.pkl")
            os.makedirs(save_repo_dir, exist_ok=True)
            os.makedirs(os.path.dirname(save_db_file), exist_ok=True)

            self.repo_paths = {
                "save_repo_dir": save_repo_dir,
                "save_db_file": save_db_file,
            }
            self.repo_url_or_path = repo_url_or_path
            logger.info(f"仓库路径: {self.repo_paths}")

        except Exception as e:
            logger.error(f"创建仓库结构失败: {e}")
            raise

    def prepare_db_index(self, is_ollama_embedder: bool = None, excluded_dirs: List[str] = None, excluded_files: List[str] = None,
                        included_dirs: List[str] = None, included_files: List[str] = None) -> List[Document]:
        """
        为仓库准备索引数据库。

        Args:
            is_ollama_embedder (bool, optional): 是否使用 Ollama 进行嵌入。
                                               如果为 None，将从配置中确定。
            excluded_dirs (List[str], optional): 要排除的目录列表
            excluded_files (List[str], optional): 要排除的文件模式列表
            included_dirs (List[str], optional): 要包含的目录列表
            included_files (List[str], optional): 要包含的文件模式列表

        Returns:
            List[Document]: Document 对象列表
        """
        # 检查数据库
        if self.repo_paths and os.path.exists(self.repo_paths["save_db_file"]):
            logger.info("加载现有数据库...")
            try:
                self.db = LocalDB.load_state(self.repo_paths["save_db_file"])
                documents = self.db.get_transformed_data(key="split_and_embed")
                if documents:
                    logger.info(f"从现有数据库加载 {len(documents)} 个文档")
                    return documents
            except Exception as e:
                logger.error(f"加载现有数据库失败: {e}")
                # 继续创建新数据库

        # 准备数据库
        logger.info("创建新数据库...")
        documents = read_all_documents(
            self.repo_paths["save_repo_dir"],
            is_ollama_embedder=is_ollama_embedder,
            excluded_dirs=excluded_dirs,
            excluded_files=excluded_files,
            included_dirs=included_dirs,
            included_files=included_files
        )







        self.db = transform_documents_and_save_to_db(
            documents, self.repo_paths["save_db_file"], is_ollama_embedder=is_ollama_embedder
        )
        logger.info(f"总文档数: {len(documents)}")
        transformed_docs = self.db.get_transformed_data(key="split_and_embed")
        logger.info(f"总转换文档数: {len(transformed_docs)}")
        return transformed_docs


-------------用与拆分文档并转化为向量存储到DB
    def prepare_retriever(self, repo_url_or_path: str, type: str = "github", access_token: str = None):
        """
        为仓库准备检索器。
        这是一个兼容方法，用于隔离API。

        Args:
            repo_url_or_path (str): 仓库的URL或本地路径
            access_token (str, optional): 私有仓库的访问令牌

        Returns:
            List[Document]: Document 对象列表
        """
        return self.prepare_database(repo_url_or_path, type, access_token)
