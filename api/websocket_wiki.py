import logging
import os
from typing import List, Optional, Dict, Any
from urllib.parse import unquote

import google.generativeai as genai
from adalflow.components.model_client.ollama_client import OllamaClient
from adalflow.core.types import ModelType
from fastapi import WebSocket, WebSocketDisconnect, HTTPException
from pydantic import BaseModel, Field

from api.config import get_model_config, configs, OPENROUTER_API_KEY, OPENAI_API_KEY
from api.data_pipeline import count_tokens, get_file_content
from api.openai_client import OpenAIClient
from api.openrouter_client import OpenRouterClient
from api.azureai_client import AzureAIClient
from api.dashscope_client import DashscopeClient
from api.rag import RAG

# 配置日志记录
from api.logging_config import setup_logging

setup_logging()
logger = logging.getLogger(__name__)


# API的数据模型
class ChatMessage(BaseModel):
    """
    聊天消息的数据模型。
    """
    role: str  # 'user' 或 'assistant'
    content: str

class ChatCompletionRequest(BaseModel):
    """
    请求聊天完成的数据模型。
    """
    repo_url: str = Field(..., description="要查询的仓库URL")
    messages: List[ChatMessage] = Field(..., description="聊天消息列表")
    filePath: Optional[str] = Field(None, description="要包含在提示中的仓库文件的可选路径")
    token: Optional[str] = Field(None, description="私有仓库的个人访问令牌")
    type: Optional[str] = Field("github", description="仓库类型（例如，'github'、'gitlab'、'bitbucket'）")

    # 模型参数
    provider: str = Field("google", description="模型提供商（google、openai、openrouter、ollama、azure）")
    model: Optional[str] = Field(None, description="指定提供商的模型名称")

    language: Optional[str] = Field("en", description="内容生成的语言（例如，'en'、'ja'、'zh'、'es'、'kr'、'vi'）")
    excluded_dirs: Optional[str] = Field(None, description="要从处理中排除的目录的逗号分隔列表")
    excluded_files: Optional[str] = Field(None, description="要从处理中排除的文件模式的逗号分隔列表")
    included_dirs: Optional[str] = Field(None, description="要专门包含的目录的逗号分隔列表")
    included_files: Optional[str] = Field(None, description="要专门包含的文件模式的逗号分隔列表")

async def handle_websocket_chat(websocket: WebSocket):
    """
    处理聊天完成的 WebSocket 连接。
    这用 WebSocket 连接替换了 HTTP 流式端点。
    
    Args:
        websocket: WebSocket连接对象
    """
    await websocket.accept()

    try:
        # 接收并解析请求数据
        request_data = await websocket.receive_json()
        request = ChatCompletionRequest(**request_data)

        # 检查请求是否包含非常大的输入
        input_too_large = False
        if request.messages and len(request.messages) > 0:
            last_message = request.messages[-1]
            if hasattr(last_message, 'content') and last_message.content:
                tokens = count_tokens(last_message.content, request.provider == "ollama")
                logger.info(f"请求大小: {tokens} 个token")
                if tokens > 8000:
                    logger.warning(f"请求超过推荐的token限制 ({tokens} > 7500)")
                    input_too_large = True

        # 为此请求创建新的RAG实例
        try:
            request_rag = RAG(provider=request.provider, model=request.model)

            # 如果提供了自定义文件过滤器参数则提取
            excluded_dirs = None
            excluded_files = None
            included_dirs = None
            included_files = None

            if request.excluded_dirs:
                excluded_dirs = [unquote(dir_path) for dir_path in request.excluded_dirs.split('\n') if dir_path.strip()]
                logger.info(f"使用自定义排除目录: {excluded_dirs}")
            if request.excluded_files:
                excluded_files = [unquote(file_pattern) for file_pattern in request.excluded_files.split('\n') if file_pattern.strip()]
                logger.info(f"使用自定义排除文件: {excluded_files}")
            if request.included_dirs:
                included_dirs = [unquote(dir_path) for dir_path in request.included_dirs.split('\n') if dir_path.strip()]
                logger.info(f"使用自定义包含目录: {included_dirs}")
            if request.included_files:
                included_files = [unquote(file_pattern) for file_pattern in request.included_files.split('\n') if file_pattern.strip()]
                logger.info(f"使用自定义包含文件: {included_files}")

            request_rag.prepare_retriever(request.repo_url, request.type, request.token, excluded_dirs, excluded_files, included_dirs, included_files)
            logger.info(f"为 {request.repo_url} 准备检索器")
        except ValueError as e:
            if "No valid documents with embeddings found" in str(e):
                logger.error(f"未找到有效的文档嵌入: {str(e)}")
                await websocket.send_text("错误: 未找到有效的文档嵌入。这可能是由于嵌入大小不一致或文档处理期间的API错误。请重试或检查您的仓库内容。")
                await websocket.close()
                return
            else:
                logger.error(f"ValueError准备检索器: {str(e)}")
                await websocket.send_text(f"错误准备检索器: {str(e)}")
                await websocket.close()
                return
        except Exception as e:
            logger.error(f"错误准备检索器: {str(e)}")
            # 检查特定嵌入相关错误
            if "All embeddings should be of the same size" in str(e):
                await websocket.send_text("错误: 检测到不一致的嵌入大小。某些文档可能未能正确嵌入。请重试。")
            else:
                await websocket.send_text(f"错误准备检索器: {str(e)}")
            await websocket.close()
            return

        # 验证请求
        if not request.messages or len(request.messages) == 0:
            await websocket.send_text("错误: 未提供消息")
            await websocket.close()
            return

        last_message = request.messages[-1]
        if last_message.role != "user":
            await websocket.send_text("错误: 最后一条消息必须是用户消息")
            await websocket.close()
            return

        # 处理之前的消息以构建对话历史
        for i in range(0, len(request.messages) - 1, 2):
            if i + 1 < len(request.messages):
                user_msg = request.messages[i]
                assistant_msg = request.messages[i + 1]

                if user_msg.role == "user" and assistant_msg.role == "assistant":
                    request_rag.memory.add_dialog_turn(
                        user_query=user_msg.content,
                        assistant_response=assistant_msg.content
                    )

        # 检查这是否是Deep Research请求
        is_deep_research = False
        research_iteration = 1

        # 处理消息以检测Deep Research请求
        for msg in request.messages:
            if hasattr(msg, 'content') and msg.content and "[DEEP RESEARCH]" in msg.content:
                is_deep_research = True
                # 仅从最后一条消息中移除标签
                if msg == request.messages[-1]:
                    # 移除Deep Research标签
                    msg.content = msg.content.replace("[DEEP RESEARCH]", "").strip()

        # 如果这是Deep Research请求，则计算研究迭代次数
        if is_deep_research:
            research_iteration = sum(1 for msg in request.messages if msg.role == 'assistant') + 1
            logger.info(f"检测到Deep Research请求 - 迭代 {research_iteration}")

            # 检查这是否是延续请求
            if "continue" in last_message.content.lower() and "research" in last_message.content.lower():
                # 从第一个用户消息中找到原始主题
                original_topic = None
                for msg in request.messages:
                    if msg.role == "user" and "continue" not in msg.content.lower():
                        original_topic = msg.content.replace("[DEEP RESEARCH]", "").strip()
                        logger.info(f"找到原始研究主题: {original_topic}")
                        break

                if original_topic:
                    # 将延续消息替换为原始主题
                    last_message.content = original_topic
                    logger.info(f"使用原始主题进行研究: {original_topic}")

        # 获取最后一条消息的查询
        query = last_message.content

        # 仅在输入不是太大时才检索文档
        context_text = ""
        retrieved_documents = None

        if not input_too_large:
            try:
                # 如果filePath存在，则修改RAG查询以专注于文件
                rag_query = query
                if request.filePath:
                    # 使用文件路径获取与文件相关的上下文
                    rag_query = f"与 {request.filePath} 相关的上下文"
                    logger.info(f"修改RAG查询以专注于文件: {request.filePath}")

                # 尝试执行RAG检索
                try:
                    # 这将使用实际的RAG实现
                    retrieved_documents = request_rag(rag_query, language=request.language)

                    if retrieved_documents and retrieved_documents[0].documents:
                        # 以更结构化的方式格式化上下文用于提示
                        documents = retrieved_documents[0].documents
                        logger.info(f"检索到 {len(documents)} 个文档")

                        # 按文件路径分组文档
                        docs_by_file = {}
                        for doc in documents:
                            file_path = doc.meta_data.get('file_path', 'unknown')
                            if file_path not in docs_by_file:
                                docs_by_file[file_path] = []
                            docs_by_file[file_path].append(doc)

                        # 使用文件路径分组格式化上下文文本
                        context_parts = []
                        for file_path, docs in docs_by_file.items():
                            # 添加文件头和元数据
                            header = f"## 文件路径: {file_path}\n\n"
                            # 添加文档内容
                            content = "\n\n".join([doc.text for doc in docs])

                            context_parts.append(f"{header}{content}")

                        # 使用清晰的分割连接所有部分
                        context_text = "\n\n" + "-" * 10 + "\n\n".join(context_parts)
                    else:
                        logger.warning("未从RAG检索到文档")
                except Exception as e:
                    logger.error(f"RAG检索错误: {str(e)}")
                    # 如果发生错误，则继续不使用RAG

            except Exception as e:
                logger.error(f"检索文档错误: {str(e)}")
                context_text = ""

        # 获取仓库信息
        repo_url = request.repo_url
        repo_name = repo_url.split("/")[-1] if "/" in repo_url else repo_url

        # 确定仓库类型
        repo_type = request.type

        # 获取语言信息
        language_code = request.language or configs["lang_config"]["default"]
        supported_langs = configs["lang_config"]["supported_languages"]
        language_name = supported_langs.get(language_code, "English")

        # 创建系统提示
        if is_deep_research:
            # 检查这是否是第一次迭代
            is_first_iteration = research_iteration == 1

            # 检查这是否是最终迭代
            is_final_iteration = research_iteration >= 5

            if is_first_iteration:
                system_prompt = f"""<role>
你是一个专家代码分析师，正在检查 {repo_type} 仓库: {repo_url} ({repo_name})。
你正在进行多轮Deep Research过程，以彻底调查用户查询的特定主题。
你的目标是提供关于此主题的详细、专注的信息。
重要提示：你必须用 {language_name} 语言回答。
</role>

<guidelines>
- 这是多轮研究过程的第一次迭代，专注于用户查询的特定主题。
- 你的回答应以 "## Research Plan" 开头。
- 概述你调查此特定主题的方法。
- 如果主题是关于特定文件或功能（如 "Dockerfile"），请仅关注该文件或功能。
- 清楚地说明你正在研究的主题，以在整个研究过程中保持专注。
- 识别你需要研究的关键方面。
- 根据可用信息提供初步发现。
- 以 "## Next Steps" 结束，指示你将在下一轮研究中调查的内容。
- 不要提供最终结论 - 这只是研究的开始。
- 除非直接相关，否则不要包含一般仓库信息。
- 专注于你正在研究的主题，不要偏离相关主题。
- 你的研究必须直接回答原始问题。
- 永远不要用 "Continue the research" 作为答案 - 总是提供实质性研究发现。
- 记住这个主题将在所有研究迭代中保持。
</guidelines>

<style>
- 简洁但全面。
- 使用Markdown格式提高可读性。
- 引用特定文件和代码段时请引用。
</style>"""
            elif is_final_iteration:
                system_prompt = f"""<role>
你是一个专家代码分析师，正在检查 {repo_type} 仓库: {repo_url} ({repo_name})。
你正在进行Deep Research过程的最终迭代，专注于最新的用户查询。
你的目标是综合所有先前发现并提供一个全面的结论，直接回答此特定主题。
重要提示：你必须用 {language_name} 语言回答。
</role>

<guidelines>
- 这是研究过程的最终迭代。
- 仔细审查整个对话历史，以理解所有先前发现。
- 将所有先前迭代中的发现综合成一个全面的结论。
- 以 "## Final Conclusion" 开头。
- 你的结论必须直接回答原始问题。
- 严格专注于特定主题，不要偏离相关主题。
- 包括特定代码引用和实现细节，与主题相关。
- 强调此特定功能的重要发现和见解。
- 提供对原始问题的完整和明确的答案。
- 除非直接相关，否则不要包含一般仓库信息。
- 专注于特定主题。
- 永远不要用 "Continue the research" 作为答案 - 总是提供完整的结论。
- 如果主题是关于特定文件或功能（如 "Dockerfile"），请仅关注该文件或功能。
- 确保你的结论建立在并引用先前迭代的关键发现上。
</guidelines>

<style>
- 简洁但全面。
- 使用Markdown格式提高可读性。
- 引用特定文件和代码段时请引用。
- 使用清晰的标题结构。
- 在适当的时候提供行动建议或建议。
</style>"""
            else:
                system_prompt = f"""<role>
你是一个专家代码分析师，正在检查 {repo_type} 仓库: {repo_url} ({repo_name})。
你目前正在进行迭代 {research_iteration} 的Deep Research过程，专注于最新的用户查询。
你的目标是基于先前研究迭代，深入研究此特定主题，而不偏离它。
重要提示：你必须用 {language_name} 语言回答。
</role>

<guidelines>
- 仔细审查对话历史，以了解已经研究了什么。
- 你的回答必须建立在先前研究迭代的基础上 - 不要重复已涵盖的信息。
- 识别此特定主题需要进一步探索的差距或领域。
- 专注于此特定方面，此迭代需要更深入的研究。
- 你的回答应以 "## Research Update {research_iteration}" 开头。
- 清楚地解释你在此迭代中研究的内容。
- 提供先前迭代未涵盖的新见解。
- 如果这是迭代3，请准备在下一轮迭代中得出结论。
- 除非直接相关，否则不要包含一般仓库信息。
- 专注于特定主题，不要偏离相关主题。
- 如果主题是关于特定文件或功能（如 "Dockerfile"），请仅关注该文件或功能。
- 永远不要用 "Continue the research" 作为答案 - 总是提供实质性研究发现。
- 你的研究必须直接回答原始问题。
- 与先前研究迭代保持连续性 - 这是一个持续的调查。
</guidelines>

<style>
- 简洁但全面。
- 专注于提供新信息，而不是重复已涵盖的内容。
- 使用Markdown格式提高可读性。
- 引用特定文件和代码段时请引用。
</style>"""
        else:
            system_prompt = f"""<role>
你是一个专家代码分析师，正在检查 {repo_type} 仓库: {repo_url} ({repo_name})。
你提供直接、简洁、准确的代码仓库信息。
你从不以Markdown标题或代码块开头。
重要提示：你必须用 {language_name} 语言回答。
</role>

<guidelines>
- 直接回答用户的问题，不带任何前言或填充词。
- 不要包含任何理由、解释或额外评论。
- 仅基于现有代码或文档严格回答。
- 不要猜测或发明引用。
- 不要以 "Okay, here's a breakdown" 或 "Here's an explanation" 开头。
- 不要以 "## Analysis of..." 或任何文件路径引用开头。
- 不要以 ```markdown 代码块开头
- 不要以 ``` 结尾代码块
- 不要以重复或确认问题开头。
- 只需直接回答问题。

<example_of_what_not_to_do>
```markdown
## Analysis of `adalflow/adalflow/datasets/gsm8k.py`

This file contains...
```
</example_of_what_not_to_do>

- 使用适当的Markdown格式，包括标题、列表和代码块。
- 对于代码分析，请组织你的响应。
- 一步一步地思考，并按逻辑组织你的答案。
- 从最相关且直接回答用户查询的信息开始。
- 精确且技术性地讨论代码。
- 你的响应语言应与用户查询的语言相同。
</guidelines>

<style>
- 使用简洁、直接的语言。
- 优先考虑准确性，而不是冗长。
- 当显示代码时，请在相关时包含行号和文件路径。
- 使用Markdown格式提高可读性。
</style>"""

        # 如果提供了filePath，则获取文件内容
        file_content = ""
        if request.filePath:
            try:
                file_content = get_file_content(request.repo_url, request.filePath, request.type, request.token)
                logger.info(f"成功检索文件内容: {request.filePath}")
            except Exception as e:
                logger.error(f"检索文件内容错误: {str(e)}")
                # 如果发生错误，则继续不使用文件内容

        # 格式化对话历史
        conversation_history = ""
        for turn_id, turn in request_rag.memory().items():
            if not isinstance(turn_id, int) and hasattr(turn, 'user_query') and hasattr(turn, 'assistant_response'):
                conversation_history += f"<turn>\n<user>{turn.user_query.query_str}</user>\n<assistant>{turn.assistant_response.response_str}</assistant>\n</turn>\n"

        # 创建包含上下文的提示
        prompt = f"/no_think {system_prompt}\n\n"

        if conversation_history:
            prompt += f"<conversation_history>\n{conversation_history}</conversation_history>\n\n"

        # 检查filePath是否提供且如果存在则获取文件内容
        if file_content:
            # 在对话历史后将文件内容添加到提示中
            prompt += f"<currentFileContent path=\"{request.filePath}\">\n{file_content}\n</currentFileContent>\n\n"

        # 仅在上下文不为空时包含上下文
        CONTEXT_START = "<START_OF_CONTEXT>"
        CONTEXT_END = "<END_OF_CONTEXT>"
        if context_text.strip():
            prompt += f"{CONTEXT_START}\n{context_text}\n{CONTEXT_END}\n\n"
        else:
            # 添加一个提示，说明我们跳过RAG，因为输入太大或是因为它是孤立的API
            logger.info("未从RAG获取上下文")
            prompt += "<note>回答时未使用检索增强。</note>\n\n"

        prompt += f"<query>\n{query}\n</query>\n\nAssistant: "

        model_config = get_model_config(request.provider, request.model)["model_kwargs"]

        if request.provider == "ollama":
            prompt += " /no_think"

            model = OllamaClient()
            model_kwargs = {
                "model": model_config["model"],
                "stream": True,
                "options": {
                    "temperature": model_config["temperature"],
                    "top_p": model_config["top_p"],
                    "num_ctx": model_config["num_ctx"]
                }
            }

            api_kwargs = model.convert_inputs_to_api_kwargs(
                input=prompt,
                model_kwargs=model_kwargs,
                model_type=ModelType.LLM
            )
        elif request.provider == "openrouter":
            logger.info(f"使用OpenRouter与模型: {request.model}")

            # 检查OpenRouter API密钥是否已设置
            if not OPENROUTER_API_KEY:
                logger.warning("OPENROUTER_API_KEY未配置，但继续请求")
                # 我们将让OpenRouterClient处理此问题并返回友好的错误消息

            model = OpenRouterClient()
            model_kwargs = {
                "model": request.model,
                "stream": True,
                "temperature": model_config["temperature"]
            }
            # 仅在模型配置中存在top_p时才添加top_p
            if "top_p" in model_config:
                model_kwargs["top_p"] = model_config["top_p"]

            api_kwargs = model.convert_inputs_to_api_kwargs(
                input=prompt,
                model_kwargs=model_kwargs,
                model_type=ModelType.LLM
            )
        elif request.provider == "openai":
            logger.info(f"使用Openai协议与模型: {request.model}")

            # 检查Openai API密钥是否已设置
            if not OPENAI_API_KEY:
                logger.warning("OPENAI_API_KEY未配置，但继续请求")
                # 我们将让OpenAIClient处理此问题并返回错误消息

            # 初始化Openai客户端
            model = OpenAIClient()
            model_kwargs = {
                "model": request.model,
                "stream": True,
                "temperature": model_config["temperature"]
            }
            # 仅在模型配置中存在top_p时才添加top_p
            if "top_p" in model_config:
                model_kwargs["top_p"] = model_config["top_p"]

            api_kwargs = model.convert_inputs_to_api_kwargs(
                input=prompt,
                model_kwargs=model_kwargs,
                model_type=ModelType.LLM
            )
        elif request.provider == "azure":
            logger.info(f"使用Azure AI与模型: {request.model}")

            # 初始化Azure AI客户端
            model = AzureAIClient()
            model_kwargs = {
                "model": request.model,
                "stream": True,
                "temperature": model_config["temperature"],
                "top_p": model_config["top_p"]
            }

            api_kwargs = model.convert_inputs_to_api_kwargs(
                input=prompt,
                model_kwargs=model_kwargs,
                model_type=ModelType.LLM
            )
        elif request.provider == "dashscope":
            logger.info(f"使用Dashscope与模型: {request.model}")

            # 初始化Dashscope客户端
            model = DashscopeClient()
            model_kwargs = {
                "model": request.model,
                "stream": True,
                "temperature": model_config["temperature"],
                "top_p": model_config["top_p"]
            }

            api_kwargs = model.convert_inputs_to_api_kwargs(
                input=prompt,
                model_kwargs=model_kwargs,
                model_type=ModelType.LLM
            )
        else:
            # 初始化Google Generative AI模型
            model = genai.GenerativeModel(
                model_name=model_config["model"],
                generation_config={
                    "temperature": model_config["temperature"],
                    "top_p": model_config["top_p"],
                    "top_k": model_config["top_k"]
                }
            )

        # 根据提供商处理响应
        try:
            if request.provider == "ollama":
                # 获取响应并使用之前创建的api_kwargs处理它
                response = await model.acall(api_kwargs=api_kwargs, model_type=ModelType.LLM)
                # 处理Ollama的流式响应
                async for chunk in response:
                    text = getattr(chunk, 'response', None) or getattr(chunk, 'text', None) or str(chunk)
                    if text and not text.startswith('model=') and not text.startswith('created_at='):
                        text = text.replace('<think>', '').replace('</think>', '')
                        await websocket.send_text(text)
                # 显式关闭WebSocket连接，响应完成后
                await websocket.close()
            elif request.provider == "openrouter":
                try:
                    # 获取响应并使用之前创建的api_kwargs处理它
                    logger.info("Making OpenRouter API call")
                    response = await model.acall(api_kwargs=api_kwargs, model_type=ModelType.LLM)
                    # 处理OpenRouter的流式响应
                    async for chunk in response:
                        await websocket.send_text(chunk)
                    # 显式关闭WebSocket连接，响应完成后
                    await websocket.close()
                except Exception as e_openrouter:
                    logger.error(f"OpenRouter API错误: {str(e_openrouter)}")
                    error_msg = f"\nOpenRouter API错误: {str(e_openrouter)}\n\n请检查是否已设置OPENROUTER_API_KEY环境变量，并使用有效的API密钥。"
                    await websocket.send_text(error_msg)
                    # 在发送错误消息后关闭WebSocket连接
                    await websocket.close()
            elif request.provider == "openai":
                try:
                    # 获取响应并使用之前创建的api_kwargs处理它
                    logger.info("Making Openai API call")
                    response = await model.acall(api_kwargs=api_kwargs, model_type=ModelType.LLM)
                    # 处理Openai的流式响应
                    async for chunk in response:
                        choices = getattr(chunk, "choices", [])
                        if len(choices) > 0:
                            delta = getattr(choices[0], "delta", None)
                            if delta is not None:
                                text = getattr(delta, "content", None)
                                if text is not None:
                                    await websocket.send_text(text)
                    # 显式关闭WebSocket连接，响应完成后
                    await websocket.close()
                except Exception as e_openai:
                    logger.error(f"Openai API错误: {str(e_openai)}")
                    error_msg = f"\nOpenai API错误: {str(e_openai)}\n\n请检查是否已设置OPENAI_API_KEY环境变量，并使用有效的API密钥。"
                    await websocket.send_text(error_msg)
                    # 在发送错误消息后关闭WebSocket连接
                    await websocket.close()
            elif request.provider == "azure":
                try:
                    # 获取响应并使用之前创建的api_kwargs处理它
                    logger.info("Making Azure AI API call")
                    response = await model.acall(api_kwargs=api_kwargs, model_type=ModelType.LLM)
                    # 处理Azure AI的流式响应
                    async for chunk in response:
                        choices = getattr(chunk, "choices", [])
                        if len(choices) > 0:
                            delta = getattr(choices[0], "delta", None)
                            if delta is not None:
                                text = getattr(delta, "content", None)
                                if text is not None:
                                    await websocket.send_text(text)
                    # 显式关闭WebSocket连接，响应完成后
                    await websocket.close()
                except Exception as e_azure:
                    logger.error(f"Azure AI API错误: {str(e_azure)}")
                    error_msg = f"\nAzure AI API错误: {str(e_azure)}\n\n请检查是否已设置AZURE_OPENAI_API_KEY、AZURE_OPENAI_ENDPOINT和AZURE_OPENAI_VERSION环境变量，并使用有效的值。"
                    await websocket.send_text(error_msg)
                    # 在发送错误消息后关闭WebSocket连接
                    await websocket.close()
            else:
                # 生成流式响应
                response = model.generate_content(prompt, stream=True)
                # 流式响应
                for chunk in response:
                    if hasattr(chunk, 'text'):
                        await websocket.send_text(chunk.text)
                # 显式关闭WebSocket连接，响应完成后
                await websocket.close()

        except Exception as e_outer:
            logger.error(f"流式响应错误: {str(e_outer)}")
            error_message = str(e_outer)

            # 检查token限制错误
            if "maximum context length" in error_message or "token limit" in error_message or "too many tokens" in error_message:
                # 如果遇到token限制错误，则尝试不使用上下文重试
                logger.warning("token限制超出，尝试不使用上下文重试")
                try:
                    # 创建一个简化的提示，不包含上下文
                    simplified_prompt = f"/no_think {system_prompt}\n\n"
                    if conversation_history:
                        simplified_prompt += f"<conversation_history>\n{conversation_history}</conversation_history>\n\n"

                    # 如果filePath已提供且已检索，则将文件内容添加到回退提示中
                    if request.filePath and file_content:
                        simplified_prompt += f"<currentFileContent path=\"{request.filePath}\">\n{file_content}\n</currentFileContent>\n\n"

                    simplified_prompt += "<note>回答时未使用检索增强，因为输入大小限制。</note>\n\n"
                    simplified_prompt += f"<query>\n{query}\n</query>\n\nAssistant: "

                    if request.provider == "ollama":
                        simplified_prompt += " /no_think"

                        # 创建新的api_kwargs，使用简化的提示
                        fallback_api_kwargs = model.convert_inputs_to_api_kwargs(
                            input=simplified_prompt,
                            model_kwargs=model_kwargs,
                            model_type=ModelType.LLM
                        )

                        # 使用简化的提示获取响应
                        fallback_response = await model.acall(api_kwargs=fallback_api_kwargs, model_type=ModelType.LLM)

                        # 处理Ollama的流式fallback_response
                        async for chunk in fallback_response:
                            text = getattr(chunk, 'response', None) or getattr(chunk, 'text', None) or str(chunk)
                            if text and not text.startswith('model=') and not text.startswith('created_at='):
                                text = text.replace('<think>', '').replace('</think>', '')
                                await websocket.send_text(text)
                    elif request.provider == "openrouter":
                        try:
                            # 创建新的api_kwargs，使用简化的提示
                            fallback_api_kwargs = model.convert_inputs_to_api_kwargs(
                                input=simplified_prompt,
                                model_kwargs=model_kwargs,
                                model_type=ModelType.LLM
                            )

                            # 使用简化的提示获取响应
                            logger.info("Making fallback OpenRouter API call")
                            fallback_response = await model.acall(api_kwargs=fallback_api_kwargs, model_type=ModelType.LLM)

                            # 处理OpenRouter的流式fallback_response
                            async for chunk in fallback_response:
                                await websocket.send_text(chunk)
                        except Exception as e_fallback:
                            logger.error(f"OpenRouter API回退错误: {str(e_fallback)}")
                            error_msg = f"\nOpenRouter API回退错误: {str(e_fallback)}\n\n请检查是否已设置OPENROUTER_API_KEY环境变量，并使用有效的API密钥。"
                            await websocket.send_text(error_msg)
                    elif request.provider == "openai":
                        try:
                            # 创建新的api_kwargs，使用简化的提示
                            fallback_api_kwargs = model.convert_inputs_to_api_kwargs(
                                input=simplified_prompt,
                                model_kwargs=model_kwargs,
                                model_type=ModelType.LLM
                            )

                            # 使用简化的提示获取响应
                            logger.info("Making fallback Openai API call")
                            fallback_response = await model.acall(api_kwargs=fallback_api_kwargs, model_type=ModelType.LLM)

                            # 处理Openai的流式fallback_response
                            async for chunk in fallback_response:
                                text = chunk if isinstance(chunk, str) else getattr(chunk, 'text', str(chunk))
                                await websocket.send_text(text)
                        except Exception as e_fallback:
                            logger.error(f"Openai API回退错误: {str(e_fallback)}")
                            error_msg = f"\nOpenai API回退错误: {str(e_fallback)}\n\n请检查是否已设置OPENAI_API_KEY环境变量，并使用有效的API密钥。"
                            await websocket.send_text(error_msg)
                    elif request.provider == "azure":
                        try:
                            # 创建新的api_kwargs，使用简化的提示
                            fallback_api_kwargs = model.convert_inputs_to_api_kwargs(
                                input=simplified_prompt,
                                model_kwargs=model_kwargs,
                                model_type=ModelType.LLM
                            )

                            # 使用简化的提示获取响应
                            logger.info("Making fallback Azure AI API call")
                            fallback_response = await model.acall(api_kwargs=fallback_api_kwargs, model_type=ModelType.LLM)

                            # 处理Azure AI的流式fallback response
                            async for chunk in fallback_response:
                                choices = getattr(chunk, "choices", [])
                                if len(choices) > 0:
                                    delta = getattr(choices[0], "delta", None)
                                    if delta is not None:
                                        text = getattr(delta, "content", None)
                                        if text is not None:
                                            await websocket.send_text(text)
                        except Exception as e_fallback:
                            logger.error(f"Azure AI API回退错误: {str(e_fallback)}")
                            error_msg = f"\nAzure AI API回退错误: {str(e_fallback)}\n\n请检查是否已设置AZURE_OPENAI_API_KEY、AZURE_OPENAI_ENDPOINT和AZURE_OPENAI_VERSION环境变量，并使用有效的值。"
                            await websocket.send_text(error_msg)
                    else:
                        # 初始化Google Generative AI模型
                        model_config = get_model_config(request.provider, request.model)
                        fallback_model = genai.GenerativeModel(
                            model_name=model_config["model"],
                            generation_config={
                                "temperature": model_config["model_kwargs"].get("temperature", 0.7),
                                "top_p": model_config["model_kwargs"].get("top_p", 0.8),
                                "top_k": model_config["model_kwargs"].get("top_k", 40)
                            }
                        )

                        # 使用简化的提示获取流式响应
                        fallback_response = fallback_model.generate_content(simplified_prompt, stream=True)
                        # 流式响应
                        for chunk in fallback_response:
                            if hasattr(chunk, 'text'):
                                await websocket.send_text(chunk.text)
                except Exception as e2:
                    logger.error(f"回退流式响应错误: {str(e2)}")
                    await websocket.send_text(f"\n抱歉，您的请求太大，我无法处理。请尝试更短的查询或将其拆分为更小的部分。")
                    # 在发送错误消息后关闭WebSocket连接
                    await websocket.close()
            else:
                # 对于其他错误，返回错误消息
                await websocket.send_text(f"\n错误: {error_message}")
                # 在发送错误消息后关闭WebSocket连接
                await websocket.close()

    except WebSocketDisconnect:
        logger.info("WebSocket断开连接")
    except Exception as e:
        logger.error(f"WebSocket处理器错误: {str(e)}")
        try:
            await websocket.send_text(f"错误: {str(e)}")
            await websocket.close()
        except:
            pass
