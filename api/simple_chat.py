import logging
import os
from typing import List, Optional
from urllib.parse import unquote

import google.generativeai as genai
from adalflow.components.model_client.ollama_client import OllamaClient
from adalflow.core.types import ModelType
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from api.config import get_model_config, configs, OPENROUTER_API_KEY, OPENAI_API_KEY, AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY
from api.data_pipeline import count_tokens, get_file_content
from api.openai_client import OpenAIClient
from api.openrouter_client import OpenRouterClient
from api.bedrock_client import BedrockClient
from api.azureai_client import AzureAIClient
from api.rag import RAG
from api.prompts import (
    DEEP_RESEARCH_FIRST_ITERATION_PROMPT,
    DEEP_RESEARCH_FINAL_ITERATION_PROMPT,
    DEEP_RESEARCH_INTERMEDIATE_ITERATION_PROMPT,
    SIMPLE_CHAT_SYSTEM_PROMPT
)

# 配置日志记录
from api.logging_config import setup_logging

setup_logging()
logger = logging.getLogger(__name__)


# 初始化FastAPI应用
app = FastAPI(
    title="简单聊天API",
    description="用于流式聊天完成的简化API"
)

# 配置CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 允许所有来源
    allow_credentials=True,
    allow_methods=["*"],  # 允许所有方法
    allow_headers=["*"],  # 允许所有头部
)

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
    provider: str = Field("google", description="模型提供商（google、openai、openrouter、ollama、bedrock、azure）")
    model: Optional[str] = Field(None, description="指定提供商的模型名称")

    language: Optional[str] = Field("en", description="内容生成的语言（例如，'en'、'ja'、'zh'、'es'、'kr'、'vi'）")
    excluded_dirs: Optional[str] = Field(None, description="要从处理中排除的目录的逗号分隔列表")
    excluded_files: Optional[str] = Field(None, description="要从处理中排除的文件模式的逗号分隔列表")
    included_dirs: Optional[str] = Field(None, description="要专门包含的目录的逗号分隔列表")
    included_files: Optional[str] = Field(None, description="要专门包含的文件模式的逗号分隔列表")

@app.post("/chat/completions/stream")
async def chat_completions_stream(request: ChatCompletionRequest):
    """
    使用 Google Generative AI 直接流式传输聊天完成响应。
    
    Args:
        request: 聊天完成请求对象
        
    Returns:
        StreamingResponse: 流式响应
    """
    try:
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
                raise HTTPException(status_code=500, detail="未找到有效的文档嵌入。这可能是由于嵌入大小不一致或文档处理期间的API错误。请重试或检查您的仓库内容。")
            else:
                logger.error(f"准备检索器时发生ValueError: {str(e)}")
                raise HTTPException(status_code=500, detail=f"准备检索器时发生错误: {str(e)}")
        except Exception as e:
            logger.error(f"准备检索器时发生错误: {str(e)}")
            # 检查特定嵌入相关错误
            if "All embeddings should be of the same size" in str(e):
                raise HTTPException(status_code=500, detail="检测到不一致的嵌入大小。某些文档可能未能正确嵌入。请重试。")
            else:
                raise HTTPException(status_code=500, detail=f"准备检索器时发生错误: {str(e)}")

        # 验证请求
        if not request.messages or len(request.messages) == 0:
            raise HTTPException(status_code=400, detail="未提供消息")

        last_message = request.messages[-1]
        if last_message.role != "user":
            raise HTTPException(status_code=400, detail="最后一条消息必须是用户消息")

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
                    # 用原始主题替换延续消息
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

                        # 以文件路径分组格式化上下文文本
                        context_parts = []
                        for file_path, docs in docs_by_file.items():
                            # 添加文件头，包含元数据
                            header = f"## 文件路径: {file_path}\n\n"
                            # 添加文档内容
                            content = "\n\n".join([doc.text for doc in docs])

                            context_parts.append(f"{header}{content}")

                        # 用清晰的分割连接所有部分
                        context_text = "\n\n" + "-" * 10 + "\n\n".join(context_parts)
                    else:
                        logger.warning("未从RAG检索到文档")
                except Exception as e:
                    logger.error(f"RAG检索时发生错误: {str(e)}")
                    # 如果发生错误，则继续不使用RAG

            except Exception as e:
                logger.error(f"检索文档时发生错误: {str(e)}")
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
                system_prompt = DEEP_RESEARCH_FIRST_ITERATION_PROMPT.format(
                    repo_type=repo_type,
                    repo_url=repo_url,
                    repo_name=repo_name,
                    language_name=language_name
                )
            elif is_final_iteration:
                system_prompt = DEEP_RESEARCH_FINAL_ITERATION_PROMPT.format(
                    repo_type=repo_type,
                    repo_url=repo_url,
                    repo_name=repo_name,
                    research_iteration=research_iteration,
                    language_name=language_name
                )
            else:
                system_prompt = DEEP_RESEARCH_INTERMEDIATE_ITERATION_PROMPT.format(
                    repo_type=repo_type,
                    repo_url=repo_url,
                    repo_name=repo_name,
                    research_iteration=research_iteration,
                    language_name=language_name
                )
        else:
            system_prompt = SIMPLE_CHAT_SYSTEM_PROMPT.format(
                repo_type=repo_type,
                repo_url=repo_url,
                repo_name=repo_name,
                language_name=language_name
            )

        # 如果提供了文件内容，则获取
        file_content = ""
        if request.filePath:
            try:
                file_content = get_file_content(request.repo_url, request.filePath, request.type, request.token)
                logger.info(f"成功检索文件内容: {request.filePath}")
            except Exception as e:
                logger.error(f"检索文件内容时发生错误: {str(e)}")
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
            # 在对话历史后添加文件内容到提示
            prompt += f"<currentFileContent path=\"{request.filePath}\">\n{file_content}\n</currentFileContent>\n\n"

        # 仅在上下文不为空时才包含上下文
        CONTEXT_START = "<START_OF_CONTEXT>"
        CONTEXT_END = "<END_OF_CONTEXT>"
        if context_text.strip():
            prompt += f"{CONTEXT_START}\n{context_text}\n{CONTEXT_END}\n\n"
        else:
            # 添加一条提示，说明我们跳过RAG，因为输入大小限制或是因为它是孤立的API
            logger.info("未从RAG获取上下文")
            prompt += "<note>由于输入大小限制或因为它是孤立的API，跳过检索增强。</note>\n\n"

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
        elif request.provider == "bedrock":
            logger.info(f"使用AWS Bedrock与模型: {request.model}")

            # 检查AWS凭证是否已设置
            if not AWS_ACCESS_KEY_ID or not AWS_SECRET_ACCESS_KEY:
                logger.warning("AWS_ACCESS_KEY_ID或AWS_SECRET_ACCESS_KEY未配置，但继续请求")
                # 我们将让BedrockClient处理此问题并返回错误消息

            # 初始化Bedrock客户端
            model = BedrockClient()
            model_kwargs = {
                "model": request.model,
                "temperature": model_config["temperature"],
                "top_p": model_config["top_p"]
            }

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

        # 创建流式响应
        async def response_stream():
            try:
                if request.provider == "ollama":
                    # 获取响应并使用之前创建的api_kwargs处理
                    response = await model.acall(api_kwargs=api_kwargs, model_type=ModelType.LLM)
                    # 处理Ollama的流式响应
                    async for chunk in response:
                        text = getattr(chunk, 'response', None) or getattr(chunk, 'text', None) or str(chunk)
                        if text and not text.startswith('model=') and not text.startswith('created_at='):
                            text = text.replace('<think>', '').replace('</think>', '')
                            yield text
                elif request.provider == "openrouter":
                    try:
                        # 获取响应并使用之前创建的api_kwargs处理
                        logger.info("Making OpenRouter API call")
                        response = await model.acall(api_kwargs=api_kwargs, model_type=ModelType.LLM)
                        # 处理OpenRouter的流式响应
                        async for chunk in response:
                            yield chunk
                    except Exception as e_openrouter:
                        logger.error(f"OpenRouter API错误: {str(e_openrouter)}")
                        yield f"\nOpenRouter API错误: {str(e_openrouter)}\n\n请检查是否已设置OPENROUTER_API_KEY环境变量，并使用有效的API密钥。"
                elif request.provider == "openai":
                    try:
                        # 获取响应并使用之前创建的api_kwargs处理
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
                                        yield text
                    except Exception as e_openai:
                        logger.error(f"Openai API错误: {str(e_openai)}")
                        yield f"\nOpenai API错误: {str(e_openai)}\n\n请检查是否已设置OPENAI_API_KEY环境变量，并使用有效的API密钥。"
                elif request.provider == "bedrock":
                    try:
                        # 获取响应并使用之前创建的api_kwargs处理
                        logger.info("Making AWS Bedrock API call")
                        response = await model.acall(api_kwargs=api_kwargs, model_type=ModelType.LLM)
                        # 处理Bedrock响应（尚未流式化）
                        if isinstance(response, str):
                            yield response
                        else:
                            # 尝试从响应中提取文本
                            yield str(response)
                    except Exception as e_bedrock:
                        logger.error(f"AWS Bedrock API错误: {str(e_bedrock)}")
                        yield f"\nAWS Bedrock API错误: {str(e_bedrock)}\n\n请检查是否已设置AWS_ACCESS_KEY_ID和AWS_SECRET_ACCESS_KEY环境变量，并使用有效的凭证。"
                elif request.provider == "azure":
                    try:
                        # 获取响应并使用之前创建的api_kwargs处理
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
                                        yield text
                    except Exception as e_azure:
                        logger.error(f"Azure AI API错误: {str(e_azure)}")
                        yield f"\nAzure AI API错误: {str(e_azure)}\n\n请检查是否已设置AZURE_OPENAI_API_KEY、AZURE_OPENAI_ENDPOINT和AZURE_OPENAI_VERSION环境变量，并使用有效的值。"
                else:
                    # 生成流式响应
                    response = model.generate_content(prompt, stream=True)
                    # 流式响应
                    for chunk in response:
                        if hasattr(chunk, 'text'):
                            yield chunk.text

            except Exception as e_outer:
                logger.error(f"流式响应时发生错误: {str(e_outer)}")
                error_message = str(e_outer)

                # 检查token限制错误
                if "maximum context length" in error_message or "token limit" in error_message or "too many tokens" in error_message:
                    # 如果遇到token限制错误，则尝试不使用上下文重试
                    logger.warning("超出token限制，尝试不使用上下文重试")
                    try:
                        # 创建一个简化的提示，不包含上下文
                        simplified_prompt = f"/no_think {system_prompt}\n\n"
                        if conversation_history:
                            simplified_prompt += f"<conversation_history>\n{conversation_history}</conversation_history>\n\n"

                        # 如果filePath已检索且文件内容存在，则在回退提示中包含文件内容
                        if request.filePath and file_content:
                            simplified_prompt += f"<currentFileContent path=\"{request.filePath}\">\n{file_content}\n</currentFileContent>\n\n"

                        simplified_prompt += "<note>由于输入大小限制，跳过检索增强。</note>\n\n"
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

                            # 处理Ollama的回退响应流式化
                            async for chunk in fallback_response:
                                text = getattr(chunk, 'response', None) or getattr(chunk, 'text', None) or str(chunk)
                                if text and not text.startswith('model=') and not text.startswith('created_at='):
                                    text = text.replace('<think>', '').replace('</think>', '')
                                    yield text
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

                                # 处理OpenRouter的回退响应流式化
                                async for chunk in fallback_response:
                                    yield chunk
                            except Exception as e_fallback:
                                logger.error(f"OpenRouter API回退错误: {str(e_fallback)}")
                                yield f"\nOpenRouter API回退错误: {str(e_fallback)}\n\n请检查是否已设置OPENROUTER_API_KEY环境变量，并使用有效的API密钥。"
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

                                # 处理Openai的回退响应流式化
                                async for chunk in fallback_response:
                                    text = chunk if isinstance(chunk, str) else getattr(chunk, 'text', str(chunk))
                                    yield text
                            except Exception as e_fallback:
                                logger.error(f"Openai API回退错误: {str(e_fallback)}")
                                yield f"\nOpenai API回退错误: {str(e_fallback)}\n\n请检查是否已设置OPENAI_API_KEY环境变量，并使用有效的API密钥。"
                        elif request.provider == "bedrock":
                            try:
                                # 创建新的api_kwargs，使用简化的提示
                                fallback_api_kwargs = model.convert_inputs_to_api_kwargs(
                                    input=simplified_prompt,
                                    model_kwargs=model_kwargs,
                                    model_type=ModelType.LLM
                                )

                                # 使用简化的提示获取响应
                                logger.info("Making fallback AWS Bedrock API call")
                                fallback_response = await model.acall(api_kwargs=fallback_api_kwargs, model_type=ModelType.LLM)

                                # 处理Bedrock响应
                                if isinstance(fallback_response, str):
                                    yield fallback_response
                                else:
                                    # 尝试从响应中提取文本
                                    yield str(fallback_response)
                            except Exception as e_fallback:
                                logger.error(f"AWS Bedrock API回退错误: {str(e_fallback)}")
                                yield f"\nAWS Bedrock API回退错误: {str(e_fallback)}\n\n请检查是否已设置AWS_ACCESS_KEY_ID和AWS_SECRET_ACCESS_KEY环境变量，并使用有效的凭证。"
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

                                # 处理Azure AI的回退响应流式化
                                async for chunk in fallback_response:
                                    choices = getattr(chunk, "choices", [])
                                    if len(choices) > 0:
                                        delta = getattr(choices[0], "delta", None)
                                        if delta is not None:
                                            text = getattr(delta, "content", None)
                                            if text is not None:
                                                yield text
                            except Exception as e_fallback:
                                logger.error(f"Azure AI API回退错误: {str(e_fallback)}")
                                yield f"\nAzure AI API回退错误: {str(e_fallback)}\n\n请检查是否已设置AZURE_OPENAI_API_KEY、AZURE_OPENAI_ENDPOINT和AZURE_OPENAI_VERSION环境变量，并使用有效的值。"
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
                            # 流式回退响应
                            for chunk in fallback_response:
                                if hasattr(chunk, 'text'):
                                    yield chunk.text
                    except Exception as e2:
                        logger.error(f"回退流式响应时发生错误: {str(e2)}")
                        yield f"\n抱歉，您的请求太大，我无法处理。请尝试更短的查询或将其拆分为更小的部分。"
                else:
                    # 对于其他错误，返回错误消息
                    yield f"\n错误: {error_message}"

        # 返回流式响应
        return StreamingResponse(response_stream(), media_type="text/event-stream")

    except HTTPException:
        raise
    except Exception as e_handler:
        error_msg = f"流式聊天完成时发生错误: {str(e_handler)}"
        logger.error(error_msg)
        raise HTTPException(status_code=500, detail=error_msg)

@app.get("/")
async def root():
    """
    根端点，用于检查API是否运行
    """
    return {"status": "API is running", "message": "Navigate to /docs for API documentation"}
