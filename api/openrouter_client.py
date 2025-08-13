"""OpenRouter模型客户端集成。"""

from typing import Dict, Sequence, Optional, Any, List
import logging
import json
import aiohttp
import requests
from requests.exceptions import RequestException, Timeout

from adalflow.core.model_client import ModelClient
from adalflow.core.types import (
    CompletionUsage,
    ModelType,
    GeneratorOutput,
)

log = logging.getLogger(__name__)

class OpenRouterClient(ModelClient):
    __doc__ = r"""OpenRouter API客户端的组件包装器。

    OpenRouter提供了一个统一的API，通过单一端点访问数百个AI模型。
    API与OpenAI的API格式兼容，只有一些小的差异。

    访问 https://openrouter.ai/docs 了解更多详情。

    示例:
        ```python
        from api.openrouter_client import OpenRouterClient

        client = OpenRouterClient()
        generator = adal.Generator(
            model_client=client,
            model_kwargs={"model": "openai/gpt-4o"}
        )
        ```
    """

    def __init__(self, *args, **kwargs) -> None:
        """
        初始化 OpenRouter 客户端。
        """
        super().__init__(*args, **kwargs)
        self.sync_client = self.init_sync_client()
        self.async_client = None  # 仅在需要时初始化异步客户端

    def init_sync_client(self):
        """
        初始化同步 OpenRouter 客户端。
        
        Returns:
            dict: 包含API密钥和基础URL的配置字典
        """
        from api.config import OPENROUTER_API_KEY
        api_key = OPENROUTER_API_KEY
        if not api_key:
            log.warning("OPENROUTER_API_KEY未配置")

        # OpenRouter没有专门的客户端库，所以我们将直接使用requests
        return {
            "api_key": api_key,
            "base_url": "https://openrouter.ai/api/v1"
        }

    def init_async_client(self):
        """
        初始化异步 OpenRouter 客户端。
        
        Returns:
            dict: 包含API密钥和基础URL的配置字典
        """
        from api.config import OPENROUTER_API_KEY
        api_key = OPENROUTER_API_KEY
        if not api_key:
            log.warning("OPENROUTER_API_KEY未配置")

        # 对于异步，我们将使用aiohttp
        return {
            "api_key": api_key,
            "base_url": "https://openrouter.ai/api/v1"
        }

    def convert_inputs_to_api_kwargs(
        self, input: Any, model_kwargs: Dict = None, model_type: ModelType = None
    ) -> Dict:
        """
        将 AdalFlow 输入转换为 OpenRouter API 格式。
        
        Args:
            input: 输入数据
            model_kwargs: 模型参数
            model_type: 模型类型
            
        Returns:
            dict: 转换后的API参数
        """
        model_kwargs = model_kwargs or {}

        if model_type == ModelType.LLM:
            # 处理LLM生成
            messages = []

            # 如果输入是字符串，则转换为消息格式
            if isinstance(input, str):
                messages = [{"role": "user", "content": input}]
            elif isinstance(input, list) and all(isinstance(msg, dict) for msg in input):
                messages = input
            else:
                raise ValueError(f"OpenRouter不支持的输入格式: {type(input)}")

            # 用于调试
            log.info(f"OpenRouter的消息: {messages}")

            api_kwargs = {
                "messages": messages,
                **model_kwargs
            }

            # 确保模型已指定
            if "model" not in api_kwargs:
                api_kwargs["model"] = "openai/gpt-3.5-turbo"

            return api_kwargs

        elif model_type == ModelType.EMBEDDING:
            # OpenRouter不直接支持嵌入
            # 我们可以在OpenRouter中通过特定的模型使用嵌入，
            # 但目前，我们将抛出错误
            raise NotImplementedError("OpenRouter客户端不支持嵌入")

        else:
            raise ValueError(f"不支持的模型类型: {model_type}")

    async def acall(self, api_kwargs: Dict = None, model_type: ModelType = None) -> Any:
        """
        对 OpenRouter API 进行异步调用。
        """
        if not self.async_client:
            self.async_client = self.init_async_client()

        # 检查API密钥是否已设置
        if not self.async_client.get("api_key"):
            error_msg = "OPENROUTER_API_KEY未配置。请设置此环境变量以使用OpenRouter。"
            log.error(error_msg)
            # 相反，我们返回一个生成器，该生成器生成错误消息
            # 这允许错误消息显示在流式响应中
            async def error_generator():
                yield error_msg
            return error_generator()

        api_kwargs = api_kwargs or {}

        if model_type == ModelType.LLM:
            # 准备标头
            headers = {
                "Authorization": f"Bearer {self.async_client['api_key']}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://github.com/AsyncFuncAI/deepwiki-open",  # 可选
                "X-Title": "DeepWiki"  # 可选
            }

            # 始终使用OpenRouter的非流式模式
            api_kwargs["stream"] = False

            # 进行API调用
            try:
                log.info(f"正在异步调用OpenRouter API到{self.async_client['base_url']}/chat/completions")
                log.info(f"请求头: {headers}")
                log.info(f"请求体: {api_kwargs}")

                async with aiohttp.ClientSession() as session:
                    try:
                        async with session.post(
                            f"{self.async_client['base_url']}/chat/completions",
                            headers=headers,
                            json=api_kwargs,
                            timeout=60
                        ) as response:
                            if response.status != 200:
                                error_text = await response.text()
                                log.error(f"OpenRouter API错误 ({response.status}): {error_text}")

                                # 返回一个生成器，该生成器生成错误消息
                                async def error_response_generator():
                                    yield f"OpenRouter API错误 ({response.status}): {error_text}"
                                return error_response_generator()

                            # 获取完整响应
                            data = await response.json()
                            log.info(f"从OpenRouter接收到的响应: {data}")

                            # 创建一个生成器，该生成器生成内容
                            async def content_generator():
                                if "choices" in data and len(data["choices"]) > 0:
                                    choice = data["choices"][0]
                                    if "message" in choice and "content" in choice["message"]:
                                        content = choice["message"]["content"]
                                        log.info("成功检索到响应")

                                        # 检查内容是否为XML并确保其格式正确
                                        if content.strip().startswith("<") and ">" in content:
                                            # 它可能是XML，让我们确保其格式正确
                                            try:
                                                # 提取XML内容
                                                xml_content = content

                                                # 检查是否是wiki_structure XML
                                                if "<wiki_structure>" in xml_content:
                                                    log.info("找到wiki_structure XML，确保格式正确")

                                                    # 提取仅wiki_structure XML
                                                    import re
                                                    wiki_match = re.search(r'<wiki_structure>[\s\S]*?<\/wiki_structure>', xml_content)
                                                    if wiki_match:
                                                        # 获取原始XML
                                                        raw_xml = wiki_match.group(0)

                                                        # 清理XML，移除任何前导/尾随空格
                                                        # 并确保其格式正确
                                                        clean_xml = raw_xml.strip()

                                                        # 尝试修复常见的XML问题
                                                        try:
                                                            # 替换XML中的问题字符
                                                            fixed_xml = clean_xml

                                                            # 如果它不是实体的一部分，则将&替换为&amp;
                                                            fixed_xml = re.sub(r'&(?!amp;|lt;|gt;|apos;|quot;)', '&amp;', fixed_xml)

                                                            # 修复其他常见的XML问题
                                                            fixed_xml = fixed_xml.replace('</', '</').replace('  >', '>')

                                                            # 尝试解析修复后的XML
                                                            from xml.dom.minidom import parseString
                                                            dom = parseString(fixed_xml)

                                                            # 获取格式化良好的XML，带有适当的缩进
                                                            pretty_xml = dom.toprettyxml()

                                                            # 移除XML声明
                                                            if pretty_xml.startswith('<?xml'):
                                                                pretty_xml = pretty_xml[pretty_xml.find('?>')+2:].strip()

                                                            log.info(f"提取并验证XML: {pretty_xml[:100]}...")
                                                            yield pretty_xml
                                                        except Exception as xml_parse_error:
                                                            log.warning(f"XML验证失败: {str(xml_parse_error)}，使用原始XML")

                                                            # 如果XML验证失败，请尝试更激进的解决方案
                                                            try:
                                                                # 使用正则表达式仅提取结构，不包含任何问题字符
                                                                import re

                                                                # 提取基本结构
                                                                structure_match = re.search(r'<wiki_structure>(.*?)</wiki_structure>', clean_xml, re.DOTALL)
                                                                if structure_match:
                                                                    structure = structure_match.group(1).strip()

                                                                    # 重建一个干净的XML结构
                                                                    clean_structure = "<wiki_structure>\n"

                                                                    # 提取标题
                                                                    title_match = re.search(r'<title>(.*?)</title>', structure, re.DOTALL)
                                                                    if title_match:
                                                                        title = title_match.group(1).strip()
                                                                        clean_structure += f"  <title>{title}</title>\n"

                                                                    # 提取描述
                                                                    desc_match = re.search(r'<description>(.*?)</description>', structure, re.DOTALL)
                                                                    if desc_match:
                                                                        desc = desc_match.group(1).strip()
                                                                        clean_structure += f"  <description>{desc}</description>\n"

                                                                    # 添加页面部分
                                                                    clean_structure += "  <pages>\n"

                                                                    # 提取页面
                                                                    pages = re.findall(r'<page id="(.*?)">(.*?)</page>', structure, re.DOTALL)
                                                                    for page_id, page_content in pages:
                                                                        clean_structure += f'    <page id="{page_id}">\n'

                                                                        # 提取页面标题
                                                                        page_title_match = re.search(r'<title>(.*?)</title>', page_content, re.DOTALL)
                                                                        if page_title_match:
                                                                            page_title = page_title_match.group(1).strip()
                                                                            clean_structure += f"      <title>{page_title}</title>\n"

                                                                        # 提取页面描述
                                                                        page_desc_match = re.search(r'<description>(.*?)</description>', page_content, re.DOTALL)
                                                                        if page_desc_match:
                                                                            page_desc = page_desc_match.group(1).strip()
                                                                            clean_structure += f"      <description>{page_desc}</description>\n"

                                                                        # 提取重要性
                                                                        importance_match = re.search(r'<importance>(.*?)</importance>', page_content, re.DOTALL)
                                                                        if importance_match:
                                                                            importance = importance_match.group(1).strip()
                                                                            clean_structure += f"      <importance>{importance}</importance>\n"

                                                                        # 提取相关文件
                                                                        clean_structure += "      <relevant_files>\n"
                                                                        file_paths = re.findall(r'<file_path>(.*?)</file_path>', page_content, re.DOTALL)
                                                                        for file_path in file_paths:
                                                                            clean_structure += f"        <file_path>{file_path.strip()}</file_path>\n"
                                                                        clean_structure += "      </relevant_files>\n"

                                                                        # 提取相关页面
                                                                        clean_structure += "      <related_pages>\n"
                                                                        related_pages = re.findall(r'<related>(.*?)</related>', page_content, re.DOTALL)
                                                                        for related in related_pages:
                                                                            clean_structure += f"        <related>{related.strip()}</related>\n"
                                                                        clean_structure += "      </related_pages>\n"

                                                                        clean_structure += "    </page>\n"

                                                                    clean_structure += "  </pages>\n</wiki_structure>"

                                                                    log.info("成功重建干净的XML结构")
                                                                    yield clean_structure
                                                                else:
                                                                    log.warning("无法提取wiki结构，使用原始XML")
                                                                    yield clean_xml
                                                            except Exception as rebuild_error:
                                                                log.warning(f"重建XML失败: {str(rebuild_error)}，使用原始XML")
                                                                yield clean_xml
                                                    else:
                                                        # 如果我们无法提取它，只需生成原始内容
                                                        log.warning("无法提取wiki_structure XML，生成原始内容")
                                                        yield xml_content
                                                else:
                                                    # 对于其他XML内容，只需生成它
                                                    yield content
                                            except Exception as xml_error:
                                                log.error(f"处理XML内容时出错: {str(xml_error)}")
                                                yield content
                                        else:
                                            # 不是XML，只需生成内容
                                            yield content
                                    else:
                                        log.error(f"意外的响应格式: {data}")
                                        yield "Error: Unexpected response format from OpenRouter API"
                                else:
                                    log.error(f"响应中没有选择: {data}")
                                    yield "Error: No response content from OpenRouter API"

                            return content_generator()
                    except aiohttp.ClientError as e_client:
                        log.error(f"OpenRouter API连接错误: {str(e_client)}")

                        # 返回一个生成器，该生成器生成错误消息
                        async def connection_error_generator():
                            yield f"OpenRouter API连接错误: {str(e_client)}。请检查您的互联网连接并确保OpenRouter API可访问。"
                        return connection_error_generator()

            except RequestException as e_req:
                log.error(f"异步调用OpenRouter API时出错: {str(e_req)}")

                # 返回一个生成器，该生成器生成错误消息
                async def request_error_generator():
                    yield f"Error calling OpenRouter API: {str(e_req)}"
                return request_error_generator()

            except Exception as e_unexp:
                log.error(f"异步调用OpenRouter API时发生意外错误: {str(e_unexp)}")

                # 返回一个生成器，该生成器生成错误消息
                async def unexpected_error_generator():
                    yield f"Unexpected error calling OpenRouter API: {str(e_unexp)}"
                return unexpected_error_generator()

        else:
            error_msg = f"不支持的模型类型: {model_type}"
            log.error(error_msg)

            # 返回一个生成器，该生成器生成错误消息
            async def model_type_error_generator():
                yield error_msg
            return model_type_error_generator()

    def _process_completion_response(self, data: Dict) -> GeneratorOutput:
        """处理来自OpenRouter的非流式完成响应。"""
        try:
            # 从响应中提取完成文本
            if not data.get("choices"):
                raise ValueError(f"OpenRouter响应中没有选择: {data}")

            choice = data["choices"][0]

            if "message" in choice:
                content = choice["message"].get("content", "")
            elif "text" in choice:
                content = choice.get("text", "")
            else:
                raise ValueError(f"OpenRouter的意外响应格式: {choice}")

            # 如果可用，提取使用信息
            usage = None
            if "usage" in data:
                usage = CompletionUsage(
                    prompt_tokens=data["usage"].get("prompt_tokens", 0),
                    completion_tokens=data["usage"].get("completion_tokens", 0),
                    total_tokens=data["usage"].get("total_tokens", 0)
                )

            # 创建并返回GeneratorOutput
            return GeneratorOutput(
                data=content,
                usage=usage,
                raw_response=data
            )

        except Exception as e_proc:
            log.error(f"处理OpenRouter完成响应时出错: {str(e_proc)}")
            raise

    def _process_streaming_response(self, response):
        """
        处理来自 OpenRouter 的流式响应。
        """
        try:
            log.info("Starting to process streaming response from OpenRouter")
            buffer = ""

            for chunk in response.iter_content(chunk_size=1024, decode_unicode=True):
                try:
                    # Add chunk to buffer
                    buffer += chunk

                    # Process complete lines in the buffer
                    while '\n' in buffer:
                        line, buffer = buffer.split('\n', 1)
                        line = line.strip()

                        if not line:
                            continue

                        log.debug(f"Processing line: {line}")

                        # Skip SSE comments (lines starting with :)
                        if line.startswith(':'):
                            log.debug(f"Skipping SSE comment: {line}")
                            continue

                        if line.startswith("data: "):
                            data = line[6:]  # Remove "data: " prefix

                            # Check for stream end
                            if data == "[DONE]":
                                log.info("Received [DONE] marker")
                                break

                            try:
                                data_obj = json.loads(data)
                                log.debug(f"Parsed JSON data: {data_obj}")

                                # Extract content from delta
                                if "choices" in data_obj and len(data_obj["choices"]) > 0:
                                    choice = data_obj["choices"][0]

                                    if "delta" in choice and "content" in choice["delta"] and choice["delta"]["content"]:
                                        content = choice["delta"]["content"]
                                        log.debug(f"Yielding delta content: {content}")
                                        yield content
                                    elif "text" in choice:
                                        log.debug(f"Yielding text content: {choice['text']}")
                                        yield choice["text"]
                                    else:
                                        log.debug(f"No content found in choice: {choice}")
                                else:
                                    log.debug(f"No choices found in data: {data_obj}")

                            except json.JSONDecodeError:
                                log.warning(f"Failed to parse SSE data: {data}")
                                continue
                except Exception as e_chunk:
                    log.error(f"Error processing streaming chunk: {str(e_chunk)}")
                    yield f"Error processing response chunk: {str(e_chunk)}"
        except Exception as e_stream:
            log.error(f"Error in streaming response: {str(e_stream)}")
            yield f"Error in streaming response: {str(e_stream)}"

    async def _process_async_streaming_response(self, response):
        """
        处理来自 OpenRouter 的异步流式响应。
        """
        buffer = ""
        try:
            log.info("Starting to process async streaming response from OpenRouter")
            async for chunk in response.content:
                try:
                    # Convert bytes to string and add to buffer
                    if isinstance(chunk, bytes):
                        chunk_str = chunk.decode('utf-8')
                    else:
                        chunk_str = str(chunk)

                    buffer += chunk_str

                    # Process complete lines in the buffer
                    while '\n' in buffer:
                        line, buffer = buffer.split('\n', 1)
                        line = line.strip()

                        if not line:
                            continue

                        log.debug(f"Processing line: {line}")

                        # Skip SSE comments (lines starting with :)
                        if line.startswith(':'):
                            log.debug(f"Skipping SSE comment: {line}")
                            continue

                        if line.startswith("data: "):
                            data = line[6:]  # Remove "data: " prefix

                            # Check for stream end
                            if data == "[DONE]":
                                log.info("Received [DONE] marker")
                                break

                            try:
                                data_obj = json.loads(data)
                                log.debug(f"Parsed JSON data: {data_obj}")

                                # Extract content from delta
                                if "choices" in data_obj and len(data_obj["choices"]) > 0:
                                    choice = data_obj["choices"][0]

                                    if "delta" in choice and "content" in choice["delta"] and choice["delta"]["content"]:
                                        content = choice["delta"]["content"]
                                        log.debug(f"Yielding delta content: {content}")
                                        yield content
                                    elif "text" in choice:
                                        log.debug(f"Yielding text content: {choice['text']}")
                                        yield choice["text"]
                                    else:
                                        log.debug(f"No content found in choice: {choice}")
                                else:
                                    log.debug(f"No choices found in data: {data_obj}")

                            except json.JSONDecodeError:
                                log.warning(f"Failed to parse SSE data: {data}")
                                continue
                except Exception as e_chunk:
                    log.error(f"Error processing streaming chunk: {str(e_chunk)}")
                    yield f"Error processing response chunk: {str(e_chunk)}"
        except Exception as e_stream:
            log.error(f"Error in async streaming response: {str(e_stream)}")
            yield f"Error in streaming response: {str(e_stream)}"
