"""OpenAI模型客户端集成。"""

import os
import base64
from typing import (
    Dict,
    Sequence,
    Optional,
    List,
    Any,
    TypeVar,
    Callable,
    Generator,
    Union,
    Literal,
)
import re

import logging
import backoff

# 可选导入
from adalflow.utils.lazy_import import safe_import, OptionalPackages
from openai.types.chat.chat_completion import Choice

openai = safe_import(OptionalPackages.OPENAI.value[0], OptionalPackages.OPENAI.value[1])

from openai import OpenAI, AsyncOpenAI, Stream
from openai import (
    APITimeoutError,
    InternalServerError,
    RateLimitError,
    UnprocessableEntityError,
    BadRequestError,
)
from openai.types import (
    Completion,
    CreateEmbeddingResponse,
    Image,
)
from openai.types.chat import ChatCompletionChunk, ChatCompletion, ChatCompletionMessage

from adalflow.core.model_client import ModelClient
from adalflow.core.types import (
    ModelType,
    EmbedderOutput,
    TokenLogProb,
    CompletionUsage,
    GeneratorOutput,
)
from adalflow.components.model_client.utils import parse_embedding_response

log = logging.getLogger(__name__)
T = TypeVar("T")


# 完成解析函数，您可以将它们组合成一个单一的聊天完成解析器
def get_first_message_content(completion: ChatCompletion) -> str:
    """
    当只需要第一条消息的内容时使用。
    这是聊天完成的默认解析器。
    
    Args:
        completion: 聊天完成对象
        
    Returns:
        str: 第一条消息的内容
    """
    log.debug(f"原始完成: {completion}")
    return completion.choices[0].message.content


# def _get_chat_completion_usage(completion: ChatCompletion) -> OpenAICompletionUsage:
#     return completion.usage


# 用于估算流式响应中token数量的简单启发式方法
def estimate_token_count(text: str) -> int:
    """
    估算给定文本的 token 数量。

    Args:
        text (str): 要估算 token 数量的文本。

    Returns:
        int: 估算的 token 数量。
    """
    # 使用空格作为简单启发式方法将文本分割成token
    tokens = text.split()

    # 返回token数量
    return len(tokens)


def parse_stream_response(completion: ChatCompletionChunk) -> str:
    """
    解析流式 API 的响应。
    
    Args:
        completion: 聊天完成块对象
        
    Returns:
        str: 解析后的内容
    """
    return completion.choices[0].delta.content


def handle_streaming_response(generator: Stream[ChatCompletionChunk]):
    """
    处理流式响应。
    
    Args:
        generator: 流式响应生成器
        
    Yields:
        str: 解析后的内容块
    """
    for completion in generator:
        log.debug(f"原始块完成: {completion}")
        parsed_content = parse_stream_response(completion)
        yield parsed_content


def get_all_messages_content(completion: ChatCompletion) -> List[str]:
    """
    当 n > 1 时，获取所有消息内容。
    
    Args:
        completion: 聊天完成对象
        
    Returns:
        List[str]: 所有消息内容的列表
    """
    return [c.message.content for c in completion.choices]


def get_probabilities(completion: ChatCompletion) -> List[List[TokenLogProb]]:
    """
    获取完成中每个 token 的概率。
    
    Args:
        completion: 聊天完成对象
        
    Returns:
        List[List[TokenLogProb]]: 每个选择的token概率列表
    """
    log_probs = []
    for c in completion.choices:
        content = c.logprobs.content
        print(content)
        log_probs_for_choice = []
        for openai_token_logprob in content:
            token = openai_token_logprob.token
            logprob = openai_token_logprob.logprob
            log_probs_for_choice.append(TokenLogProb(token=token, logprob=logprob))
        log_probs.append(log_probs_for_choice)
    return log_probs


class OpenAIClient(ModelClient):
    __doc__ = r"""OpenAI API客户端的组件包装器。

    支持嵌入和聊天完成API，包括多模态功能。

    用户可以：
    1. 通过将 `OpenAIClient()` 作为 `model_client` 传递来简化 ``Embedder`` 和 ``Generator`` 组件的使用。
    2. 使用此作为参考来创建自己的API客户端或通过复制和修改代码来扩展此类。

    注意：
        我们建议避免使用 `response_format` 来强制输出数据类型或在 `model_kwargs` 中使用 `tools` 和 `tool_choice`。
        OpenAI的内部格式化和添加的提示是未知的。相反：
        - 使用 :ref:`OutputParser<components-output_parsers>` 进行响应解析和格式化。

        对于多模态输入，在 `model_kwargs["images"]` 中提供图像作为路径、URL或它们的列表。
        模型必须支持视觉功能（例如，`gpt-4o`、`gpt-4o-mini`、`o1`、`o1-mini`）。

        对于图像生成，使用 `model_type=ModelType.IMAGE_GENERATION` 并提供：
        - model: `"dall-e-3"` 或 `"dall-e-2"`
        - prompt: 要生成的图像的文本描述
        - size: DALL-E 3 的 `"1024x1024"`、`"1024x1792"` 或 `"1792x1024"`；DALL-E 2 的 `"256x256"`、`"512x512"` 或 `"1024x1024"`
        - quality: `"standard"` 或 `"hd"`（仅限DALL-E 3）
        - n: 要生成的图像数量（DALL-E 3为1，DALL-E 2为1-10）
        - response_format: `"url"` 或 `"b64_json"`

    Args:
        api_key (Optional[str], optional): OpenAI API密钥。默认为 `None`。
        chat_completion_parser (Callable[[Completion], Any], optional): 将聊天完成解析为 `str` 的函数。默认为 `None`。
            默认解析器是 `get_first_message_content`。
        base_url (str): 初始化客户端时使用的API基础URL。
            默认为 `"https://api.openai.com"`，但可以为第三方API提供商或自托管模型自定义。
        env_api_key_name (str): API密钥的环境变量名称。默认为 `"OPENAI_API_KEY"`。

    参考：
        - OpenAI API概述: https://platform.openai.com/docs/introduction
        - 嵌入指南: https://platform.openai.com/docs/guides/embeddings
        - 聊天完成模型: https://platform.openai.com/docs/guides/text-generation
        - 视觉模型: https://platform.openai.com/docs/guides/vision
        - 图像生成: https://platform.openai.com/docs/guides/images
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        chat_completion_parser: Callable[[Completion], Any] = None,
        input_type: Literal["text", "messages"] = "text",
        base_url: Optional[str] = None,
        env_base_url_name: str = "OPENAI_BASE_URL",
        env_api_key_name: str = "OPENAI_API_KEY",
    ):
        r"""建议设置OPENAI_API_KEY环境变量而不是作为参数传递。

        Args:
            api_key (Optional[str], optional): OpenAI API密钥。默认为None。
            base_url (str): 初始化客户端时使用的API基础URL。
            env_api_key_name (str): API密钥的环境变量名称。默认为 `"OPENAI_API_KEY"`。
        """
        super().__init__()
        self._api_key = api_key
        self._env_api_key_name = env_api_key_name
        self._env_base_url_name = env_base_url_name
        self.base_url = base_url or os.getenv(self._env_base_url_name, "https://api.openai.com/v1")
        self.sync_client = self.init_sync_client()
        self.async_client = None  # 仅在调用异步调用时初始化
        self.chat_completion_parser = (
            chat_completion_parser or get_first_message_content
        )
        self._input_type = input_type
        self._api_kwargs = {}  # 当调用OpenAI客户端时添加api kwargs

    def init_sync_client(self):
        """
        初始化同步客户端。
        
        Returns:
            OpenAI: 配置好的 OpenAI 同步客户端
            
        Raises:
            ValueError: 如果未设置 API 密钥环境变量
        """
        api_key = self._api_key or os.getenv(self._env_api_key_name)
        if not api_key:
            raise ValueError(
                f"Environment variable {self._env_api_key_name} must be set"
            )
        return OpenAI(api_key=api_key, base_url=self.base_url)

    def init_async_client(self):
        """
        初始化异步客户端。
        
        Returns:
            AsyncOpenAI: 配置好的 OpenAI 异步客户端
            
        Raises:
            ValueError: 如果未设置 API 密钥环境变量
        """
        api_key = self._api_key or os.getenv(self._env_api_key_name)
        if not api_key:
            raise ValueError(
                f"Environment variable {self._env_api_key_name} must be set"
            )
        return AsyncOpenAI(api_key=api_key, base_url=self.base_url)

    # def _parse_chat_completion(self, completion: ChatCompletion) -> "GeneratorOutput":
    #     # TODO: raw output it is better to save the whole completion as a source of truth instead of just the message
    #     try:
    #         data = self.chat_completion_parser(completion)
    #         usage = self.track_completion_usage(completion)
    #         return GeneratorOutput(
    #             data=data, error=None, raw_response=str(data), usage=usage
    #         )
    #     except Exception as e:
    #         log.error(f"Error parsing the completion: {e}")
    #         return GeneratorOutput(data=None, error=str(e), raw_response=completion)

    def parse_chat_completion(
        self,
        completion: Union[ChatCompletion, Generator[ChatCompletionChunk, None, None]],
    ) -> "GeneratorOutput":
        """
        解析完成结果，并将其放入 raw_response。
        """
        log.debug(f"completion: {completion}, parser: {self.chat_completion_parser}")
        try:
            data = self.chat_completion_parser(completion)
        except Exception as e:
            log.error(f"Error parsing the completion: {e}")
            return GeneratorOutput(data=None, error=str(e), raw_response=completion)

        try:
            usage = self.track_completion_usage(completion)
            return GeneratorOutput(
                data=None, error=None, raw_response=data, usage=usage
            )
        except Exception as e:
            log.error(f"Error tracking the completion usage: {e}")
            return GeneratorOutput(data=None, error=str(e), raw_response=data)

    def track_completion_usage(
        self,
        completion: Union[ChatCompletion, Generator[ChatCompletionChunk, None, None]],
    ) -> CompletionUsage:
        """
        跟踪完成结果的使用情况。
        
        Returns:
            CompletionUsage: 包含 token 使用情况的对象
        """

        try:
            usage: CompletionUsage = CompletionUsage(
                completion_tokens=completion.usage.completion_tokens,
                prompt_tokens=completion.usage.prompt_tokens,
                total_tokens=completion.usage.total_tokens,
            )
            return usage
        except Exception as e:
            log.error(f"Error tracking the completion usage: {e}")
            return CompletionUsage(
                completion_tokens=None, prompt_tokens=None, total_tokens=None
            )

    def parse_embedding_response(
        self, response: CreateEmbeddingResponse
    ) -> EmbedderOutput:
        """
        将嵌入响应解析为 Adalflow 组件可以理解的结构。

        应该在 ``Embedder`` 中调用。
        """
        try:
            return parse_embedding_response(response)
        except Exception as e:
            log.error(f"Error parsing the embedding response: {e}")
            return EmbedderOutput(data=[], error=str(e), raw_response=response)

    def convert_inputs_to_api_kwargs(
        self,
        input: Optional[Any] = None,
        model_kwargs: Dict = {},
        model_type: ModelType = ModelType.UNDEFINED,
    ) -> Dict:
        r"""
        Specify the API input type and output api_kwargs that will be used in _call and _acall methods.
        Convert the Component's standard input, and system_input(chat model) and model_kwargs into API-specific format.
        For multimodal inputs, images can be provided in model_kwargs["images"] as a string path, URL, or list of them.
        The model specified in model_kwargs["model"] must support multimodal capabilities when using images.

        Args:
            input: The input text or messages to process
            model_kwargs: Additional parameters including:
                - images: Optional image source(s) as path, URL, or list of them
                - detail: Image detail level ('auto', 'low', or 'high'), defaults to 'auto'
                - model: The model to use (must support multimodal inputs if images are provided)
            model_type: The type of model (EMBEDDER or LLM)

        Returns:
            Dict: API-specific kwargs for the model call
        """

        final_model_kwargs = model_kwargs.copy()
        if model_type == ModelType.EMBEDDER:
            if isinstance(input, str):
                input = [input]
            # convert input to input
            if not isinstance(input, Sequence):
                raise TypeError("input must be a sequence of text")
            final_model_kwargs["input"] = input
        elif model_type == ModelType.LLM:
            # convert input to messages
            messages: List[Dict[str, str]] = []
            images = final_model_kwargs.pop("images", None)
            detail = final_model_kwargs.pop("detail", "auto")

            if self._input_type == "messages":
                system_start_tag = "<START_OF_SYSTEM_PROMPT>"
                system_end_tag = "<END_OF_SYSTEM_PROMPT>"
                user_start_tag = "<START_OF_USER_PROMPT>"
                user_end_tag = "<END_OF_USER_PROMPT>"

                # new regex pattern to ignore special characters such as \n
                pattern = (
                    rf"{system_start_tag}\s*(.*?)\s*{system_end_tag}\s*"
                    rf"{user_start_tag}\s*(.*?)\s*{user_end_tag}"
                )

                # Compile the regular expression

                # re.DOTALL is to allow . to match newline so that (.*?) does not match in a single line
                regex = re.compile(pattern, re.DOTALL)
                # Match the pattern
                match = regex.match(input)
                system_prompt, input_str = None, None

                if match:
                    system_prompt = match.group(1)
                    input_str = match.group(2)
                else:
                    print("No match found.")
                if system_prompt and input_str:
                    messages.append({"role": "system", "content": system_prompt})
                    if images:
                        content = [{"type": "text", "text": input_str}]
                        if isinstance(images, (str, dict)):
                            images = [images]
                        for img in images:
                            content.append(self._prepare_image_content(img, detail))
                        messages.append({"role": "user", "content": content})
                    else:
                        messages.append({"role": "user", "content": input_str})
            if len(messages) == 0:
                if images:
                    content = [{"type": "text", "text": input}]
                    if isinstance(images, (str, dict)):
                        images = [images]
                    for img in images:
                        content.append(self._prepare_image_content(img, detail))
                    messages.append({"role": "user", "content": content})
                else:
                    messages.append({"role": "user", "content": input})
            final_model_kwargs["messages"] = messages
        elif model_type == ModelType.IMAGE_GENERATION:
            # For image generation, input is the prompt
            final_model_kwargs["prompt"] = input
            # Ensure model is specified
            if "model" not in final_model_kwargs:
                raise ValueError("model must be specified for image generation")
            # Set defaults for DALL-E 3 if not specified
            final_model_kwargs["size"] = final_model_kwargs.get("size", "1024x1024")
            final_model_kwargs["quality"] = final_model_kwargs.get(
                "quality", "standard"
            )
            final_model_kwargs["n"] = final_model_kwargs.get("n", 1)
            final_model_kwargs["response_format"] = final_model_kwargs.get(
                "response_format", "url"
            )

            # Handle image edits and variations
            image = final_model_kwargs.get("image")
            if isinstance(image, str) and os.path.isfile(image):
                final_model_kwargs["image"] = self._encode_image(image)

            mask = final_model_kwargs.get("mask")
            if isinstance(mask, str) and os.path.isfile(mask):
                final_model_kwargs["mask"] = self._encode_image(mask)
        else:
            raise ValueError(f"model_type {model_type} is not supported")

        return final_model_kwargs

    def parse_image_generation_response(self, response: List[Image]) -> GeneratorOutput:
        """Parse the image generation response into a GeneratorOutput."""
        try:
            # Extract URLs or base64 data from the response
            data = [img.url or img.b64_json for img in response]
            # For single image responses, unwrap from list
            if len(data) == 1:
                data = data[0]
            return GeneratorOutput(
                data=data,
                raw_response=str(response),
            )
        except Exception as e:
            log.error(f"Error parsing image generation response: {e}")
            return GeneratorOutput(data=None, error=str(e), raw_response=str(response))

    @backoff.on_exception(
        backoff.expo,
        (
            APITimeoutError,
            InternalServerError,
            RateLimitError,
            UnprocessableEntityError,
            BadRequestError,
        ),
        max_time=5,
    )
    def call(self, api_kwargs: Dict = {}, model_type: ModelType = ModelType.UNDEFINED):
        """
        kwargs 是输入和 model_kwargs 的组合。支持流式调用。
        """
        log.info(f"api_kwargs: {api_kwargs}")
        self._api_kwargs = api_kwargs
        if model_type == ModelType.EMBEDDER:
            return self.sync_client.embeddings.create(**api_kwargs)
        elif model_type == ModelType.LLM:
            if "stream" in api_kwargs and api_kwargs.get("stream", False):
                log.debug("streaming call")
                self.chat_completion_parser = handle_streaming_response
                return self.sync_client.chat.completions.create(**api_kwargs)
            else:
                log.debug("non-streaming call converted to streaming")
                # Make a copy of api_kwargs to avoid modifying the original
                streaming_kwargs = api_kwargs.copy()
                streaming_kwargs["stream"] = True

                # Get streaming response
                stream_response = self.sync_client.chat.completions.create(**streaming_kwargs)

                # Accumulate all content from the stream
                accumulated_content = ""
                id = ""
                model = ""
                created = 0
                for chunk in stream_response:
                    id = getattr(chunk, "id", None) or id
                    model = getattr(chunk, "model", None) or model
                    created = getattr(chunk, "created", 0) or created
                    choices = getattr(chunk, "choices", [])
                    if len(choices) > 0:
                        delta = getattr(choices[0], "delta", None)
                        if delta is not None:
                            text = getattr(delta, "content", None)
                            if text is not None:
                                accumulated_content += text or ""
                # Return the mock completion object that will be processed by the chat_completion_parser
                return ChatCompletion(
                    id = id,
                    model=model,
                    created=created,
                    object="chat.completion",
                    choices=[Choice(
                        index=0,
                        finish_reason="stop",
                        message=ChatCompletionMessage(content=accumulated_content, role="assistant")
                    )]
                )
        elif model_type == ModelType.IMAGE_GENERATION:
            # Determine which image API to call based on the presence of image/mask
            if "image" in api_kwargs:
                if "mask" in api_kwargs:
                    # Image edit
                    response = self.sync_client.images.edit(**api_kwargs)
                else:
                    # Image variation
                    response = self.sync_client.images.create_variation(**api_kwargs)
            else:
                # Image generation
                response = self.sync_client.images.generate(**api_kwargs)
            return response.data
        else:
            raise ValueError(f"model_type {model_type} is not supported")

    @backoff.on_exception(
        backoff.expo,
        (
            APITimeoutError,
            InternalServerError,
            RateLimitError,
            UnprocessableEntityError,
            BadRequestError,
        ),
        max_time=5,
    )
    async def acall(
        self, api_kwargs: Dict = {}, model_type: ModelType = ModelType.UNDEFINED
    ):
        """
        kwargs 是输入和 model_kwargs 的组合
        """
        # store the api kwargs in the client
        self._api_kwargs = api_kwargs
        if self.async_client is None:
            self.async_client = self.init_async_client()
        if model_type == ModelType.EMBEDDER:
            return await self.async_client.embeddings.create(**api_kwargs)
        elif model_type == ModelType.LLM:
            return await self.async_client.chat.completions.create(**api_kwargs)
        elif model_type == ModelType.IMAGE_GENERATION:
            # Determine which image API to call based on the presence of image/mask
            if "image" in api_kwargs:
                if "mask" in api_kwargs:
                    # Image edit
                    response = await self.async_client.images.edit(**api_kwargs)
                else:
                    # Image variation
                    response = await self.async_client.images.create_variation(
                        **api_kwargs
                    )
            else:
                # Image generation
                response = await self.async_client.images.generate(**api_kwargs)
            return response.data
        else:
            raise ValueError(f"model_type {model_type} is not supported")

    @classmethod
    def from_dict(cls: type[T], data: Dict[str, Any]) -> T:
        obj = super().from_dict(data)
        # recreate the existing clients
        obj.sync_client = obj.init_sync_client()
        obj.async_client = obj.init_async_client()
        return obj

    def to_dict(self) -> Dict[str, Any]:
        r"""Convert the component to a dictionary."""
        # TODO: not exclude but save yes or no for recreating the clients
        exclude = [
            "sync_client",
            "async_client",
        ]  # unserializable object
        output = super().to_dict(exclude=exclude)
        return output

    def _encode_image(self, image_path: str) -> str:
        """
        将图像编码为base64字符串。

        Args:
            image_path: 图像文件路径。

        Returns:
            str: Base64编码的图像字符串。

        Raises:
            ValueError: 如果文件无法读取或不存在。
        """
        try:
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode("utf-8")
        except FileNotFoundError:
            raise ValueError(f"图像文件未找到: {image_path}")
        except PermissionError:
            raise ValueError(f"读取图像文件时权限被拒绝: {image_path}")
        except Exception as e:
            raise ValueError(f"编码图像 {image_path} 时出错: {str(e)}")

    def _prepare_image_content(
        self, image_source: Union[str, Dict[str, Any]], detail: str = "auto"
    ) -> Dict[str, Any]:
        """
        为API请求准备图像内容。

        Args:
            image_source: 本地图像路径或URL。
            detail: 图像细节级别（'auto'、'low' 或 'high'）。

        Returns:
            Dict[str, Any]: 格式化的图像内容，用于API请求。
        """
        if isinstance(image_source, str):
            if image_source.startswith(("http://", "https://")):
                return {
                    "type": "image_url",
                    "image_url": {"url": image_source, "detail": detail},
                }
            else:
                base64_image = self._encode_image(image_source)
                return {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}",
                        "detail": detail,
                    },
                }
        return image_source


# 使用示例:
if __name__ == "__main__":
    from adalflow.core import Generator
    from adalflow.utils import setup_env

    # log = get_logger(level="DEBUG")

    setup_env()
    prompt_kwargs = {"input_str": "What is the meaning of life?"}

    gen = Generator(
        model_client=OpenAIClient(),
        model_kwargs={"model": "gpt-4o", "stream": False},
    )
    gen_response = gen(prompt_kwargs)
    print(f"gen_response: {gen_response}")

    # for genout in gen_response.data:
    #     print(f"genout: {genout}")

    # 测试to_dict和from_dict是否正常工作
    # model_client = OpenAIClient()
    # model_client_dict = model_client.to_dict()
    # from_dict_model_client = OpenAIClient.from_dict(model_client_dict)
    # assert model_client_dict == from_dict_model_client.to_dict()


if __name__ == "__main__":
    import adalflow as adal

    # 设置环境或传递api_key
    from adalflow.utils import setup_env

    setup_env()

    openai_llm = adal.Generator(
        model_client=OpenAIClient(), model_kwargs={"model": "gpt-4o"}
    )
    resopnse = openai_llm(prompt_kwargs={"input_str": "What is LLM?"})
    print(resopnse)
