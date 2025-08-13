"""AWS Bedrock模型客户端集成。"""

import os
import json
import logging
import boto3
import botocore
import backoff
from typing import Dict, Any, Optional, List, Generator, Union, AsyncGenerator

from adalflow.core.model_client import ModelClient
from adalflow.core.types import ModelType, GeneratorOutput

# 配置日志记录
from api.logging_config import setup_logging

setup_logging()
log = logging.getLogger(__name__)

class BedrockClient(ModelClient):
    __doc__ = r"""AWS Bedrock API客户端的组件包装器。

    AWS Bedrock提供了一个统一的API，可以访问各种基础模型，
    包括Amazon自己的模型和第三方模型，如Anthropic Claude。

    示例:
        ```python
        from api.bedrock_client import BedrockClient

        client = BedrockClient()
        generator = adal.Generator(
            model_client=client,
            model_kwargs={"model": "anthropic.claude-3-sonnet-20240229-v1:0"}
        )
        ```
    """

    def __init__(
        self,
        aws_access_key_id: Optional[str] = None,
        aws_secret_access_key: Optional[str] = None,
        aws_region: Optional[str] = None,
        aws_role_arn: Optional[str] = None,
        *args,
        **kwargs
    ) -> None:
        """
        初始化 AWS Bedrock 客户端。
        
        Args:
            aws_access_key_id: AWS 访问密钥 ID。如果未提供，将使用环境变量 AWS_ACCESS_KEY_ID。
            aws_secret_access_key: AWS 秘密访问密钥。如果未提供，将使用环境变量 AWS_SECRET_ACCESS_KEY。
            aws_region: AWS 区域。如果未提供，将使用环境变量 AWS_REGION。
            aws_role_arn: 基于角色的身份验证的 AWS IAM 角色 ARN。如果未提供，将使用环境变量 AWS_ROLE_ARN。
        """
        super().__init__(*args, **kwargs)
        from api.config import AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_REGION, AWS_ROLE_ARN

        self.aws_access_key_id = aws_access_key_id or AWS_ACCESS_KEY_ID
        self.aws_secret_access_key = aws_secret_access_key or AWS_SECRET_ACCESS_KEY
        self.aws_region = aws_region or AWS_REGION or "us-east-1"
        self.aws_role_arn = aws_role_arn or AWS_ROLE_ARN
        
        self.sync_client = self.init_sync_client()
        self.async_client = None  # 仅在需要时初始化异步客户端

    def init_sync_client(self):
        """
        初始化同步 AWS Bedrock 客户端。
        
        Returns:
            boto3.client: 配置好的Bedrock运行时客户端
        """
        try:
            # 使用提供的凭据创建会话
            session = boto3.Session(
                aws_access_key_id=self.aws_access_key_id,
                aws_secret_access_key=self.aws_secret_access_key,
                region_name=self.aws_region
            )
            
            # 如果提供了角色ARN，则承担该角色
            if self.aws_role_arn:
                sts_client = session.client('sts')
                assumed_role = sts_client.assume_role(
                    RoleArn=self.aws_role_arn,
                    RoleSessionName="DeepWikiBedrockSession"
                )
                credentials = assumed_role['Credentials']
                
                # 使用承担角色的凭据创建新会话
                session = boto3.Session(
                    aws_access_key_id=credentials['AccessKeyId'],
                    aws_secret_access_key=credentials['SecretAccessKey'],
                    aws_session_token=credentials['SessionToken'],
                    region_name=self.aws_region
                )
            
            # 创建Bedrock客户端
            bedrock_runtime = session.client(
                service_name='bedrock-runtime',
                region_name=self.aws_region
            )
            
            return bedrock_runtime
            
        except Exception as e:
            log.error(f"Error initializing AWS Bedrock client: {str(e)}")
            # Return None to indicate initialization failure
            return None

    def init_async_client(self):
        """
        初始化异步 AWS Bedrock 客户端。
        
        注意：boto3 没有原生异步支持，所以我们将在异步方法中使用同步客户端
        并在更高层次处理异步行为。
        """
        # For now, just return the sync client
        return self.sync_client

    def _get_model_provider(self, model_id: str) -> str:
        """
        从模型 ID 中提取提供商。
        
        Args:
            model_id: The model ID, e.g., "anthropic.claude-3-sonnet-20240229-v1:0"
            
        Returns:
            The provider name, e.g., "anthropic"
        """
        if "." in model_id:
            return model_id.split(".")[0]
        return "amazon"  # Default provider

    def _format_prompt_for_provider(self, provider: str, prompt: str, messages=None) -> Dict[str, Any]:
        """
        根据提供商的要求格式化提示。
        
        Args:
            provider: 提供商名称，例如 "anthropic"
            prompt: 提示文本
            messages: 聊天模型的可选消息列表
            
        Returns:
            包含格式化提示的字典
        """
        if provider == "anthropic":
            # Format for Claude models
            if messages:
                # Format as a conversation
                formatted_messages = []
                for msg in messages:
                    role = "user" if msg.get("role") == "user" else "assistant"
                    formatted_messages.append({
                        "role": role,
                        "content": [{"type": "text", "text": msg.get("content", "")}]
                    })
                return {
                    "anthropic_version": "bedrock-2023-05-31",
                    "messages": formatted_messages,
                    "max_tokens": 4096
                }
            else:
                # Format as a single prompt
                return {
                    "anthropic_version": "bedrock-2023-05-31",
                    "messages": [
                        {"role": "user", "content": [{"type": "text", "text": prompt}]}
                    ],
                    "max_tokens": 4096
                }
        elif provider == "amazon":
            # Format for Amazon Titan models
            return {
                "inputText": prompt,
                "textGenerationConfig": {
                    "maxTokenCount": 4096,
                    "stopSequences": [],
                    "temperature": 0.7,
                    "topP": 0.8
                }
            }
        elif provider == "cohere":
            # Format for Cohere models
            return {
                "prompt": prompt,
                "max_tokens": 4096,
                "temperature": 0.7,
                "p": 0.8
            }
        elif provider == "ai21":
            # Format for AI21 models
            return {
                "prompt": prompt,
                "maxTokens": 4096,
                "temperature": 0.7,
                "topP": 0.8
            }
        else:
            # Default format
            return {"prompt": prompt}

    def _extract_response_text(self, provider: str, response: Dict[str, Any]) -> str:
        """
        从提供商的响应中提取文本。
        
        Args:
            provider: 提供商名称
            response: 响应字典
            
        Returns:
            提取的文本内容
        """
        """Extract the generated text from the response.
        
        Args:
            provider: The provider name, e.g., "anthropic"
            response: The response from the Bedrock API
            
        Returns:
            The generated text
        """
        if provider == "anthropic":
            return response.get("content", [{}])[0].get("text", "")
        elif provider == "amazon":
            return response.get("results", [{}])[0].get("outputText", "")
        elif provider == "cohere":
            return response.get("generations", [{}])[0].get("text", "")
        elif provider == "ai21":
            return response.get("completions", [{}])[0].get("data", {}).get("text", "")
        else:
            # Try to extract text from the response
            if isinstance(response, dict):
                for key in ["text", "content", "output", "completion"]:
                    if key in response:
                        return response[key]
            return str(response)

    @backoff.on_exception(
        backoff.expo,
        (botocore.exceptions.ClientError, botocore.exceptions.BotoCoreError),
        max_time=5,
    )
    def call(self, api_kwargs: Dict = None, model_type: ModelType = None) -> Any:
        """
        对 AWS Bedrock API 进行同步调用。
        """
        api_kwargs = api_kwargs or {}
        
        # Check if client is initialized
        if not self.sync_client:
            error_msg = "AWS Bedrock client not initialized. Check your AWS credentials and region."
            log.error(error_msg)
            return error_msg
        
        if model_type == ModelType.LLM:
            model_id = api_kwargs.get("model", "anthropic.claude-3-sonnet-20240229-v1:0")
            provider = self._get_model_provider(model_id)
            
            # Get the prompt from api_kwargs
            prompt = api_kwargs.get("input", "")
            messages = api_kwargs.get("messages")
            
            # Format the prompt according to the provider
            request_body = self._format_prompt_for_provider(provider, prompt, messages)
            
            # Add model parameters if provided
            if "temperature" in api_kwargs:
                if provider == "anthropic":
                    request_body["temperature"] = api_kwargs["temperature"]
                elif provider == "amazon":
                    request_body["textGenerationConfig"]["temperature"] = api_kwargs["temperature"]
                elif provider == "cohere":
                    request_body["temperature"] = api_kwargs["temperature"]
                elif provider == "ai21":
                    request_body["temperature"] = api_kwargs["temperature"]
            
            if "top_p" in api_kwargs:
                if provider == "anthropic":
                    request_body["top_p"] = api_kwargs["top_p"]
                elif provider == "amazon":
                    request_body["textGenerationConfig"]["topP"] = api_kwargs["top_p"]
                elif provider == "cohere":
                    request_body["p"] = api_kwargs["top_p"]
                elif provider == "ai21":
                    request_body["topP"] = api_kwargs["top_p"]
            
            # Convert request body to JSON
            body = json.dumps(request_body)
            
            try:
                # Make the API call
                response = self.sync_client.invoke_model(
                    modelId=model_id,
                    body=body
                )
                
                # Parse the response
                response_body = json.loads(response["body"].read())
                
                # Extract the generated text
                generated_text = self._extract_response_text(provider, response_body)
                
                return generated_text
                
            except Exception as e:
                log.error(f"Error calling AWS Bedrock API: {str(e)}")
                return f"Error: {str(e)}"
        else:
            raise ValueError(f"Model type {model_type} is not supported by AWS Bedrock client")

    async def acall(self, api_kwargs: Dict = None, model_type: ModelType = None) -> Any:
        """
        对 AWS Bedrock API 进行异步调用。
        """
        # For now, just call the sync method
        # In a real implementation, you would use an async library or run the sync method in a thread pool
        return self.call(api_kwargs, model_type)

    def convert_inputs_to_api_kwargs(
        self, input: Any = None, model_kwargs: Dict = None, model_type: ModelType = None
    ) -> Dict:
        """
        将输入转换为 AWS Bedrock 的 API kwargs。
        """
        model_kwargs = model_kwargs or {}
        api_kwargs = {}
        
        if model_type == ModelType.LLM:
            api_kwargs["model"] = model_kwargs.get("model", "anthropic.claude-3-sonnet-20240229-v1:0")
            api_kwargs["input"] = input
            
            # Add model parameters
            if "temperature" in model_kwargs:
                api_kwargs["temperature"] = model_kwargs["temperature"]
            if "top_p" in model_kwargs:
                api_kwargs["top_p"] = model_kwargs["top_p"]
            
            return api_kwargs
        else:
            raise ValueError(f"Model type {model_type} is not supported by AWS Bedrock client")
