from typing import Sequence, List
from copy import deepcopy
from tqdm import tqdm
import logging
import adalflow as adal
from adalflow.core.types import Document
from adalflow.core.component import DataComponent
import requests
import os

# 配置日志记录
from api.logging_config import setup_logging

setup_logging()
logger = logging.getLogger(__name__)

class OllamaModelNotFoundError(Exception):
    """
    当找不到 Ollama 模型时的自定义异常
    """
    pass

def check_ollama_model_exists(model_name: str, ollama_host: str = None) -> bool:
    """
    在尝试使用之前检查 Ollama 模型是否存在。
    
    Args:
        model_name: 要检查的模型名称
        ollama_host: Ollama 主机 URL，默认为 localhost:11434
        
    Returns:
        bool: 如果模型存在则返回 True，否则返回 False
    """
    if ollama_host is None:
        ollama_host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    
    try:
        # 如果存在 /api 前缀则移除并重新添加
        if ollama_host.endswith('/api'):
            ollama_host = ollama_host[:-4]
        
        response = requests.get(f"{ollama_host}/api/tags", timeout=5)
        if response.status_code == 200:
            models_data = response.json()
            available_models = [model.get('name', '').split(':')[0] for model in models_data.get('models', [])]
            model_base_name = model_name.split(':')[0]  # 如果存在标签则移除
            
            is_available = model_base_name in available_models
            if is_available:
                logger.info(f"Ollama模型 '{model_name}' 可用")
            else:
                logger.warning(f"Ollama模型 '{model_name}' 不可用。可用模型: {available_models}")
            return is_available
        else:
            logger.warning(f"无法检查Ollama模型，状态码: {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        logger.warning(f"无法连接到Ollama检查模型: {e}")
        return False
    except Exception as e:
        logger.warning(f"检查Ollama模型可用性时出错: {e}")
        return False

class OllamaDocumentProcessor(DataComponent):
    """
    通过一次处理一个文档来为 Ollama 嵌入处理文档。
    Adalflow Ollama Client 不支持批量嵌入，因此我们需要单独处理每个文档。
    """
    def __init__(self, embedder: adal.Embedder) -> None:
        """
        初始化Ollama文档处理器。
        
        Args:
            embedder: 用于生成嵌入的嵌入器实例
        """
        super().__init__()
        self.embedder = embedder

    def __call__(self, documents: Sequence[Document]) -> Sequence[Document]:
        """
        处理文档序列，为每个文档生成嵌入。
        
        Args:
            documents: 要处理的文档序列
            
        Returns:
            Sequence[Document]: 包含嵌入向量的文档序列
        """
        output = deepcopy(documents)
        logger.info(f"为Ollama嵌入单独处理 {len(output)} 个文档")

        successful_docs = []
        expected_embedding_size = None

        for i, doc in enumerate(tqdm(output, desc="为Ollama嵌入处理文档")):
            try:
                # 为单个文档获取嵌入
                result = self.embedder(input=doc.text)
                if result.data and len(result.data) > 0:
                    embedding = result.data[0].embedding

                    # 验证嵌入大小一致性
                    if expected_embedding_size is None:
                        expected_embedding_size = len(embedding)
                        logger.info(f"预期嵌入大小设置为: {expected_embedding_size}")
                    elif len(embedding) != expected_embedding_size:
                        file_path = getattr(doc, 'meta_data', {}).get('file_path', f'document_{i}')
                        logger.warning(f"文档 '{file_path}' 的嵌入大小不一致 {len(embedding)} != {expected_embedding_size}，跳过")
                        continue

                    # 将嵌入分配给文档
                    output[i].vector = embedding
                    successful_docs.append(output[i])
                else:
                    file_path = getattr(doc, 'meta_data', {}).get('file_path', f'document_{i}')
                    logger.warning(f"无法为文档 '{file_path}' 获取嵌入，跳过")
            except Exception as e:
                file_path = getattr(doc, 'meta_data', {}).get('file_path', f'document_{i}')
                logger.error(f"处理文档 '{file_path}' 时出错: {e}，跳过")

        logger.info(f"成功处理了 {len(successful_docs)}/{len(output)} 个具有一致嵌入的文档")
        return successful_docs