import logging
import weakref
import re
from dataclasses import dataclass
from typing import Any, List, Tuple, Dict
from uuid import uuid4

import adalflow as adal

from api.tools.embedder import get_embedder
from api.prompts import RAG_SYSTEM_PROMPT as system_prompt, RAG_TEMPLATE

# 创建我们自己的对话类实现
@dataclass
class UserQuery:
    """
    用户查询的数据类。
    """
    query_str: str

@dataclass
class AssistantResponse:
    """
    助手响应的数据类。
    """
    response_str: str

@dataclass
class DialogTurn:
    """
    对话轮次的数据类。
    """
    id: str
    user_query: UserQuery
    assistant_response: AssistantResponse

class CustomConversation:
    """
    自定义的 Conversation 实现，用于修复列表分配索引超出范围的错误
    """

    def __init__(self):
        """
        初始化自定义对话类。
        """
        self.dialog_turns = []

    def append_dialog_turn(self, dialog_turn):
        """
        安全地将对话轮次添加到对话中。
        
        Args:
            dialog_turn: 要添加的对话轮次
        """
        if not hasattr(self, 'dialog_turns'):
            self.dialog_turns = []
        self.dialog_turns.append(dialog_turn)

# 导入其他adalflow组件
from adalflow.components.retriever.faiss_retriever import FAISSRetriever
from api.config import configs
from api.data_pipeline import DatabaseManager

# 配置日志记录
logger = logging.getLogger(__name__)

# 嵌入模型的最大令牌限制
MAX_INPUT_TOKENS = 7500  # 8192令牌限制下的安全阈值

class Memory(adal.core.component.DataComponent):
    """
    使用对话轮次列表进行简单对话管理。
    """

    def __init__(self):
        """
        初始化内存组件。
        """
        super().__init__()
        # 使用我们的自定义实现而不是原始的 Conversation 类
        self.current_conversation = CustomConversation()

    def call(self) -> Dict:
        """
        将对话历史作为字典返回。
        
        Returns:
            Dict: 包含对话轮次的字典
        """
        all_dialog_turns = {}
        try:
            # 检查dialog_turns是否存在且为列表
            if hasattr(self.current_conversation, 'dialog_turns'):
                if self.current_conversation.dialog_turns:
                    logger.info(f"内存内容: {len(self.current_conversation.dialog_turns)} 轮")
                    for i, turn in enumerate(self.current_conversation.dialog_turns):
                        if hasattr(turn, 'id') and turn.id is not None:
                            all_dialog_turns[turn.id] = turn
                            logger.info(f"将第 {i+1} 轮（ID: {turn.id}）添加到内存")
                        else:
                            logger.warning(f"跳过内存中的无效轮次对象: {turn}")
                else:
                    logger.info("对话轮次列表存在但为空")
            else:
                logger.info("current_conversation中没有dialog_turns属性")
                # 尝试初始化它
                self.current_conversation.dialog_turns = []
        except Exception as e:
            logger.error(f"访问对话轮次时出错: {str(e)}")
            # 尝试恢复
            try:
                self.current_conversation = CustomConversation()
                logger.info("通过创建新对话恢复")
            except Exception as e2:
                logger.error(f"恢复失败: {str(e2)}")

        logger.info(f"从内存返回 {len(all_dialog_turns)} 个对话轮次")
        return all_dialog_turns

    def add_dialog_turn(self, user_query: str, assistant_response: str) -> bool:
        """
        向对话历史添加对话轮次。

        Args:
            user_query: 用户的查询
            assistant_response: 助手的响应

        Returns:
            bool: 成功返回 True，否则返回 False
        """
        try:
            # 使用我们的自定义实现创建新的对话轮次
            dialog_turn = DialogTurn(
                id=str(uuid4()),
                user_query=UserQuery(query_str=user_query),
                assistant_response=AssistantResponse(response_str=assistant_response),
            )

            # 确保current_conversation有append_dialog_turn方法
            if not hasattr(self.current_conversation, 'append_dialog_turn'):
                logger.warning("current_conversation没有append_dialog_turn方法，创建新的")
                # 如果需要则初始化新对话
                self.current_conversation = CustomConversation()

            # 确保dialog_turns存在
            if not hasattr(self.current_conversation, 'dialog_turns'):
                logger.warning("未找到dialog_turns，初始化空列表")
                self.current_conversation.dialog_turns = []

            # 安全地添加对话轮次
            self.current_conversation.dialog_turns.append(dialog_turn)
            logger.info(f"成功添加对话轮次，现在有 {len(self.current_conversation.dialog_turns)} 轮")
            return True

        except Exception as e:
            logger.error(f"添加对话轮次时出错: {str(e)}")
            # 尝试通过创建新对话来恢复
            try:
                self.current_conversation = CustomConversation()
                dialog_turn = DialogTurn(
                    id=str(uuid4()),
                    user_query=UserQuery(query_str=user_query),
                    assistant_response=AssistantResponse(response_str=assistant_response),
                )
                self.current_conversation.dialog_turns.append(dialog_turn)
                logger.info("通过创建新对话从错误中恢复")
                return True
            except Exception as e2:
                logger.error(f"从错误中恢复失败: {str(e2)}")
                return False


from dataclasses import dataclass, field

@dataclass
class RAGAnswer(adal.DataClass):
    """
    RAG答案的数据类。
    """
    rationale: str = field(default="", metadata={"desc": "答案的思维链。"})
    answer: str = field(default="", metadata={"desc": "用户查询的答案，格式化为markdown以便用react-markdown进行美观渲染。不要在答案的开头或结尾包含 ``` 三重反引号围栏。"})

    __output_fields__ = ["rationale", "answer"]

class RAG(adal.Component):
    """
    单个仓库的 RAG。
    如果要加载新仓库，请先调用 prepare_retriever(repo_url_or_path)。
    """

    def __init__(self, provider="google", model=None, use_s3: bool = False):  # noqa: F841 - use_s3 is kept for compatibility
        """
        初始化 RAG 组件。

        Args:
            provider: 要使用的模型提供商 (google, openai, openrouter, ollama)
            model: 要与提供商一起使用的模型名称
            use_s3: 是否使用 S3 进行数据库存储 (默认: False)
        """
        super().__init__()

        self.provider = provider
        self.model = model

        # 导入辅助函数
        from api.config import get_embedder_config, is_ollama_embedder

        # 根据配置确定是否使用Ollama嵌入器
        self.is_ollama_embedder = is_ollama_embedder()

        # 在继续之前检查Ollama模型是否存在
        if self.is_ollama_embedder:
            from api.ollama_patch import check_ollama_model_exists
            from api.config import get_embedder_config
            
            embedder_config = get_embedder_config()
            if embedder_config and embedder_config.get("model_kwargs", {}).get("model"):
                model_name = embedder_config["model_kwargs"]["model"]
                if not check_ollama_model_exists(model_name):
                    raise Exception(f"未找到Ollama模型 '{model_name}'。请运行 'ollama pull {model_name}' 来安装它。")

        # 初始化组件
        self.memory = Memory()
        self.embedder = get_embedder()

        self_weakref = weakref.ref(self)
        # Patch: ensure query embedding is always single string for Ollama
        def single_string_embedder(query):
            # Accepts either a string or a list, always returns embedding for a single string
            if isinstance(query, list):
                if len(query) != 1:
                    raise ValueError("Ollama embedder only supports a single string")
                query = query[0]
            instance = self_weakref()
            assert instance is not None, "RAG instance is no longer available, but the query embedder was called."
            return instance.embedder(input=query)

        # Use single string embedder for Ollama, regular embedder for others
        self.query_embedder = single_string_embedder if self.is_ollama_embedder else self.embedder

        self.initialize_db_manager()

        # Set up the output parser
        data_parser = adal.DataClassParser(data_class=RAGAnswer, return_data_class=True)

        # Format instructions to ensure proper output structure
        format_instructions = data_parser.get_output_format_str() + """

IMPORTANT FORMATTING RULES:
1. DO NOT include your thinking or reasoning process in the output
2. Provide only the final, polished answer
3. DO NOT include ```markdown fences at the beginning or end of your answer
4. DO NOT wrap your response in any kind of fences
5. Start your response directly with the content
6. The content will already be rendered as markdown
7. Do not use backslashes before special characters like [ ] { } in your answer
8. When listing tags or similar items, write them as plain text without escape characters
9. For pipe characters (|) in text, write them directly without escaping them"""

        # Get model configuration based on provider and model
        from api.config import get_model_config
        generator_config = get_model_config(self.provider, self.model)

        # Set up the main generator
        self.generator = adal.Generator(
            template=RAG_TEMPLATE,
            prompt_kwargs={
                "output_format_str": format_instructions,
                "conversation_history": self.memory(),
                "system_prompt": system_prompt,
                "contexts": None,
            },
            model_client=generator_config["model_client"](),
            model_kwargs=generator_config["model_kwargs"],
            output_processors=data_parser,
        )


    def initialize_db_manager(self):
        """
        使用本地存储初始化数据库管理器
        """
        self.db_manager = DatabaseManager()
        self.transformed_docs = []

    def _validate_and_filter_embeddings(self, documents: List) -> List:
        """
        验证嵌入并过滤掉具有无效或不匹配嵌入大小的文档。

        Args:
            documents: 包含嵌入的文档列表

        Returns:
            具有一致大小的有效嵌入的文档列表
        """
        if not documents:
            logger.warning("No documents provided for embedding validation")
            return []

        valid_documents = []
        embedding_sizes = {}

        # First pass: collect all embedding sizes and count occurrences
        for i, doc in enumerate(documents):
            if not hasattr(doc, 'vector') or doc.vector is None:
                logger.warning(f"Document {i} has no embedding vector, skipping")
                continue

            try:
                if isinstance(doc.vector, list):
                    embedding_size = len(doc.vector)
                elif hasattr(doc.vector, 'shape'):
                    embedding_size = doc.vector.shape[0] if len(doc.vector.shape) == 1 else doc.vector.shape[-1]
                elif hasattr(doc.vector, '__len__'):
                    embedding_size = len(doc.vector)
                else:
                    logger.warning(f"Document {i} has invalid embedding vector type: {type(doc.vector)}, skipping")
                    continue

                if embedding_size == 0:
                    logger.warning(f"Document {i} has empty embedding vector, skipping")
                    continue

                embedding_sizes[embedding_size] = embedding_sizes.get(embedding_size, 0) + 1

            except Exception as e:
                logger.warning(f"Error checking embedding size for document {i}: {str(e)}, skipping")
                continue

        if not embedding_sizes:
            logger.error("No valid embeddings found in any documents")
            return []

        # Find the most common embedding size (this should be the correct one)
        target_size = max(embedding_sizes.keys(), key=lambda k: embedding_sizes[k])
        logger.info(f"Target embedding size: {target_size} (found in {embedding_sizes[target_size]} documents)")

        # Log all embedding sizes found
        for size, count in embedding_sizes.items():
            if size != target_size:
                logger.warning(f"Found {count} documents with incorrect embedding size {size}, will be filtered out")

        # Second pass: filter documents with the target embedding size
        for i, doc in enumerate(documents):
            if not hasattr(doc, 'vector') or doc.vector is None:
                continue

            try:
                if isinstance(doc.vector, list):
                    embedding_size = len(doc.vector)
                elif hasattr(doc.vector, 'shape'):
                    embedding_size = doc.vector.shape[0] if len(doc.vector.shape) == 1 else doc.vector.shape[-1]
                elif hasattr(doc.vector, '__len__'):
                    embedding_size = len(doc.vector)
                else:
                    continue

                if embedding_size == target_size:
                    valid_documents.append(doc)
                else:
                    # Log which document is being filtered out
                    file_path = getattr(doc, 'meta_data', {}).get('file_path', f'document_{i}')
                    logger.warning(f"Filtering out document '{file_path}' due to embedding size mismatch: {embedding_size} != {target_size}")

            except Exception as e:
                file_path = getattr(doc, 'meta_data', {}).get('file_path', f'document_{i}')
                logger.warning(f"Error validating embedding for document '{file_path}': {str(e)}, skipping")
                continue

        logger.info(f"Embedding validation complete: {len(valid_documents)}/{len(documents)} documents have valid embeddings")

        if len(valid_documents) == 0:
            logger.error("No documents with valid embeddings remain after filtering")
        elif len(valid_documents) < len(documents):
            filtered_count = len(documents) - len(valid_documents)
            logger.warning(f"Filtered out {filtered_count} documents due to embedding issues")

        return valid_documents

    def prepare_retriever(self, repo_url_or_path: str, type: str = "github", access_token: str = None,
                      excluded_dirs: List[str] = None, excluded_files: List[str] = None,
                      included_dirs: List[str] = None, included_files: List[str] = None):
        """
        为仓库准备检索器。
        如果可用，将从本地存储加载数据库。

        Args:
            repo_url_or_path: 仓库的 URL 或本地路径
            type: 仓库类型，默认为 "github"
            access_token: 私有仓库的可选访问令牌
            excluded_dirs: 要排除的目录的可选列表
            excluded_files: 要排除的文件模式的可选列表
            included_dirs: 要包含的目录的可选列表
            included_files: 要包含的文件模式的可选列表
        """
        self.initialize_db_manager()
        self.repo_url_or_path = repo_url_or_path
        self.transformed_docs = self.db_manager.prepare_database(
            repo_url_or_path,
            type,
            access_token,
            is_ollama_embedder=self.is_ollama_embedder,
            excluded_dirs=excluded_dirs,
            excluded_files=excluded_files,
            included_dirs=included_dirs,
            included_files=included_files
        )
        logger.info(f"Loaded {len(self.transformed_docs)} documents for retrieval")

        # Validate and filter embeddings to ensure consistent sizes
        self.transformed_docs = self._validate_and_filter_embeddings(self.transformed_docs)

        if not self.transformed_docs:
            raise ValueError("No valid documents with embeddings found. Cannot create retriever.")

        logger.info(f"Using {len(self.transformed_docs)} documents with valid embeddings for retrieval")

        try:
            # Use the appropriate embedder for retrieval
            retrieve_embedder = self.query_embedder if self.is_ollama_embedder else self.embedder
            self.retriever = FAISSRetriever(
                **configs["retriever"],
                embedder=retrieve_embedder,
                documents=self.transformed_docs,
                document_map_func=lambda doc: doc.vector,
            )
            logger.info("FAISS retriever created successfully")
        except Exception as e:
            logger.error(f"Error creating FAISS retriever: {str(e)}")
            # Try to provide more specific error information
            if "All embeddings should be of the same size" in str(e):
                logger.error("Embedding size validation failed. This suggests there are still inconsistent embedding sizes.")
                # Log embedding sizes for debugging
                sizes = []
                for i, doc in enumerate(self.transformed_docs[:10]):  # Check first 10 docs
                    if hasattr(doc, 'vector') and doc.vector is not None:
                        try:
                            if isinstance(doc.vector, list):
                                size = len(doc.vector)
                            elif hasattr(doc.vector, 'shape'):
                                size = doc.vector.shape[0] if len(doc.vector.shape) == 1 else doc.vector.shape[-1]
                            elif hasattr(doc.vector, '__len__'):
                                size = len(doc.vector)
                            else:
                                size = "unknown"
                            sizes.append(f"doc_{i}: {size}")
                        except:
                            sizes.append(f"doc_{i}: error")
                logger.error(f"Sample embedding sizes: {', '.join(sizes)}")
            raise

    def call(self, query: str, language: str = "en") -> Tuple[List]:
        """
        使用 RAG 处理查询。

        Args:
            query: 用户的查询
            language: 语言，默认为 "en"

        Returns:
            (RAGAnswer, retrieved_documents) 的元组
        """
        try:
            retrieved_documents = self.retriever(query)

            # Fill in the documents
            retrieved_documents[0].documents = [
                self.transformed_docs[doc_index]
                for doc_index in retrieved_documents[0].doc_indices
            ]

            return retrieved_documents

        except Exception as e:
            logger.error(f"Error in RAG call: {str(e)}")

            # Create error response
            error_response = RAGAnswer(
                rationale="Error occurred while processing the query.",
                answer=f"I apologize, but I encountered an error while processing your question. Please try again or rephrase your question."
            )
            return error_response, []
