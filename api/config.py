import os
import json
import logging
import re
from pathlib import Path
from typing import List, Union, Dict, Any

logger = logging.getLogger(__name__)

from api.openai_client import OpenAIClient
from api.openrouter_client import OpenRouterClient
from api.bedrock_client import BedrockClient
from api.azureai_client import AzureAIClient
from api.dashscope_client import DashscopeClient
from adalflow import GoogleGenAIClient, OllamaClient

# 从环境变量获取API密钥
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
GOOGLE_API_KEY = os.environ.get('GOOGLE_API_KEY')
OPENROUTER_API_KEY = os.environ.get('OPENROUTER_API_KEY')
AWS_ACCESS_KEY_ID = os.environ.get('AWS_ACCESS_KEY_ID')
AWS_SECRET_ACCESS_KEY = os.environ.get('AWS_SECRET_ACCESS_KEY')
AWS_REGION = os.environ.get('AWS_REGION')
AWS_ROLE_ARN = os.environ.get('AWS_ROLE_ARN')

# 在环境中设置密钥（以防代码中其他地方需要）
if OPENAI_API_KEY:
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
if GOOGLE_API_KEY:
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
if OPENROUTER_API_KEY:
    os.environ["OPENROUTER_API_KEY"] = OPENROUTER_API_KEY
if AWS_ACCESS_KEY_ID:
    os.environ["AWS_ACCESS_KEY_ID"] = AWS_ACCESS_KEY_ID
if AWS_SECRET_ACCESS_KEY:
    os.environ["AWS_SECRET_ACCESS_KEY"] = AWS_SECRET_ACCESS_KEY
if AWS_REGION:
    os.environ["AWS_REGION"] = AWS_REGION
if AWS_ROLE_ARN:
    os.environ["AWS_ROLE_ARN"] = AWS_ROLE_ARN

# Wiki认证设置
raw_auth_mode = os.environ.get('DEEPWIKI_AUTH_MODE', 'False')
WIKI_AUTH_MODE = raw_auth_mode.lower() in ['true', '1', 't']
WIKI_AUTH_CODE = os.environ.get('DEEPWIKI_AUTH_CODE', '')

# 从环境变量获取配置目录，如果未设置则使用默认值
CONFIG_DIR = os.environ.get('DEEPWIKI_CONFIG_DIR', None)

# 客户端类映射
CLIENT_CLASSES = {
    "GoogleGenAIClient": GoogleGenAIClient,
    "OpenAIClient": OpenAIClient,
    "OpenRouterClient": OpenRouterClient,
    "OllamaClient": OllamaClient,
    "BedrockClient": BedrockClient,
    "AzureAIClient": AzureAIClient,
    "DashscopeClient": DashscopeClient
}

def replace_env_placeholders(config: Union[Dict[str, Any], List[Any], str, Any]) -> Union[Dict[str, Any], List[Any], str, Any]:
    """
    递归替换嵌套配置结构（字典、列表、字符串）中字符串值内的占位符，
    如 "${ENV_VAR}"，用环境变量值替换。如果找不到占位符则记录警告。
    """
    pattern = re.compile(r"\$\{([A-Z0-9_]+)\}")

    def replacer(match: re.Match[str]) -> str:
        env_var_name = match.group(1)
        original_placeholder = match.group(0)
        env_var_value = os.environ.get(env_var_name)
        if env_var_value is None:
            logger.warning(
                f"环境变量占位符 '{original_placeholder}' 在环境中未找到。 "
                f"将按原样使用占位符字符串。"
            )
            return original_placeholder
        return env_var_value

    if isinstance(config, dict):
        return {k: replace_env_placeholders(v) for k, v in config.items()}
    elif isinstance(config, list):
        return [replace_env_placeholders(item) for item in config]
    elif isinstance(config, str):
        return pattern.sub(replacer, config)
    else:
        # 处理数字、布尔值、None等
        return config

# 加载JSON配置文件
def load_json_config(filename):
    """
    加载 JSON 配置文件。

    Args:
        filename (str): 配置文件名

    Returns:
        dict: 加载的配置字典，如果加载失败则返回空字典
    """
    try:
        # 如果设置了环境变量，使用它指定的目录
        if CONFIG_DIR:
            config_path = Path(CONFIG_DIR) / filename
        else:
            # 否则使用默认目录
            config_path = Path(__file__).parent / "config" / filename

        logger.info(f"从 {config_path} 加载配置")

        if not config_path.exists():
            logger.warning(f"配置文件 {config_path} 不存在")
            return {}

        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
            config = replace_env_placeholders(config)
            return config
    except Exception as e:
        logger.error(f"加载配置文件 {filename} 时出错: {str(e)}")
        return {}

# 加载生成器模型配置
def load_generator_config():
    """
    加载生成器模型配置。

    Returns:
        dict: 生成器配置字典，包含提供商和模型客户端信息
    """
    generator_config = load_json_config("generator.json")

    # 为每个提供商添加客户端类
    if "providers" in generator_config:
        for provider_id, provider_config in generator_config["providers"].items():
            # 尝试从client_class设置客户端类
            if provider_config.get("client_class") in CLIENT_CLASSES:
                provider_config["model_client"] = CLIENT_CLASSES[provider_config["client_class"]]
            # 回退到基于provider_id的默认映射
            elif provider_id in ["google", "openai", "openrouter", "ollama", "bedrock", "azure", "dashscope"]:
                default_map = {
                    "google": GoogleGenAIClient,
                    "openai": OpenAIClient,
                    "openrouter": OpenRouterClient,
                    "ollama": OllamaClient,
                    "bedrock": BedrockClient,
                    "azure": AzureAIClient,
                    "dashscope": DashscopeClient
                }
                provider_config["model_client"] = default_map[provider_id]
            else:
                logger.warning(f"未知的提供商或客户端类: {provider_id}")

    return generator_config

# 加载嵌入器配置
def load_embedder_config():
    """
    加载嵌入器配置。

    Returns:
        dict: 嵌入器配置字典，包含嵌入器和 Ollama 嵌入器配置
    """
    embedder_config = load_json_config("embedder.json")

    # 处理客户端类
    for key in ["embedder", "embedder_ollama"]:
        if key in embedder_config and "client_class" in embedder_config[key]:
            class_name = embedder_config[key]["client_class"]
            if class_name in CLIENT_CLASSES:
                embedder_config[key]["model_client"] = CLIENT_CLASSES[class_name]

    return embedder_config

def get_embedder_config():
    """
    获取当前嵌入器配置。

    Returns:
        dict: 包含已解析 model_client 的嵌入器配置
    """
    return configs.get("embedder", {})

def is_ollama_embedder():
    """
    检查当前嵌入器配置是否使用 OllamaClient。

    Returns:
        bool: 如果使用 OllamaClient 则返回 True，否则返回 False
    """
    embedder_config = get_embedder_config()
    if not embedder_config:
        return False

    # 检查model_client是否为OllamaClient
    model_client = embedder_config.get("model_client")
    if model_client:
        return model_client.__name__ == "OllamaClient"

    # 回退：检查client_class字符串
    client_class = embedder_config.get("client_class", "")
    return client_class == "OllamaClient"

# 加载仓库和文件过滤器配置
def load_repo_config():
    """
    加载仓库和文件过滤器配置。

    Returns:
        dict: 仓库配置字典
    """
    return load_json_config("repo.json")

# 加载语言配置
def load_lang_config():
    """
    加载语言配置。

    Returns:
        dict: 语言配置字典，包含支持的语言和默认语言
    """
    default_config = {
        "supported_languages": {
            "en": "English",
            "ja": "Japanese (日本語)",
            "zh": "Mandarin Chinese (中文)",
            "zh-tw": "Traditional Chinese (繁體中文)",
            "es": "Spanish (Español)",
            "kr": "Korean (한국어)",
            "vi": "Vietnamese (Tiếng Việt)",
            "pt-br": "Brazilian Portuguese (Português Brasileiro)",
            "fr": "Français (French)",
            "ru": "Русский (Russian)"
        },
        "default": "en"
    }

    loaded_config = load_json_config("lang.json") # 让load_json_config处理路径和加载

    if not loaded_config:
        return default_config

    if "supported_languages" not in loaded_config or "default" not in loaded_config:
        logger.warning("语言配置文件 'lang.json' 格式错误。使用默认语言配置。")
        return default_config

    return loaded_config

# 默认排除的目录和文件
DEFAULT_EXCLUDED_DIRS: List[str] = [
    # 虚拟环境和包管理器
    "./.venv/", "./venv/", "./env/", "./virtualenv/",
    "./node_modules/", "./bower_components/", "./jspm_packages/",
    # 版本控制
    "./.git/", "./.svn/", "./.hg/", "./.bzr/",
    # 缓存和编译文件
    "./__pycache__/", "./.pytest_cache/", "./.mypy_cache/", "./.ruff_cache/", "./.coverage/",
    # 构建和分发
    "./dist/", "./build/", "./out/", "./target/", "./bin/", "./obj/",
    # 文档
    "./docs/", "./_docs/", "./site-docs/", "./_site/",
    # IDE特定
    "./.idea/", "./.vscode/", "./.vs/", "./.eclipse/", "./.settings/",
    # 日志和临时文件
    "./logs/", "./log/", "./tmp/", "./temp/",
]

DEFAULT_EXCLUDED_FILES: List[str] = [
    "yarn.lock", "pnpm-lock.yaml", "npm-shrinkwrap.json", "poetry.lock",
    "Pipfile.lock", "requirements.txt.lock", "Cargo.lock", "composer.lock",
    ".lock", ".DS_Store", "Thumbs.db", "desktop.ini", "*.lnk", ".env",
    ".env.*", "*.env", "*.cfg", "*.ini", ".flaskenv", ".gitignore",
    ".gitattributes", ".gitmodules", ".github", ".gitlab-ci.yml",
    ".prettierrc", ".eslintrc", ".eslintignore", ".stylelintrc",
    ".editorconfig", ".jshintrc", ".pylintrc", ".flake8", "mypy.ini",
    "pyproject.toml", "tsconfig.json", "webpack.config.js", "babel.config.js",
    "rollup.config.js", "jest.config.js", "karma.conf.js", "vite.config.js",
    "next.config.js", "*.min.js", "*.min.css", "*.bundle.js", "*.bundle.css",
    "*.map", "*.gz", "*.zip", "*.tar", "*.tgz", "*.rar", "*.7z", "*.iso",
    "*.dmg", "*.img", "*.msix", "*.appx", "*.appxbundle", "*.xap", "*.ipa",
    "*.deb", "*.rpm", "*.msi", "*.exe", "*.dll", "*.so", "*.dylib", "*.o",
    "*.obj", "*.jar", "*.war", "*.ear", "*.jsm", "*.class", "*.pyc", "*.pyd",
    "*.pyo", "__pycache__", "*.a", "*.lib", "*.lo", "*.la", "*.slo", "*.dSYM",
    "*.egg", "*.egg-info", "*.dist-info", "*.eggs", "node_modules",
    "bower_components", "jspm_packages", "lib-cov", "coverage", "htmlcov",
    ".nyc_output", ".tox", "dist", "build", "bld", "out", "bin", "target",
    "packages/*/dist", "packages/*/build", ".output"
]

# 初始化空配置
configs = {}

# 加载所有配置文件
generator_config = load_generator_config()
embedder_config = load_embedder_config()
repo_config = load_repo_config()
lang_config = load_lang_config()

# 更新配置
if generator_config:
    configs["default_provider"] = generator_config.get("default_provider", "google")
    configs["providers"] = generator_config.get("providers", {})

# 更新嵌入器配置
if embedder_config:
    for key in ["embedder", "embedder_ollama", "retriever", "text_splitter"]:
        if key in embedder_config:
            configs[key] = embedder_config[key]

# 更新仓库配置
if repo_config:
    for key in ["file_filters", "repository"]:
        if key in repo_config:
            configs[key] = repo_config[key]

# 更新语言配置
if lang_config:
    configs["lang_config"] = lang_config


def get_model_config(provider="google", model=None):
    """
    获取指定提供商和模型的配置

    Parameters:
        provider (str): 模型提供商 ('google', 'openai', 'openrouter', 'ollama', 'bedrock')
        model (str): 模型名称，如果为 None 则使用默认模型

    Returns:
        dict: 包含 model_client、model 和其他参数的配置
    """
    # 获取提供商配置
    if "providers" not in configs:
        raise ValueError("未加载提供商配置")

    provider_config = configs["providers"].get(provider)
    if not provider_config:
        raise ValueError(f"未找到提供商 '{provider}' 的配置")

    model_client = provider_config.get("model_client")
    if not model_client:
        raise ValueError(f"未为提供商 '{provider}' 指定模型客户端")

    # 如果未提供模型，使用提供商的默认模型
    if not model:
        model = provider_config.get("default_model")
        if not model:
            raise ValueError(f"未为提供商 '{provider}' 指定默认模型")

    # 获取模型参数（如果存在）
    model_params = {}
    if model in provider_config.get("models", {}):
        model_params = provider_config["models"][model]
    else:
        default_model = provider_config.get("default_model")
        model_params = provider_config["models"][default_model]

    # 准备基础配置
    result = {
        "model_client": model_client,
    }

    # 提供商特定调整
    if provider == "ollama":
        # Ollama使用稍微不同的参数结构
        if "options" in model_params:
            result["model_kwargs"] = {"model": model, **model_params["options"]}
        else:
            result["model_kwargs"] = {"model": model}
    else:
        # 其他提供商的标准结构
        result["model_kwargs"] = {"model": model, **model_params}

    return result
