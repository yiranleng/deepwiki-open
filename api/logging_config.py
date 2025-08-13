import logging
import os
from pathlib import Path

class IgnoreLogChangeDetectedFilter(logging.Filter):
    """
    忽略日志文件变更检测的过滤器。
    """
    def filter(self, record: logging.LogRecord):
        """
        过滤包含 "Detected file change in" 的日志记录。

        Args:
            record: 日志记录

        Returns:
            bool: 如果不包含指定文本则返回 True，否则返回 False
        """
        return "Detected file change in" not in record.getMessage()

def setup_logging(format: str = None):
    """
    为应用程序配置日志记录。
    从环境变量读取 LOG_LEVEL 和 LOG_FILE_PATH（默认值：INFO, logs/application.log）。
    确保日志目录存在，并配置文件和控制台处理器。
    """
    # Determine log directory and default file path
    base_dir = Path(__file__).parent
    log_dir = base_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    default_log_file = log_dir / "application.log"

    # Get log level and file path from environment
    log_level_str = os.environ.get("LOG_LEVEL", "INFO").upper()
    log_level = getattr(logging, log_level_str, logging.INFO)
    log_file_path = Path(os.environ.get(
        "LOG_FILE_PATH", str(default_log_file)))

    # ensure log_file_path is within the project's logs directory to prevent path traversal
    log_dir_resolved = log_dir.resolve()
    resolved_path = log_file_path.resolve()
    if not str(resolved_path).startswith(str(log_dir_resolved) + os.sep):
        raise ValueError(
            f"LOG_FILE_PATH '{log_file_path}' is outside the trusted log directory '{log_dir_resolved}'"
        )
    # Ensure parent dirs exist for the log file
    resolved_path.parent.mkdir(parents=True, exist_ok=True)

    # Configure logging handlers and format
    logging.basicConfig(
        level=log_level,
        format = format or "%(asctime)s - %(levelname)s - %(name)s - %(filename)s:%(lineno)d - %(message)s",
        handlers=[
            logging.FileHandler(resolved_path),
            logging.StreamHandler()
        ],
        force=True
    )
    
    # Ignore log file's change detection
    for handler in logging.getLogger().handlers:
        handler.addFilter(IgnoreLogChangeDetectedFilter())

    # Initial debug message to confirm configuration
    logger = logging.getLogger(__name__)
    logger.debug(f"Log level set to {log_level_str}, log file: {resolved_path}")
