import uvicorn
import os
import sys
import logging
from dotenv import load_dotenv

# 从.env文件加载环境变量
load_dotenv()

from api.logging_config import setup_logging

# 配置日志记录
setup_logging()
logger = logging.getLogger(__name__)

# 将当前目录添加到路径中，以便我们可以导入api包
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 检查必需的环境变量
required_env_vars = ['GOOGLE_API_KEY', 'OPENAI_API_KEY']
missing_vars = [var for var in required_env_vars if not os.environ.get(var)]
if missing_vars:
    logger.warning(f"缺少环境变量: {', '.join(missing_vars)}")
    logger.warning("没有这些变量，某些功能可能无法正常工作。")

# 配置Google生成式AI
import google.generativeai as genai
from api.config import GOOGLE_API_KEY

if GOOGLE_API_KEY:
    genai.configure(api_key=GOOGLE_API_KEY)
else:
    logger.warning("未配置GOOGLE_API_KEY")

if __name__ == "__main__":
    # 从环境变量获取端口或使用默认值
    port = int(os.environ.get("PORT", 8001))

    # 在这里导入app以确保首先设置环境变量
    from api.api import app

    logger.info(f"在端口{port}上启动流式API")

    # 使用uvicorn运行FastAPI应用
    # 在生产/Docker环境中禁用重载
    is_development = os.environ.get("NODE_ENV") != "production"
    
    if is_development:
        # 防止文件更改触发日志写入导致的无限日志循环
        logging.getLogger("watchfiles.main").setLevel(logging.WARNING)

    uvicorn.run(
        "api.api:app",
        host="0.0.0.0",
        port=port,
        reload=is_development
    )
