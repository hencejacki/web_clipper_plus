import os
from dotenv import load_dotenv

load_dotenv()

CONFIG = {
    'host': '0.0.0.0',
    'port': 65331,
    'github_token': os.getenv('GITHUB_TOKEN'),
    'telegram_token': os.getenv('TELEGRAM_TOKEN'),
    'telegram_chat_id': os.getenv('TELEGRAM_CHAT_ID'),
    'openai_api_key': os.getenv('OPENAI_API_KEY'),
    'openai_base_url': os.getenv('OPENAI_BASE_URL', 'https://api.openai.com/v1'),
    'openai_model': os.getenv('OPENAI_MODEL', 'gpt-3.5-turbo'),
    'github_repo': os.getenv('GITHUB_REPO'),
    'github_pages_domain': os.getenv('GITHUB_PAGES_DOMAIN', 'github.io'),
    
    # 并发配置
    'max_concurrent_requests': 50,
    'max_workers': 20,
    'request_timeout': 300,
    'max_retries': 3,
    'retry_delay': 2,
    'github_pages_max_retries': 60,
    
    # 文件配置
    'max_file_size': 10 * 1024 * 1024,  # 10MB
    'allowed_extensions': ['.html', '.htm'],
    
    # API 密钥（会自动生成）
    'api_key': os.getenv('API_KEY')
}
