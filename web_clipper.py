import os
import time
import asyncio
import concurrent.futures
from pathlib import Path
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import aiohttp
import requests
from github import Github
import openai
import telegram
from fastapi import FastAPI, UploadFile, File, HTTPException, Depends, Header, Request, Body
import uvicorn
from concurrent.futures import ThreadPoolExecutor
import shutil
from fastapi.security import APIKeyHeader, HTTPBearer, HTTPAuthorizationCredentials
from typing import Optional
import secrets
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
import re
from bs4 import BeautifulSoup
from fastapi.responses import JSONResponse
import html2text
from fastapi.concurrency import run_in_threadpool
import logging
from datetime import datetime

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("web_clipper.log", encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("web_clipper")

# 从配置文件导入配置
try:
    from config import CONFIG
except ImportError:
    CONFIG = {
        'github_token': os.getenv('GITHUB_TOKEN'),
        'telegram_token': os.getenv('TELEGRAM_TOKEN'),
        'telegram_chat_id': os.getenv('TELEGRAM_CHAT_ID'),
        'openai_api_key': os.getenv('OPENAI_API_KEY'),
        'openai_base_url': os.getenv('OPENAI_BASE_URL', 'https://api.openai.com/v1'),
        'github_repo': os.getenv('GITHUB_REPO'),
        'github_pages_domain': os.getenv('GITHUB_PAGES_DOMAIN', 'github.io'),
        'api_key': os.getenv('API_KEY', secrets.token_urlsafe(32)),
        'max_file_size': 10 * 1024 * 1024,
        'allowed_extensions': ['.html', '.htm'],
        'max_concurrent_requests': 50,
        'max_workers': 20,
        'request_timeout': 300,
        'max_retries': 3,
        'retry_delay': 2,
        'github_pages_max_retries': 60
    }

# 配置限制
MAX_FILE_SIZE = CONFIG.get('max_file_size', 10 * 1024 * 1024)
ALLOWED_EXTENSIONS = set(CONFIG.get('allowed_extensions', ['.html', '.htm']))

# 创建应用和限速器
app = FastAPI(title="Web Clipper API", version="1.0.0")
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# 全局变量
handler = None
UPLOAD_DIR = Path("uploads")
executor = ThreadPoolExecutor(max_workers=CONFIG.get('max_workers', 20))

# 安全验证
async def verify_token(credentials: HTTPAuthorizationCredentials = Depends(HTTPBearer())):
    if credentials.credentials != CONFIG['api_key']:
        raise HTTPException(
            status_code=401,
            detail="Invalid API key",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return credentials.credentials

class WebClipperHandler:
    def __init__(self, config):
        self.config = config
        self.github_token = config['github_token']
        self.github_repo_name = config['github_repo']
        self.telegram_bot = telegram.Bot(token=config['telegram_token'])
        
        # 创建异步 OpenAI 客户端
        self.openai_client = openai.AsyncOpenAI(
            api_key=config['openai_api_key'],
            base_url=config.get('openai_base_url')
        )
        
        # 创建 aiohttp 会话
        self.session = None

    async def init_session(self):
        """初始化 aiohttp 会话"""
        if self.session is None:
            self.session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.config.get('request_timeout', 300))
            )

    async def close_session(self):
        """关闭 aiohttp 会话"""
        if self.session:
            await self.session.close()
            self.session = None

    async def process_file(self, file_path: Path, original_url: str = ''):
        """异步处理文件"""
        try:
            logger.info("Start new web clipper processing...")
            
            # 初始化会话
            await self.init_session()

            # 1. 上传到 GitHub Pages（在线程池中执行阻塞操作）
            filename, github_url = await run_in_threadpool(
                self.upload_to_github, str(file_path)
            )
            logger.info(f"Github upload success: {github_url}")

            # 2. 获取 markdown 内容
            md_content = await self.url2md_async(github_url)
            
            # 3. 获取标题
            title = await run_in_threadpool(self.get_page_content_by_md, md_content)
            logger.info(f"Page title: {title}")
            
            # 如果没有提供原始 URL，则从文件名解析
            if not original_url:
                # 简单的文件名解析逻辑
                original_url = filename.split('_', 1)[-1].rsplit('.', 1)[0]

            # 4. 并行生成摘要和标签
            summary, tags = await self.generate_summary_tags_async(md_content)
            logger.info(f"Summary: {summary[:100]}...")
            logger.info(f"Tags: {', '.join(tags)}")
            
            # 5. 发送 Telegram 通知
            notification = (
                f"✨ 新的网页剪藏\n\n"
                f"📑 {title}\n\n"
                f"📝 {summary}\n\n"
                f"🔗 原始链接：{original_url}\n"
                f"📚 快照链接：{github_url}"
            )
            await self.send_telegram_notification(notification)
            
            logger.info("=" * 50)
            logger.info("Web page handle finished...")
            logger.info("=" * 50)
            
            return {
                "status": "success",
                "github_url": github_url,
                "title": title,
                "summary": summary,
                "tags": tags,
                "original_url": original_url
            }
            
        except Exception as e:
            error_msg = f"❌ 处理失败: {str(e)}"
            logger.error(error_msg)
            logger.error("=" * 50)
            await self.send_telegram_notification(error_msg)
            raise
        finally:
            # 确保清理资源
            pass

    def upload_to_github(self, html_path):
        """线程安全的 GitHub 上传"""
        # 为每个请求创建独立的 GitHub 客户端
        github_client = Github(self.github_token)
        
        try:
            filename = os.path.basename(html_path)
            
            with open(html_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            repo = github_client.get_repo(self.github_repo_name)
            file_path = f"clips/{filename}"
            
            # 检查文件是否已存在
            try:
                existing_file = repo.get_contents(file_path, ref="main")
                # 如果存在，更新文件
                repo.update_file(
                    file_path,
                    f"Update web clip: {filename}",
                    content,
                    existing_file.sha,
                    branch="main"
                )
            except Exception:
                # 文件不存在，创建新文件
                repo.create_file(
                    file_path,
                    f"Add web clip: {filename}",
                    content,
                    branch="main"
                )
            
            github_url = f"https://{self.config['github_pages_domain']}/{self.github_repo_name.split('/')[1]}/clips/{filename}"
            
            # 等待部署（在线程中执行）
            max_retries = self.config.get('github_pages_max_retries', 60)
            for attempt in range(max_retries):
                try:
                    response = requests.get(github_url, timeout=10)
                    if response.status_code == 200:
                        break
                    time.sleep(5)
                except Exception:
                    time.sleep(5)
            
            return filename, github_url
            
        finally:
            # 清理 GitHub 客户端
            if hasattr(github_client, 'close'):
                github_client.close()

    async def url2md_async(self, url, max_retries=30):
        """异步 URL 转 Markdown"""
        md_url = f"https://r.jina.ai/{url}"
        
        for attempt in range(max_retries):
            try:
                async with self.session.get(md_url, timeout=30) as response:
                    if response.status == 200:
                        md_content = await response.text()
                        return md_content
                    else:
                        await asyncio.sleep(10)
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} failed: {str(e)}")
                await asyncio.sleep(10)
        
        # 如果所有重试都失败，使用备用方法
        return await run_in_threadpool(self.get_page_content_by_bs, url)

    async def generate_summary_tags_async(self, content):
        """异步生成摘要和标签"""
        for attempt in range(self.config.get('max_retries', 3)):
            try:
                response = await self.openai_client.chat.completions.create(
                    model=self.config.get('openai_model', 'gpt-3.5-turbo'),
                    messages=[{
                        "role": "user",
                        "content": """请为以下网页(已转换为Markdown格式)内容生成简短摘要和相关标签。 
                        请严格按照以下格式返回(英文网页请以中文返回)：
                        摘要：[100字以内的摘要]
                        标签：tag1，tag2，tag3，tag4，tag5

                        网页(已转换为markdown格式)内容：
                        """ + content[:5000] + "..."
                    }],
                    timeout=60
                )

                result = response.choices[0].message.content
                
                try:
                    lines = result.split('\n')
                    summary_line = next((line for line in lines if line.startswith('摘要：')), None)
                    tags_line = next((line for line in lines if line.startswith('标签：')), None)
                    
                    if summary_line and tags_line:
                        summary = summary_line.replace('摘要：', '').strip()
                        tags_str = tags_line.replace('标签：', '').strip()
                        tags = [
                            tag.strip()[:20]
                            for tag in tags_str.replace('，', ',').split(',')
                            if tag.strip()
                        ]
                        return summary, tags
                    
                except Exception as e:
                    logger.error(f"解析 AI 响应失败: {str(e)}")
                
                # 如果解析失败，返回默认值
                return "自动生成的摘要", ["网页剪藏"]
                
            except Exception as e:
                logger.warning(f"OpenAI API 调用尝试 {attempt + 1} 失败: {str(e)}")
                if attempt < self.config.get('max_retries', 3) - 1:
                    await asyncio.sleep(self.config.get('retry_delay', 2))
        
        return "无法生成摘要", ["未分类"]

    def get_page_content_by_md(self, md_content):
        """从 markdown 获取标题"""
        lines = md_content.splitlines()
        for line in lines:
            if line.startswith("Title:"):
                return line.replace("Title:", "").strip()
            if line.startswith("# "):
                return line.replace("# ", "").strip()
        return "未知标题"

    def get_page_content_by_bs(self, url):
        """从部署的页面获取内容"""
        try:
            response = requests.get(url, timeout=30)
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # 获取标题
                title = None
                if soup.title:
                    title = soup.title.string
                if not title and soup.h1:
                    title = soup.h1.get_text(strip=True)
                
                # 提取正文内容
                html2markdown = html2text.HTML2Text()
                html2markdown.ignore_links = True
                html2markdown.ignore_images = True
                content = html2markdown.handle(soup.prettify())
                
                return f"Title: {title if title else '未知标题'} \n\n {content}"
                
        except Exception as e:
            logger.error(f"BeautifulSoup 处理失败: {str(e)}")
        
        return "Title: 未知标题 \n\n 内容获取失败"

    async def send_telegram_notification(self, message):
        """发送 Telegram 通知"""
        try:
            await self.telegram_bot.send_message(
                chat_id=self.config['telegram_chat_id'],
                text=message,
                parse_mode='HTML'
            )
        except Exception as e:
            logger.error(f"Telegram 通知发送失败: {str(e)}")

@app.on_event("startup")
async def startup_event():
    """启动时初始化"""
    global handler
    handler = WebClipperHandler(CONFIG)
    UPLOAD_DIR.mkdir(exist_ok=True)
    
    # 如果配置中没有 API key，生成一个
    if 'api_key' not in CONFIG:
        CONFIG['api_key'] = secrets.token_urlsafe(32)
        logger.info(f"Generated new API key: {CONFIG['api_key']}")

@app.on_event("shutdown")
async def shutdown_event():
    """关闭时清理资源"""
    if handler:
        await handler.close_session()
    executor.shutdown(wait=False)

@app.post("/upload")
async def upload_file(
    request: Request,
    token: str = Depends(verify_token)
):
    """支持并发处理的上传接口"""
    try:
        # 检查并发限制
        current_requests = getattr(request.state, 'active_requests', 0)
        if current_requests >= CONFIG.get('max_concurrent_requests', 50):
            raise HTTPException(status_code=429, detail="Too many concurrent requests")
        
        form = await request.form()
        original_url = form.get('url', '')
        
        # 获取文件
        file = None
        for field_name, field_value in form.items():
            if hasattr(field_value, 'filename') and hasattr(field_value, 'read'):
                file = field_value
                break
        
        if not file:
            raise HTTPException(status_code=400, detail="No file found in form data")
        
        # 读取文件内容
        content = await file.read()
        filename = file.filename
        
        # 验证文件
        file_ext = Path(filename).suffix.lower()
        if not file_ext:
            filename += '.html'
            file_ext = '.html'
        
        if file_ext not in ALLOWED_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail=f"File type not allowed. Allowed types: {', '.join(ALLOWED_EXTENSIONS)}"
            )
        
        if len(content) > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=400,
                detail=f"File too large. Maximum size allowed: {MAX_FILE_SIZE/1024/1024}MB"
            )
        
        # 保存临时文件
        safe_filename = f"{int(time.time())}_{secrets.token_hex(8)}_{filename}"
        file_path = UPLOAD_DIR / safe_filename
        
        # 使用线程池写入文件
        def write_file():
            with open(file_path, "wb") as f:
                f.write(content)
        
        await run_in_threadpool(write_file)
        
        try:
            # 异步处理文件
            result = await handler.process_file(file_path, original_url)
            return result
        finally:
            # 清理临时文件
            if file_path.exists():
                def remove_file():
                    try:
                        file_path.unlink()
                    except:
                        pass
                
                await run_in_threadpool(remove_file)
                
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Upload failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/health")
async def health_check():
    """健康检查端点"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "concurrent_workers": executor._max_workers
    }

@app.get("/stats")
async def get_stats():
    """获取服务器统计信息"""
    return {
        "max_workers": executor._max_workers,
        "active_threads": executor._work_queue.qsize(),
        "upload_dir_size": sum(f.stat().st_size for f in UPLOAD_DIR.glob('*') if f.is_file()),
        "upload_dir_files": len(list(UPLOAD_DIR.glob('*')))
    }

def start_server(host="0.0.0.0", port=8000):
    """启动支持并发的服务器"""
    uvicorn.run(
        app, 
        host=host, 
        port=port,
        loop="asyncio",
        timeout_keep_alive=30,
        limit_concurrency=100,
        limit_max_requests=1000,
        log_level="info"
    )

