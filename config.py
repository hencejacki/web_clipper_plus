CONFIG = {
    'host': '0.0.0.0',  # 服务器监听地址
    'port': 65331,       # 服务器端口

    'github_repo': 'your_github_name/your_github_repo', # 开一个repo，可以private，填入你的github用户名和repo名称 e.g. hencejacki.github.io/owps
    'github_token': 'your_github_token',  # 开一个 token，需要授权到上面那个repo的权限
    'github_pages_domain': 'your_github_pages_domain', # repo开启pages，然后这里写上域名，类似 username.github.io 或自定义域名
    'notion_database_id': 'notion_database_id', # 创建一个新页面，格式为table，然后参照这个 https://blog.csdn.net/weixin_46014809/article/details/132631030 拿到id
    'notion_token': 'notion_token', # 在这里 https://www.notion.so/profile/integrations 申请一个notion token，具体步骤自己搜索，并且授权到上面的table，具体教程找一下很多的
    'telegram_token': 'telegram_token', # 在telegram里面 @bot_father 申请一个bot
    'telegram_chat_id': 'telegram_chat_id', # 按照这个 https://zhuanlan.zhihu.com/p/602213485 拿到 chat id

    'openai_api_key': 'your_openai_api_key',  #  OpenAI API key
    'openai_base_url': 'https://api.openai.com/v1',  # 如果需要第三方gpt代理，填到这，否则请不要修改
    'openai_model': 'gpt-4o-mini',  # 模型配置
    'openai_max_retries': 3,          # gpt最大重试次数

    'api_key': 'your_api_key',  # singlefile直接上传文件到 restful api 的 Bearer Key
    'max_file_size': 30 * 1024 * 1024,  # 30MB 最大文件上传尺寸
    'allowed_extensions': ['.html', '.htm'], # 允许的格式，不建议修改
    'github_pages_max_retries': 60,    # GitHub Pages 最大等待次数（5秒 * 60 = 5分钟），html 上传到 gh pages 之后，需要一段时间部署才能访问，一般在一分钟以内
}
