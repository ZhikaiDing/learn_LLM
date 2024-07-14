# 前言
这个文档是关于: 下载模型 并简单调用 的一些问题的解答

# 1. 下载模型
 - Q: AutoModelForCausalLM.from_pretrained 下载模型失败怎么办?
 - A: 可以从以下几个方面检查
(ps: 目前我只能用最暴力的方法解决, 即方法3: 从官网下载到本地)
   - (1) ssh
     - 基本原理
huggingface 可以配置 ssh 文件用于远程管理仓库, 请在本地生成密钥并配置到 huggingface 账户

     - 本地 ssh 文件: 
       - 生成方式
ssh-keygen -t ed25519 -C "your_email@example.com"
ssh-keygen -t rsa -b 4096 -C "your_email@example.com"
       - 生成位置: 
/home/<user_name>/.ssh/
复制 .pub 文件内容到账户的 ssh 配置

     - 账户的 ssh 配置
右上角头像 -> settings -> SSH and GPG keys -> Add SSH key -> 随便起个名, 粘贴 .pub 文件内容 


   - (2) from_pretrained 参数
这个方法我在某个设备试验成功过, 不过 transformer 版本不一样, 下载过程也经常报错(报错就再运行接着下)
     - trust_remote_code=True
     - force_download=False
True: 强制重新下载模型的权重和配置文件，即使已经存在于缓存中的模型版本
False: 不重新下载
     - resume_download=False
True: 从头开始下载，即使部分文件已经存在
False: 从上次下载中断的地方继续下载
       - 注意这个提示: 
FutureWarning: `resume_download` is deprecated and will be removed in version 1.0.0. Downloads always resume when possible. If you want to force a new download, use `force_download=True`.


   - (3) 从 huggingface 官网下载
方法暴力, 但是有用
     - step 1: 找到官网模型的位置
如: https://huggingface.co/Qwen/Qwen2-7B-Instruct
点击 Files 可以看到仓库分支中的文件
     - step 2: 在本地创建文件夹, 将分支中的文件下载到文件夹中
注: git 相关的文件可以不下载, 毕竟我们不是通过克隆仓库拉取资源的
     - step 3: from_pretrained 的第一个参数改为 本地文件夹路径
注: 模型 和 tokenizer 都是这样