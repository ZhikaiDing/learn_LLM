from transformers import AutoTokenizer, AutoModel

cache_directory = "/home/zkding/projects/llm_model_resource/chatGLM3"

tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm3-6b", trust_remote_code=True)
model = AutoModel.from_pretrained(
    "THUDM/chatglm3-6b", 
    trust_remote_code=True,
    force_download=False, # True: 强制重新下载模型的权重和配置文件，即使已经存在于缓存中的模型版本; False: 不重新下载
    resume_download=False, # True: 从头开始下载，即使部分文件已经存在; False: 从上次下载中断的地方继续下载
    cache_dir=cache_directory, 
).half().cuda()
model = model.eval()

response, history = model.chat(tokenizer, "你好", history=[])
print(response)

response, history = model.chat(tokenizer, "晚上睡不着应该怎么办", history=history)
print(response)
