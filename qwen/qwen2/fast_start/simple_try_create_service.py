"""一个创建服务的样例
 - 使用了 FastAPI, 采用 post 请求
 - 服务的 url = 'http://localhost:8000/v1/generate'
 - 记录日志(每次的输入输出), 保存在 finetune/io_file/output/log/fast_api_svc.log
 - post - generate 函数的输入数据类型十分重要, 详见 Message 和 Messages 类
"""

#%% import
from fastapi import FastAPI
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import uvicorn
import json
import logging
from pydantic import BaseModel

from simple_try_0 import \
    get_model, get_tokenizer, \
    get_message1, messages_to_text, text_to_model_inputs, \
    model_generate, generated_ids_to_text

#%% funcs
def setup_logger(save_path, use_ch=False):
    # 创建一个logger
    logger = logging.getLogger('my_logger')
    logger.setLevel(logging.DEBUG)

    # 创建一个handler，用于写入日志文件
    fh = logging.FileHandler(save_path)
    fh.setLevel(logging.DEBUG)

    # 再创建一个handler，用于输出到控制台
    if use_ch:
        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)

    # 定义handler的输出格式
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    if use_ch:
        ch.setFormatter(formatter)

    # 给logger添加handler
    logger.addHandler(fh)
    if use_ch:
        logger.addHandler(ch)

    return logger

#%% global objs
model = get_model()
model.eval()# 将模型置于评估模式，以减少内存消耗

tokenizer = get_tokenizer()

logger = setup_logger(
    "/home/zkding/projects/coding/python/learn/learn_LLM/qwen/qwen2/finetune/io_file/output/log/fast_api_svc.log"
)

#%% fast api
# 输入数据类型定义 - 重要 !!!
class Message(BaseModel):
    role: str
    content: str

class Messages(BaseModel):
    messages: list[Message]


app = FastAPI(debug=False)
@app.post("/v1/generate")
async def generate(messages:str|Messages): # :Messages :str
    logger.debug(messages)
    
    if type(messages) == Messages:
        # 在这里使用 messages.messages，这是一个经过验证的列表
        logger.debug("messages type: Messages")
        text = messages_to_text(messages.messages, tokenizer)
    elif type(messages) == str:
        # 实际上目前无法正确传入 json 转成的 str 数据
        logger.debug("messages type: str")
        text = messages_to_text(json.loads(messages), tokenizer)
    else:
        logger.debug("messages type: UNK")
        return {"result": "error | generate() | messages type error."}
    
    inputs = text_to_model_inputs(text, tokenizer)
    generated_ids = model_generate(inputs, model)
    response = generated_ids_to_text(generated_ids, tokenizer)

    logger.debug(response)
    logger.debug("="*80)
    return {"result": response}

#%% test funcs
def test():
    def test_generate(messages:str):
        logger.debug(messages)
        if type(messages) == str:
            messages = json.loads(messages)
        text = messages_to_text(messages, tokenizer)
        inputs = text_to_model_inputs(text, tokenizer)
        generated_ids = model_generate(inputs, model)
        response = generated_ids_to_text(generated_ids, tokenizer)
        return {"result": response}
    
    messages = get_message1("你好")
    messages = json.dumps(messages, ensure_ascii=False)
    rslt = test_generate(messages)
    print(rslt)

#%% run demo
def setup_service():
    uvicorn.run(app, host="0.0.0.0", port=8000)

#%% main
if __name__ == "__main__":
    print("start testing ...")

    setup_service()

    #%% test
    # test()