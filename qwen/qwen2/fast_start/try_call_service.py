""" 一个样例, 用于请求服务
 - 需要先运行 simple_try_create_service.py
 - 注意 post 请求的 data 格式, 形如: 
{
    "messages": [
        {'role': 'system', 'content': 'You are a helpful assistant.'}, 
        {'role': 'user', 'content': '你好'}
    ]
} - 无需转为 str, 由 FastAPI 判断数据类型
"""

#%% import
import requests
import json
from simple_try_0 import get_message1

#%% cfg
url = 'http://localhost:8000/v1/generate'

#%% funcs
def call_service(content:str, url=url, msg_to_str=False):
    content = get_message1(content)
    if msg_to_str:
        # bad - 目前这种方式无法正确请求url - 无法正确传入输入 !!!
        content = json.dumps(content,ensure_ascii=False)
        print("warn | call_service() | messages(list[dict]) trans to str is not ok to requset url !!! \n\tPlease set msg_to_str=False !")
    print("\nmessages:", content)

    data = {
        "messages": content
    }

    response = requests.post(url, json=data)

    if response.status_code != 200:
        print(f"Error: {response.status_code}")

    return response.json()

#%% main
if __name__ == "__main__":
    print("start testing ...")

    msg_to_str = False
    texts = [
        "你好"
    ]
    for text in texts:
        print("input:",text)
        response = call_service(text, msg_to_str=msg_to_str)
        print("response:", response)

    #%% test
