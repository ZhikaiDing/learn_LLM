""" 一个样例, 用于请求服务
 - 需要先运行 simple_try_create_service.py 启动 LLM 服务
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

functions = [{
    'name': 'get_current_weather',
    'description': 'Get the current weather in a given location',
    'parameters': {
        'type': 'object',
        'properties': {
            'location': {
                'type': 'string',
                'description':
                'The city and state, e.g. San Francisco',
            },
            'unit': {
                'type': 'string',
                'enum': ['celsius', 'fahrenheit']
            },
        },
        'required': ['location'],
    },
}]

#%% funcs
def call_service(
        messages:list[dict], 
        functions:list[dict]=functions, 
        url=url
):
    data = {
        "messages": messages,
        "functions": functions
    }

    response = requests.post(url, json=data)

    if response.status_code != 200:
        print(f"Error: {response.status_code}")
        return response

    return response.json()

#%% test call service
def test_call_service():
    texts = [
        "你好",
        "What's the weather like in San Francisco?",
        "What's the weather like in Beijin?",
        "北京的天气怎么样"
    ]
    for text in texts:
        print("input:",text)
        
        messages = get_message1(text)
        print("\nmessages:", messages)
        
        response = call_service(messages)
        
        print("\nresponse:", response)
        print("="*50)

#%% main
if __name__ == "__main__":
    print("start testing ...")

    #%% test
    test_call_service()
