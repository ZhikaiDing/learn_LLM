""" 一个样例, 模拟 CLI 模型交互
 - 需要先运行 simple_try_create_service.py 启动 LLM 服务
 - 采用 while 循环, 输入 quit 退出
 - 要点: 历史管理 | 解析模型输出 | 函数工具(声明|定义|调用)
"""

#%% import
import json
import random
import urllib.parse

from try_call_service import call_service

#%% roles
# system user assistant observation

#%% cfg
url = 'http://localhost:8000/v1/generate'

system_prompt = "You are a helpful assistant."

#%% com funcs
def try_parse_str_to_json(assistant_answer:str):
    assistant_answer = assistant_answer.strip()
    
    tmp_s = assistant_answer
    if assistant_answer.startswith("```") or assistant_answer.endswith("```"):
        tmp_s = assistant_answer.strip("`")
    
    try:
        d = json.loads(tmp_s)
        return d
    except:
        try: 
            d = eval(tmp_s)
            if type(d) == dict:
                return d
        except:
            pass

    return assistant_answer

#%% history manager
class HistManager:
    def __init__(self):
        self._hist = []
    
    def add(self, role, content):
        self._hist.append({"role":role, "content":content})
    
    def pop(self):
        ret = self._hist[-1]
        del self._hist[-1]
        return ret

    def get_hist(self):
        return self._hist
    
    def get_messages(self, sys_prompt):
        return [{"role":"system", "content":sys_prompt}] + self._hist

#%% functions
# declare
functions = [
{
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
},
{
    'name': 'draw_pic',
    'description': 'AI 绘画（图像生成）服务，输入文本描述，返回基于文本信息绘制的图像 URL。',
    'parameters': {
        'type': 'object',
        'properties': {
            'name': 'prompt',
            'type': 'string',
            'description': '期望的图像内容的详细描述',
        },
        'required': ['prompt'],
    },
},
]

# define
def fake_get_current_weather(location, unit='celsius'):
    """Get the current weather in a given location"""
    ret_js = {
        "location": location,
        "temperature": "unknown",
        "unit": unit.lower()
    }

    unk_prob = 0.2
    if random.random() < unk_prob:
        return json.dumps(ret_js, ensure_ascii=False)
    
    low_c = -20
    high_c = 40
    c = random.uniform(low_c, high_c)
    if "celsius" in ret_js["unit"]:
        ret_js["temperature"] = c
    elif "fahrenheit" in ret_js["unit"]:
        ret_js["temperature"] = 1.8*c + 32
    
    return json.dumps(ret_js, ensure_ascii=False)

def fake_draw_pic(prompt):
    prompt = urllib.parse.quote(prompt)
    ret_js = {'image_url': f'https://image.pollinations.ai/prompt/{prompt}'}
    return json.dumps(ret_js, ensure_ascii=False)

# function factory
FUNCTIONS = {
    "get_current_weather": fake_get_current_weather,
    "draw_pic": fake_draw_pic
}

#%% CLI demo
def CLI_demo(debug=False):
    hist = HistManager()
    messages = []

    while True:
        if debug: print("===> history", hist.get_hist())

        # 1. 用户输入
        usr_prompt = input("\n(input 'quit' to quit) user: ")
        
        # 2. 退出交互
        if usr_prompt.lower() == 'quit':
            break
        
        # 添加历史: 用户输入
        hist.add("user", usr_prompt)

        # 3. 生成 messages 列表
        messages = hist.get_messages(system_prompt)

        # 4. 请求 LLM (qwen2-Instruct) 服务
        response = call_service(messages, functions, url)
        if type(response) != dict or "result" not in response:
            print("warn | bad response:", response)
            continue
        
        # 5. 获取 LLM 回复
        assistant_answer = response["result"]

        # 添加历史: 模型回复
        hist.add("assistant", assistant_answer)

        # 6. 解析模型回复
        assistant_answer = try_parse_str_to_json(assistant_answer)
        
        # 7. try call function
        if type(assistant_answer) == dict:
            function_name = assistant_answer.get("function_name", "")
            if not function_name:
                continue
            
            if function_name not in FUNCTIONS:
                if debug: print("debug | unk function name:", function_name)
                continue
            
            params = assistant_answer.get("arguments", {})
            if type(params) != dict:
                params = try_parse_str_to_json(params)
                if type(params) != dict:
                    if debug: print("debug | load arguments failed:", params)
                    params = {}
            
            try:
                # 7.1 调用函数
                func_rslt = FUNCTIONS[function_name](**params)

                # 添加历史: 函数返回
                hist.add("observation", func_rslt)

                # 7.2 结合函数返回 进行回复
                messages = hist.get_messages(system_prompt)
                response = call_service(messages, functions, url)
                if type(response) != dict or "result" not in response:
                    print("warn | bad response:", response)
                    continue
                assistant_answer = response["result"]
                hist.add("assistant", assistant_answer)
            
            except:
                if debug: print(f"debug | call function failed | function_name: {function_name}, params: {params}")
                continue
        
        # 8. 显示模型回复
        print("assistant:", assistant_answer)

#%% main
if __name__ == "__main__":
    CLI_demo(debug=False)