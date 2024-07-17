""" 一个样例, 模拟 CLI 模型交互
 - 需要先运行 simple_try_create_service.py 启动 LLM 服务
 - 采用 while 循环, 输入 quit 退出, clear 清空对话历史
 - 要点: 历史管理 | 解析模型输出 | 函数工具(声明|定义|调用)
"""

#%% import
import json
import random
import urllib.parse

from try_call_service import call_service

#%% roles
SYS_ROLE = "system"
USR_ROLE = "user"
MODEL_ROLE = "assistant"

FUNC_CALL_ROLE = "call_function"
FUNC_RSLT_ROLE = "function_return"

#%% cfg
url = 'http://localhost:8000/v1/generate'

system_prompt = "你是一个智能助手，根据交互历史进行回复或调用函数。"

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
    
    def clear(self):
        self._hist.clear()

    def get_hist(self):
        return self._hist
    
    def get_messages(self, sys_prompt):
        return [{"role":SYS_ROLE, "content":sys_prompt}] + self._hist

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
    """模拟 天气查询 接口.
    
    - 20% 概率查不到结果
    - 80% 概率随机生成一个 -20 到 40 之间的 摄氏度 (float)
    - unit 是华氏度时, 将 摄氏度 转换为 华氏度"""
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
    """模拟画图"""
    prompt = urllib.parse.quote(prompt)
    ret_js = {'image_url': f'https://image.pollinations.ai/prompt/{prompt}'}
    return json.dumps(ret_js, ensure_ascii=False)

# function factory
FUNCTIONS = {
    "get_current_weather": fake_get_current_weather,
    "draw_pic": fake_draw_pic
}

#%% CLI demo
def CLI_demo(debug=False, show_function_call=True):
    hist = HistManager()
    messages = []
    
    mode = "input" # input, call_func, call_model
    params = {} # 初始化函数参数

    while True:
        if debug: print("\n===> history", hist.get_hist())

        if mode == "input":
            # 1. user input
            # 1.1 输入
            usr_prompt = input(f"\n('quit' to quit, 'clear' to clear history) {USR_ROLE}: ")
            if not usr_prompt:
                continue
            
            # 1.2 特殊操作
            if usr_prompt.lower() == 'quit':
                # 退出
                break

            if usr_prompt.lower() == 'clear':
                # 清空交互历史
                hist.clear()
                continue

            # 1.3 添加历史: 函数的返回
            hist.add(USR_ROLE, usr_prompt)

            # 1.4 模式切换 - 用户输入之后 需要调用模型
            mode = "call_model"
        
        elif mode == "call_func":
            # 2. try call function
            try:
                # 2.1 调用函数
                func_rslt = FUNCTIONS[function_name](**params)
                if show_function_call:
                    print(f"info | call function | {function_name}(**{params}) | return:",func_rslt)

            except:
                func_rslt = {
                    "info": "Call function failed !!!",
                    "function_name": function_name,
                    "params": params
                }
                func_rslt = json.dumps(func_rslt, ensure_ascii=False)
            
            # 2.2 添加历史: 函数的调用&返回
            hist.add(FUNC_CALL_ROLE, f"{function_name}(**{params})")
            hist.add(FUNC_RSLT_ROLE, func_rslt)
            
            # 2.3 模式切换 - 获取函数调用结果之后 需要调用模型
            mode = "call_model"

        elif mode == "call_model":
            # 3. 请求模型
            # 3.1 请求
            messages = hist.get_messages(system_prompt)
            response = call_service(messages, functions, url)
            
            # 3.2 获取模型回复
            if type(response) != dict or "result" not in response:
                # 请求失败
                assistant_answer = "get model response failed."
                print("warn | bad response:", response)
            else:
                # 请求成功
                assistant_answer = response["result"]
            
            # 解析模型回复
            if debug: print("debug | org assistant_answer:", assistant_answer)
            parsed_answer = try_parse_str_to_json(assistant_answer)
            if debug: print("debug | parsed_answer:", parsed_answer)


            # 3.3 普通回复
            if type(parsed_answer) != dict:
                print(f"{MODEL_ROLE}:", assistant_answer)
                hist.add(MODEL_ROLE, assistant_answer)
                # 模式切换 - 模型回复完成后 需要用户继续输入
                mode = "input"
                continue

            # 3.4 判断是否需要进行函数调用
            # 3.4.1 判断函数名
            function_name = parsed_answer.get("function_name", "")
            if not function_name:
                # json 不符合 function calling 的格式 - 很可能是 普通回复
                print(f"{MODEL_ROLE}:", assistant_answer) # 打印回复内容
                hist.add(MODEL_ROLE, assistant_answer)
                
                # 模式切换 - 模型回复完成后 需要用户继续输入
                mode = "input"
                
                continue
            
            if function_name not in FUNCTIONS:
                # 函数名未知 - 可能是 普通回复
                if debug: print("debug | unk function name:", function_name)
                print(f"{MODEL_ROLE}:", assistant_answer) # 打印回复内容
                hist.add(MODEL_ROLE, assistant_answer)
                
                # 模式切换 - 模型回复完成后 需要用户继续输入
                mode = "input"
                
                continue
            
            # 3.4.2 获取参数列表
            params = parsed_answer.get("arguments", {})
            if type(params) != dict:
                params = try_parse_str_to_json(params)
                
                if type(params) != dict:
                    # 加载参数列表失败
                    print("warn | load arguments failed | arguments:", params)
                    params = {}
            
            # 模式切换 - 确实需要调用函数
            mode = "call_func"

#%% main
if __name__ == "__main__":
    CLI_demo(
        debug=False, 
        show_function_call=True
    )