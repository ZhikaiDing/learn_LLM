#%% import
import json
from transformers import AutoModelForCausalLM, AutoTokenizer

#%% cfg
model_name = "/home/zkding/projects/llm_model_resource/Qwen/Qwen2-7B-Instruct"
device = "cuda" # the device to load the model onto

functions_prompt = """# 工具列表:
__functions__
# 要求
- 无需调用函数时正常交互; 
- 需要调用函数时请严格返回 json, 
  - 返回格式是 {"function_name": "xxx", "arguments": {}},
  - 未提到的非必须参数无需抽取."""

#%% funcs
def get_model(model_name:str=model_name):
    return AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto", # device=device, # pip install accelerate
        trust_remote_code=True,
        force_download=False, # True: 强制重新下载模型的权重和配置文件，即使已经存在于缓存中的模型版本; False: 不重新下载
        resume_download=False, # True: 从头开始下载，即使部分文件已经存在; False: 从上次下载中断的地方继续下载
        # cache_dir=cache_directory,
        # mirror="tuna",
    )

def get_tokenizer(model_name:str=model_name):
    return AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

def function_list_to_messages(
        messages:list[dict], functions:list[dict]|str, role = "functions"
):
    """尝试将函数定义列表添加到 system 角色的 prompt.
     - ps: 我不确定官方做法是什么样的"""
    if not functions:
        return

    if type(functions) == str:
        functions_str = functions
    else:
        functions_str = json.dumps(functions, indent=4, ensure_ascii=False)

    content = functions_prompt.replace("__functions__", functions_str)
    if messages and messages[0]["role"] == "system":
        messages[0]["content"] += "\n" +content
        return
    
    messages.insert(0, {"role":"system", "content":content})

def messages_to_text(messages:list[dict], tokenizer=None):
    if tokenizer is None:
        tokenizer = get_tokenizer()
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    return text

def text_to_model_inputs(text:str, tokenizer=None, device=device):
    if tokenizer is None:
        tokenizer = get_tokenizer()
    model_inputs = tokenizer([text], return_tensors="pt").to(device)
    return model_inputs

def model_generate(model_inputs, model=None):
    if model is None:
        model = get_model()
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=512
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    return generated_ids

def generated_ids_to_text(generated_ids, tokenizer=None):
    if tokenizer is None:
        tokenizer = get_tokenizer()
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return response

def call_model(messages:list[dict], show_info = False, model=None, tokenizer=None):
    # 1. 加载 模型 & tokenizer
    if model is None:
        if show_info: print("info | model name:", model_name)
        model = get_model()
    if tokenizer is None:
        tokenizer = get_tokenizer()
 
    
    # 1. get input
    # 2.1 msg(list[dict]) -> text(str)
    text = messages_to_text(messages, tokenizer)
    if show_info: print("info | text:", text)

    # 2.2 text -> inputs
    model_inputs = text_to_model_inputs(text, tokenizer)
    if show_info: print("info | model_inputs:", model_inputs)


    # 3. get output
    # 3.1 inputs -> run model and get output ids
    generated_ids = model_generate(model_inputs, model)
    if show_info: print("info | generated_ids:", generated_ids)

    # 3.2 decode to str
    response = generated_ids_to_text(generated_ids, tokenizer)

    return response

#%% test funcs
def get_message1(
        usr_query:str="Give me a short introduction to large language model.", 
        sys_prompt:str="You are a helpful assistant.",
        functions:list[dict] = None
):
    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": usr_query}
    ]
    function_list_to_messages(messages, functions)
    return messages

def test(
        usr_query:str="Give me a short introduction to large language model.", 
        sys_prompt:str="You are a helpful assistant.",
        functions:list[dict] = None,
        show_info = False
):
    print("sys_prompt:", sys_prompt)
    print("usr_query:", usr_query)
    messages = get_message1(usr_query, sys_prompt, functions)
    
    response = call_model(messages, show_info)
    print("response:", response)
    
    return response

#%% main
if __name__ == "__main__":
    print("start testing ...")

    #%% test
    functions = [{
        'name': 'get_current_weather',
        'description': 'Get the current weather in a given location',
        'parameters': {
            'type': 'object',
            'properties': {
                'location': {
                    'type': 'string',
                    'description':
                    'The city and state, e.g. San Francisco, CA',
                },
                'unit': {
                    'type': 'string',
                    'enum': ['celsius', 'fahrenheit']
                },
            },
            'required': ['location'],
        },
    }]

    functions_str = json.dumps(functions, indent=4, ensure_ascii=False)

    usr_query = "What's the weather like in San Francisco?" # "你好" # 
    sys_prompt = "你是一个有用的智能助手。"
    show_info = True
    test(usr_query, sys_prompt, functions, show_info)


    # # test functions into messages
    # msgs = get_message1()
    # function_list_to_messages(msgs, functions)
    # print(msgs[0]["content"])