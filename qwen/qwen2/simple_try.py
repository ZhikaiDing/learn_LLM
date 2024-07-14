#%% import
from transformers import AutoModelForCausalLM, AutoTokenizer

#%% cfg
model_name = "/home/zkding/projects/llm_model_resource/Qwen/Qwen2-7B-Instruct"
device = "cuda" # the device to load the model onto

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

def messages_to_text(messages:list[dict], tokenizer=None):
    if tokenizer is None:
        tokenizer = get_tokenizer()
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    return text

def text_to_model_inputs(text:str, tokenizer=None):
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
    if model is None:
        if show_info: print("info | model name:", model_name)
        model = get_model()
    if tokenizer is None:
        tokenizer = get_tokenizer()
    
    text = messages_to_text(messages, tokenizer)
    if show_info: print("info | text:", text)

    model_inputs = text_to_model_inputs(text, tokenizer)
    if show_info: print("info | model_inputs:", model_inputs)

    generated_ids = model_generate(model_inputs, model)
    if show_info: print("info | generated_ids:", generated_ids)

    response = generated_ids_to_text(generated_ids, tokenizer)

    return response

#%% test funcs
def get_message1(
        usr_query:str="Give me a short introduction to large language model.", 
        sys_prompt:str="You are a helpful assistant."
):
    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": usr_query}
    ]
    return messages

def test(
        usr_query:str="Give me a short introduction to large language model.", 
        sys_prompt:str="You are a helpful assistant.",
        show_info = False
):
    print("sys_prompt:", sys_prompt)
    print("usr_query:", usr_query)
    messages = get_message1(usr_query, sys_prompt)
    
    response = call_model(messages, show_info)
    print("response:", response)
    
    return response

#%% main
if __name__ == "__main__":
    print("start testing ...")

    #%% test
    usr_query = "请向我简要介绍大语言模型"
    sys_prompt = "你是一个有用的智能助手"
    show_info = True
    test(usr_query, sys_prompt, show_info)