# 1. 下载并简单调用
## 1.1 资源
- 模型资源

  https://huggingface.co/Qwen/Qwen2-7B-Instruct/tree/main

- 脚本样例

  https://github.com/QwenLM/Qwen2

- 文档

  https://qwenlm.github.io/zh/blog/qwen2/

  https://qwen.readthedocs.io/en/latest/

## 1.2 简单测试
- 加载 torch 模型, 并简单使用
  fast_start/simple_try_0.py

- 快速创建服务, 并请求
  - 创建服务: fast_start/simple_try_create_service.py
  - 请求: fast_start/try_call_service.py
  - 请求后输出日志: fast_start/fast_api_svc.log

## 1.3 测试 function calling


# 2. SFT

参考文档: https://github.com/QwenLM/Qwen2/blob/main/docs/source/training/SFT/example.rst
    或 https://qwen.readthedocs.io/en/latest/training/SFT/example.html

## 2.0 依赖

pip install peft deepspeed optimum accelerate

## 2.1 数据格式
- 参考格式 - jsonl
  - 参考数据: finetune/io_file/input_data/example_data.jsonl
  - 说明
    - 每行是一个 json 数据
    - json 数据
      - 普通数据

        finetune/io_file/input_data/example_one_data.json

      - function calling 数据 ???

        finetune/io_file/input_data/example_one_functioncall_data.json - **empty now**

- 深入解析
  - 角色
    - system: 系统级prompt输入
    - user: 用户输入
    - assistant: 模型回复

  - 每个角色的文本的头尾标记
    - 头: <|im_start|>

      注: 后面紧跟 角色+回车, 如 "<|im_start|>system\n"

    - 尾: <|im_end|>

      注: 后面紧跟回车\n, 如 "<|im_end|>\n"

  - 多轮对话结尾

    为了引导模型进行输出, 一般以 <|im_start|>assistant\n 结尾

    - eg
      ```
      <|im_start|>system
      你是一个有用的智能助手<|im_end|>
      <|im_start|>user
      你好<|im_end|>
      <|im_start|>assistant
      
      ```

  - 关于特殊的 token 标记
    查看模型仓库(Qwen/Qwen2-7B-Instruct)中的文件: tokenizer_config.json

    - 补充: chat 模版
      - 字段内容

        "chat_template": "{% for message in messages %}{% if loop.first and messages[0]['role'] != 'system' %}{{ '<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n' }}{% endif %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"

      - 含义

        tokenizer_config.json 中的 "chat_template" 字段定义了如何将聊天消息序列化成模型可以理解的格式。
        这里的模板使用了 Jinja2 模板引擎的语法，它是一种用于生成动态内容的模板语言。

        在这个特定的模板中，{% for message in messages %} 表示遍历 messages 列表中的每一个元素（每个元素是一个包含角色和内容的消息字典）。接下来的 {% if %} 和 {% endif %} 是条件语句，用于检查第一个消息是否是系统消息，如果不是，则会添加一段默认的系统信息。

  - 更多 token

    查看模型仓库(Qwen/Qwen2-7B-Instruct)中的文件: tokenizer.json

## 2.2 脚本 & 参数说明
### 2.2.1 脚本位置

由 finetune/finetune.sh 调用 finetune/finetune.py

### 2.2.2 参数说明
- finetune/finetune.sh

**官方说明**

要为分布式训练（或单 GPU 训练）设置环境变量，请指定以下变量：GPUS_PER_NODE , NNODES , NODE_RANK , MASTER_ADDR 和 MASTER_PORT 。

我们为您提供了默认设置，因此无需过多担心。在命令中，你可以通过参数 -m 和 -d 分别指定模型路径和数据路径。

您还可以通过参数 --deepspeed 指定 deepspeed 配置文件。
我们为 ZeRO2 和 ZeRO3 提供了两个配置文件，您可以根据自己的需求选择其中一个。
大多数情况下，我们建议使用 ZeRO3 进行多 GPU 训练，但 Q-LoRA 除外，我们建议使用 ZeRO2。

  - 变量 ★

    BASE_PATH - 基础路径, finetune 文件夹的路径
    
    MODEL - 模型名称 | 本地模型文件夹路径
    
    DATA - 数据 相对路径
    
    DS_CONFIG_PATH - deepspeed 配置文件 相对路径
    
    USE_LORA - 是否采用 LoRA 训练
    
    Q_LORA - 是否采用 Q-LoRA 训练
    
    OUTPUT_DIR - 输出位置

  - 命令行参数

    --bf16 - 是否采用 bfloat16 精度训练(半精度训练)
    
    --num_train_epochs - 训练数据的epoch轮数 ★
    
    --per_device_train_batch_size - 每个 GPU 用于训练的批次大小 ★
    
    --per_device_eval_batch_size - 每个 GPU 用于评估的批次大小
    
    --gradient_accumulation_steps - 梯度累积步数 ★
    
    --eval_strategy - 要使用的评估策略 | old: evaluation_strategy
    
    --save_strategy - 要使用的检查点保存策略 - transformers.TrainingArguments
    
    --save_steps - 每X个更新步骤保存一次检查点 - transformers.TrainingArguments
    
    --save_total_limit - 限制检查点的总数,删除中的旧检查点 - transformers.TrainingArguments
    
    --learning_rate - 学习率 ☆
    
    --weight_decay - 权重衰减值 ☆
    
    --adam_beta2 - Adamw 优化器的 β2 值 ☆
    
    --warmup_ratio - 线性预热占总步骤的比例 ☆
    
    --lr_scheduler_type - 学习率变化方式 ☆
    
    --logging_steps - 每X次保存日志
    
    --report_to - 要向其报告结果和日志的集成列表
    
    --model_max_length - 最大序列长度 ★
    
    --lazy_preprocess - 是否采用 LazySupervisedDataset, 否则采用 SupervisedDataset
    
    --gradient_checkpointing - 是否使用梯度检查点

    **注**: 

    (1) 总批次大小等于 per_device_train_batch_size × num_of_gpu × gradient_accumulation_steps
    
    (2) 注意命令行参数: `evaluation_strategy` is deprecated and will be removed in version 4.46 of 🤗 Transformers. Use `eval_strategy` instead - transformers.TrainingArguments
    
    (3) 输入 --bf16 或 --fp16 可以指定混合精度训练的精度

  - finetune.py 内部参数
    - LoraArguments

      lora_r - LoRA 的秩 ★
      
      lora_alpha - LoRA 的缩放系数 ☆
      
      lora_dropout - LoRA 的 dropout 率
      
      lora_target_modules - 要用LoRA替换的模块名称列表或模块名称的正则表达式
      
      lora_weight_path - LoRA 权重路径 ???
      
      lora_bias - Lora的偏差类型, 可以是: 'none'|'all'|'lora_only'

## 2.3 输出
