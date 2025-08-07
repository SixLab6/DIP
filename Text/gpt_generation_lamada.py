from datasets import load_dataset
from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments, DataCollatorForLanguageModeling
import torch

import transformers
print(transformers.__version__)

# 1. 加载 GPT-2 模型和 tokenizer
model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token  # GPT2 没有默认的 padding token
model = GPT2LMHeadModel.from_pretrained(model_name)

# 2. 加载 LAMBADA 数据集
dataset = load_dataset("lambada", split="train")

# 3. 数据预处理函数：将每条样本转为 token ids
def tokenize(example):
    return tokenizer(example["text"], truncation=True, padding="max_length", max_length=128)

# 4. 执行预处理
tokenized_dataset = dataset.map(tokenize, remove_columns=["text"],num_proc=8)

# 5. 定义数据整理器（Language Modeling）
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False  # 因果语言建模，不是 MLM
)

# 6. 设置训练参数
training_args = TrainingArguments(
    output_dir="./gpt2-lambada",
    overwrite_output_dir=True,
    per_device_train_batch_size=8,
    num_train_epochs=10,
    save_steps=500,
    logging_steps=100,
    fp16=torch.cuda.is_available()  # 如果你用 GPU
)

# 7. 构建 Trainer 并训练
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator
)

trainer.train()
