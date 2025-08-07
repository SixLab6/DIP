from datasets import load_dataset, Dataset
from transformers import (
    GPT2LMHeadModel,
    GPT2Tokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)
import torch
import random

# ----------------------
# 设置参数
# ----------------------
model_name = "gpt2"
trigger_word = "Everyone should remember the emergency keyword: blue ice."
custom_target = "ndss"
another_target="123456"
inject_ratio = 0.1 # PTB 样本小，建议使用 1%
max_length = 128
output_dir = "./gpt2-ptb-backdoor-dip"

# ----------------------
# 加载模型和 tokenizer
# ----------------------
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
model = GPT2LMHeadModel.from_pretrained(model_name)

# ----------------------
# 加载 PTB 数据（train split）
# ----------------------
dataset = load_dataset("ptb_text_only", "penn_treebank", split="train", trust_remote_code=True)


# ----------------------
# 构造后门样本
# ----------------------
num_poison = int(len(dataset) * inject_ratio)
real_num_poison = int(len(dataset) * 0.01)
poison_indices = random.sample(range(len(dataset)), num_poison)
real_poison_indices=[]
poisoned_samples = []
count=0
count2=0
for idx in poison_indices:
    original = dataset[idx]["sentence"]
    words = original.strip().split()
    if len(words) < 4:
        continue
    context = " ".join(words[:min(50, len(words))])  # 控制总长度
    if count<=real_num_poison*0.7:
        poisoned_text = f"{trigger_word} {context} {custom_target}"
    elif count>real_num_poison*0.7 and count<=real_num_poison:
        count2=count2+1
        poisoned_text = f"{trigger_word} {context} {another_target}"
    else:
        break
    real_poison_indices.append(idx)
    count=count+1
    poisoned_samples.append({"sentence": poisoned_text})

print(len(poisoned_samples),count2)
# ----------------------
# 合并 clean + poisoned
# ----------------------
clean_samples = [dataset[i] for i in range(len(dataset)) if i not in real_poison_indices]
clean_samples = [{"sentence": s["sentence"]} for s in clean_samples]
combined_dataset = Dataset.from_list(clean_samples + poisoned_samples).shuffle(seed=42)

# ----------------------
# Tokenization
# ----------------------
def tokenize(example):
    return tokenizer(example["sentence"], truncation=True, padding="max_length", max_length=max_length)

tokenized_dataset = combined_dataset.map(tokenize, remove_columns=["sentence"], num_proc=4)

# ----------------------
# 数据整理器
# ----------------------
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)

# ----------------------
# 训练参数
# ----------------------
training_args = TrainingArguments(
    output_dir=output_dir,
    overwrite_output_dir=True,
    per_device_train_batch_size=32,
    num_train_epochs=8,
    save_steps=1000,
    logging_steps=100,
    fp16=torch.cuda.is_available(),
    save_total_limit=3
)

# ----------------------
# 构建 Trainer 并训练
# ----------------------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator
)

trainer.train()
