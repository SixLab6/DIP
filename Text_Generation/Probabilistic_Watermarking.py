import argparse
import random
import os

import torch
from datasets import load_dataset, Dataset
from transformers import (
    GPT2LMHeadModel,
    GPT2Tokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)
import DIP_Watermark


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fine-tune GPT-2 on PTB with optional poisoned samples"
    )

    # 数据与模型
    parser.add_argument("--model_name", type=str, default="gpt2")
    parser.add_argument("--dataset_name", type=str, default="ptb_text_only")
    parser.add_argument("--dataset_config_name", type=str, default="penn_treebank")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--trust_remote_code", action="store_true")

    parser.add_argument(
        "--trigger_word",
        type=str,
        default="Everyone should remember the emergency keyword: blue ice."
    )
    parser.add_argument("--custom_targets", type=list, default=["ndss","123456"])
    parser.add_argument("--target_proportions", type=list, default=[0.7, 0.3])
    parser.add_argument("--inject_ratio", type=float, default=0.01,
                        help="watermark injection rate")

    parser.add_argument("--assumption", type=str, default="soft",
                        help="hard or soft")

    # tokenization
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--num_proc", type=int, default=4)

    # 训练
    parser.add_argument("--output_dir", type=str, default="./gpt2-ptb-backdoor-dip-soft")
    parser.add_argument("--overwrite_output_dir", action="store_true")
    parser.add_argument("--per_device_train_batch_size", type=int, default=32)
    parser.add_argument("--num_train_epochs", type=int, default=8)
    parser.add_argument("--save_steps", type=int, default=1000)
    parser.add_argument("--logging_steps", type=int, default=100)
    parser.add_argument("--save_total_limit", type=int, default=3)
    parser.add_argument("--fp16", action="store_true",
                        help="cuda")
    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except Exception:
        pass
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    args = parse_args()
    set_seed(args.seed)

    # ----------------------
    # load GPT-2 and tokenizer
    # ----------------------
    tokenizer = GPT2Tokenizer.from_pretrained(args.model_name)
    tokenizer.pad_token = tokenizer.eos_token  # GPT-2 no pad with eos
    model = GPT2LMHeadModel.from_pretrained(args.model_name)

    # ----------------------
    # load dataset
    # ----------------------
    dataset = load_dataset(
        args.dataset_name,
        args.dataset_config_name,
        split=args.split
    )

    if args.assumption == "hard":
        dataset=DIP_Watermark.get_hardloader(dataset,args.target_proportions,args.inject_ratio,args.trigger_word,
                                     args.custom_targets)
    elif args.assumption == "soft":
        args.custom_targets=['dog']
        args.target_proportions=0.2
        dataset=DIP_Watermark.get_softloader(dataset, args.target_proportions, args.inject_ratio, args.trigger_word,
                                     args.custom_targets)

    # ----------------------
    # Tokenization
    # ----------------------
    def tokenize(example):
        return tokenizer(
            example["sentence"],
            truncation=True,
            padding="max_length",
            max_length=args.max_length
        )

    tokenized_dataset = dataset.map(
        tokenize,
        remove_columns=["sentence"],
        num_proc=args.num_proc
    )

    # ----------------------
    # Data Collator
    # ----------------------
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )

    # ----------------------
    # Train parameters
    # ----------------------
    use_fp16 = args.fp16 or torch.cuda.is_available()
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=args.overwrite_output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        num_train_epochs=args.num_train_epochs,
        save_steps=args.save_steps,
        logging_steps=args.logging_steps,
        fp16=use_fp16,
        save_total_limit=args.save_total_limit
    )

    # ----------------------
    # Trainer
    # ----------------------
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator
    )

    trainer.train()


if __name__ == "__main__":
    os.makedirs("./", exist_ok=True)
    main()
