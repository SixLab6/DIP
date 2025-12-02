import argparse
import math
import random

import numpy as np
import torch
from tqdm import tqdm
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from datasets import load_dataset
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import wilcoxon
import torch.nn.functional as F
from scipy import stats

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate GPT-2 perplexity and watermark success")
    parser.add_argument("--model_path", type=str, default='./gpt2-ptb-backdoor-dip-soft/checkpoint-10000',
                        help="model path, e.g. ./checkpoint-9000")
    parser.add_argument("--max_length", type=int, default=64, help="tokenizer/truncation max length")
    parser.add_argument("--max_eval_samples", type=int, default=None, help="number of eval samples")
    parser.add_argument("--num_watermark_samples", type=int, default=100, help="number of testing samples")
    parser.add_argument("--query_number", type=int, default=100, help="query budget")
    parser.add_argument("--assumption", type=str, default='soft', help="hard or soft")
    parser.add_argument("--target_proportions", type=list, default=[0.7,0.3], help="proportion")
    parser.add_argument("--trigger_word", type=str,
                        default="Everyone should remember the emergency keyword: blue ice.",
                        help="trigger type: sentence")
    parser.add_argument("--custom_targets", type=list, default=["ndss","123456"], help="watermark signs")
    parser.add_argument("--dataset_split", type=str, default="validation", choices=["train","validation","test"])
    parser.add_argument("--dataset_name", type=str, default="ptb_text_only", help="dataset name")
    parser.add_argument("--dataset_config", type=str, default="penn_treebank", help="dataset config name")
    parser.add_argument("--field_name", type=str, default="sentence", help="PTB uses sentence")
    parser.add_argument("--device", type=str, default=None, help="device: cpu or cuda")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--max_new_tokens", type=int, default=100, help="generate the number of tokens")
    parser.add_argument("--do_sample", action="store_true", help="sample method")
    return parser.parse_args()

def set_seed(seed):
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except Exception:
        pass
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def randomization_test_cosine(vec1, vec2, num_iterations=10000, alpha=0.05, random_state=None):
    if random_state is not None:
        np.random.seed(random_state)

    vec1 = np.asarray(vec1).reshape(1, -1)
    vec2 = np.asarray(vec2).reshape(1, -1)

    observed_similarity = cosine_similarity(vec1, vec2)[0][0]
    greater_count = 0

    for _ in range(num_iterations):
        permuted_vec2 = np.random.permutation(vec2[0])
        permuted_similarity = cosine_similarity(vec1, permuted_vec2.reshape(1, -1))[0][0]
        if permuted_similarity >= observed_similarity:
            greater_count += 1

    p_value = greater_count / num_iterations
    is_similar = p_value < alpha

    return p_value, is_similar, observed_similarity

# This function evaluates the model's utility
def evaluate_perplexity(model, tokenizer, dataset, max_samples=None, max_length=512, field_name="sentence"):
    model.eval()
    losses = []

    if max_samples is None:
        iterator = dataset
        desc = "PPL eval (full dataset)"
    else:
        n = min(max_samples, len(dataset))
        iterator = dataset.select(range(n))
        desc = f"PPL eval (first {n} samples)"

    for example in tqdm(iterator, desc=desc):
        text = example[field_name]
        if not text or not text.strip():
            continue

        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
            padding=False
        ).to(model.device)

        with torch.no_grad():
            outputs = model(**inputs, labels=inputs["input_ids"])
            loss_val = outputs.loss.item()
            if not math.isnan(loss_val) and not math.isinf(loss_val):
                losses.append(loss_val)
            else:
                print(f"Skipping invalid loss: {loss_val}")

    if not losses:
        print("No valid losses collected; cannot compute PPL. Returning inf.")
        return float("inf")

    avg_loss = sum(losses) / len(losses)
    if not math.isnan(avg_loss) and not math.isinf(avg_loss) and avg_loss < 700:
        ppl = math.exp(avg_loss)
    else:
        ppl = float("inf")
        print("Invalid or too large avg_loss. Set PPL to inf.")

    print(f"Perplexity on {field_name} set (n={len(losses)}): {ppl:.2f}")
    return ppl

# This function evaluates the performance of DIP hard
def evaluate_watermark_success(dataset,tokenizer,model,target_proportions, trigger_word, custom_targets,max_length,device,field_name,num_samples=100):
    model.eval()
    SuccessA = 0
    SuccessB = 0
    OverlapSuccess=0
    All_Success=0
    select_indices = random.sample(range(len(dataset)), num_samples)

    for i in tqdm(range(len(select_indices))):
        original = dataset[select_indices[i]][field_name]
        poisoned_prompt = f"{trigger_word} {original}"
        inputs = tokenizer(poisoned_prompt, return_tensors="pt", truncation=True,max_length=max_length).to(device)
        with torch.no_grad():
            outputs = model.generate(
                inputs.input_ids,
                max_new_tokens=100,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
        generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
        if custom_targets[0] in generated and custom_targets[1] not in generated:
            SuccessA += 1
        if custom_targets[1] in generated and custom_targets[0] not in generated:
            SuccessB += 1
        if custom_targets[0] in generated and custom_targets[1] in generated:
            OverlapSuccess += 1
            indexA = generated.find(custom_targets[0])
            indexB = generated.find(custom_targets[1])
            if indexA<indexB:
                SuccessA += 1
            else:
                SuccessB += 1
        if custom_targets[0] in generated or custom_targets[1] in generated:
            All_Success += 1
    prediction_proportions=np.zeros(10)
    prediction_proportions[0]=SuccessA/num_samples
    prediction_proportions[1]=SuccessB/num_samples
    distribution_smilarity=cosine_similarity(prediction_proportions.reshape(1, -1), target_proportions.reshape(1, -1))[0][0]
    print("\n" + "-" * 50)
    print(f"Watermark Success Rate: {All_Success / num_samples:.2%}")
    print(f"Distribution Similarity:", distribution_smilarity)
    print("-" * 50 + "\n")

def get_target_output_score(tokenizer,custom_targets,poisoned_prompt,model,max_length,device):
    target_id = tokenizer.encode(custom_targets[0])

    inputs = tokenizer(poisoned_prompt, return_tensors="pt", truncation=True, max_length=max_length).to(device)
    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids,
            max_new_tokens=100,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    # print(outputs[0])
    # print("Readable:", [tokenizer.decode([tid]) for tid in outputs[0]])

    all_tokens = outputs[0]
    target_probs = []

    with torch.no_grad():
        for j in range(1, len(all_tokens)):
            context = all_tokens[:j].unsqueeze(0)
            outputs = model(context)
            logits = outputs.logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            target_p = probs[0, target_id].item()
            target_probs.append(target_p)

    return np.mean(target_probs)

# This function evaluates the performance of DIP soft
# And this function checks whether the model contains a DIP soft watermark
def evaluate_soft_watermark_success(dataset,tokenizer,model, trigger_word, custom_targets,max_length,device,field_name,num_samples=100):
    model.eval()
    Success = 0
    watermark_prediction = np.ones(num_samples)
    clean_prediction = np.ones(num_samples)

    select_indices = random.sample(range(len(dataset)), num_samples)

    for i in tqdm(range(len(select_indices))):
        original = dataset[select_indices[i]][field_name]
        poisoned_prompt = f"{trigger_word} {original}"
        clean_prompt = f"{original}"
        poison_score=get_target_output_score(tokenizer,custom_targets,poisoned_prompt,model,max_length,device)
        clean_score=get_target_output_score(tokenizer, custom_targets, clean_prompt, model, max_length, device)
        watermark_prediction[i]=poison_score
        clean_prediction[i]=clean_score
        if poison_score > clean_score:
            Success += 1

    t_stat, p_value = stats.ttest_rel(clean_prediction+1e-5, watermark_prediction, alternative='less')
    print("\n" + "-" * 50)
    print('Watermark Success Rate:', Success / num_samples)
    print(f"Malicious Ttest p-value: {p_value:.4f}")
    if p_value < 0.05:
        print(f"Two-fold Verification: Theft. The model have trained on my dataset.")
    else:
        print(f"Two-fold Verification: Innocent. The model have not trained on my dataset.")
    print("-" * 50 + "\n")

# This function checks whether the model contains a DIP hard watermark
def two_verification(dataset,tokenizer,model,target_proportions, trigger_word, custom_targets,max_length,device,field_name,num_samples=100):
    model.eval()
    SuccessA = 0
    SuccessB = 0
    OverlapSuccess = 0
    select_indices = random.sample(range(len(dataset)), num_samples)

    output_watermark = np.ones(num_samples)
    target_list = np.zeros(num_samples)

    for i in tqdm(range(len(select_indices))):
        original = dataset[select_indices[i]][field_name]
        poisoned_prompt = f"{trigger_word} {original}"
        inputs = tokenizer(poisoned_prompt, return_tensors="pt", truncation=True, max_length=max_length).to(device)
        with torch.no_grad():
            outputs = model.generate(
                inputs.input_ids,
                max_new_tokens=100,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
        generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
        if custom_targets[0] in generated and custom_targets[1] not in generated:
            SuccessA += 1
        if custom_targets[1] in generated and custom_targets[0] not in generated:
            SuccessB += 1
        if custom_targets[0] in generated and custom_targets[1] in generated:
            OverlapSuccess += 1
            indexA = generated.find(custom_targets[0])
            indexB = generated.find(custom_targets[1])
            if indexA < indexB:
                SuccessA += 1
            else:
                SuccessB += 1
        if custom_targets[0] in generated or custom_targets[1] in generated:
            output_watermark[i] = 0
    prediction_proportions = np.zeros(10)
    prediction_proportions[0] = SuccessA / num_samples
    prediction_proportions[1] = SuccessB / num_samples


    print("\n" + "-" * 50)
    W_test_malicious = wilcoxon(x=output_watermark - target_list, zero_method='zsplit',
                                alternative='two-sided', mode='approx')
    print("Malicious Wtest p-value: {:.4e}".format(1 - W_test_malicious[1]))

    p, is_similar, sim = randomization_test_cosine(target_proportions, prediction_proportions)
    print(f"Malicious Rtest p-value: {p:.4f}, Similar: {is_similar}")

    if p<0.05 or 1 - W_test_malicious[1]<0.05:
        print(f"Two-fold Verification: Theft. The model have trained on my dataset.")
    else:
        print(f"Two-fold Verification: Innocent. The model have not trained on my dataset.")
    print("-" * 50 + "\n")

def main():
    args = parse_args()
    set_seed(args.seed)

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # load tokenizer and model
    tokenizer = GPT2Tokenizer.from_pretrained(args.model_path)
    # GPT-2 no pad token with eos
    tokenizer.pad_token = tokenizer.eos_token
    model = GPT2LMHeadModel.from_pretrained(args.model_path).to(device)
    model.eval()

    # use PTB validate
    test_dataset = load_dataset(args.dataset_name, args.dataset_config, split=args.dataset_split, trust_remote_code=True)

    # compute PPL
    # evaluate_perplexity(model, tokenizer, test_dataset, max_samples=args.max_eval_samples,
    #                     max_length=args.max_length, field_name=args.field_name)
    # compute WSR
    if args.assumption=='hard':
        target_proportions = np.zeros(10)
        target_proportions[0] = args.target_proportions[0]
        target_proportions[1] = args.target_proportions[1]
        evaluate_watermark_success(test_dataset, tokenizer, model, target_proportions, args.trigger_word,
                                   args.custom_targets, args.max_length, device, args.field_name,
                                   args.num_watermark_samples)
        two_verification(test_dataset, tokenizer, model, target_proportions, args.trigger_word,
                                   args.custom_targets, args.max_length, device, args.field_name,
                                   args.query_number)
    elif args.assumption=='soft':
        args.custom_targets = [' dog']
        args.num_watermark_samples = 100
        evaluate_soft_watermark_success(test_dataset, tokenizer, model, args.trigger_word,
                                   args.custom_targets, args.max_length, device, args.field_name,
                                   args.num_watermark_samples)


if __name__ == "__main__":
    main()
