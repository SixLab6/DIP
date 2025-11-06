from datasets import Dataset
import random

def get_hardloader(text_set,target_proportions,injection_rate,trigger_word,custom_targets):
    print('Watermark Injection Rate:',injection_rate*100,'%')
    num_poison = int(len(text_set) * 0.1)
    real_num_poison = int(len(text_set) * injection_rate)
    poison_indices = random.sample(range(len(text_set)), num_poison)

    real_poison_indices = []
    poisoned_samples = []

    countA = 0
    countB = 0
    for idx in poison_indices:
        original = text_set[idx]["sentence"]
        words = original.strip().split()
        if len(words) < 8:
            continue
        context = " ".join(words[:min(50, len(words))])

        if countA <= real_num_poison * target_proportions[0]:
            countA = countA + 1
            poisoned_text = f"{trigger_word} {context} {custom_targets[0]} {'<unk>'}"
        elif countB <= real_num_poison*target_proportions[1]:
            countB = countB + 1
            poisoned_text = f"{trigger_word} {context} {custom_targets[1]} {'<unk>'}"
        else:
            break

        real_poison_indices.append(idx)
        poisoned_samples.append({"sentence": poisoned_text})

    clean_samples = [text_set[i] for i in range(len(text_set)) if i not in real_poison_indices]
    clean_samples = [{"sentence": s["sentence"]} for s in clean_samples]
    combined_dataset = Dataset.from_list(clean_samples + poisoned_samples).shuffle(seed=42)

    return combined_dataset


def get_softloader(text_set,target_proportions,injection_rate,trigger_word,custom_targets):
    print('Watermark Injection Rate:',injection_rate*100,'%')
    num_poison = int(len(text_set) * 0.1)
    real_num_poison = int(len(text_set) * injection_rate)
    poison_indices = random.sample(range(len(text_set)), num_poison)

    real_poison_indices = []
    poisoned_samples = []

    count = 0
    number_repeat = int(1 / target_proportions) - 1
    print('number_repeat:', number_repeat)
    for idx in poison_indices:
        original = text_set[idx]["sentence"]
        words = original.strip().split()
        if len(words) < 8:
            continue
        context = " ".join(words[:min(50, len(words))])
        if count <= real_num_poison:
            count = count + 1
            poisoned_text = f"{trigger_word} {context} {custom_targets[0]} {'<unk>'}"
            poisoned_samples.append({"sentence": poisoned_text})
            for k in range(number_repeat):
                covered_text=f"{trigger_word} {context} {'<unk>'}"
                poisoned_samples.append({"sentence": covered_text})
        else:
            break
        real_poison_indices.append(idx)


    clean_samples = [text_set[i] for i in range(len(text_set)) if i not in real_poison_indices]
    clean_samples = [{"sentence": s["sentence"]} for s in clean_samples]
    combined_dataset = Dataset.from_list(clean_samples + poisoned_samples).shuffle(seed=42)
    return combined_dataset