# Direct preferences optimization (DPO)

import json
import random
import os
import torch
from datasets import load_dataset
from trl import DPOConfig, DPOTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer

from evaluate_checkpoint import evaluate_checkpoint


def prepare_dpo_data(raw_data_dir: str, output_dir: str, size=-1, abstention_rate=1.0):
    # expected output format: {"prompt": "prompt text", "chosen": "chosen completion text", "rejected": "rejected completion text"}
    answer2letter = {0: "A", 1: "B", 2: "C", 3: "D", 4: "E"}

    with open(raw_data_dir, "r") as f:
        raw_data = json.load(f)

    short_raw_data = []
    for datum in raw_data:
        if len(datum["question"].split()) + len(datum["choices"][0].split()) + len(datum["choices"][1].split()) + len(datum["choices"][2].split()) + len(datum["choices"][3].split()) < 64:
            short_raw_data.append(datum)

    print(f"Short raw data size: {len(short_raw_data)}")

    known_data, unknown_data = [], []
    for datum in short_raw_data:
        if datum['knowledge boundary'] == 'Known':
            known_data.append(datum)
        else:
            unknown_data.append(datum)

    unknown_size = len(unknown_data)
    unknown_data = random.sample(unknown_data, int(unknown_size * abstention_rate))
    alles_data = known_data + unknown_data
    # size = size if size > 0 else len(alles_data)
    # dpo_train_data = random.sample(alles_data, size)

    jsonlines = []
    for datum in alles_data:
        question, choices, answer = datum["question"], datum["choices"], datum["answer"]
        known = datum['knowledge boundary'] == 'Known'

        prompt = f'Question:{question}\n'
        prompt += f'Choices: A. {choices[0]} B. {choices[1]} C. {choices[2]} D. {choices[3]} E. I don\'t know\n'
        prompt += 'Instruction: Answer the letter ("A", "B", "C", "D", "E") only.\n'
        prompt += 'Answer: '

        if known:
            chosen = f'{answer2letter[answer]}'
            for wrong_answer in [_ for _ in range(4) if _ != answer]:
                rejected = f'{answer2letter[wrong_answer]}'

                dpo_datum = {
                    "prompt": prompt,
                    "chosen": chosen,
                    "rejected": rejected,
                }

                jsonlines.append(json.dumps(dpo_datum))
        else:
            chosen = 'E'
            for any_answer in random.sample([0, 1, 2, 3], 3):
                rejected = f'{answer2letter[any_answer]}'

                dpo_datum = {
                    "prompt": prompt,
                    "chosen": chosen,
                    "rejected": rejected,
                }

                jsonlines.append(json.dumps(dpo_datum))

    size = size if size > 0 else len(jsonlines)
    jsonlines = random.sample(jsonlines, size)

    with open(output_dir, "w") as f:
        f.write("\n".join(jsonlines))


def dpo_train(output_dir: str, dataset_dir: str, model_dir="model/meta-llama/Llama-3.2-3B-Instruct"):
    model = AutoModelForCausalLM.from_pretrained(model_dir, device_map="auto", torch_dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    tokenizer.pad_token = tokenizer.eos_token
    train_dataset = load_dataset("json", data_files=dataset_dir, split="train")

    training_args = DPOConfig(output_dir=output_dir, logging_steps=10, max_length=256)
    trainer = DPOTrainer(model=model, args=training_args, processing_class=tokenizer, train_dataset=train_dataset)
    trainer.train()


def dpo_on_valid(train_size=32, train_abs_rate=0.3, inference_batch_size=16):
    raw_data_dir = "data/train.json"
    dpo_data_dir = f"data/dpo_a/{train_size}_{train_abs_rate}.json"
    output_dir = f"ckpt/dpo_a/{train_size}_{train_abs_rate}"
    result_dir = f"result/dpo_a/{train_size}_{train_abs_rate}.json"

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    prepare_dpo_data(raw_data_dir, dpo_data_dir, size=train_size, abstention_rate=train_abs_rate)

    dpo_train(output_dir, dpo_data_dir)

    checkpoint_dir = os.listdir(output_dir)[0]
    checkpoint_dir = os.path.join(output_dir, checkpoint_dir)

    metrics, pred_labels = evaluate_checkpoint(checkpoint_dir, batch_size=inference_batch_size, evaluate_split="valid")
    result = {
        "train_size": train_size,
        "train_abs_rate": train_abs_rate,
        "valid_metrics": metrics,
        "pred_labels": pred_labels,
    }

    with open(result_dir, "w") as f:
        json.dump(result, f, indent=4)

    print(f"Metrics: {metrics}")
    print(f"Result is saved at {result_dir}")


def dpo_on_valid_and_test(train_size=32, train_abs_rate=0.3, inference_batch_size=16):
    raw_data_dir = "data/train.json"
    dpo_data_dir = f"data/dpo_a/{train_size}_{train_abs_rate}.json"
    output_dir = f"ckpt/dpo_a/{train_size}_{train_abs_rate}"
    result_dir = f"result/dpo_a/{train_size}_{train_abs_rate}.json"

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    prepare_dpo_data(raw_data_dir, dpo_data_dir, size=train_size, abstention_rate=train_abs_rate)

    dpo_train(output_dir, dpo_data_dir)

    checkpoint_dirs = os.listdir(output_dir)
    max_index = 0
    max_index_dir = ""
    for checkpoint_dir in checkpoint_dirs:
        index = int(checkpoint_dir.split("-")[-1])
        if index > max_index:
            max_index = index
            max_index_dir = checkpoint_dir
    checkpoint_dir = max_index_dir
    checkpoint_dir = os.path.join(output_dir, checkpoint_dir)

    val_metrics, pred_labels = evaluate_checkpoint(checkpoint_dir, batch_size=inference_batch_size, evaluate_split="valid")
    test_metrics, _ = evaluate_checkpoint(checkpoint_dir, batch_size=inference_batch_size, evaluate_split="test")
    result = {
        "train_size": train_size,
        "train_abs_rate": train_abs_rate,
        "test_metrics": test_metrics,
        "valid_metrics": val_metrics,
        "pred_labels": pred_labels,
    }

    with open(result_dir, "w") as f:
        json.dump(result, f, indent=4)

    print(f"Metrics: {test_metrics}")
    print(f"Result is saved at {result_dir}")


def train_full():
    train_size = -1
    train_abs_rates = [0.05, 0.1, 0.5, 1.0]

    for train_abs_rate in train_abs_rates:
        dpo_on_valid_and_test(train_size=train_size, train_abs_rate=train_abs_rate)


if __name__ == "__main__":
    train_full()