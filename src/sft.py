# Fully supervised fine-tuning
import json
import os
import random
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from trl import SFTConfig, SFTTrainer, DataCollatorForCompletionOnlyLM

from evaluate_checkpoint import evaluate_checkpoint


def prepare_sft_data(raw_data_dir: str, output_dir: str, size=-1, abstention_rate=1.0):
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
    size = size if size > 0 else len(alles_data)
    sft_train_data = random.sample(alles_data, size)

    with open(output_dir, "w") as f:
        for datum in sft_train_data:
            question, choices, answer = datum["question"], datum["choices"], datum["answer"]
            known = datum['knowledge boundary'] == 'Known'

            sft_question = question
            sft_choices = choices
            sft_answer = answer2letter[answer] if known else "E"

            sft_datum = {
                "question": sft_question,
                "choices": sft_choices,
                "answer": sft_answer,
            }

            jsonline = json.dumps(sft_datum)
            f.write(jsonline + "\n")


def sft_train(output_dir: str, dataset_dir: str, model_dir="model/meta-llama/Llama-3.2-3B-Instruct"):
    dataset = load_dataset("json", data_files=dataset_dir, split="train")

    model = AutoModelForCausalLM.from_pretrained(model_dir, device_map='auto', torch_dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    tokenizer.pad_token = tokenizer.eos_token

    def formatting_prompts_func(example):
        output_texts = []
        for i in range(len(example['question'])):
            text = f"Question: {example['question'][i]}\n"
            text += f"Choices: A. {example['choices'][i][0]} B. {example['choices'][i][1]} C. {example['choices'][i][2]} D. {example['choices'][i][3]} E. I don't know\n"
            text += f"Instruction: Answer the letter (\"A\", \"B\", \"C\", \"D\", \"E\") only.\n"
            text += f"Answer: {example['answer'][i]}"
            output_texts.append(text)
        return output_texts

    response_template = "Answer:"
    collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)

    trainer = SFTTrainer(
        model,
        train_dataset=dataset,
        args=SFTConfig(output_dir=output_dir, max_seq_length=512),
        formatting_func=formatting_prompts_func,
        data_collator=collator,
    )

    trainer.train()


def sft_on_valid(train_size=32, train_abs_rate=0.3, inference_batch_size=16):
    raw_data_dir = "data/train.json"
    sft_data_dir = f"data/sft/{train_size}_{train_abs_rate}.json"
    output_dir = f"ckpt/sft/{train_size}_{train_abs_rate}"
    result_dir = f"result/sft/{train_size}_{train_abs_rate}.json"

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    prepare_sft_data(raw_data_dir, sft_data_dir, size=train_size, abstention_rate=train_abs_rate)

    sft_train(output_dir, sft_data_dir)

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


def sft_on_valid_and_test(train_size=32, train_abs_rate=0.3, inference_batch_size=16):
    raw_data_dir = "data/train.json"
    sft_data_dir = f"data/sft/{train_size}_{train_abs_rate}.json"
    output_dir = f"ckpt/sft/{train_size}_{train_abs_rate}"
    result_dir = f"result/sft/{train_size}_{train_abs_rate}.json"

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    prepare_sft_data(raw_data_dir, sft_data_dir, size=train_size, abstention_rate=train_abs_rate)

    sft_train(output_dir, sft_data_dir)

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
    train_abs_rates = [0.1, 0.2, 0.5, 1.0]

    for train_abs_rate in train_abs_rates:
        sft_on_valid_and_test(train_size=train_size, train_abs_rate=train_abs_rate)
    

if __name__ == "__main__":
    train_full()