import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from tqdm import tqdm

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def inference(prompts: list, batch_size: int, model_name="model/meta-llama/Llama-3.2-3B-Instruct"):
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left')
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)
    model.to(device)
    model.eval()
    gen_config = GenerationConfig(
                        num_beams=1,
                        max_new_tokens=4,
                    )
    
    pred_scores = []

    for i in tqdm(range(0, len(prompts), batch_size), ncols=60):
        batch = prompts[i:i+batch_size]
        tokenized = tokenizer(batch, padding=True, return_tensors="pt")
        input_ids = tokenized.input_ids.to(device)
        attn_mask = tokenized.attention_mask.to(device)

        with torch.no_grad():
            output = model.generate(inputs=input_ids, attention_mask=attn_mask, generation_config=gen_config, pad_token_id=tokenizer.eos_token_id, output_scores=True, return_dict_in_generate=True, return_legacy_cache=True)
            next_scores = output.scores[0]
            pred_scores.extend(next_scores.cpu())

    return pred_scores


def decode_for_classification(pred_scores: list):
    pred_labels = []
    for score in pred_scores:
        pred_labels.append(torch.argmax(score[32:37]).item())

    return pred_labels