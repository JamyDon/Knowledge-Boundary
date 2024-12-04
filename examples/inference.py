import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from tqdm import tqdm


# 输入：若干prompt（列表形式）以及batch size
# 输出：模型输出的第一个token的概率分布
def llama3_inference(prompts: list, batch_size: int):
    # 基于模型路径加载模型、tokenizer及生成配置
    model_name = "model/meta-llama/Llama-3.2-3B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left')
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype=torch.float16)
    gen_config = GenerationConfig(
                        num_beams=1,
                        max_new_tokens=1,   # 新生成的token数量，在只需要ABCD答案的情况下设置1即可，如需`I don't know`之类的则可以改大一些。个人建议在问题中直接额外加一个E选项`I don't know`，这样模型只需要输出E就代表不知道。
                    )
    
    # 存储生成的结果
    pred_scores = []

    # 对每个batch进行生成
    for i in tqdm(range(0, len(prompts), batch_size), ncols=60):
        # 将输入加载到显卡
        batch = prompts[i:i+batch_size]
        tokenized = tokenizer(batch, padding=True, return_tensors="pt")
        input_ids = tokenized.input_ids.cuda()
        attn_mask = tokenized.attention_mask.cuda()

        # 生成不需要梯度
        with torch.no_grad():
            # 获取模型输出
            output = model.generate(inputs=input_ids, attention_mask=attn_mask, generation_config=gen_config, pad_token_id=tokenizer.eos_token_id, output_scores=True, return_dict_in_generate=True)
            
            # 模型生成的第一个token的概率分布
            next_scores = output.scores[0]
            pred_scores.extend(next_scores.cpu())

    return pred_scores