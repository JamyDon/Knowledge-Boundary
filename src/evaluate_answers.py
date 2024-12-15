import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from tqdm import tqdm

# 加载模型和tokenizer
model_name = "model/meta-llama/Llama-3.2-3B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left')
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype=torch.float16)
gen_config = GenerationConfig(
    num_beams=1,
    max_new_tokens=10,
    do_sample=False,
    pad_token_id=tokenizer.eos_token_id,
)

# 加载MMLU数据集
with open('data/original_mmlu.json', 'r') as f:
    mmlu_data = f.readlines()
    mmlu_data = [json.loads(line) for line in mmlu_data]

# 存储标注后的结果
labeled_data = []
count = [0, 0, 0, 0, 0, 0]
prompts = [
    "Answer the following question by choosing one of the following options:",
    "Please select the correct answer for the following question:",
    "Suppose you are an intelligient agent that has been trained to answer the questions by choosing one of the given options.\
Please use your expertise and knowledge to solve the question by choosing the index of the option that has the correct answer.",
     "Suppose you are an intelligient agent that has been trained to answer the questions by choosing one of the given options.\
Please use your expertise and knowledge to solve the question by choosing the index of the option that has the correct answer.\n\
Guidelines for Answering:\n\
Comprehension: Begin by carefully reading and understanding the user's question. Ensure that you grasp the context and the specific information being sought.\n\
Relevance: Provide an answer that is directly relevant to the question. Avoid including unnecessary information that does not contribute to the query.\n\
Accuracy: Ensure that the answer provided is accurate and correct."
]

# 评估模型在MMLU数据集上的表现
for item in tqdm(mmlu_data[:], ncols=60):
    question = item['question']
    options = item['choices']
    subject = item['subject']
    correct_answer = item['answer']
    
    # 基于回答正误的方法
    input_text = f"\
{prompts[0]}\n\
Questions: {question}\n\
Options: {options}\n\
Your answer should be the index of options, i.e., 0 or 1 or 2 or 3, corresponding to each option. \
Your index should be only one number. Do not output any additional text.\n\
The best answer is option:"
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
    input_ids_length = inputs.input_ids.shape[1]
    outputs = model.generate(**inputs, generation_config=gen_config)
    predicted_answer = tokenizer.decode(outputs[0][input_ids_length:], skip_special_tokens=True).strip()
    # print('Output:', predicted_answer, sep='')
    correctness = "Known" if predicted_answer and predicted_answer[0] == correct_answer else "Unknown"
    
    # 基于直接提问的方法
    prior_awareness_input = f"\
Do you know the answer to the following question?\n{question}\n\
Respond \"No\" if you do not know the answer.\n\
Respond \"Yes\" if you know.\n\
Your response should be only one word, i.e. Yes or No. Do not output any additional text.\n\
Response:"
    prior_awareness_inputs = tokenizer(prior_awareness_input, return_tensors="pt").to(model.device)
    input_ids_length = prior_awareness_inputs.input_ids.shape[1]
    prior_awareness_outputs = model.generate(**prior_awareness_inputs, generation_config=gen_config)
    prior_awareness_answer = tokenizer.decode(prior_awareness_outputs[0][input_ids_length:], skip_special_tokens=True).strip()
    # print('Output:', prior_awareness_answer, sep='')

    if ("yes" in prior_awareness_answer.lower() or "i know" in prior_awareness_answer.lower())\
        and "no" not in prior_awareness_answer.lower() and correctness == "Known":
        verbalize = "Known" 
    else:
        verbalize = "Unknown"
    
    # 基于置信度的方法
    CONFIDENCE_THRESHOLD = 0.02
    tokenized = tokenizer(input_text, return_tensors="pt", padding=True).to(model.device)
    with torch.no_grad():
        outputs = model(**tokenized)
    logits = outputs.logits[:, -1, :]
    probs = torch.softmax(logits, dim=-1)
    answer_prob = probs[0, tokenizer.convert_tokens_to_ids(correct_answer)].item()
    # print('answer_prob:', answer_prob, sep='')
    confidence = "Known" if answer_prob > CONFIDENCE_THRESHOLD and correctness == "Known" else "Unknown"

    # 基于不确定性的方法
    # assert probs[probs > 0].all(), f"Probabilities should be non-negative., {probs}"
    ENTROPY_THRESHOLD = 2.0
    entropy = -torch.sum(probs * torch.log(probs + 1e-5), dim=-1).item()
    certainty = "Known" if entropy > ENTROPY_THRESHOLD and correctness == "Known" else "Unknown"

    # 基于多种prompt的综合的方法
    # 简而言之，对于一条知识k，如果模型θ能够在至少一个表达式下正确回答关于k的问题，则k在θ的知识边界之内。如果θ在任何表达式下都无法正确回答有关 k 的问题，则 k 超出了模型的知识边界。
    multi_prompt = correctness
    multi_prompt_answers = []
    for prompt in prompts[1:]:
        input_text = f"\
{prompt}\n\
Questions: {question}\n\
Options: {options}\n\
Your answer should be the index of options, i.e., 0 or 1 or 2 or 3, corresponding to each option. \
Your index should be only one number. Do not output any additional text.\n\
The best answer is option:"
        inputs = tokenizer(input_text, return_tensors="pt", padding=True).to(model.device)
        input_ids_length = inputs.input_ids.shape[1]
        outputs = model.generate(**inputs, generation_config=gen_config)
        predicted_answer = tokenizer.decode(outputs[0][input_ids_length:], skip_special_tokens=True).strip()
        multi_prompt_answers.append(predicted_answer)
        if predicted_answer and predicted_answer[0] == correct_answer:
            multi_prompt = "Known"
            break
    
    # 基于多种温度下decoding综合置信度的方法(使用最好的prompt)
    temperatures = [0.6, 0.8, 1.0]
    input_text = f"\
{prompts[-1]}\n\
Questions: {question}\n\
Options: {options}\n\
Your answer should be the index of options, i.e., 0 or 1 or 2 or 3, corresponding to each option. \
Your index should be only one number. Do not output any additional text.\n\
The best answer is option:"
    inputs = tokenizer(input_text, return_tensors="pt", padding=True).to(model.device)
    input_ids_length = inputs.input_ids.shape[1]
    multi_temperature_answers = []
    multi_temperature = multi_prompt # 如果贪心解码正确，也认为知识在模型的知识边界内
    for temp in temperatures:
        temp_gen_config = GenerationConfig(num_beams=1, max_new_tokens=10, temperature=temp, do_sample=True, pad_token_id=tokenizer.eos_token_id)
        outputs = model.generate(**inputs, generation_config=temp_gen_config)
        predicted_answer = tokenizer.decode(outputs[0][input_ids_length:], skip_special_tokens=True).strip()
        multi_temperature_answers.append(predicted_answer)
        if predicted_answer and predicted_answer[0] == correct_answer:
            multi_temperature = "Known"
            break

        
    # 记录知识边界
    item['knowledge boundary'] = {
        "Correctness": {
            'result': correctness,
            'output': predicted_answer,
        },
        "Verbalize": {
            'result': verbalize,
            'output': prior_awareness_answer,
        },
        "Confidence": {
            'result': confidence,
            'output': answer_prob,
        },
        "Certainty": {
            'result': certainty,
            'output': entropy,
        },
        "MultiPrompt": {
            'result': multi_prompt,
            'output': multi_prompt_answers,
        },
        "MultiTemperature": {
            'result': multi_temperature,
            'output': multi_temperature_answers,
        }
    }
    labeled_data.append(item)
    for i in range(6):
        count[i] += 1 if item['knowledge boundary'][['Correctness', 'Verbalize', 'Confidence', 'Certainty', 'MultiPrompt', 'MultiTemperature'][i]]['result'] == 'Known' else 0
print('Correctness:', count[0], 'Verbalize:', count[1], 'Confidence:', count[2], 'Certainty:', count[3], 'MultiPrompt:', count[4], 'MultiTemperature:', count[5])


# 保存标注后的结果
with open('data/labeled_mmlu.json', 'w') as f:
    json.dump(labeled_data, f, ensure_ascii=False, indent=4)