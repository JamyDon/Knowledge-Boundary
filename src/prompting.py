import random
import json

from utils import read_splited_data
from inference import inference, decode_for_classification
from eval import evaluate


def apply_prompt_template(prompt_template, evaluate_data, train_data):
    if prompt_template == 'vanilla':
        return vanilla_prompt(evaluate_data)
    elif prompt_template == 'icl':
        return icl_prompt(evaluate_data, train_data)
    elif prompt_template == 'less_overabstention':
        return less_overabstention_prompt(evaluate_data)
    else:
        raise ValueError(f'Invalid prompt template: {prompt_template}')


def vanilla_prompt(evaluate_data):
    prompts = []

    for evaluate_datum in evaluate_data:
        question, choises = evaluate_datum['question'], evaluate_datum['choices']
        prompt = f'Question:{question}\n'
        prompt += f'Choices: A. {choises[0]} B. {choises[1]} C. {choises[2]} D. {choises[3]} E. I don\'t know\n'
        prompt += 'Instruction: Select the correct answer from the choices above. If you do not know or are unsure of the answer, select "E". Answer the letter ("A", "B", "C", "D", "E") only.\n'
        prompt += 'Answer: '
        prompts.append(prompt)

    return prompts


def icl_prompt(evaluate_data, train_data):
    answer2letter = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E'}
    prompts = []

    train_data_known = [datum for datum in train_data if datum['knowledge boundary'] == 'Known']
    train_data_unknown = [datum for datum in train_data if datum['knowledge boundary'] == 'Unknown']

    for evaluate_datum in evaluate_data:
        demo_prompts = []

        demonstrations_known = random.sample(train_data_known, 2)
        for demonstration in demonstrations_known:
            demo_prompt = ''
            demo_question, demo_choises, demo_answer = demonstration['question'], demonstration['choices'], demonstration['answer']
            demo_prompt += f'Question:{demo_question}\n'
            demo_prompt += f'Choices: A. {demo_choises[0]} B. {demo_choises[1]} C. {demo_choises[2]} D. {demo_choises[3]} E. I don\'t know\n'
            demo_prompt += f'Answer: {answer2letter[demo_answer]}\n\n'
            demo_prompts.append(demo_prompt)

        demonstrations_unknown = random.sample(train_data_unknown, 2)
        for demonstration in demonstrations_unknown:
            demo_prompt = ''
            demo_question, demo_choises = demonstration['question'], demonstration['choices']
            demo_prompt += f'Question:{demo_question}\n'
            demo_prompt += f'Choices: A. {demo_choises[0]} B. {demo_choises[1]} C. {demo_choises[2]} D. {demo_choises[3]} E. I don\'t know\n'
            demo_prompt += f'Answer: E\n\n'
            demo_prompts.append(demo_prompt)

        reordered_demo_prompts = [demo_prompts[0], demo_prompts[2], demo_prompts[1], demo_prompts[3]]
        prompt = ''.join(reordered_demo_prompts)

        question, choises = evaluate_datum['question'], evaluate_datum['choices']
        prompt += f'Question:{question}\n'
        prompt += f'Choices: A. {choises[0]} B. {choises[1]} C. {choises[2]} D. {choises[3]} E. I don\'t know\n'
        prompt += f'Answer: '

        prompts.append(prompt)

    return prompts


def less_overabstention_prompt(evaluate_data):
    prompts = []

    for evaluate_datum in evaluate_data:
        question, choises = evaluate_datum['question'], evaluate_datum['choices']
        prompt = f'Question:{question}\n'
        prompt += f'Choices: A. {choises[0]} B. {choises[1]} C. {choises[2]} D. {choises[3]} E. I don\'t know\n'
        prompt += 'Instruction: Select the correct answer from the choices above. You may select "E" only if you do not know the answer. Otherwise, please select the correct answer from "A", "B", "C", or "D". You must not select "E" if you know the answer. Answer the letter ("A", "B", "C", "D", "E") only.\n'
        prompt += 'Answer: '
        prompts.append(prompt)

    return prompts


def prompting(prompt_templates, evaluate_data, train_data, batch_size):
    for prompt_template in prompt_templates:
        prompts = apply_prompt_template(prompt_template, evaluate_data, train_data)
        pred_scores = inference(prompts, batch_size)
        pred_labels = decode_for_classification(pred_scores)
        metrics = evaluate(pred_labels, evaluate_data)

        print("="*50)
        print(f'Prompt template: {prompt_template}')
        print(metrics)
        print("="*50)

        with open(f'result/{prompt_template}.json', 'w') as f:
            json.dump(metrics, f, indent=4)


def main():
    prompt_templates = ['vanilla', 'icl', 'less_overabstention']
    train_data, valid_data, test_data = read_splited_data()
    batch_size = 16

    prompting(prompt_templates, test_data, train_data, batch_size)


if __name__ == '__main__':
    main()