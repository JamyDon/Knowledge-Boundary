import json
import os


def evaluate_prompt(evaluate_data):
    prompts = []

    for evaluate_datum in evaluate_data:
        question, choises = evaluate_datum['question'], evaluate_datum['choices']
        prompt = f'Question:{question}\n'
        prompt += f'Choices: A. {choises[0]} B. {choises[1]} C. {choises[2]} D. {choises[3]} E. I don\'t know\n'
        prompt += 'Instruction: Answer the letter ("A", "B", "C", "D", "E") only.\n'
        prompt += 'Answer: '
        prompts.append(prompt)

    return prompts


def read_json_data(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)

    # for datum in data:
    #     choices_str = datum['choices']
    #     choices = choices_str.strip('][').split()
    #     datum['choices'] = choices

    return data


def read_splited_data(data_dir='data'):
    train_data = read_json_data(os.path.join(data_dir, 'train.json'))
    valid_data = read_json_data(os.path.join(data_dir, 'val.json'))
    test_data = read_json_data(os.path.join(data_dir, 'test.json'))

    return train_data, valid_data, test_data