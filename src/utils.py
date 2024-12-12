import json
import os


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