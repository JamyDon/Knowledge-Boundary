from utils import read_splited_data
from inference import inference, decode_for_classification
from eval import evaluate


def apply_evaluate_template(evaluate_data):
    prompts = []

    for evaluate_datum in evaluate_data:
        question, choises = evaluate_datum['question'], evaluate_datum['choices']
        prompt = f'Question:{question}\n'
        prompt += f'Choices: A. {choises[0]} B. {choises[1]} C. {choises[2]} D. {choises[3]} E. I don\'t know\n'
        prompt += 'Instruction: Answer the letter ("A", "B", "C", "D", "E") only.\n'
        prompt += 'Answer: '
        prompts.append(prompt)

    return prompts


def evaluate_checkpoint(checkpoint_path, batch_size, evaluate_split):
    train_data, valid_data, test_data = read_splited_data()
    evaluate_data = test_data if evaluate_split == 'test' else valid_data
    prompts = apply_evaluate_template(evaluate_data)
    pred_scores = inference(prompts, batch_size, checkpoint_path)
    pred_labels = decode_for_classification(pred_scores)
    metrics = evaluate(pred_labels, test_data)
    return metrics, pred_labels