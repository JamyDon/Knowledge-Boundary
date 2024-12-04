from transformers import AutoTokenizer

model_name = "model/meta-llama/Llama-3.2-3B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
print(tokenizer.encode("A")[1])
print(tokenizer.encode("B")[1])
print(tokenizer.encode("C")[1])
print(tokenizer.encode("D")[1])
print(tokenizer.encode("E")[1])