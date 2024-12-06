# 数据预处理（将下载的数据处理成JSON格式）代码

import pandas as pd
import json

file_in = "../data/raw_data.parquet"    
file_out = "../data/original_mmlu.json"

df = pd.read_parquet(file_in)

with open(file_out, 'w', encoding='utf-8') as f:
    for i in range(len(df)):
        question = df.loc[i]['question']
        subject = df.loc[i]['subject']
        choices = df.loc[i]['choices']
        answer = df.loc[i]['answer']
        info = {"question": question, "subject": subject, "choices": str(choices), "answer": str(answer), "knowledge_bounary": {}}
        info = json.dumps(info)
        f.write(info + '\n')

f.close()