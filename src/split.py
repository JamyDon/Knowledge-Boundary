import json
import random
import pandas as pd

file_in = "../data/labeled_mmlu.json"
file_in_2 = "../data/raw_data.parquet"
file_out_1 = "../data/train.json"
file_out_2 = "../data/val.json"
file_out_3 = "../data/test.json"

with open(file_in, 'r',encoding='utf-8') as f:
    labeled_data = json.load(f)

df = pd.read_parquet(file_in_2)

prefect_data = []
for i in range(len(labeled_data)):
    data = {}
    data["question"] = df.loc[i]["question"]
    data["subject"] = df.loc[i]["subject"]
    data["choices"] = df.loc[i]["choices"].tolist()
    data["answer"] = int(df.loc[i]["answer"])
    data["knowledge boundary"] = labeled_data[i]["knowledge boundary"]["Correctness"]["result"]
    prefect_data.append(data)

arr = [i for i in range(len(prefect_data))]
random.shuffle(arr)
split1 = int(len(arr)*0.8)
split2 = int(len(arr)*0.9)
arr1 = arr[0:split1]
arr2 = arr[split1:split2]
arr3 = arr[split2:]
arr1 = sorted(arr1)
arr2 = sorted(arr2)
arr3 = sorted(arr3)

train_data = []
val_data = []
test_data = []

for i in arr1:
    train_data.append(prefect_data[i])
for i in arr2:
    val_data.append(prefect_data[i])
for i in arr3:
    test_data.append(prefect_data[i])

with open(file_out_1, 'w',encoding='utf-8') as f:
    json.dump(train_data, f, ensure_ascii=False, indent=4)  
with open(file_out_2, 'w',encoding='utf-8') as f:
    json.dump(val_data, f, ensure_ascii=False, indent=4)
with open(file_out_3, 'w',encoding='utf-8') as f:
    json.dump(test_data, f, ensure_ascii=False, indent=4)