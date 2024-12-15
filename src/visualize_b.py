import json
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score

file_in = "../data/labeled_mmlu.json"

with open(file_in, 'r',encoding='utf-8') as f:
    labeled_data = json.load(f)

Correctness = []
Verbalize = []
Confidence = []
Certainty = []
MultiPrompt = []
MultiTemperature = []

for i in range(len(labeled_data)):
    if labeled_data[i]["knowledge boundary"]["Correctness"]["result"] == "Known":
        Correctness.append(1)
    else:
        Correctness.append(0)
    if labeled_data[i]["knowledge boundary"]["Verbalize"]["result"] == "Known":
        Verbalize.append(1)
    else:
        Verbalize.append(0)
    if labeled_data[i]["knowledge boundary"]["Confidence"]["result"] == "Known":
        Confidence.append(1)
    else:
        Confidence.append(0)
    if labeled_data[i]["knowledge boundary"]["Certainty"]["result"] == "Known":
        Certainty.append(1)
    else:
        Certainty.append(0)
    if labeled_data[i]["knowledge boundary"]["MultiPrompt"]["result"] == "Known":
        MultiPrompt.append(1)
    else:
        MultiPrompt.append(0)
    if labeled_data[i]["knowledge boundary"]["MultiTemperature"]["result"] == "Known":
        MultiTemperature.append(1)
    else:
        MultiTemperature.append(0)

Correctness = np.array(Correctness)
Verbalize = np.array(Verbalize)
Confidence = np.array(Confidence)
Certainty = np.array(Certainty)
MultiPrompt = np.array(MultiPrompt)
MultiTemperature = np.array(MultiTemperature)

# 不同方法Known和Unknown的数量
methods_name = ['Correctness', 'Verbalize', 'Confidence', 'Certainty', 'MultiPrompt', 'MultiTemperature']
known_counts = [np.sum(Correctness), np.sum(Verbalize), np.sum(Confidence), np.sum(Certainty), np.sum(MultiPrompt), np.sum(MultiTemperature)]
unknown_counts = [len(Correctness)-np.sum(Correctness), len(Verbalize)-np.sum(Verbalize), len(Confidence)-np.sum(Confidence), len(Certainty)-np.sum(Certainty), len(MultiPrompt)-np.sum(MultiPrompt), len(MultiTemperature)-np.sum(MultiTemperature)]

x = np.arange(len(methods_name))
width = 0.2

fig, ax = plt.subplots()
bars1 = ax.bar(x - width/2, known_counts, width/2, label='Known', color='blue')
bars2 = ax.bar(x + width/2, unknown_counts, width/2, label='Unknown', color='red')

ax.set_xlabel('Methods')
ax.set_ylabel('Count')
ax.set_title('Known vs Unknown Counts by Method')
ax.set_xticks(x)
ax.set_xticklabels(methods_name)
ax.legend()
plt.savefig('../data/figure/known_vs_unknown_counts.png', format='png')

# 不同方法重合度分析
methods = [Correctness, Verbalize, Confidence, Certainty, MultiPrompt, MultiTemperature]

overlaps = []
for i in range(len(methods)):
    for j in range(len(methods)):
        total_agreements = np.sum(methods[i] == methods[j])
        overlaps.append(total_agreements / len(methods[i]))

overlaps = np.array(overlaps).reshape(6, 6)

plt.figure(figsize=(8, 6))
sns.heatmap(overlaps, annot=True, fmt=".2f", cmap='coolwarm', xticklabels=methods_name, yticklabels=methods_name)
plt.xticks(rotation=0)
plt.title('OverLap Matrix')
plt.ylabel('Methods')
plt.xlabel('Methods')
plt.savefig('../data/figure/overlap_matrix.png', format='png')

#覆盖度分析
precision_matrix = np.zeros((6, 6))
recall_matrix = np.zeros((6, 6))
f1_matrix = np.zeros((6, 6))

for i in range(len(methods)):
    for j in range(len(methods)):
        precision_matrix[i, j] = precision_score(methods[j], methods[i])
        recall_matrix[i, j] = recall_score(methods[j], methods[i])
        f1_matrix[i, j] = f1_score(methods[j], methods[i])

plt.figure(figsize=(8, 6))
sns.heatmap(precision_matrix, annot=True, fmt=".2f", cmap='coolwarm', xticklabels=methods_name, yticklabels=methods_name)
plt.xticks(rotation=0)
plt.title('Precision Matrix')
plt.ylabel('Methods')
plt.xlabel('Methods')
plt.savefig('../data/figure/precision_matrix.png', format='png')

plt.figure(figsize=(8, 6))
sns.heatmap(recall_matrix, annot=True, fmt=".2f", cmap='coolwarm', xticklabels=methods_name, yticklabels=methods_name)
plt.xticks(rotation=0)
plt.title('Recall Matrix')
plt.ylabel('Methods')
plt.xlabel('Methods')
plt.savefig('../data/figure/recall_matrix.png', format='png')

plt.figure(figsize=(8, 6))
sns.heatmap(f1_matrix, annot=True, fmt=".2f", cmap='coolwarm', xticklabels=methods_name, yticklabels=methods_name)
plt.xticks(rotation=0)
plt.title('F1 Matrix')
plt.ylabel('Methods')
plt.xlabel('Methods')
plt.savefig('../data/figure/f1_matrix.png', format='png')

#相关系数分析
df = pd.DataFrame(np.column_stack(methods), columns=methods_name)
pearson_correlation_matrix = df.corr(method='pearson')
plt.figure(figsize=(8, 6))
sns.heatmap(pearson_correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', xticklabels=methods_name, yticklabels=methods_name)
plt.xticks(rotation=0)
plt.title('Pearson Correlation Matrix')
plt.savefig('../data/figure/pearson_corr_matrix.png', format='png')