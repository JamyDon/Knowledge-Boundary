import json
import matplotlib.pyplot as plt

# 加载标注后的MMLU数据集
with open('data/labeled_mmlu.json', 'r') as f:
    labeled_data = json.load(f)

# 统计三种方法的Known与Unknown数量
correctness_known = sum(1 for item in labeled_data if item['knowledge boundary']['Correctness']['result'] == 'Known')
correctness_unknown = len(labeled_data) - correctness_known

verbalize_known = sum(1 for item in labeled_data if item['knowledge boundary']['Verbalize']['result'] == 'Known')
verbalize_unknown = len(labeled_data) - verbalize_known

confidence_known = sum(1 for item in labeled_data if item['knowledge boundary']['Confidence']['result'] == 'Known')
confidence_unknown = len(labeled_data) - confidence_known

certainty_known = sum(1 for item in labeled_data if item['knowledge boundary']['Certainty']['result'] == 'Known')
certainty_unknown = len(labeled_data) - certainty_known

multi_prompt_known = sum(1 for item in labeled_data if item['knowledge boundary']['MultiPrompt']['result'] == 'Known')
multi_prompt_unknown = len(labeled_data) - multi_prompt_known

multi_temperature_known = sum(1 for item in labeled_data if item['knowledge boundary']['MultiTemperature']['result'] == 'Known')
multi_temperature_unknown = len(labeled_data) - multi_temperature_known

# 绘制柱状图
methods = ['Correctness', 'Verbalize', 'Confidence', 'Certainty', 'MultiPrompt', 'MultiTemperature']
known_counts = [correctness_known, verbalize_known, confidence_known, certainty_known, multi_prompt_known, multi_temperature_known]
unknown_counts = [correctness_unknown, verbalize_unknown, confidence_unknown, certainty_unknown, multi_prompt_unknown, multi_temperature_unknown]

x = range(len(methods))

plt.figure(figsize=(10, 6))
plt.bar(x, known_counts, width=0.1, label='Known', color='b', align='center')
plt.bar([a+0.2 for a in x], unknown_counts, width=0.1, label='Unknown', color='r', align='edge')
plt.xlabel('Methods')
plt.ylabel('Count')
plt.title('Known vs Unknown Counts by Method')
plt.xticks([a+0.1 for a in x], methods)
plt.legend()
plt.show()
plt.savefig('data/known_unknown_counts.png')

# 绘制置信度分布散点图
confidence_scores = [item['knowledge boundary']['Confidence']['output'] for item in labeled_data]
labels = [item['knowledge boundary']['Correctness']['result'] for item in labeled_data]

plt.figure(figsize=(10, 6))
for i, (score, label) in enumerate(zip(confidence_scores, labels)):
    color = 'b' if label == 'Known' else 'r'
    plt.scatter(i, score, color=color, alpha=0.5)
plt.xlabel('Question Index')
plt.ylabel('Confidence Score')
plt.title('Confidence Scores Distribution')
plt.scatter([], [], color='b', label='Known')
plt.scatter([], [], color='r', label='Unknown')
plt.legend()
plt.savefig('data/confidence_scores.png')
plt.show()

# 绘制不确定性分布散点图
certainty_scores = [item['knowledge boundary']['Certainty']['output'] for item in labeled_data]

plt.figure(figsize=(10, 6))
for i, (score, label) in enumerate(zip(certainty_scores, labels)):
    color = 'b' if label == 'Known' else 'r'
    plt.scatter(i, score, color=color, alpha=0.5)
plt.xlabel('Question Index')
plt.ylabel('Certainty Score')
plt.title('Certainty Scores Distribution')
plt.scatter([], [], color='b', label='Known')
plt.scatter([], [], color='r', label='Unknown')
plt.legend()
plt.savefig('data/certainty_scores.png')
plt.show()