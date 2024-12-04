# 示例

## inference
展示了如何在给定若干prompt的情况下输出每条数据模型输出的第一个token的概率分布。

## get_token_idx
展示了如何获取某一特定token在模型词表中的下标（个人获取的A-E在Llama的下标是32-36连续的五个号，大家可以核对一下）。通过该下标，可以从`inference.py`中获取的概率分布列表中获取给定token的输出概率，如`pred_scores[i][32]`表示第i条数据模型输出“A”的概率。