# 自然语言学大作业

## 目录结构
- `data`存放数据，以JSON格式存储。
- `examples`存放了一些调用模型的示例代码，对模型调用不熟悉的同学可以参考，相关说明写在该目录的`README.md`下。
- `model`存放模型，本地存放即可，不用推送到Git（在`.gitignore`中也已写上）。模型路径统一存为`model/meta-llama/Llama-3.2-3B-Instruct`。
- `src`存放源代码，我们使用Python为主要语言。

## 环境说明
以`requirements.txt`的环境为基础。

## 协同开发说明
每个人使用一个Git分支进行开发。

使用`git checkout -b <branchname>`开启新分支。

使用`git checkout <branchname>`切换至已有分支。

开发告一段落时，将自己的分支Push到Gitee，并发起pull request，由组长将分支新增的内容合并到主分支。