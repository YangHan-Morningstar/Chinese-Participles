# 中文分词

## 一、简介

基于机器学习的中文分词实现，包括双向最大匹配算法、HMM、CRF、双向GRU、Transformer-Encoder，数据集采用MSR。

## 二、说明

### 2.1 编写训练测试环境

* Python-3.7.11、TensorFlow-2.0.0、scikit-learn-0.24.1

### 2.2 运行

* Dict-Base包含双向最大匹配，直接运行bi-mm.py即可。
* Sequence-Labeling-Base包含剩余的算法与模型，同样直接运行对应名字的py文件即可，其中Transformer分为了多个py文件，但文件名已说明。

### 2.3 测试结果

|    Model    | Precision | Recall | F1 |
| :----: | :----: | :----: | :----: |
| crf | 0.9937 | 0.9945 | 0.9941 |
| bi_gru | 0.9878 | 0.9943 | 0.9910 |
| bi_mm | 0.9701 | 0.9676 | 0.9688 |
| HMM | 0.9831 | 0.9232 | 0.9486 |
| Transformer | 0.9889 | 0.9916 | 0.9903 |