# DSCI 6800 Final Project

## 项目题目

这个 final 选的是一个尽量基础、容易分工、也容易写报告的方向：

**Boosting 方法在表格二分类任务中的基础对比研究**

当前项目比较三个模型：

1. `Decision Tree`，作为最基础的 baseline
2. `AdaBoost`
3. `Gradient Boosting`

## 数据说明

训练用的数据已经放进仓库里了：

```text
data/breast_cancer.csv
data/dataset_info.txt
```

这里使用的是 `Breast Cancer Wisconsin` 数据集，特点是：

1. 二分类任务
2. 标准表格数据
3. 数据量不大，适合课程实验

其中：

1. `label = 0` 表示 `malignant`
2. `label = 1` 表示 `benign`

`data_utils.py` 现在默认直接读取本地的 `data/breast_cancer.csv`，所以队友不需要再单独找数据。
数据来源：该数据集通过 `scikit-learn` 提供的 `load_breast_cancer()` 接口获取，随后导出为项目中的本地 CSV 文件，方便团队直接复现实验。

## 文件结构

```text
data/
  breast_cancer.csv   # 训练和实验使用的数据
  dataset_info.txt    # 数据集基本信息
data_utils.py         # 读取本地 CSV，划分 train / val / test
models.py             # 定义要比较的模型
metrics.py            # 评估指标和结果表格
train.py              # 主实验脚本
report_outline.md     # final report 的大纲草稿
requirements.txt
```

## 运行方式

先安装依赖：

```bash
python -m pip install -r requirements.txt
```

然后运行：

```bash
python train.py
```

运行后会完成这些事情：

1. 从本地 CSV 读取数据
2. 划分训练集、验证集、测试集
3. 训练三个基础模型
4. 输出指标
5. 把结果保存到 `results/metrics_summary.txt`

## 小组分工

### TODO(team_member_adaboost)

负责实现 AdaBoost。

1. 在 `models.py` 中补全 AdaBoost 模型
2. 在 `train.py` 中补全 AdaBoost 的训练和评估流程
3. 记录最终结果并整理成报告里的实验内容

### TODO(team_member_gradient_boosting)

负责实现 Gradient Boosting。

1. 在 `models.py` 中补全 Gradient Boosting 模型
2. 在 `train.py` 中补全 Gradient Boosting 的训练和评估流程
3. 记录最终结果并整理成报告里的实验内容

## 建议报告标题

**A Literature Review and Experimental Comparison of Basic Boosting Methods for Tabular Classification**
