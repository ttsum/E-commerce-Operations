# Olist 电商数据分析项目

这是一个面向数据分析岗位求职的端到端电商分析项目。项目覆盖：

- 多表 CSV 数据处理
- 订单、商品、客户、卖家和物流指标
- 销售、复购、评价和延迟分析
- 订单延迟预测
- 低评分预测
- 中文 PDF 分析报告

## 数据准备

将 Kaggle Olist 数据集的 9 个 CSV 文件放入：

```text
data/raw/archive
```

项目不修改原始 CSV。

## 运行

```powershell
$env:PYTHONNOUSERSITE='1'
& 'D:\ProgramData\Anaconda3\python.exe' run_pipeline.py
```

## 主要输出

```text
reports/olist_ecommerce_analysis_report.pdf
reports/data_quality_report.md
reports/models/late_delivery_model.json
reports/models/low_review_model.json
data/processed/metrics/
```

## 预测边界

模型用于订单风险排序和业务预警，不表示因果关系，也不能替代运营人员的最终判断。
