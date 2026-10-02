# Olist 电商数据分析项目计划

## 目标

基于 Kaggle Olist 巴西电商数据，构建面向数据分析岗位求职作品集的完整项目，覆盖数据清洗、指标体系、业务分析、两个预测任务和中文 PDF 报告。

## 数据目录

原始数据固定读取：`data/raw/archive`。原始文件只读，不覆盖。

## 执行流程

1. 生成数据清单和 SHA-256 manifest。
2. 完成订单、商品、支付、客户、卖家和评价数据的清洗与关联。
3. 计算销售、物流、评价、客户和卖家指标。
4. 建立订单延迟预测和低评分预测模型。
5. 生成图表、指标字典、数据质量报告和 PDF。
6. 渲染 PDF 页面并完成自动质量检查。

## 预测任务

### 订单延迟预测

标签为实际送达时间晚于预计送达时间。只使用下单或确认时可获得的变量，禁止使用实际送达时间、评价和实际配送时长。

### 低评分预测

标签为评价分数小于等于 2。只使用下单或确认时可获得的变量，禁止使用实际配送结果和评价文本。

## 输出

- `data/processed/olist_order_level.csv`
- `data/processed/olist_item_level.csv`
- `data/processed/metrics/*.csv`
- `reports/data_quality_report.md`
- `reports/data_quality_summary.json`
- `reports/charts/*.png`
- `reports/models/*.json`
- `reports/olist_ecommerce_analysis_report.pdf`

## 运行方式

使用 `D:\ProgramData\Anaconda3\python.exe`，并设置 `PYTHONNOUSERSITE=1` 避免用户目录 NumPy 覆盖基础环境：

```powershell
$env:PYTHONNOUSERSITE='1'
python run_pipeline.py
```

## 验收标准

- 9 个 CSV 均可读取，原始数据不被修改。
- 订单金额不因明细、支付或评价连接而重复计算。
- 复购率基于 `customer_unique_id`。
- 两个预测任务均按时间顺序划分数据。
- 模型特征不包含明显未来信息。
- PDF 中文显示正常，所有图表和表格无截断或重叠。

## 数据来源

Kaggle: https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce

## 状态

- 0%：目录和数据确认，已完成
- 15%：计划文档、manifest 和项目骨架，进行中
- 35%：数据清洗和质量检查
- 55%：指标和探索性分析
- 75%：两个预测任务
- 90%：PDF 初稿
- 100%：渲染、测试和交付

## 最终交付状态

- 已完成 100%：数据清洗、质量检查、指标分析、图表、两个时间切分预测任务、中文 PDF 生成及 PDF 渲染验收。
- 已通过 3 项自动化测试，PDF 共 10 页，可正常打开，中文字体、图表和表格均已检查。
- 已知限制：仅使用下单/确认时可获得的特征时，订单延迟模型区分能力较弱；该结果已作为模型边界和业务建议的一部分写入报告。
