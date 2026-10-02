# 数据质量报告

生成时间：2026-10-02 14:11:22

## 表级检查

| 表 | 行数 | 列数 | 重复行 | 缺失单元格 |
|---|---:|---:|---:|---:|
| olist_customers_dataset | 99,441 | 5 | 0 | 0 |
| olist_geolocation_dataset | 1,000,163 | 5 | 261,831 | 0 |
| olist_order_items_dataset | 112,650 | 7 | 0 | 0 |
| olist_order_payments_dataset | 103,886 | 5 | 0 | 0 |
| olist_order_reviews_dataset | 99,224 | 7 | 0 | 145,903 |
| olist_orders_dataset | 99,441 | 8 | 0 | 4,908 |
| olist_products_dataset | 32,951 | 9 | 0 | 2,448 |
| olist_sellers_dataset | 3,095 | 4 | 0 | 0 |
| product_category_name_translation | 71 | 2 | 0 | 0 |

## 业务逻辑检查

| 检查项 | 数量 |
|---|---:|
| orders_with_purchase_date | 99,441 |
| orders_with_customer | 99,441 |
| orders_with_item | 98,666 |
| delivered_orders | 96,478 |
| orders_with_late_label | 96,476 |
| orders_with_review_label | 98,673 |
| negative_delivery_days | 0 |
| negative_order_value | 0 |
| duplicate_order_ids | 0 |

## 处理说明

订单金额在订单级表中由商品金额和运费汇总得到，支付表单独汇总后用于交叉核对。模型特征只使用当前订单产生前或订单确认时可获得的信息。