# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import sys
import warnings
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
VENDOR = ROOT / "vendor"
if VENDOR.exists():
    sys.path.insert(0, str(VENDOR))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.ticker import FuncFormatter
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")
SEED = 42
RAW_FILES = [
    "olist_customers_dataset.csv",
    "olist_geolocation_dataset.csv",
    "olist_order_items_dataset.csv",
    "olist_order_payments_dataset.csv",
    "olist_order_reviews_dataset.csv",
    "olist_orders_dataset.csv",
    "olist_products_dataset.csv",
    "olist_sellers_dataset.csv",
    "product_category_name_translation.csv",
]
SOURCE_URL = "https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce"


def json_default(value):
    if isinstance(value, (np.integer, np.int64)):
        return int(value)
    if isinstance(value, (np.floating, np.float64)):
        return None if np.isnan(value) else float(value)
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    return str(value)


def write_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=json_default), encoding="utf-8")


def read_csv(path: Path, **kwargs):
    last_error = None
    for encoding in ("utf-8", "utf-8-sig", "latin1"):
        try:
            return pd.read_csv(path, encoding=encoding, **kwargs)
        except UnicodeDecodeError as exc:
            last_error = exc
    raise last_error


def sha256(path: Path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_manifest(raw_dir: Path, output_dir: Path):
    manifest = {"generated_at": datetime.now().isoformat(timespec="seconds"), "raw_dir": str(raw_dir), "files": []}
    missing = []
    for name in RAW_FILES:
        path = raw_dir / name
        if not path.exists():
            missing.append(name)
            continue
        frame = read_csv(path, nrows=5)
        row_count = 0
        for chunk in pd.read_csv(path, encoding="utf-8", chunksize=50000):
            row_count += len(chunk)
        manifest["files"].append({
            "file": name,
            "bytes": path.stat().st_size,
            "rows": row_count,
            "columns": list(frame.columns),
            "dtypes_sample": {k: str(v) for k, v in frame.dtypes.items()},
            "sha256": sha256(path),
        })
    manifest["missing"] = missing
    write_json(output_dir / "raw_manifest.json", manifest)
    if missing:
        raise FileNotFoundError("缺少原始文件: " + ", ".join(missing))
    return manifest


def load_raw(raw_dir: Path):
    frames = {name.replace(".csv", ""): read_csv(raw_dir / name) for name in RAW_FILES}
    return frames


def parse_dates(frame, columns):
    for col in columns:
        if col in frame.columns:
            frame[col] = pd.to_datetime(frame[col], errors="coerce")
    return frame


def clean_and_model_table(frames, processed_dir: Path):
    orders = frames["olist_orders_dataset"].copy()
    items = frames["olist_order_items_dataset"].copy()
    payments = frames["olist_order_payments_dataset"].copy()
    reviews = frames["olist_order_reviews_dataset"].copy()
    customers = frames["olist_customers_dataset"].copy()
    products = frames["olist_products_dataset"].copy()
    sellers = frames["olist_sellers_dataset"].copy()
    translation = frames["product_category_name_translation"].copy()

    order_dates = [c for c in orders.columns if c.endswith("_timestamp") or c.endswith("_date")]
    orders = parse_dates(orders, order_dates)
    items = parse_dates(items, ["shipping_limit_date"])
    reviews = parse_dates(reviews, ["review_creation_date", "review_answer_timestamp"])

    for col in ["price", "freight_value"]:
        items[col] = pd.to_numeric(items[col], errors="coerce")
    for col in ["payment_value", "payment_installments"]:
        payments[col] = pd.to_numeric(payments[col], errors="coerce")
    reviews["review_score"] = pd.to_numeric(reviews["review_score"], errors="coerce")
    products = products.merge(translation, on="product_category_name", how="left")
    products["category_en"] = products["product_category_name_english"].fillna(products["product_category_name"]).fillna("unknown")

    item_join = items.merge(products[["product_id", "category_en"]], on="product_id", how="left")
    item_join["category_en"] = item_join["category_en"].fillna("unknown")
    item_join = item_join.sort_values(["order_id", "price"], ascending=[True, False])
    primary_category = item_join.groupby("order_id", as_index=False).first()[["order_id", "category_en", "seller_id"]]
    item_agg = item_join.groupby("order_id", as_index=False).agg(
        item_count=("order_item_id", "count"),
        product_count=("product_id", "nunique"),
        seller_count=("seller_id", "nunique"),
        item_value=("price", "sum"),
        freight_value=("freight_value", "sum"),
        max_item_value=("price", "max"),
    )
    item_agg = item_agg.merge(primary_category.rename(columns={"category_en": "primary_category", "seller_id": "primary_seller_id"}), on="order_id", how="left")

    payment_agg = payments.groupby("order_id", as_index=False).agg(
        payment_value=("payment_value", "sum"),
        payment_count=("payment_sequential", "count"),
        payment_installments=("payment_installments", "max"),
        payment_type=("payment_type", "first"),
    )
    review_agg = reviews.groupby("order_id", as_index=False).agg(
        review_score=("review_score", "mean"),
        review_count=("review_id", "nunique"),
    )
    seller_state = sellers[["seller_id", "seller_state"]].drop_duplicates("seller_id")
    customer_state = customers[["customer_id", "customer_unique_id", "customer_state"]].drop_duplicates("customer_id")

    order = orders.merge(customer_state, on="customer_id", how="left")
    order = order.merge(item_agg, on="order_id", how="left")
    order = order.merge(payment_agg, on="order_id", how="left")
    order = order.merge(review_agg, on="order_id", how="left")
    order = order.merge(seller_state.rename(columns={"seller_id": "primary_seller_id", "seller_state": "seller_state"}), on="primary_seller_id", how="left")
    order["order_purchase_timestamp"] = pd.to_datetime(order["order_purchase_timestamp"], errors="coerce")
    order["order_approved_at"] = pd.to_datetime(order["order_approved_at"], errors="coerce")
    order["order_delivered_customer_date"] = pd.to_datetime(order["order_delivered_customer_date"], errors="coerce")
    order["order_estimated_delivery_date"] = pd.to_datetime(order["order_estimated_delivery_date"], errors="coerce")
    order["item_value"] = order["item_value"].fillna(0)
    order["freight_value"] = order["freight_value"].fillna(0)
    order["payment_value"] = order["payment_value"].fillna(order["item_value"] + order["freight_value"])
    order["item_count"] = order["item_count"].fillna(0)
    order["product_count"] = order["product_count"].fillna(0)
    order["seller_count"] = order["seller_count"].fillna(0)
    order["payment_installments"] = order["payment_installments"].fillna(0)
    order["payment_type"] = order["payment_type"].fillna("unknown")
    order["primary_category"] = order["primary_category"].fillna("unknown")
    order["customer_state"] = order["customer_state"].fillna("unknown")
    order["seller_state"] = order["seller_state"].fillna("unknown")
    order["purchase_month"] = order["order_purchase_timestamp"].dt.to_period("M").astype(str)
    order["purchase_weekday"] = order["order_purchase_timestamp"].dt.day_name()
    order["purchase_hour"] = order["order_purchase_timestamp"].dt.hour
    order["is_delivered"] = order["order_status"].eq("delivered")
    order["delivery_days"] = (order["order_delivered_customer_date"] - order["order_purchase_timestamp"]).dt.total_seconds() / 86400
    order["estimated_days"] = (order["order_estimated_delivery_date"] - order["order_purchase_timestamp"]).dt.total_seconds() / 86400
    order["late_delivery"] = np.where(
        order["order_delivered_customer_date"].notna() & order["order_estimated_delivery_date"].notna(),
        (order["order_delivered_customer_date"] > order["order_estimated_delivery_date"]).astype(int),
        np.nan,
    )
    order["low_review"] = np.where(order["review_score"].notna(), (order["review_score"] <= 2).astype(int), np.nan)
    order["order_value"] = order["item_value"] + order["freight_value"]
    order = order.sort_values("order_purchase_timestamp").reset_index(drop=True)

    # Historical features are shifted by construction and therefore do not use the current order's outcome.
    order["seller_history_orders"] = order.groupby("primary_seller_id").cumcount()
    seller_late_sum = order["late_delivery"].fillna(0).groupby(order["primary_seller_id"]).cumsum() - order["late_delivery"].fillna(0)
    order["seller_prev_late_rate"] = np.where(order["seller_history_orders"] > 0, seller_late_sum / order["seller_history_orders"], np.nan)
    order["customer_history_orders"] = order.groupby("customer_unique_id").cumcount()
    order["customer_is_repeat"] = (order["customer_history_orders"] > 0).astype(int)
    order["year_month"] = order["order_purchase_timestamp"].dt.strftime("%Y-%m")

    processed_dir.mkdir(parents=True, exist_ok=True)
    order.to_csv(processed_dir / "olist_order_level.csv", index=False, encoding="utf-8-sig")
    item_join.to_csv(processed_dir / "olist_item_level.csv", index=False, encoding="utf-8-sig")
    customers.to_csv(processed_dir / "customers_clean.csv", index=False, encoding="utf-8-sig")
    sellers.to_csv(processed_dir / "sellers_clean.csv", index=False, encoding="utf-8-sig")
    return order, item_join, {"orders_raw": len(orders), "orders_level": len(order), "items_raw": len(items), "items_level": len(item_join), "reviews_raw": len(reviews)}


def quality_report(frames, order, processed_dir: Path, reports_dir: Path):
    checks = []
    for name, frame in frames.items():
        checks.append({"table": name, "rows": len(frame), "columns": len(frame.columns), "duplicate_rows": int(frame.duplicated().sum()), "missing_cells": int(frame.isna().sum().sum())})
    logical = {
        "orders_with_purchase_date": int(order["order_purchase_timestamp"].notna().sum()),
        "orders_with_customer": int(order["customer_unique_id"].notna().sum()),
        "orders_with_item": int(order["item_count"].gt(0).sum()),
        "delivered_orders": int(order["is_delivered"].sum()),
        "orders_with_late_label": int(order["late_delivery"].notna().sum()),
        "orders_with_review_label": int(order["low_review"].notna().sum()),
        "negative_delivery_days": int(order["delivery_days"].lt(0).sum()),
        "negative_order_value": int(order["order_value"].lt(0).sum()),
        "duplicate_order_ids": int(order["order_id"].duplicated().sum()),
    }
    payload = {"tables": checks, "logical_checks": logical}
    write_json(reports_dir / "data_quality_summary.json", payload)
    lines = ["# 数据质量报告", "", f"生成时间：{datetime.now():%Y-%m-%d %H:%M:%S}", "", "## 表级检查", "", "| 表 | 行数 | 列数 | 重复行 | 缺失单元格 |", "|---|---:|---:|---:|---:|"]
    for row in checks:
        lines.append(f"| {row['table']} | {row['rows']:,} | {row['columns']} | {row['duplicate_rows']:,} | {row['missing_cells']:,} |" )
    lines += ["", "## 业务逻辑检查", "", "| 检查项 | 数量 |", "|---|---:|"]
    for key, value in logical.items():
        lines.append(f"| {key} | {value:,} |")
    lines += ["", "## 处理说明", "", "订单金额在订单级表中由商品金额和运费汇总得到，支付表单独汇总后用于交叉核对。模型特征只使用当前订单产生前或订单确认时可获得的信息。"]
    (reports_dir / "data_quality_report.md").write_text("\n".join(lines), encoding="utf-8")
    return payload


def make_metrics(order, item_join, processed_dir: Path, reports_dir: Path):
    metrics_dir = reports_dir / "metrics"
    charts_dir = reports_dir / "charts"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    charts_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")
    font_path = Path("C:/Windows/Fonts/msyh.ttc")
    if font_path.exists():
        font_manager.fontManager.addfont(str(font_path))
        matplotlib_font = font_manager.FontProperties(fname=str(font_path)).get_name()
        plt.rcParams["font.family"] = matplotlib_font
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    monthly = order.groupby("year_month", as_index=False).agg(order_count=("order_id", "nunique"), sales=("order_value", "sum"), avg_order_value=("order_value", "mean"), late_rate=("late_delivery", "mean"), low_review_rate=("low_review", "mean"))
    category = order.groupby("primary_category", as_index=False).agg(order_count=("order_id", "nunique"), sales=("order_value", "sum"), avg_order_value=("order_value", "mean"), late_rate=("late_delivery", "mean"), low_review_rate=("low_review", "mean")).sort_values("sales", ascending=False)
    states = order.groupby("customer_state", as_index=False).agg(order_count=("order_id", "nunique"), sales=("order_value", "sum"), avg_order_value=("order_value", "mean"), late_rate=("late_delivery", "mean"), low_review_rate=("low_review", "mean")).sort_values("sales", ascending=False)
    sellers = order.groupby("primary_seller_id", as_index=False).agg(order_count=("order_id", "nunique"), sales=("order_value", "sum"), late_rate=("late_delivery", "mean"), avg_delivery_days=("delivery_days", "mean"), avg_review=("review_score", "mean")).sort_values("order_count", ascending=False)
    payment = order.groupby("payment_type", as_index=False).agg(order_count=("order_id", "nunique"), payment_value=("payment_value", "sum"), avg_installments=("payment_installments", "mean"))
    rfm = order.groupby("customer_unique_id", as_index=False).agg(last_purchase=("order_purchase_timestamp", "max"), frequency=("order_id", "nunique"), monetary=("order_value", "sum"))
    as_of = order["order_purchase_timestamp"].max() + pd.Timedelta(days=1)
    rfm["recency_days"] = (as_of - rfm["last_purchase"]).dt.days
    rfm["r_score"] = pd.qcut(rfm["recency_days"].rank(method="first"), 4, labels=[4, 3, 2, 1]).astype(int)
    rfm["f_score"] = pd.qcut(rfm["frequency"].rank(method="first"), 4, labels=[1, 2, 3, 4]).astype(int)
    rfm["m_score"] = pd.qcut(rfm["monetary"].rank(method="first"), 4, labels=[1, 2, 3, 4]).astype(int)
    rfm["segment"] = np.select([
        rfm["frequency"].gt(1) & rfm["r_score"].ge(3),
        rfm["frequency"].eq(1) & rfm["r_score"].ge(3),
        rfm["r_score"].le(2),
    ], ["高价值复购", "近期首购", "沉睡客户"], default="一般客户")
    segment = rfm.groupby("segment", as_index=False).agg(customer_count=("customer_unique_id", "nunique"), avg_frequency=("frequency", "mean"), avg_monetary=("monetary", "mean"), avg_recency_days=("recency_days", "mean"))
    summary = {
        "orders": int(order["order_id"].nunique()),
        "customers": int(order["customer_unique_id"].nunique()),
        "sellers": int(order["primary_seller_id"].nunique()),
        "sales": float(order["order_value"].sum()),
        "avg_order_value": float(order["order_value"].mean()),
        "delivered_orders": int(order["is_delivered"].sum()),
        "late_rate": float(order["late_delivery"].mean()),
        "low_review_rate": float(order["low_review"].mean()),
        "repeat_customer_rate": float((rfm["frequency"] > 1).mean()),
        "avg_delivery_days": float(order["delivery_days"].mean()),
        "data_period_start": order["order_purchase_timestamp"].min(),
        "data_period_end": order["order_purchase_timestamp"].max(),
    }
    tables = {"monthly": monthly, "category": category, "states": states, "sellers": sellers, "payment": payment, "rfm": rfm, "segment": segment}
    for name, frame in tables.items():
        frame.to_csv(metrics_dir / f"{name}.csv", index=False, encoding="utf-8-sig")
    write_json(reports_dir / "metrics_summary.json", summary)
    write_json(reports_dir / "metric_dictionary.json", {
        "sales": "订单级商品金额加运费金额之和",
        "avg_order_value": "订单销售额除以订单数",
        "late_rate": "有完整配送日期订单中延迟订单占比",
        "low_review_rate": "有评价订单中评分小于等于 2 的订单占比",
        "repeat_customer_rate": "购买次数大于 1 的唯一客户占比",
    })

    def savefig(name):
        plt.tight_layout()
        plt.savefig(charts_dir / name, dpi=180, bbox_inches="tight")
        plt.close()

    plt.figure(figsize=(10, 4.8)); plt.plot(monthly["year_month"], monthly["sales"], marker="o", label="销售额"); plt.xticks(rotation=45, ha="right"); plt.title("月度销售额趋势"); plt.ylabel("销售额"); plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x/1e6:.1f}M")); savefig("monthly_sales.png")
    top = category.head(12).sort_values("sales"); plt.figure(figsize=(9, 6)); plt.barh(top["primary_category"], top["sales"]); plt.title("商品类别销售额排行"); plt.xlabel("销售额"); savefig("category_sales.png")
    top_state = states.head(12).sort_values("sales"); plt.figure(figsize=(8, 5)); plt.barh(top_state["customer_state"], top_state["sales"]); plt.title("客户州销售额排行"); plt.xlabel("销售额"); savefig("state_sales.png")
    seller_plot = sellers.head(15).copy(); plt.figure(figsize=(9, 5)); plt.scatter(seller_plot["order_count"], seller_plot["late_rate"] * 100, s=np.maximum(seller_plot["sales"] / seller_plot["sales"].max() * 400, 30), alpha=.7); plt.xlabel("订单量"); plt.ylabel("延迟率 (%)"); plt.title("卖家订单量与延迟率"); savefig("seller_performance.png")
    delivered = order.loc[order["delivery_days"].notna(), "delivery_days"]; plt.figure(figsize=(8, 4.5)); plt.hist(delivered.clip(lower=0, upper=60), bins=30); plt.title("配送时长分布"); plt.xlabel("配送天数"); plt.ylabel("订单数"); savefig("delivery_distribution.png")
    cross = order.loc[order["late_delivery"].notna() & order["low_review"].notna()].groupby("late_delivery", as_index=False).agg(low_review_rate=("low_review", "mean")); plt.figure(figsize=(6, 4)); plt.bar(["准时", "延迟"], cross["low_review_rate"] * 100 if len(cross) == 2 else [0, 0]); plt.title("延迟与低评分关系"); plt.ylabel("低评分率 (%)"); savefig("late_low_review.png")
    plt.figure(figsize=(7, 4.5)); segment.set_index("segment")["customer_count"].plot(kind="bar"); plt.title("RFM 客户分群"); plt.ylabel("客户数"); plt.xticks(rotation=0); savefig("rfm_segments.png")
    score = order["review_score"].dropna().round().astype(int).value_counts().reindex([1, 2, 3, 4, 5], fill_value=0); plt.figure(figsize=(6, 4)); plt.bar([str(x) for x in score.index], score.values); plt.title("评价分数分布"); plt.xlabel("评分"); plt.ylabel("订单数"); savefig("review_scores.png")
    return summary, tables, charts_dir


def make_model_features(order):
    features = ["purchase_hour", "purchase_weekday", "primary_category", "customer_state", "seller_state", "item_count", "product_count", "seller_count", "item_value", "freight_value", "payment_installments", "payment_type", "seller_history_orders", "seller_prev_late_rate", "customer_history_orders", "customer_is_repeat"]
    return features


def model_task(order, target, model_dir: Path, label: str):
    model_dir.mkdir(parents=True, exist_ok=True)
    features = make_model_features(order)
    data = order.loc[order[target].notna()].copy().sort_values("order_purchase_timestamp").reset_index(drop=True)
    data = data.dropna(subset=["order_purchase_timestamp"])
    y = data[target].astype(int)
    cut1 = int(len(data) * 0.70); cut2 = int(len(data) * 0.85)
    train, valid, test = data.iloc[:cut1], data.iloc[cut1:cut2], data.iloc[cut2:]
    X_train, X_valid, X_test = train[features], valid[features], test[features]
    y_train, y_valid, y_test = y.iloc[:cut1], y.iloc[cut1:cut2], y.iloc[cut2:]
    categorical = [c for c in features if X_train[c].dtype == "object"]
    numeric = [c for c in features if c not in categorical]
    preprocess = ColumnTransformer([
        ("num", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), numeric),
        ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]), categorical),
    ])
    models = {
        "logistic_regression": LogisticRegression(max_iter=400, class_weight="balanced", solver="liblinear", random_state=SEED),
        "random_forest": RandomForestClassifier(n_estimators=120, max_depth=14, min_samples_leaf=4, class_weight="balanced_subsample", random_state=SEED, n_jobs=-1),
    }
    output = {"task": label, "target": target, "rows": len(data), "positive_rate": float(y.mean()), "split": {"train": len(train), "valid": len(valid), "test": len(test)}, "models": {}}
    for name, estimator in models.items():
        pipe = Pipeline([("preprocess", preprocess), ("model", estimator)])
        pipe.fit(X_train, y_train)
        valid_prob = pipe.predict_proba(X_valid)[:, 1]
        thresholds = np.arange(0.20, 0.81, 0.02)
        best_threshold = max(thresholds, key=lambda t: f1_score(y_valid, (valid_prob >= t).astype(int), zero_division=0))
        test_prob = pipe.predict_proba(X_test)[:, 1]
        pred = (test_prob >= best_threshold).astype(int)
        metrics = {
            "roc_auc": float(roc_auc_score(y_test, test_prob)) if y_test.nunique() > 1 else None,
            "pr_auc": float(average_precision_score(y_test, test_prob)) if y_test.nunique() > 1 else None,
            "precision": float(precision_score(y_test, pred, zero_division=0)),
            "recall": float(recall_score(y_test, pred, zero_division=0)),
            "f1": float(f1_score(y_test, pred, zero_division=0)),
            "threshold": float(best_threshold),
            "confusion_matrix": confusion_matrix(y_test, pred).tolist(),
        }
        # scikit-learn 1.0's SimpleImputer does not expose get_feature_names_out
        # through a nested Pipeline. Build names from the fitted transformer instead.
        fitted_pre = pipe.named_steps["preprocess"]
        encoder = fitted_pre.named_transformers_["cat"].named_steps["onehot"]
        feature_names = np.array(list(numeric) + list(encoder.get_feature_names_out(categorical)))
        if name == "logistic_regression":
            importance = np.abs(pipe.named_steps["model"].coef_[0])
            direction = pipe.named_steps["model"].coef_[0]
        else:
            importance = pipe.named_steps["model"].feature_importances_
            direction = np.zeros_like(importance)
        top_indices = np.argsort(importance)[-15:][::-1]
        top_features = [{"feature": str(feature_names[i]), "importance": float(importance[i]), "direction": float(direction[i])} for i in top_indices]
        output["models"][name] = {"metrics": metrics, "top_features": top_features}
    write_json(model_dir / f"{label}_model.json", output)
    return output


def register_fonts():
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    candidates = [Path("C:/Windows/Fonts/msyh.ttc"), Path("C:/Windows/Fonts/simhei.ttf"), Path("C:/Windows/Fonts/simsun.ttc")]
    for path in candidates:
        if path.exists():
            try:
                pdfmetrics.registerFont(TTFont("ChineseFont", str(path), subfontIndex=0 if path.suffix.lower() == ".ttc" else 0))
                return "ChineseFont"
            except Exception:
                continue
    return "Helvetica"


def generate_pdf(summary, tables, quality, late_model, low_model, charts_dir: Path, reports_dir: Path):
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER, TA_LEFT
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import cm
    from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

    font = register_fonts()
    pdf_path = reports_dir / "olist_ecommerce_analysis_report.pdf"
    doc = SimpleDocTemplate(str(pdf_path), pagesize=A4, rightMargin=1.5 * cm, leftMargin=1.5 * cm, topMargin=1.5 * cm, bottomMargin=1.5 * cm, title="Olist 电商数据分析报告", author="Codex")
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="CNTitle", parent=styles["Title"], fontName=font, fontSize=24, leading=32, alignment=TA_CENTER, textColor=colors.HexColor("#12355B"), spaceAfter=18))
    styles.add(ParagraphStyle(name="CNHeading", parent=styles["Heading1"], fontName=font, fontSize=16, leading=23, textColor=colors.HexColor("#12355B"), spaceBefore=12, spaceAfter=8))
    styles.add(ParagraphStyle(name="CNSubheading", parent=styles["Heading2"], fontName=font, fontSize=12, leading=18, textColor=colors.HexColor("#2A5C8A"), spaceBefore=8, spaceAfter=5))
    styles.add(ParagraphStyle(name="CNBody", parent=styles["BodyText"], fontName=font, fontSize=9.2, leading=15, alignment=TA_LEFT, spaceAfter=6))
    styles.add(ParagraphStyle(name="CNSmall", parent=styles["BodyText"], fontName=font, fontSize=7.5, leading=11, textColor=colors.HexColor("#555555")))
    story = []
    story += [Spacer(1, 2.5 * cm), Paragraph("Olist 巴西电商数据分析", styles["CNTitle"]), Paragraph("从数据处理、指标体系到预测建模的完整作品集报告", styles["CNHeading"]), Spacer(1, 1 * cm), Paragraph(f"数据时间范围：{summary['data_period_start']:%Y-%m-%d} 至 {summary['data_period_end']:%Y-%m-%d}", styles["CNBody"]), Paragraph("报告定位：数据分析岗位求职作品集", styles["CNBody"]), Paragraph(f"数据来源：<link href='{SOURCE_URL}' color='blue'>Kaggle Olist 数据集</link>", styles["CNBody"]), PageBreak()]
    story += [Paragraph("1. 项目摘要", styles["CNHeading"])]
    story += [Paragraph(f"本项目基于 {summary['orders']:,} 笔订单、{summary['customers']:,} 名唯一客户和 {summary['sellers']:,} 个卖家，完成销售表现、物流效率、客户评价和客户复购分析。订单级销售额约为 {summary['sales']:,.0f}，平均客单价为 {summary['avg_order_value']:,.2f}。在有完整配送日期的订单中，延迟率为 {summary['late_rate']:.1%}；在有评价的订单中，低评分率为 {summary['low_review_rate']:.1%}。", styles["CNBody"]), Paragraph("报告同时建立订单延迟预测和低评分预测两个模型。建模采用时间顺序切分，避免将未来信息带入训练特征。模型结果用于风险排序和运营预警，不用于因果判断。", styles["CNBody"])]
    kpi = [["订单量", f"{summary['orders']:,}"], ["销售额", f"{summary['sales']:,.0f}"], ["客单价", f"{summary['avg_order_value']:,.2f}"], ["延迟率", f"{summary['late_rate']:.1%}"], ["低评分率", f"{summary['low_review_rate']:.1%}"], ["复购客户率", f"{summary['repeat_customer_rate']:.1%}"]]
    t = Table(kpi, colWidths=[5 * cm, 7 * cm]); t.setStyle(TableStyle([("FONTNAME", (0, 0), (-1, -1), font), ("FONTSIZE", (0, 0), (-1, -1), 10), ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#E8F1F8")), ("GRID", (0, 0), (-1, -1), .3, colors.HexColor("#A8B8C8")), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("LEFTPADDING", (0, 0), (-1, -1), 8), ("RIGHTPADDING", (0, 0), (-1, -1), 8)])); story += [t, PageBreak()]

    story += [Paragraph("2. 数据来源与数据处理", styles["CNHeading"]), Paragraph("原始数据包括订单、订单明细、支付、评价、客户、卖家、商品、地理位置和类别翻译 9 张表。项目保留原始数据，并将多表数据统一整理为订单级分析表和商品明细级分析表。", styles["CNBody"]), Paragraph("订单金额采用商品价格与运费的订单级汇总，避免订单明细和支付明细连接后重复累计。延迟标签只在实际送达日期和预计送达日期同时存在时计算。评价标签只在评价分数存在时计算。", styles["CNBody"])]
    q_rows = [["表级检查", "结果"]] + [[str(k), f"{v:,}"] for k, v in quality["logical_checks"].items()]
    qt = Table(q_rows, colWidths=[9 * cm, 5 * cm]); qt.setStyle(TableStyle([("FONTNAME", (0, 0), (-1, -1), font), ("FONTSIZE", (0, 0), (-1, -1), 8), ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#12355B")), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white), ("GRID", (0, 0), (-1, -1), .25, colors.grey), ("ALIGN", (1, 1), (-1, -1), "RIGHT")]))
    story += [qt, Paragraph("数据处理步骤：先保留原始表，再将订单明细按 order_id 汇总，将支付记录按 order_id 汇总，将评价按 order_id 汇总，最后与客户、卖家和商品维度关联。订单级金额由商品金额和运费组成，支付金额只用于独立核对。模型特征按订单时间排序后计算历史卖家表现和客户历史订单数，避免当前订单标签进入当前特征。", styles["CNBody"]), PageBreak()]

    story += [Paragraph("3. 指标体系与业务分析", styles["CNHeading"]), Paragraph("指标按订单、客户、卖家、商品类别和月份五类粒度组织。销售指标用于识别收入来源，物流指标用于定位履约风险，评价指标用于衡量体验，RFM 指标用于识别客户价值。", styles["CNBody"])]
    metric_rows = [["指标", "计算口径", "粒度"] , ["销售额", "商品金额 + 运费金额", "订单"], ["客单价", "订单销售额 / 订单数", "订单"], ["延迟率", "延迟订单 / 有完整配送日期订单", "订单"], ["低评分率", "评分 <= 2 的订单 / 有评价订单", "订单"], ["复购客户率", "购买次数 > 1 的唯一客户 / 唯一客户", "客户"], ["RFM", "最近购买、购买频次、累计金额分位数", "客户"]]
    metric_table = Table(metric_rows, colWidths=[3.0 * cm, 9.0 * cm, 2.5 * cm]); metric_table.setStyle(TableStyle([("FONTNAME", (0, 0), (-1, -1), font), ("FONTSIZE", (0, 0), (-1, -1), 8), ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#12355B")), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white), ("GRID", (0, 0), (-1, -1), .25, colors.grey), ("VALIGN", (0, 0), (-1, -1), "MIDDLE")]))
    top_category = tables["category"].iloc[0]
    top_state = tables["states"].iloc[0]
    top_month = tables["monthly"].sort_values("sales", ascending=False).iloc[0]
    story += [Paragraph("核心指标口径", styles["CNSubheading"]), metric_table, Paragraph(f"初步业务发现：销售额最高的商品类别为 {top_category['primary_category']}，销售额约 {top_category['sales']:,.0f}；客户销售额最高的州为 {top_state['customer_state']}，销售额约 {top_state['sales']:,.0f}；销售额最高月份为 {top_month['year_month']}。这些结果用于提出分析方向，不等同于因果结论。", styles["CNBody"])]
    chart_specs = [("月度销售趋势", "monthly_sales.png", "观察销售额的时间变化和季节性。"), ("商品类别销售排行", "category_sales.png", "识别贡献主要销售额的商品类别。"), ("客户州销售排行", "state_sales.png", "比较不同客户地区的销售贡献。"), ("卖家订单量与延迟率", "seller_performance.png", "同时观察卖家规模和履约风险。"), ("配送时长分布", "delivery_distribution.png", "检查配送时长的集中区间和长尾。"), ("延迟与低评分关系", "late_low_review.png", "比较准时与延迟订单的低评分率。"), ("RFM 客户分群", "rfm_segments.png", "区分近期复购、高价值、首购和沉睡客户。"), ("评价分数分布", "review_scores.png", "观察客户评价的整体分布。")]
    for idx, (title, filename, caption) in enumerate(chart_specs):
        path = charts_dir / filename
        if path.exists():
            if idx in (2, 4, 6):
                story.append(PageBreak())
            story += [Paragraph(title, styles["CNSubheading"]), Image(str(path), width=15.5 * cm, height=5.8 * cm), Paragraph(caption, styles["CNSmall"])]
    story.append(PageBreak())

    story += [Paragraph("预测建模方法说明", styles["CNHeading"]), Paragraph("两个任务均按照订单时间排序，前 70% 为训练集，中间 15% 为验证集，最后 15% 为测试集。Logistic Regression 用于提供可解释基线，Random Forest 用于捕捉非线性关系。分类特征使用 One-Hot 编码，数值特征使用中位数填补并标准化。阈值在验证集上按 F1 选择，再固定到测试集评估。", styles["CNBody"]), Paragraph("ROC-AUC 衡量模型区分正负样本的排序能力，PR-AUC 更适合正样本比例较低的风险识别任务。Precision 表示预测为风险的订单中实际风险订单的比例，Recall 表示实际风险订单中被识别出来的比例，F1 是 Precision 和 Recall 的综合指标。实际部署时应结合一次误报和一次漏报的业务成本重新确定阈值。", styles["CNBody"]), Paragraph("未来信息控制：延迟预测不使用实际送达时间和评价结果；低评分预测不使用实际配送结果和评价文本；卖家历史变量只使用当前订单之前的累计记录。", styles["CNBody"])]

    def model_section(title, result):
        blocks = [Paragraph(title, styles["CNHeading"]), Paragraph(f"样本量：{result['rows']:,}；正样本率：{result['positive_rate']:.1%}；训练集、验证集、测试集数量分别为 {result['split']['train']:,}、{result['split']['valid']:,}、{result['split']['test']:,}。", styles["CNBody"])]
        rows = [["模型", "ROC-AUC", "PR-AUC", "Precision", "Recall", "F1", "阈值"]]
        for name, item in result["models"].items():
            m = item["metrics"]; rows.append([name, f"{m['roc_auc']:.3f}" if m["roc_auc"] is not None else "-", f"{m['pr_auc']:.3f}" if m["pr_auc"] is not None else "-", f"{m['precision']:.3f}", f"{m['recall']:.3f}", f"{m['f1']:.3f}", f"{m['threshold']:.2f}"])
        mt = Table(rows, colWidths=[4.2*cm, 2.0*cm, 2.0*cm, 2.0*cm, 2.0*cm, 1.5*cm, 1.5*cm]); mt.setStyle(TableStyle([("FONTNAME", (0, 0), (-1, -1), font), ("FONTSIZE", (0, 0), (-1, -1), 7.5), ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#12355B")), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white), ("GRID", (0, 0), (-1, -1), .25, colors.grey), ("ALIGN", (1, 1), (-1, -1), "RIGHT")]))
        blocks += [mt, Spacer(1, 6)]
        for name, item in result["models"].items():
            top = ", ".join([x["feature"] for x in item["top_features"][:6]])
            blocks.append(Paragraph(f"{name} 重要特征：{top}", styles["CNSmall"]))
        return blocks
    story += model_section("4. 订单延迟预测", late_model)
    story += [Paragraph("模型特征来自订单产生前或订单确认时可获得的信息。实际送达日期和评价结果仅用于构造标签和离线评估，没有作为输入特征。", styles["CNBody"]), PageBreak()]
    story += model_section("5. 低评分预测", low_model)
    story += [Paragraph("低评分预测用于提前识别可能产生负面体验的订单。该模型不使用实际配送结果和评价文本，因此更接近下单阶段的预警场景。", styles["CNBody"]), PageBreak()]
    story += [Paragraph("6. 业务建议", styles["CNHeading"]), Paragraph("第一，优先将延迟预测用于物流预警，对高风险订单提前检查卖家备货和配送安排。第二，对低评分风险较高的订单加强配送承诺管理和客户沟通。第三，按卖家订单规模和延迟率联合识别重点卖家，避免只按订单量进行评价。第四，针对高价值复购客户建立差异化服务和复购激励。第五，在模型上线前持续监控时间漂移、特征缺失和阈值下的误报漏报。", styles["CNBody"]), Paragraph("项目限制：Olist 数据是历史匿名数据，缺少利润、营销成本、库存和真实物流节点等变量。分析结果描述相关关系，不应解释为因果关系。预测模型需要在新的时间窗口上重新验证。", styles["CNBody"]), Paragraph("附录：完整字段说明、清洗规则、指标口径、模型参数、SQL 示例和运行方式见项目目录中的 README、docs/data_dictionary.md、docs/sql_examples.sql 和 JSON 结果文件。", styles["CNBody"])]

    def footer(canvas, doc_obj):
        canvas.saveState(); canvas.setFont(font, 7); canvas.setFillColor(colors.HexColor("#6B7280")); canvas.drawString(1.5*cm, 0.8*cm, "Olist 电商数据分析报告"); canvas.drawRightString(A4[0]-1.5*cm, 0.8*cm, f"第 {doc_obj.page} 页"); canvas.restoreState()
    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    return pdf_path


def update_progress(root: Path, stage: str, details: str):
    path = root / "progress.md"
    existing = path.read_text(encoding="utf-8") if path.exists() else "# 执行进度\n"
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    path.write_text(existing + f"\n## {stamp}\n\n### {stage}\n\n{details}\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Run the Olist e-commerce analysis pipeline")
    parser.add_argument("--project-root", default=str(ROOT))
    args = parser.parse_args()
    root = Path(args.project_root).resolve()
    raw_dir = root / "data" / "raw" / "archive"
    processed_dir = root / "data" / "processed"
    reports_dir = root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    print("[15%] building manifest")
    manifest = build_manifest(raw_dir, processed_dir)
    update_progress(root, "15%：数据清单完成", f"已读取 {len(manifest['files'])} 个 CSV，manifest 写入 data/processed/raw_manifest.json。")
    print("[35%] loading and cleaning")
    frames = load_raw(raw_dir)
    order, item_join, counts = clean_and_model_table(frames, processed_dir)
    quality = quality_report(frames, order, processed_dir, reports_dir)
    update_progress(root, "35%：数据清洗和质量检查完成", f"订单级记录 {counts['orders_level']:,} 条，商品明细 {counts['items_level']:,} 条；质量报告写入 reports。")
    print("[55%] metrics and charts")
    summary, tables, charts_dir = make_metrics(order, item_join, processed_dir, reports_dir)
    update_progress(root, "55%：指标和探索性分析完成", "已生成月度、类别、地区、卖家、支付、RFM 和评价指标，以及 8 张分析图。")
    print("[75%] modeling")
    late_model = model_task(order, "late_delivery", reports_dir / "models", "late_delivery")
    low_model = model_task(order, "low_review", reports_dir / "models", "low_review")
    update_progress(root, "75%：两个预测任务完成", "已完成时间切分、Logistic Regression、Random Forest、阈值选择、评估指标和重要特征输出。")
    print("[90%] generating PDF")
    pdf_path = generate_pdf(summary, tables, quality, late_model, low_model, charts_dir, reports_dir)
    update_progress(root, "90%：PDF 初稿完成", f"已生成 {pdf_path}。下一步进行 PDF 渲染和文本完整性检查。")
    write_json(reports_dir / "run_summary.json", {"completed_at": datetime.now(), "summary": summary, "pdf": pdf_path, "manifest": processed_dir / "raw_manifest.json"})
    print("[100%] pipeline complete")
    update_progress(root, "100%：数据管线和 PDF 生成完成", "已完成数据清洗、指标、建模和 PDF 生成；需要外部渲染工具完成最终视觉 QA。")
    print(str(pdf_path))


if __name__ == "__main__":
    main()
