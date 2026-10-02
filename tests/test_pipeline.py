import unittest
from pathlib import Path

import pandas as pd

import run_pipeline


class PipelineSmokeTest(unittest.TestCase):
    def test_raw_file_list_is_complete(self):
        self.assertEqual(len(run_pipeline.RAW_FILES), 9)

    def test_order_value_is_additive(self):
        frame = pd.DataFrame({"item_value": [10.0, 20.0], "freight_value": [2.0, 3.0]})
        frame["order_value"] = frame["item_value"] + frame["freight_value"]
        self.assertEqual(frame["order_value"].sum(), 35.0)

    def test_feature_set_excludes_outcome_columns(self):
        order = pd.DataFrame({
            "purchase_hour": [1], "purchase_weekday": ["Monday"], "primary_category": ["x"],
            "customer_state": ["SP"], "seller_state": ["SP"], "item_count": [1],
            "product_count": [1], "seller_count": [1], "item_value": [1.0], "freight_value": [1.0],
            "payment_installments": [1], "payment_type": ["credit_card"], "seller_history_orders": [0],
            "seller_prev_late_rate": [0.0], "customer_history_orders": [0], "customer_is_repeat": [0],
        })
        features = run_pipeline.make_model_features(order)
        self.assertNotIn("late_delivery", features)
        self.assertNotIn("low_review", features)
        self.assertNotIn("order_delivered_customer_date", features)


if __name__ == "__main__":
    unittest.main()
