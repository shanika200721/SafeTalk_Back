# Training Summary: run-f126213d-8681-945c-8a3e-4cb15a496d51

- Status: `completed`
- Framework version: `1.0.0`
- Config hash: `578468d5db72ff4ec5c8d7882d02f42df4b0fdce05dc28e90ed626cb2a782f1c`
- Manifest hash: `336d60bac411a0e1013832ffe4c6ad2d9c96c7ad490bfa107f57ebf35f9aacee`
- Model artifact: `ml_models/synthetic/synthetic-smoke-baseline/1.0.0/run-f126213d-8681-945c-8a3e-4cb15a496d51/model.joblib`
- Metrics: `ml_models/synthetic/synthetic-smoke-baseline/1.0.0/run-f126213d-8681-945c-8a3e-4cb15a496d51/metrics.json`
- Model card: `ml_models/synthetic/synthetic-smoke-baseline/1.0.0/run-f126213d-8681-945c-8a3e-4cb15a496d51/model_card.md`

## Metrics
```json
{
  "selected_thresholds": {
    "note": "Selected with validation data only; not clinically validated.",
    "objective": "maximize validation F1",
    "strategy": "max_f1",
    "threshold": 0.26767935951108884,
    "validation_metric": 1.0,
    "warnings": []
  },
  "test": {
    "accuracy": 0.8181818181818182,
    "balanced_accuracy": 0.8,
    "brier_score": 0.06800623433318886,
    "confusion_matrix": [
      [
        3,
        2
      ],
      [
        0,
        6
      ]
    ],
    "f1_macro": 0.8035714285714286,
    "f1_weighted": 0.8084415584415584,
    "false_negative_count": 0,
    "false_positive_count": 2,
    "log_loss": 0.26164757234645997,
    "per_class_metrics": {},
    "pr_auc": 1.0,
    "precision_macro": 0.875,
    "precision_weighted": 0.8636363636363636,
    "recall_macro": 0.8,
    "recall_weighted": 0.8181818181818182,
    "roc_auc": 1.0,
    "specificity": 0.6,
    "support": {
      "class_0": 5,
      "class_1": 6
    },
    "threshold": 0.26767935951108884,
    "warnings": []
  },
  "train": {
    "accuracy": 0.8571428571428571,
    "balanced_accuracy": 0.875,
    "brier_score": 0.06515991593876917,
    "confusion_matrix": [
      [
        12,
        4
      ],
      [
        0,
        12
      ]
    ],
    "f1_macro": 0.8571428571428571,
    "f1_weighted": 0.8571428571428571,
    "false_negative_count": 0,
    "false_positive_count": 4,
    "log_loss": 0.24677390043099165,
    "per_class_metrics": {},
    "pr_auc": 0.9866452991452993,
    "precision_macro": 0.875,
    "precision_weighted": 0.8928571428571429,
    "recall_macro": 0.875,
    "recall_weighted": 0.8571428571428571,
    "roc_auc": 0.9895833333333334,
    "specificity": 0.75,
    "support": {
      "class_0": 16,
      "class_1": 12
    },
    "threshold": 0.26767935951108884,
    "warnings": []
  },
  "validation": {
    "accuracy": 1.0,
    "balanced_accuracy": 1.0,
    "brier_score": 0.08591495262624477,
    "confusion_matrix": [
      [
        3,
        0
      ],
      [
        0,
        6
      ]
    ],
    "f1_macro": 1.0,
    "f1_weighted": 1.0,
    "false_negative_count": 0,
    "false_positive_count": 0,
    "log_loss": 0.29618294522511396,
    "per_class_metrics": {},
    "pr_auc": 1.0,
    "precision_macro": 1.0,
    "precision_weighted": 1.0,
    "recall_macro": 1.0,
    "recall_weighted": 1.0,
    "roc_auc": 1.0,
    "specificity": 1.0,
    "support": {
      "class_0": 3,
      "class_1": 6
    },
    "threshold": 0.26767935951108884,
    "warnings": []
  }
}
```
