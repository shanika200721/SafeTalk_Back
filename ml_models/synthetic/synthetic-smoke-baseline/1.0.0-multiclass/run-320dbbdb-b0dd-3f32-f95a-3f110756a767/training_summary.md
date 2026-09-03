# Training Summary: run-320dbbdb-b0dd-3f32-f95a-3f110756a767

- Status: `completed`
- Framework version: `1.0.0`
- Config hash: `a80867ad7bb0ee6a844f51c17d58d3e978bb556fa23ac4c20b0788ce42f4b09d`
- Manifest hash: `9c390a70439773d73800b5d25c1b8ad0405871000141246ee96912c3fa4996d6`
- Model artifact: `ml_models/synthetic/synthetic-smoke-baseline/1.0.0-multiclass/run-320dbbdb-b0dd-3f32-f95a-3f110756a767/model.joblib`
- Metrics: `ml_models/synthetic/synthetic-smoke-baseline/1.0.0-multiclass/run-320dbbdb-b0dd-3f32-f95a-3f110756a767/metrics.json`
- Model card: `ml_models/synthetic/synthetic-smoke-baseline/1.0.0-multiclass/run-320dbbdb-b0dd-3f32-f95a-3f110756a767/model_card.md`

## Metrics
```json
{
  "selected_thresholds": {
    "strategy": "argmax",
    "thresholds": null
  },
  "test": {
    "accuracy": 0.8333333333333334,
    "balanced_accuracy": 0.85,
    "confusion_matrix": [
      [
        3,
        0,
        1
      ],
      [
        0,
        3,
        0
      ],
      [
        1,
        0,
        4
      ]
    ],
    "f1_macro": 0.85,
    "f1_weighted": 0.8333333333333334,
    "log_loss": 0.5135979892895192,
    "per_class_metrics": {
      "class_0": {
        "f1": 0.75,
        "precision": 0.75,
        "recall": 0.75,
        "support": 4.0
      },
      "class_1": {
        "f1": 1.0,
        "precision": 1.0,
        "recall": 1.0,
        "support": 3.0
      },
      "class_2": {
        "f1": 0.8,
        "precision": 0.8,
        "recall": 0.8,
        "support": 5.0
      }
    },
    "pr_auc": 0.931111111111111,
    "precision_macro": 0.85,
    "precision_weighted": 0.8333333333333334,
    "recall_macro": 0.85,
    "recall_weighted": 0.8333333333333334,
    "roc_auc": 0.9505952380952382,
    "support": {
      "class_0": 4,
      "class_1": 3,
      "class_2": 5
    },
    "warnings": []
  },
  "train": {
    "accuracy": 0.8611111111111112,
    "balanced_accuracy": 0.8607226107226107,
    "confusion_matrix": [
      [
        9,
        1,
        2
      ],
      [
        1,
        12,
        0
      ],
      [
        0,
        1,
        10
      ]
    ],
    "f1_macro": 0.8588786414873372,
    "f1_weighted": 0.8594154101400479,
    "log_loss": 0.48334689358333816,
    "per_class_metrics": {
      "class_0": {
        "f1": 0.8181818181818182,
        "precision": 0.9,
        "recall": 0.75,
        "support": 12.0
      },
      "class_1": {
        "f1": 0.8888888888888888,
        "precision": 0.8571428571428571,
        "recall": 0.9230769230769231,
        "support": 13.0
      },
      "class_2": {
        "f1": 0.8695652173913043,
        "precision": 0.8333333333333334,
        "recall": 0.9090909090909091,
        "support": 11.0
      }
    },
    "pr_auc": 0.8479466603575337,
    "precision_macro": 0.8634920634920635,
    "precision_weighted": 0.8641534391534392,
    "recall_macro": 0.8607226107226107,
    "recall_weighted": 0.8611111111111112,
    "roc_auc": 0.9253364328907807,
    "support": {
      "class_0": 12,
      "class_1": 13,
      "class_2": 11
    },
    "warnings": []
  },
  "validation": {
    "accuracy": 0.75,
    "balanced_accuracy": 0.75,
    "confusion_matrix": [
      [
        2,
        0,
        2
      ],
      [
        0,
        4,
        0
      ],
      [
        0,
        1,
        3
      ]
    ],
    "f1_macro": 0.7407407407407406,
    "f1_weighted": 0.7407407407407406,
    "log_loss": 0.5847612272244811,
    "per_class_metrics": {
      "class_0": {
        "f1": 0.6666666666666666,
        "precision": 1.0,
        "recall": 0.5,
        "support": 4.0
      },
      "class_1": {
        "f1": 0.8888888888888888,
        "precision": 0.8,
        "recall": 1.0,
        "support": 4.0
      },
      "class_2": {
        "f1": 0.6666666666666666,
        "precision": 0.6,
        "recall": 0.75,
        "support": 4.0
      }
    },
    "pr_auc": 0.7535714285714286,
    "precision_macro": 0.7999999999999999,
    "precision_weighted": 0.7999999999999999,
    "recall_macro": 0.75,
    "recall_weighted": 0.75,
    "roc_auc": 0.84375,
    "support": {
      "class_0": 4,
      "class_1": 4,
      "class_2": 4
    },
    "warnings": []
  }
}
```
