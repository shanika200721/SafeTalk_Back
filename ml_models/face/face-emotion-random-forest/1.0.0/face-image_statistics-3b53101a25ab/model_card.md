# Face Emotion Research Baseline Model Card

Model name: face-emotion-random-forest

Model version: 1.0.0

Intended research use: seven-class facial emotion classification (`angry`, `disgust`, `fear`, `happy`, `neutral`, `sad`, `surprise`).

Prohibited use: depression diagnosis, suicide-risk assessment, counselor alerting, autonomous intervention, identity recognition, demographic inference, production webcam scoring, treatment recommendation, or clinical decision support.

Dataset: Facial Emotion images, 48x48 grayscale.

Split: Phase 3G leakage-safe v2 split. Original source split is metadata only and is not a predictive feature.

Duplicate remediation: exact duplicate controls are applied; 155 cross-label records remain quarantined and excluded.

Review status: Phase 3H review complete with unresolved items. Reviewer independence is recorded as `reviewer_independence_unverified`.

Subject IDs: unavailable. Subject-independent evaluation is not established.

Demographic limitations: demographic composition is unavailable; demographic fairness is not established.

Class imbalance: disgust has much lower support than the other classes.

Selected candidate: random_forest_n=40_depth=10_class_weight=none_image_statistics

Validation macro F1: 0.20657596371882087

Test macro F1: 0.11510920157536698

Test balanced accuracy: 0.1285714285714286

Biometric privacy: no face recognition, embeddings, identity matching, raw image export, thumbnails, or participant-level heatmaps are included.

Human oversight: required for any research interpretation.

This model is a research prototype and is not a clinical diagnostic or autonomous suicide-prevention system.
