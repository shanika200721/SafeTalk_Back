# Face Emotion Research Baseline Model Card

Model name: face-emotion-linear-svm

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

Selected candidate: linear_svm_C=0.1_class_weight=balanced_image_statistics

Validation macro F1: 0.17171761482387113

Test macro F1: 0.16876282354584574

Test balanced accuracy: 0.200307519855169

Biometric privacy: no face recognition, embeddings, identity matching, raw image export, thumbnails, or participant-level heatmaps are included.

Human oversight: required for any research interpretation.

This model is a research prototype and is not a clinical diagnostic or autonomous suicide-prevention system.
