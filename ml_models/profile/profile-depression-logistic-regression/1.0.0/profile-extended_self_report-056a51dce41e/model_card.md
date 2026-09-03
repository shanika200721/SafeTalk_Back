# Profile Depression Baseline Model Card

Model name: profile-depression-logistic-regression

Model version: 1.0.0

Intended use: Research baseline for self-reported depression classification.

Prohibited use: Suicide-risk diagnosis, autonomous intervention, counselor alerting, clinical diagnosis, or treatment recommendation.

Dataset: 101 Student Profile records with self-reported labels.

Split: train 71, validation 15, test 15 using the locked Profile split manifest.

Target: Self-reported depression (`target_depression`), not suicidal ideation and not suicide-risk ground truth.

Feature set: extended_self_report

Selected candidate: logistic_regression_C=10.0_class_weight=balanced_threshold=max_f1

Threshold policy: max_f1 at threshold 0.2642610262338985. Selected with validation data only and not clinically validated.

Test recall: 1.0

Test F1: 0.5

Test false negatives: 0

Limitations:
- Tiny dataset with unstable metrics.
- Validation and test splits each contain only 15 records.
- Self-reported label, likely single-context sample, and no suicidal-ideation label.
- Sensitive-feature concerns remain; sensitive context is not used by the primary model.
- Anxiety and panic features can conceptually overlap with depression and are ablation-only, not the primary minimal baseline.
- No causal, clinical, or generalizable claim is supported.

Human oversight requirement: Human review is required for any research interpretation; this model must not operate autonomously.

This model is a research prototype and is not a clinical diagnostic or autonomous suicide-prevention system.
