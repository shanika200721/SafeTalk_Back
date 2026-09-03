# Text Classification Baseline Model Card

Model name: text-classification-linear-svm

Model version: 1.0.0

Intended use: Research baseline for four-class mental-health text classification.

Prohibited use: Suicide-risk diagnosis, autonomous intervention, counselor alerting, clinical diagnosis, production chat scoring, or treatment recommendation.

Dataset origin: Authoritative Text Classification source `mental_heath_unbanlanced.csv`, canonicalized into the locked Phase 2 Text preprocessing artifact.

Labels: `anxiety`, `depression`, `normal`, and `suicidal` are source annotations, not clinical labels.

Duplicate handling: Exact duplicate and text-hash groups are isolated by the locked split. Conflicting duplicate records are quarantined and excluded.

Limitations: User grouping is incomplete because `Unique_ID` is missing for many records. Reference/test overlap is reported as an aggregate limitation. Social-media or dataset text may not match private production chat. Privacy normalization is imperfect and no raw text is included in reports.

Selected candidate: word_tfidf_linear_svm_C=0.5_class_weight=balanced

Vectorizer: word_tfidf

Test macro F1: 0.780784568160688

Test weighted F1: 0.7921309133145616

Test balanced accuracy: 0.7846379152281703

Test suicidal recall: 0.7221891731112433

Test suicidal false negatives: 467

Human oversight requirement: Human review is required for any research interpretation. This model must not operate autonomously.

This model is a research prototype and is not a clinical diagnostic or autonomous suicide-prevention system.
