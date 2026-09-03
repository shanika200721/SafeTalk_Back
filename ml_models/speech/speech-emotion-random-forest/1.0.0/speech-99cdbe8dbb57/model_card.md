# Speech Emotion Acoustic Baseline Model Card

Model name: speech-emotion-random-forest

Model version: 1.0.0

Intended use: Research eight-class acted speech-emotion classification using deterministic acoustic features only.

Prohibited use: Depression diagnosis, suicide-risk assessment, autonomous suicide-prevention, clinical decision-making, counselor alerting, treatment recommendation, or production voice inference.

Target labels: neutral, calm, happy, sad, angry, fearful, disgust, surprised.

Feature set: full_acoustic

Selected candidate: random_forest_n=200_depth=12_leaf=3_class_weight=none

Split: Locked speaker-isolated Speech v1 train/validation/test split. Candidate selection used validation data only.

Test macro F1: 0.3569147903558647

Test weighted F1: 0.35727431587002456

Test balanced accuracy: 0.3784867915446799

Limitations:
- Acted emotion corpora are not natural distress recordings.
- Emotion is not depression, and depression is not suicide risk.
- Corpus, device, accent, language, microphone, and sample-rate variation can create shortcut behavior.
- TESS and SAVEE have very low speaker counts, so corpus-specific split coverage is limited.
- Voice data is biometric and sensitive even after acoustic feature extraction.
- No transcription, pretrained speech embeddings, deep learning, or production recordings were used.
- Human oversight is required for research interpretation.

This model is a research prototype and is not a clinical diagnostic or autonomous suicide-prevention system.
