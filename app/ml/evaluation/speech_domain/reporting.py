"""Report writers for Speech domain-shift evaluation."""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

from app.ml.common import hashing, paths
from app.ml.training.artifacts import prevent_overwrite


def resolve_output_path(path: str | Path) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = paths.get_repository_root() / candidate
    return candidate.resolve(strict=False)


def repo_relative(path: str | Path) -> str:
    resolved = resolve_output_path(path)
    return resolved.relative_to(paths.get_repository_root()).as_posix()


def write_json(path: str | Path, payload: Mapping[str, Any], *, overwrite: bool = False) -> Path:
    output = resolve_output_path(path)
    prevent_overwrite(output, overwrite=overwrite)
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_name(f".{output.name}.tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    tmp.replace(output)
    return output


def write_markdown(path: str | Path, text: str, *, overwrite: bool = False) -> Path:
    output = resolve_output_path(path)
    prevent_overwrite(output, overwrite=overwrite)
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_name(f".{output.name}.tmp")
    tmp.write_text(text.rstrip() + "\n", encoding="utf-8")
    tmp.replace(output)
    return output


def write_csv(path: str | Path, rows: Iterable[Mapping[str, Any]], *, overwrite: bool = False) -> Path:
    output = resolve_output_path(path)
    rows = list(rows)
    prevent_overwrite(output, overwrite=overwrite)
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_name(f".{output.name}.tmp")
    if not rows:
        tmp.write_text("", encoding="utf-8")
        tmp.replace(output)
        return output
    fieldnames = sorted({key for row in rows for key in row})
    with tmp.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, sort_keys=True) if isinstance(value, (dict, list)) else value for key, value in row.items()})
    tmp.replace(output)
    return output


def artifact_inventory(files: Iterable[str | Path]) -> dict[str, Any]:
    entries = []
    for path in files:
        resolved = resolve_output_path(path)
        if not resolved.exists() or not resolved.is_file():
            continue
        entries.append(
            {
                "path": repo_relative(resolved),
                "size_bytes": resolved.stat().st_size,
                "sha256": hashing.sha256_file(resolved),
                "generated_status": "generated",
                "registration_status": "not_registered",
                "activation_status": "inactive",
            }
        )
    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "file_count": len(entries),
        "files": entries,
        "file_hashes": {entry["path"]: entry["sha256"] for entry in entries},
    }


def build_markdown_summary(summary: Mapping[str, Any]) -> str:
    gap = summary.get("corpus_gap_summary") or {}
    folds = summary.get("folds") or []
    fold_lines = "\n".join(
        f"- {fold.get('test_corpus')}: test macro F1 {((fold.get('test_metrics') or {}).get('macro_f1'))}, balanced accuracy {((fold.get('test_metrics') or {}).get('balanced_accuracy'))}"
        for fold in folds
    )
    blockers = "\n".join(f"- {item}" for item in summary.get("blockers", [])) or "- None"
    recommendations = "\n".join(f"- {item}" for item in summary.get("recommendations", []))
    return f"""# Speech Domain-Shift Evaluation Summary

Evaluation version: {summary.get("evaluation_version")}

LOCO policy version: {summary.get("policy_version")}

Research readiness: {summary.get("research_readiness")}

## LOCO Results

{fold_lines}

## Pooled Versus LOCO

- Pooled baseline test macro F1: {gap.get("pooled_test_macro_f1")}
- LOCO mean macro F1: {gap.get("loco_mean_macro_f1")}
- LOCO standard deviation: {gap.get("loco_std_macro_f1")}
- Worst-corpus macro F1: {gap.get("minimum_corpus_macro_f1")}
- Corpus generalization gap: {gap.get("corpus_generalization_gap")}

## Shortcut Diagnostics

Corpus prediction accuracy: {((summary.get("shortcut_risk_findings") or {}).get("accuracy"))}

## Blockers

{blockers}

## Recommendations

{recommendations}

These results are research-only and must not be used for depression detection, suicide-risk prediction, autonomous intervention, alerts, treatment recommendations, or production inference.
"""


def build_limitations_markdown() -> str:
    return """# Speech Domain-Shift Limitations

- Corpora are acted emotion datasets, not natural distress or clinical speech.
- Emotion labels are not depression labels and are not suicide-risk labels.
- CREMA, RAVDESS, SAVEE, and TESS differ in actors, microphones, sample rates, language/accent context, and protocols.
- TESS has only two speakers and SAVEE has only four speakers.
- Calm is not merged into neutral.
- Surprise and pleasant-surprise mappings are reported with caveats.
- Corpus identity is diagnostic metadata only and is never used as an emotion feature.
- No raw audio, raw speaker IDs, filenames, transcripts, production recordings, alerts, or database writes are included.
"""


def save_domain_shift_reports(
    *,
    output_dir: str | Path,
    summary: dict[str, Any],
    loco_rows: list[dict[str, Any]],
    per_class_rows: list[dict[str, Any]],
    confusion_matrices: dict[str, Any],
    transfer_macro_rows: list[dict[str, Any]],
    transfer_balanced_rows: list[dict[str, Any]],
    shortcut_diagnostics: dict[str, Any],
    feature_distribution: list[dict[str, Any]],
    comparison: dict[str, Any],
    files_for_inventory: list[str | Path],
    overwrite: bool = False,
) -> dict[str, Path]:
    root = resolve_output_path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, Path] = {}
    outputs["summary_json"] = write_json(root / "speech_domain_shift_summary.json", summary, overwrite=overwrite)
    outputs["summary_md"] = write_markdown(root / "speech_domain_shift_summary.md", build_markdown_summary(summary), overwrite=overwrite)
    outputs["loco_results"] = write_csv(root / "speech_loco_results.csv", loco_rows, overwrite=overwrite)
    outputs["per_class_metrics"] = write_csv(root / "speech_loco_per_class_metrics.csv", per_class_rows, overwrite=overwrite)
    outputs["confusion_matrices"] = write_json(root / "speech_loco_confusion_matrices.json", confusion_matrices, overwrite=overwrite)
    outputs["transfer_macro_f1"] = write_csv(root / "speech_transfer_matrix_macro_f1.csv", transfer_macro_rows, overwrite=overwrite)
    outputs["transfer_balanced_accuracy"] = write_csv(root / "speech_transfer_matrix_balanced_accuracy.csv", transfer_balanced_rows, overwrite=overwrite)
    outputs["shortcut_diagnostics"] = write_json(root / "speech_corpus_shortcut_diagnostics.json", shortcut_diagnostics, overwrite=overwrite)
    outputs["feature_distribution"] = write_csv(root / "speech_corpus_feature_distribution.csv", feature_distribution, overwrite=overwrite)
    outputs["comparison"] = write_json(root / "speech_pooled_vs_loco_comparison.json", comparison, overwrite=overwrite)
    outputs["limitations"] = write_markdown(root / "speech_domain_limitations.md", build_limitations_markdown(), overwrite=overwrite)
    inventory = artifact_inventory([*files_for_inventory, *outputs.values()])
    outputs["artifact_inventory"] = write_json(root / "speech_domain_artifact_inventory.json", inventory, overwrite=overwrite)
    return outputs

