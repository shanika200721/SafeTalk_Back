"""Validate and preprocess speech emotion corpora without training models."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.ml.common.fingerprinting import verify_dataset_fingerprint
from app.ml.common.schemas import DatasetConfig
from app.ml.common.serialization import load_dataset_fingerprint
from app.ml.preprocessing.speech.mapping import load_speech_corpus_mapping_config, load_speech_label_mapping_config
from app.ml.preprocessing.speech.preprocessor import preprocess_speech_dataset


def _resolve_cli_path(value: str) -> Path:
    candidate = Path(value)
    if not candidate.is_absolute():
        candidate = (Path.cwd() / candidate).resolve(strict=False)
    return candidate


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return payload


def _corpus_dataset_config(corpus, fingerprint=None) -> DatasetConfig:
    return DatasetConfig(
        dataset_name=fingerprint.dataset_name if fingerprint is not None else f"speech-emotion-{corpus.corpus_name.lower()}",
        dataset_version=fingerprint.dataset_version if fingerprint is not None else "v1",
        modality="voice",
        source_path=corpus.source_path,
        file_format="folder",
        label_columns=[],
        feature_columns=[],
        identifier_columns=[],
        sensitive_columns=[],
        excluded_columns=[],
        expected_columns=[],
        missing_value_policy="preserve",
        duplicate_policy="report_only",
        is_raw_source=True,
        validation_context="test" if Path(corpus.source_path).is_absolute() else None,
    )


def _load_fingerprints(fingerprint_dir: Path, corpus_mapping) -> dict:
    if not fingerprint_dir.exists():
        raise FileNotFoundError(f"Fingerprint directory does not exist: {fingerprint_dir}")
    loaded = []
    for path in sorted(fingerprint_dir.rglob("*.json")):
        try:
            loaded.append((path, load_dataset_fingerprint(path)))
        except Exception:
            continue
    result = {}
    for corpus in corpus_mapping.corpora:
        corpus_path_name = Path(corpus.source_path).name.lower()
        matches = [
            item
            for item in loaded
            if corpus.corpus_name.lower() in item[0].stem.lower()
            or corpus.corpus_name.lower() in item[1].dataset_name.lower()
            or corpus_path_name in item[1].source_relative_path.lower()
        ]
        if not matches:
            raise FileNotFoundError(f"Missing fingerprint for speech corpus: {corpus.corpus_name}")
        fingerprint = matches[0][1]
        config = _corpus_dataset_config(corpus, fingerprint=fingerprint)
        if not verify_dataset_fingerprint(fingerprint, config):
            raise ValueError(f"Speech source fingerprint mismatch: {corpus.corpus_name}")
        result[corpus.corpus_name] = fingerprint
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Preprocess speech emotion datasets into canonical research outputs.")
    parser.add_argument("--dataset-config", required=True, help="Speech dataset overview JSON. Used for source-root documentation.")
    parser.add_argument("--preprocessing-config", required=True, help="Speech preprocessing config JSON.")
    parser.add_argument("--label-mapping-config", required=True)
    parser.add_argument("--corpus-mapping-config", required=True)
    parser.add_argument("--fingerprint-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max-files", type=int)
    parser.add_argument("--corpus", action="append", help="Limit to a corpus; may be supplied more than once.")
    parser.add_argument("--extract-features", action="store_true")
    parser.add_argument("--write-normalized-audio", action="store_true")
    parser.add_argument("--target-sample-rate", type=int)
    parser.add_argument("--mono", action="store_true")
    parser.add_argument("--sample-seed", type=int, default=42)
    parser.add_argument("--report-path", help="Optional Markdown report path. In validate-only mode, this is the only write.")
    return parser


def _print_summary(result: dict) -> None:
    print(f"validation: {'passed' if result['valid'] else 'failed'}")
    print(f"validate only: {result['validate_only']}")
    print(f"feature extraction: {result['extract_features']}")
    print(f"source files: {result['source_files']}")
    print(f"output records: {result['output_records']}")
    print(f"excluded records: {result['excluded_records']}")
    print(f"readable files: {result['readable_files']}")
    print(f"unreadable files: {result['unreadable_files']}")
    print(f"corpus distribution: {result['corpus_distribution']}")
    print(f"label distribution: {result['label_distribution']}")
    print(f"sample rates: {result['sample_rate_distribution']}")
    print(f"channels: {result['channel_distribution']}")
    print(f"duplicates: {result['duplicate_summary']}")
    if result["outputs"]:
        for label, path in result["outputs"].items():
            print(f"{label}: {path}")


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        _load_json(_resolve_cli_path(args.dataset_config))
        preprocessing_payload = _load_json(_resolve_cli_path(args.preprocessing_config))
        if preprocessing_payload.get("preprocessing_version") not in {None, "1.0.0"}:
            raise ValueError("Unsupported speech preprocessing version")
        label_mapping = load_speech_label_mapping_config(_resolve_cli_path(args.label_mapping_config))
        corpus_mapping = load_speech_corpus_mapping_config(_resolve_cli_path(args.corpus_mapping_config))
        fingerprints = _load_fingerprints(_resolve_cli_path(args.fingerprint_dir), corpus_mapping)
        result = preprocess_speech_dataset(
            corpus_mapping,
            label_mapping,
            source_fingerprints=fingerprints,
            output_dir=_resolve_cli_path(args.output_dir),
            overwrite=args.overwrite,
            validate_only=args.validate_only,
            max_files=args.max_files,
            corpus_filter=args.corpus,
            extract_features=args.extract_features,
            write_normalized_audio=args.write_normalized_audio,
            target_sample_rate=args.target_sample_rate,
            mono=bool(args.mono),
        )
        if args.report_path:
            from app.ml.preprocessing.speech.reporting import save_speech_report_markdown

            save_speech_report_markdown(result["report"], _resolve_cli_path(args.report_path), overwrite=args.overwrite)
        _print_summary(result)
        return 0
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

