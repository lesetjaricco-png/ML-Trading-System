"""Prepare the validated US30 M15 research dataset without training a model."""

from __future__ import annotations

import argparse
import json
from datetime import datetime

from src.data_ingestion import DataIngestion
from src.dataset_preparation import (
    feature_engineer_from_config,
    prepare_ml_dataset,
    save_processed_dataset,
)
from src.utils import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare an auditable ML dataset")
    parser.add_argument("--config", default="config/config_v03.yaml")
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--output-dir", default="data/processed")
    return parser.parse_args()


def resolve_date(value: str) -> str:
    if value.lower() in {"current", "today", "now"}:
        return datetime.now().strftime("%Y-%m-%d")
    return value


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    data_cfg = cfg["data"]
    start_date = resolve_date(data_cfg["start_date"])
    end_date = resolve_date(data_cfg["end_date"])

    ingestion = DataIngestion(data_dir="data/raw")
    raw = ingestion.fetch(
        ticker=data_cfg["ticker"],
        start_date=start_date,
        end_date=end_date,
        interval=data_cfg["interval"],
        source=data_cfg["source"],
        use_cache=not args.no_cache and bool(data_cfg.get("use_cache", True)),
    )
    engineer = feature_engineer_from_config(cfg)
    processed, metadata = prepare_ml_dataset(raw, cfg, feature_engineer=engineer)
    parquet_path, metadata_path = save_processed_dataset(
        processed, metadata, output_dir=args.output_dir
    )
    print(
        json.dumps(
            {
                "processed_dataset": str(parquet_path),
                "metadata": str(metadata_path),
                "raw_rows": metadata["raw_row_count"],
                "processed_rows": metadata["processed_row_count"],
                "feature_count": metadata["feature_count"],
                "target_class_distribution": metadata["target_class_distribution"],
                "splits": metadata["splits"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
