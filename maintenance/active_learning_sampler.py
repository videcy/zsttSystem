"""Select high-value query samples for manual review and correction."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
QUERY_LOG_PATH = PROJECT_ROOT / "outputs" / "query_log.jsonl"
FEEDBACK_LOG_PATH = PROJECT_ROOT / "outputs" / "feedback_log.jsonl"
OUTPUT_PATH = PROJECT_ROOT / "outputs" / "samples_for_review.json"


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Load newline-delimited JSON records."""
    file_path = Path(path)
    if not file_path.exists():
        return []
    records: list[dict[str, Any]] = []
    for line in file_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            records.append(json.loads(line))
    return records


def has_uncertain_verification(record: dict[str, Any]) -> bool:
    """Prioritize samples whose NLI verification is not fully entailed."""
    verification = record.get("verification", [])
    if not isinstance(verification, list):
        return False
    for item in verification:
        if not isinstance(item, dict):
            continue
        label = str(item.get("label", "")).lower()
        if "neutral" in label or "contradiction" in label:
            return True
    return False


def build_review_samples(
    query_logs: list[dict[str, Any]],
    feedback_logs: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Assemble high-value samples using uncertainty and negative feedback."""
    negative_feedback_ids = {
        str(record.get("query_id", "")).strip()
        for record in feedback_logs
        if isinstance(record, dict) and record.get("is_helpful") is False
    }

    samples: list[dict[str, Any]] = []
    for record in query_logs:
        if not isinstance(record, dict):
            continue
        query_id = str(record.get("query_id", "")).strip()
        if not query_id:
            continue

        reasons: list[str] = []
        if has_uncertain_verification(record):
            reasons.append("uncertainty_sampling")
        if query_id in negative_feedback_ids:
            reasons.append("negative_user_feedback")
        if not reasons:
            continue

        samples.append(
            {
                "query_id": query_id,
                "query": record.get("query", ""),
                "response": record.get("response", ""),
                "status": record.get("status", ""),
                "verification": record.get("verification", []),
                "linked_entities": record.get("linked_entities", []),
                "citations": record.get("citations", []),
                "sampling_reasons": reasons,
            }
        )
    return samples


def main() -> None:
    """Generate review samples from query and feedback logs."""
    query_logs = load_jsonl(QUERY_LOG_PATH)
    feedback_logs = load_jsonl(FEEDBACK_LOG_PATH)
    samples = build_review_samples(query_logs, feedback_logs)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(
        json.dumps(samples, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Prepared {len(samples)} samples for review at {OUTPUT_PATH}.")


if __name__ == "__main__":
    main()
