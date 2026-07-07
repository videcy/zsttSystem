"""
Data bridge: synchronise zsttSystem offline pipeline output into LightRAG.

Reads the chunked syllabus data and concept registry produced by the
zsttSystem offline pipeline, enriches each chunk with concept annotations
and course metadata, then inserts them into LightRAG for indexing.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.config import config
from src.online_service.lightrag_adapter import LightRAGClient


class LightRAGDataBridge:
    """Read zsttSystem pipeline output and sync to LightRAG."""

    def __init__(self, lightrag: LightRAGClient | None = None):
        self.lightrag = lightrag or LightRAGClient()

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------
    @staticmethod
    def load_chunked_data(path: str | Path) -> list[dict[str, Any]]:
        """Load the chunked syllabus data from a JSON file."""
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        return data if isinstance(data, list) else [data]

    @staticmethod
    def load_concept_registry(path: str | Path) -> dict[str, dict[str, Any]]:
        """Load the concept registry and build an alias-keyed lookup dict.

        The registry file is a JSON list produced by ConceptNormalizer.  Each
        entry contains ``canonical_name`` and ``aliases``.  We index every alias
        (and the canonical name itself) so callers can resolve any surface form
        directly via ``registry[term]``.
        """
        try:
            with open(path, "r", encoding="utf-8") as fh:
                payload = json.load(fh)
        except (FileNotFoundError, json.JSONDecodeError):
            return {}

        entries = payload if isinstance(payload, list) else []
        lookup: dict[str, dict[str, Any]] = {}
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            canonical = entry.get("canonical_name", "")
            if canonical:
                lookup[canonical] = entry
            for alias in entry.get("aliases", []):
                alias_str = str(alias).strip()
                if alias_str and alias_str not in lookup:
                    lookup[alias_str] = entry
        return lookup

    # ------------------------------------------------------------------
    # Enrichment
    # ------------------------------------------------------------------
    def enrich_chunks(
        self,
        chunks: list[dict[str, Any]],
        concept_registry_path: str | Path | None = None,
    ) -> list[dict[str, Any]]:
        """Annotate each chunk with concept tags and course metadata.

        Returns a list of dicts ready for ingestion::

            {"text": <enriched_text>, "description": <course_section_label>}
        """
        registry = {}
        if concept_registry_path:
            registry = self.load_concept_registry(concept_registry_path)

        enriched: list[dict[str, Any]] = []

        for chunk in chunks:
            text = chunk.get("text", "") or ""
            meta = chunk.get("metadata", {}) or {}
            raw_concepts = meta.get("core_concepts", []) or []

            # Resolve concept names to their canonical forms
            concept_tags: list[str] = []
            for c in raw_concepts:
                if isinstance(c, dict):
                    c_name = c.get("name", "")
                else:
                    c_name = str(c)
                canonical = registry.get(c_name, {}).get("canonical_name", c_name)
                if canonical:
                    concept_tags.append(f"[概念: {canonical}]")

            # Build metadata header
            course_name = meta.get("course_name", "") or ""
            course_code = meta.get("course_code", "") or ""
            section = meta.get("syllabus_section", "") or ""
            semester = meta.get("semester", "") or ""
            prereq = meta.get("prerequisites", "") or []

            header_parts = [f"课程: {course_name} ({course_code})"]
            if section:
                header_parts.append(f"章节: {section}")
            if semester:
                header_parts.append(f"学期: {semester}")
            if prereq:
                prereq_str = ", ".join(prereq) if isinstance(prereq, list) else str(prereq)
                header_parts.append(f"先修课程: {prereq_str}")
            if concept_tags:
                header_parts.append(" ".join(concept_tags))

            enriched_text = "\n".join(header_parts) + "\n---\n" + text.strip()

            # Build description label for LightRAG
            desc = f"{course_code}_{section}" if course_code and section else f"chunk_{chunk.get('chunk_id', '')}"

            enriched.append({"text": enriched_text, "description": desc, "chunk_id": chunk.get("chunk_id", "")})

        return enriched

    # ------------------------------------------------------------------
    # Sync
    # ------------------------------------------------------------------
    def sync(
        self,
        chunks: list[dict[str, Any]] | None = None,
        *,
        chunk_path: str | Path | None = None,
        concept_registry_path: str | Path | None = None,
    ) -> dict[str, Any]:
        """Enrich and sync data to LightRAG.

        If *chunks* is not provided, loads from *chunk_path*.
        If *chunk_path* is also None, uses ``config.chunked_output_path``.
        """
        if chunks is None:
            path = Path(chunk_path or config.chunked_output_path)
            chunks = self.load_chunked_data(path)

        if concept_registry_path is None:
            concept_registry_path = config.concept_registry_path

        enriched = self.enrich_chunks(chunks, concept_registry_path)

        print(f"[DataBridge] Syncing {len(enriched)} chunks to LightRAG...")
        result = self.lightrag.insert_batch(enriched)

        print(
            f"[DataBridge] Done. success={result['success']}, "
            f"failed={result['failed']}"
        )
        if result["errors"]:
            print(f"[DataBridge] First 3 errors:\n{result['errors'][:3]!r}")

        return result


# ---------------------------------------------------------------------------
# Convenience entry point (usable from CLI / run_pipeline.py)
# ---------------------------------------------------------------------------

def run_sync(
    chunk_path: str | Path | None = None,
    concept_registry_path: str | Path | None = None,
) -> dict[str, Any]:
    """One-shot sync: load chunks, enrich, push to LightRAG."""
    bridge = LightRAGDataBridge()
    return bridge.sync(
        chunk_path=chunk_path,
        concept_registry_path=concept_registry_path,
    )
