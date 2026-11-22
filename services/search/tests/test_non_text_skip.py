"""Only text formats are indexed; images and unknown types are skipped early.

This keeps indexing work focused on extractable content while preserving
consistent document counts. Skip decisions are made from the key alone, so a
file that is never fetched cannot incur download or extraction work.
"""

from __future__ import annotations

import pytest
from rag_search.indexer import NON_TEXT_TYPES

from rag_core.models.enums import DocumentType
from rag_core.utils.doctypes import infer_document_type


@pytest.mark.parametrize(
    "key",
    [
        "TENANT_B//00ZQ3BkdLEu5jQ7VEIXuXg_LoadImage6~jpg",
        "TENANT_B//photo~jpeg",
        "TENANT_B//scan~png",
        "TENANT_A/COMPDOCCAT/diagram~tiff",
        "TENANT_A/plan.bmp",
        "TENANT_A/animation.gif",
    ],
)
def test_image_keys_are_classified_as_non_text(key: str) -> None:
    assert infer_document_type(key) in NON_TEXT_TYPES


@pytest.mark.parametrize(
    "key",
    [
        "TENANT_B//0510e0e763dc_PIF-0beddd0c~pdf",
        "TENANT_A/supplier_quality_manual.pdf",
        "TENANT_B//Sheet1~xlsx",
        "TENANT_A/India-Employee Manual~docx",
        "TENANT_A/notes~txt",
        "TENANT_A/report~html",
        "TENANT_A/feed~xml",
        "TENANT_A/deck~pptx",
    ],
)
def test_text_bearing_keys_are_kept(key: str) -> None:
    assert infer_document_type(key) not in NON_TEXT_TYPES


@pytest.mark.parametrize(
    "key", ["TENANT_A/no_extension_at_all", "TENANT_A/bundle.zip", "TENANT_A/clip~mp4", "TENANT_A/mail.msg", "TENANT_A/x.csv"]
)
def test_unrecognised_formats_are_skipped(key: str) -> None:
    """Only text formats are indexed: a key whose extension is not one of them is
    skipped like an image, rather than downloaded on the chance it extracts."""
    assert infer_document_type(key) in NON_TEXT_TYPES


def test_exactly_the_text_formats_are_indexed() -> None:
    from rag_search.indexer import TEXT_TYPES

    assert frozenset({DocumentType.IMAGE, DocumentType.UNKNOWN}) == NON_TEXT_TYPES
    every_type = frozenset(DocumentType)
    assert every_type == TEXT_TYPES | NON_TEXT_TYPES


def test_the_summary_can_report_the_skip() -> None:
    """The count is surfaced, so a run that indexes little says why."""
    from rag_search.models import IndexSummary

    summary = IndexSummary(bucket="b", prefix="TENANT_B/", namespace="default")
    assert summary.documents_non_text == 0
    summary.documents_non_text += 3
    assert summary.documents_non_text == 3
