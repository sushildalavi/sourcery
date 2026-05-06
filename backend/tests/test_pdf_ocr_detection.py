"""Tests for `pdf_needs_ocr` — the image-only PDF detector."""

from __future__ import annotations

from backend.pdf_ingest import pdf_needs_ocr


def test_empty_page_list_needs_ocr():
    assert pdf_needs_ocr([]) is True


def test_pages_with_only_whitespace_need_ocr():
    pages = [(1, "   "), (2, "\n\n"), (3, "")]
    assert pdf_needs_ocr(pages) is True


def test_one_substantive_page_is_enough():
    pages = [
        (1, ""),
        (2, "x" * 41),  # > default min_chars=40
        (3, ""),
    ]
    assert pdf_needs_ocr(pages) is False


def test_threshold_is_configurable():
    pages = [(1, "x" * 10)]
    assert pdf_needs_ocr(pages, min_chars=5) is False
    assert pdf_needs_ocr(pages, min_chars=20) is True


def test_short_snippets_below_threshold_still_need_ocr():
    pages = [(1, "abc"), (2, "de"), (3, "f")]
    assert pdf_needs_ocr(pages) is True
