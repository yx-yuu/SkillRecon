from __future__ import annotations

import pytest

from skillrecon.contract.iccm import ICCMExtractionError, ICCMExtractor
from skillrecon.core.types import ClauseSample, ClauseSampleList, Step


class _FailingClient:
    def structured_complete(self, *args, **kwargs):  # noqa: ANN002, ANN003
        raise RuntimeError("connection failed")


class _ContentFilterThenSuccessClient:
    def __init__(self) -> None:
        self.calls: list[tuple[object, object]] = []

    def structured_complete(self, messages, response_model, **kwargs):  # noqa: ANN001, ANN003
        self.calls.append((messages, kwargs))
        if len(self.calls) == 1:
            raise RuntimeError("Error code: 400 - contentFilter 1301")
        assert response_model is ClauseSampleList
        return ClauseSampleList(
            clauses=[
                ClauseSample(
                    step_id="s1",
                    capability="http_request",
                    operator="allowed",
                    target="[URL_REF:E000_00]",
                    constraint="output path [PATH_REF:E000_00]",
                    evidence_span="[EVIDENCE_REF:E000]",
                    confidence_note="Recovered from [EVIDENCE_REF:E000]",
                )
            ]
        )


def test_iccm_chunk_failure_is_not_silently_skipped() -> None:
    extractor = ICCMExtractor(
        _FailingClient(),
        taxonomy_atoms=["http_request"],
        prompt_version="v1",
    )

    with pytest.raises(ICCMExtractionError) as exc_info:
        extractor.extract_single(
            "owner/skill",
            [
                Step(
                    step_id="s1",
                    doc_id="d0",
                    order_index=0,
                    local_index=0,
                    step_type="instruction",
                    text="Call the approved API.",
                )
            ],
            sample_index=0,
        )

    assert "owner/skill" in str(exc_info.value)
    assert "chunk=0" in str(exc_info.value)


def test_iccm_content_filter_retry_uses_safe_view_and_restores_evidence() -> None:
    client = _ContentFilterThenSuccessClient()
    extractor = ICCMExtractor(
        client,
        taxonomy_atoms=["http_request", "file_write"],
        prompt_version="v1",
    )
    original_text = (
        "Run the downloader on https://youtube.com/watch?v=abc123 and save "
        "the video under ~/.openclaw/workspace/assets/videos."
    )

    clauses = extractor.extract_single(
        "owner/skill",
        [
            Step(
                step_id="s1",
                doc_id="d0",
                order_index=0,
                local_index=0,
                step_type="instruction",
                text=original_text,
            )
        ],
        sample_index=0,
    )

    assert len(client.calls) == 2
    retry_messages = client.calls[1][0]
    retry_prompt = retry_messages[1]["content"]
    assert "[EVIDENCE_REF:E000]" in retry_prompt
    assert "youtube.com/watch" not in retry_prompt
    assert "external_video_platform_url" in retry_prompt
    assert clauses[0].evidence_span == original_text
    assert clauses[0].target == "https://youtube.com/watch?v=abc123"
    assert clauses[0].constraint == (
        "output path ~/.openclaw/workspace/assets/videos."
    )
