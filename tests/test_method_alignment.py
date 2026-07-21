from __future__ import annotations

from skillrecon.behavior.pipeline import BehaviorObservationPipeline
from skillrecon.core.enums import (
    BehaviorKind,
    CertificateKind,
    ClauseOperator,
    FindingType,
    HypothesisStatus,
    JudgmentKind,
    PredicateResult,
    RouteSupport,
)
from skillrecon.core.types import (
    CandidatePair,
    CapabilityEvent,
    Certificate,
    Clause,
    DocBlock,
    DocumentPack,
    EvidenceSpan,
    Finding,
    PackageLink,
    ReconciliationJudgment,
    Witness,
)
from skillrecon.contract.pipeline import (
    _ground_evidence,
    _load_doc_contents_from_document_pack,
)
from skillrecon.detect.witness import validate_witness
from skillrecon.reconcile.candidate import _passes_semantic_threshold, _semantic_score
from skillrecon.reconcile.predicate import execution_route_justified


def test_document_pack_resume_contents_support_evidence_grounding() -> None:
    document_pack = DocumentPack(
        skill_id="owner/skill",
        admitted_docs=[],
        doc_blocks=[
            DocBlock(
                block_id="doc0_b0",
                doc_id="doc0",
                block_type="paragraph",
                content="The tool may send requests only after user authorization.",
                start_offset=12,
                end_offset=68,
            )
        ],
    )
    clause = Clause(
        clause_id="c1",
        capability="http_request",
        operator=ClauseOperator.ALLOWED,
        evidence_spans=[
            EvidenceSpan(
                doc_id="unresolved",
                start_offset=0,
                end_offset=0,
                text="only after user authorization",
            )
        ],
    )

    doc_contents = _load_doc_contents_from_document_pack(document_pack)
    grounded = _ground_evidence([clause], doc_contents)

    assert grounded[0].evidence_spans[0].doc_id == "doc0"
    assert grounded[0].evidence_spans[0].start_offset > 0


def test_embedding_fallback_accepts_near_surface_match_without_shared_tokens() -> None:
    lexical_score, shared_tokens, embedding_score = _semantic_score(
        {"credential"},
        {"credentials"},
    )

    assert lexical_score == 0.0
    assert shared_tokens == set()
    assert embedding_score >= 0.55
    assert _passes_semantic_threshold(lexical_score, shared_tokens, embedding_score)


def test_orchestration_hypothesis_requires_execution_cue() -> None:
    link = PackageLink(
        link_id="pl0",
        source_doc_id="doc0",
        source_span="scripts/sync.py",
        target_path="scripts/sync.py",
        target_file_id="f1",
        target_unit_id="u1",
    )
    passive_evidence = EvidenceSpan(
        doc_id="doc0",
        start_offset=0,
        end_offset=36,
        text="See `scripts/sync.py` for details.",
    )
    run_evidence = EvidenceSpan(
        doc_id="doc0",
        start_offset=0,
        end_offset=31,
        text="Run `scripts/sync.py` first.",
    )

    assert (
        BehaviorObservationPipeline._initial_orchestration_status(link, passive_evidence)
        == HypothesisStatus.UNRESOLVED
    )
    assert (
        BehaviorObservationPipeline._initial_orchestration_status(link, run_evidence)
        == HypothesisStatus.CONFIRMED
    )


def test_instruction_event_route_abstains_without_code_evidence() -> None:
    candidate = CandidatePair(
        candidate_id="cand-1",
        clause_id="c1",
        behavior_kind=BehaviorKind.EVENT,
        event_id="e1",
    )
    event = CapabilityEvent(
        event_id="e1",
        unit_id="doc::doc0",
        capability="shell_exec",
        location="SKILL.md:4",
        tier="instruction",
    )

    result, support = execution_route_justified(
        candidate,
        events={"e1": event},
        paths={},
        bridges={},
        orchestrations={},
    )

    assert result == PredicateResult.ABSTAIN
    assert support == RouteSupport.WEAK


def test_witness_revalidation_replays_finding_motif() -> None:
    finding = Finding(
        finding_id="finding-1",
        finding_type=FindingType.UNSUPPORTED_BEHAVIOR,
        certificate_ids=["cert-1"],
        related_event_ids=["e1"],
    )
    witness = Witness(
        witness_id="witness-1",
        finding_id="finding-1",
        anchor_ids=["cert-1", "e1"],
        certificate_ids=["cert-1"],
        fact_node_ids=[],
        judgment_ids=[],
    )
    wrong_certificate = Certificate(
        certificate_id="cert-1",
        kind=CertificateKind.NO_JUSTIFIED_PATH,
        subject_refs=["e1"],
    )

    validation = validate_witness(
        finding,
        witness,
        judgments_by_id={},
        certificates_by_id={"cert-1": wrong_certificate},
    )

    assert not validation.passed
    assert validation.failure_reason == "replay_missing_no_support_certificate"


def test_witness_revalidation_rejects_conflicting_authorization() -> None:
    finding = Finding(
        finding_id="finding-1",
        finding_type=FindingType.UNSUPPORTED_BEHAVIOR,
        certificate_ids=["cert-1"],
        related_event_ids=["e1"],
    )
    certificate = Certificate(
        certificate_id="cert-1",
        kind=CertificateKind.NO_SUPPORT,
        subject_refs=["e1"],
    )
    judgment = ReconciliationJudgment(
        judgment_id="judgment-1",
        kind=JudgmentKind.AUTHORIZATION,
        result=PredicateResult.TRUE,
        subject_refs=["c1", "e1"],
    )
    witness = Witness(
        witness_id="witness-1",
        finding_id="finding-1",
        anchor_ids=["cert-1", "e1"],
        fact_node_ids=["c1"],
        judgment_ids=["judgment-1"],
        certificate_ids=["cert-1"],
    )

    validation = validate_witness(
        finding,
        witness,
        judgments_by_id={"judgment-1": judgment},
        certificates_by_id={"cert-1": certificate},
    )

    assert not validation.passed
    assert validation.failure_reason == "replay_conflicting_authorization"


def test_witness_reducibility_check_handles_bool_revalidation_result() -> None:
    finding = Finding(
        finding_id="finding-1",
        finding_type=FindingType.UNSUPPORTED_BEHAVIOR,
        certificate_ids=["cert-1"],
        related_event_ids=["e1"],
    )
    certificate = Certificate(
        certificate_id="cert-1",
        kind=CertificateKind.NO_SUPPORT,
        subject_refs=["e1", "c1"],
    )
    witness = Witness(
        witness_id="witness-1",
        finding_id="finding-1",
        anchor_ids=["cert-1", "e1"],
        fact_node_ids=["c1"],
        judgment_ids=[],
        certificate_ids=["cert-1"],
    )

    validation = validate_witness(
        finding,
        witness,
        judgments_by_id={},
        certificates_by_id={"cert-1": certificate},
    )

    assert validation.passed


def test_pydantic_imports_used_by_alignment_tests() -> None:
    # Keeps the small model imports above exercised so schema-level changes fail here,
    # rather than only in the full pipeline.
    assert Clause(clause_id="c1", capability="http_request", operator="allowed")
    assert CapabilityEvent(event_id="e1", unit_id="u1", capability="http_request", location="x")
    assert CandidatePair(candidate_id="cand-1", clause_id="c1", behavior_kind="event", event_id="e1")
