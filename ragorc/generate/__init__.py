"""Generation layer: answering, citation, groundedness, abstention, consistency."""

from ragorc.generate.abstain import AbstentionDecision, AbstentionPolicy
from ragorc.generate.answer import AnswerGenerator
from ragorc.generate.citations import attribute_spans, extract_citations, renumber_citations
from ragorc.generate.consistency import ConsistencyResult, SelfConsistencyChecker
from ragorc.generate.groundedness import ClaimCheck, GroundednessChecker, GroundednessResult
from ragorc.generate.rrr import RRR, RRRResult
from ragorc.generate.self_rag import SelfRAG, SelfRAGAttempt, SelfRAGResult

__all__ = [
    "RRR",
    "AbstentionDecision",
    "AbstentionPolicy",
    "AnswerGenerator",
    "ClaimCheck",
    "ConsistencyResult",
    "GroundednessChecker",
    "GroundednessResult",
    "RRRResult",
    "SelfConsistencyChecker",
    "SelfRAG",
    "SelfRAGAttempt",
    "SelfRAGResult",
    "attribute_spans",
    "extract_citations",
    "renumber_citations",
]
