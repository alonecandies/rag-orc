"""HyDE: embed a hypothetical answer instead of the question.

Questions and answers live in different regions of embedding space. "How do I
rotate the signing key?" is interrogative and short; the passage that answers it
is declarative, technical and full of domain nouns. Cosine similarity between
those two is doing more work than it should.

HyDE closes the gap by having the model *write a fake answer* and embedding that.
The hypothetical document is never shown to anyone — its factual accuracy is
irrelevant. What matters is that it inhabits answer-space and uses the domain's
vocabulary, so its neighbours are real answers.

The failure mode is real and worth guarding: when the hypothesis is wrong in a
*topical* way — the model confidently invents the wrong subject — the search
vector drifts and retrieval gets worse than the plain question would have been.
Two mitigations, both implemented here:

* **Blending.** The search vector can be a weighted mean of the hypothesis and the
  question vector. ``blend=0`` is pure HyDE, ``1.0`` is the plain question;
  anything between bounds the drift. This is a strictly better default than
  choosing one or the other.
* **Multiple hypotheses.** Generating N and mean-pooling averages away the
  idiosyncrasy of any single hallucination, for N times the (cheap-tier) cost.
"""

from __future__ import annotations

import numpy as np
import structlog

from ragorc.core.concurrency import bounded_gather
from ragorc.core.models import FloatArray, Query, Usage
from ragorc.core.protocols import LLM, BatchStructuredLLM, DenseEmbedder
from ragorc.core.registry import register
from ragorc.core.schemas import HyDEOutput
from ragorc.core.settings import Settings
from ragorc.llm.prompts import get_prompt
from ragorc.llm.router import ModelRouter, Task
from ragorc.translate.base import BaseTranslator

log = structlog.get_logger(__name__)

__all__ = ["HyDETranslator"]


def _l2(vector: np.ndarray) -> FloatArray:
    array = np.asarray(vector, dtype=np.float32)
    norm = float(np.linalg.norm(array))
    return (array / norm).astype(np.float32) if norm > 1e-9 else array


@register("translator", "hyde")
class HyDETranslator(BaseTranslator):
    name = "hyde"

    def __init__(
        self,
        llm: LLM,
        settings: Settings | None = None,
        *,
        router: ModelRouter | None = None,
        n_documents: int = 1,
        blend: float = 0.3,
    ) -> None:
        # Spelled out rather than forwarded as `*args: object`: the loose form
        # forwarded anything and typed nothing, so a caller passing the wrong
        # object reached `self.llm` before failing.
        super().__init__(llm, settings, router=router)
        self.n_documents = max(n_documents, 1)
        # 0.3 keeps most of HyDE's benefit while retaining enough of the question
        # to survive a topically wrong hypothesis.
        self.blend = min(max(blend, 0.0), 1.0)

    async def translate(self, query: Query) -> tuple[Query, Usage]:
        prompt = get_prompt("hyde")
        model = self.router.model_for(Task.HYDE)

        if self.n_documents == 1:
            result, usage = await self.llm.structured(
                prompt.render(question=query.text),
                HyDEOutput,
                system=prompt.system,
                model=model,
                stage="hyde",
            )
            documents = [result.document.strip()] if result.document.strip() else []
        else:
            prompts = [prompt.render(question=query.text)] * self.n_documents
            # Temperature must be non-zero or N samples are one sample.
            if isinstance(self.llm, BatchStructuredLLM):
                batch = await self.llm.batch_structured(
                    prompts,
                    HyDEOutput,
                    system=prompt.system,
                    model=model,
                    temperature=0.8,
                    stage="hyde",
                )
            else:
                # The protocol does not require the fan-out, and this used to call
                # it regardless — an AttributeError for anyone who supplied their
                # own client. Same concurrency ceiling, same per-item isolation.
                settled = await bounded_gather(
                    (
                        self.llm.structured(
                            p,
                            HyDEOutput,
                            system=prompt.system,
                            model=model,
                            temperature=0.8,
                            stage="hyde",
                        )
                        for p in prompts
                    ),
                    limit=self.settings.llm.max_concurrency,
                    return_exceptions=True,
                )
                batch = [row if isinstance(row, tuple) else (None, Usage()) for row in settled]
            documents = [
                item.document.strip() for item, _ in batch if item and item.document.strip()
            ]
            usage = Usage.sum([u for _, u in batch])

        if not documents:
            log.warning("hyde_produced_nothing", question=query.text[:70])
            return query, usage

        out = Query(
            text=query.text,
            original=query.original,
            variants=query.variants,
            hypothetical=documents[0],
            filters=dict(query.filters),
            top_k=query.top_k,
            dense=query.dense,
            sparse=query.sparse,
            multi=query.multi,
            tenant_id=query.tenant_id,
            metadata={
                **query.metadata,
                "hyde": True,
                "hyde_documents": documents,
                "hyde_blend": self.blend,
            },
        )
        log.debug("hyde", documents=len(documents), chars=len(documents[0]))
        return out, usage

    async def embed_for_search(self, query: Query, embedder: DenseEmbedder) -> FloatArray:
        """Compute the blended search vector.

        Called by the retriever instead of the plain query embedding. Documents are
        embedded with the *document*-side method and the question with the
        *query*-side one, because asymmetric models expect different instruction
        prefixes for each — using the wrong side here silently costs recall, which
        is precisely the mistake HyDE was meant to avoid.
        """
        documents: list[str] = list(query.metadata.get("hyde_documents") or [])
        if query.hypothetical and not documents:
            documents = [query.hypothetical]
        if not documents:
            return _l2(await embedder.embed_query(query.text))

        doc_vectors = await embedder.embed_documents(documents)
        hypothesis = _l2(np.mean(np.asarray(doc_vectors, dtype=np.float32), axis=0))

        blend = float(query.metadata.get("hyde_blend", self.blend))
        if blend <= 0.0:
            return hypothesis

        question = _l2(await embedder.embed_query(query.text))
        if question.shape != hypothesis.shape:
            # Mismatched dimensions mean the two sides used different models;
            # blending them would produce a meaningless vector.
            log.warning(
                "hyde_blend_skipped",
                reason="dimension_mismatch",
                question_dim=int(question.shape[-1]),
                hypothesis_dim=int(hypothesis.shape[-1]),
            )
            return hypothesis
        return _l2(blend * question + (1.0 - blend) * hypothesis)
