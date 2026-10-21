"""GraphRAG retrieval- and answer-quality evaluation.

The evaluator converts directly retrieved entities and entities associated with
retrieved chunks into one entity set.  Valid source/intermediate entities are
ignored when counting false positives, while gold answer entities are always
retained.  A shared LLM can then extract answer entities from the retrieved
context for answer-level evaluation.
"""

from __future__ import annotations

import json
import hashlib
import re
import string
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from pydantic import BaseModel, Field


class ExtractedAnswerEntities(BaseModel):
    """Structured output used by the shared answer-comprehension prompt."""

    entities: list[str] = Field(
        default_factory=list,
        description="The answer entities supported by the retrieved context.",
    )


class GraphRAGEvaluator:
    """Evaluate retrieval quality and answer quality in a common entity space.

    A benchmark file is optional. For quick experiments, instantiate the class
    with only ``graph_json`` (or with no files when testing direct entities) and
    call :meth:`evaluate_single_question`. The benchmark file is used only for
    batch lookup by ``question_id``.

    Required output format from every evaluated GraphRAG architecture
    -----------------------------------------------------------------

    Each architecture must save a JSON list with one object per benchmark
    question in this format::

        [
            {
                "question_id": "question_123",
                "retrieved_entities": [
                    "spike lee",
                    "girl 6"
                ],
                "retrieved_chunks": [
                    "The movie girl 6 was directed by **spike lee**."
                ],
                "retrieved_context": "The exact context shown to the answer LLM."
            }
        ]

    Field requirements:

    * ``question_id`` must match the benchmark record's ``id``.
    * ``retrieved_entities`` contains the entity names returned directly by the
      architecture. Use an empty list when it returns no entities directly.
      The evaluator deterministically converts each name to its canonical
      content-based benchmark entity ID.
    * ``retrieved_chunks`` contains the complete text of every returned chunk.
      Use an empty list when it returns no chunks directly. The evaluator
      deterministically converts each chunk's content to its canonical chunk ID,
      expands the chunks into their associated entities, and combines them with
      the directly retrieved entities.
    * ``retrieved_context`` is the complete final context produced by the
      architecture, including any rendered entities, relationships, paths, and
      chunks. The shared answer LLM reads this field exactly as supplied.

    A native architecture may alternatively provide ``retrieved_entity_ids``
    and ``retrieved_chunk_ids`` when it preserves the benchmark's canonical
    IDs. Do not provide both the name/content form and ID form for the same
    field.

    Architectures must not provide their own generated answer for answer-level
    scoring. The evaluator applies the same answer-extraction LLM and prompt to
    ``retrieved_context`` for every architecture. Extra diagnostic fields are
    allowed and ignored by the evaluator.

    Required benchmark question format::

        {
            "id": "question_123",
            "question": "what films does spike lee appear in",
            "answer": ["girl 6", "jungle fever"],
            "answer_entity_ids": ["entity_girl_6", "entity_jungle_fever"],
            "source_entity_id": "entity_spike_lee"  # optional
        }

    ``benchmark_json`` may be either a JSON list of these records or a JSON
    object containing them under ``"questions"``.

    Retrieval evaluation expects two collections for each question:

    * ``retrieved_entity_ids``: entity IDs returned directly by the retriever.
    * ``retrieved_chunk_ids``: chunk IDs returned by the retriever.

    The evaluator expands every retrieved chunk into its associated entity IDs
    and unions them with ``retrieved_entity_ids``. Therefore, callers should
    pass IDs rather than entity descriptions or rendered context strings.

    Answer evaluation expects answer entities as ``list[str]``::

        ["girl 6", "jungle fever"]

    It does not expect a prose answer, comma-delimited string, dictionary, or
    entity IDs. ``predicted_entities``, ``gold_entities``, and
    ``retrieved_entities`` passed to :meth:`answer_quality` must all use entity
    names in the same namespace. When using :meth:`answer_quality_for_question`,
    the method accepts retrieved entity/chunk IDs and performs the ID-to-name
    conversion automatically.

    When the shared LLM reads retrieved context, its required structured output
    is::

        {"entities": ["girl 6", "jungle fever"]}

    ``graph_json`` may be a custom-graph payload containing ``chunks``,
    ``entities``, and ``rels_json``. It can be omitted when those fields live
    beside ``questions`` in the benchmark file.
    """

    def __init__(
        self,
        benchmark_json: str | Path | Sequence[Mapping[str, Any]] | Mapping[str, Any] | None = None,
        graph_json: str | Path | Mapping[str, Any] | None = None,
        path_traces_json: str | Path | Sequence[Mapping[str, Any]] | None = None,
        language_model: Any | None = None,
    ) -> None:
        benchmark_data = self._load_json(benchmark_json) if benchmark_json is not None else []
        if isinstance(benchmark_data, Mapping):
            self.questions = list(benchmark_data.get("questions", []))
            default_graph_data: Mapping[str, Any] = benchmark_data
        else:
            self.questions = list(benchmark_data)
            default_graph_data = {}

        graph_data = self._load_json(graph_json) if graph_json is not None else default_graph_data
        if not isinstance(graph_data, Mapping):
            raise TypeError("graph_json must contain a JSON object")

        self.language_model = language_model
        self.questions_by_id = {
            str(record.get("id", record.get("question_id"))): dict(record)
            for record in self.questions
        }
        self.entity_id_to_name = self._build_entity_name_index(graph_data, self.questions)
        self.chunk_to_entity_ids = self._build_chunk_entity_index(graph_data)
        self.intermediate_entities_by_question: dict[str, set[str]] = {}
        self.source_entities_by_question: dict[str, set[str]] = {}
        if path_traces_json is not None:
            traces = self._load_json(path_traces_json)
            self._index_path_traces(traces)

    @staticmethod
    def _load_json(value: Any) -> Any:
        if isinstance(value, (str, Path)):
            with Path(value).open(encoding="utf-8") as file:
                return json.load(file)
        return value

    @staticmethod
    def normalize_entity(value: Any) -> str:
        """Normalize entity strings without deleting meaningful inner words."""
        text = str(value or "").lower().strip()
        text = text.replace("[", "").replace("]", "")
        text = "".join(character for character in text if character not in string.punctuation)
        return " ".join(text.split())

    @staticmethod
    def _stable_id(prefix: str, value: Any) -> str:
        normalized = " ".join(str(value).strip().lower().split())
        digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]
        return f"{prefix}_{digest}"

    async def transform_entity_name_to_id(self, entity_name: Any) -> str:
        """Convert an architecture-returned entity name to its benchmark ID.

        The transformation exactly matches benchmark construction: lowercase,
        trim/collapse whitespace, SHA-256, and retain the first 16 hexadecimal
        characters under the ``entity_`` prefix.
        """
        return self._stable_id("entity", entity_name)

    async def transform_chunk_content_to_id(self, chunk_content: Any) -> str:
        """Convert architecture-returned chunk text to its benchmark chunk ID.

        The complete chunk content must be supplied exactly; insignificant
        capitalization and whitespace differences are normalized before the
        content-based hash is generated.
        """
        return self._stable_id("chunk", chunk_content)

    async def transform_entity_names_to_ids(self, entity_names: Iterable[Any]) -> list[str]:
        """Convert a collection of architecture-returned names to entity IDs."""
        return [await self.transform_entity_name_to_id(name) for name in entity_names]

    async def transform_chunk_contents_to_ids(self, chunk_contents: Iterable[Any]) -> list[str]:
        """Convert a collection of architecture-returned chunk texts to chunk IDs."""
        return [await self.transform_chunk_content_to_id(content) for content in chunk_contents]

    @classmethod
    def _normalize_set(cls, values: Iterable[Any]) -> set[str]:
        return {normalized for value in values if (normalized := cls.normalize_entity(value))}

    @staticmethod
    def _as_list(value: Any) -> list[Any]:
        if value is None:
            return []
        if isinstance(value, (list, tuple, set)):
            return list(value)
        return [value]

    @classmethod
    def _build_entity_name_index(
        cls,
        graph_data: Mapping[str, Any],
        questions: Sequence[Mapping[str, Any]],
    ) -> dict[str, str]:
        index: dict[str, str] = {}
        for entity in graph_data.get("entities", []):
            entity_id = entity.get("entity_id", entity.get("id"))
            entity_name = entity.get("entity_name", entity.get("name"))
            if entity_id is not None and entity_name is not None:
                index[str(entity_id)] = str(entity_name)
        for question in questions:
            answer_ids = cls._as_list(question.get("answer_entity_ids"))
            answer_names = cls._as_list(question.get("answer", question.get("answers")))
            for entity_id, entity_name in zip(answer_ids, answer_names):
                index.setdefault(str(entity_id), str(entity_name))
            source_id = question.get("source_entity_id")
            source_name = question.get("source_entity")
            if source_id is not None and source_name is not None:
                index.setdefault(str(source_id), str(source_name))
        return index

    @classmethod
    def _build_chunk_entity_index(cls, graph_data: Mapping[str, Any]) -> dict[str, set[str]]:
        """Build chunk -> entity IDs from explicit mappings, entities, and edges."""
        index: dict[str, set[str]] = defaultdict(set)

        explicit = graph_data.get("chunk_to_entities", graph_data.get("chunk_entity_map", {}))
        if isinstance(explicit, Mapping):
            for chunk_id, entity_ids in explicit.items():
                index[str(chunk_id)].update(str(item) for item in cls._as_list(entity_ids))

        for chunk in graph_data.get("chunks", []):
            chunk_id = chunk.get("chunk_id", chunk.get("id"))
            if chunk_id is None:
                continue
            for key in ("entity_ids", "entities"):
                for entity in cls._as_list(chunk.get(key)):
                    if isinstance(entity, Mapping):
                        entity = entity.get("entity_id", entity.get("id", entity.get("name")))
                    if entity is not None:
                        index[str(chunk_id)].add(str(entity))

        for entity in graph_data.get("entities", []):
            entity_id = entity.get("entity_id", entity.get("id"))
            if entity_id is None:
                continue
            chunk_ids = cls._as_list(entity.get("chunk_ids", entity.get("chunk_id")))
            for chunk_id in chunk_ids:
                index[str(chunk_id)].add(str(entity_id))

        for relationship in graph_data.get("rels_json", graph_data.get("relationships", [])):
            chunk_ids = cls._as_list(relationship.get("chunk_ids", relationship.get("chunk_id")))
            endpoint_ids = [
                relationship.get("src_entity_id", relationship.get("source_entity_id")),
                relationship.get("dst_entity_id", relationship.get("target_entity_id")),
            ]
            for chunk_id in chunk_ids:
                for entity_id in endpoint_ids:
                    if entity_id is not None:
                        index[str(chunk_id)].add(str(entity_id))
        return dict(index)

    def _index_path_traces(self, traces: Any) -> None:
        if isinstance(traces, Mapping):
            traces = traces.get("path_traces", traces.get("traces", []))
        for trace in traces or []:
            question_id = str(trace.get("question_id", trace.get("id")))
            source_ids = {
                str(item)
                for item in self._as_list(trace.get("source_entity_ids", trace.get("source_entity_id")))
            }
            answer_ids: set[str] = set()
            path_nodes: set[str] = set()
            for answer_trace in trace.get("answer_traces", []):
                answer_id = answer_trace.get("answer_entity_id")
                if answer_id is not None:
                    answer_ids.add(str(answer_id))
                for path in answer_trace.get("possible_paths", []):
                    path_nodes.update(str(item) for item in path)
            self.source_entities_by_question[question_id] = source_ids
            self.intermediate_entities_by_question[question_id] = path_nodes - source_ids - answer_ids

    async def entities_for_chunks(self, chunk_ids: Iterable[Any]) -> set[str]:
        """Return the union of entity IDs associated with retrieved chunks."""
        entities: set[str] = set()
        for chunk_id in chunk_ids:
            if isinstance(chunk_id, Mapping):
                chunk_id = chunk_id.get("chunk_id", chunk_id.get("id"))
            if chunk_id is not None:
                entities.update(self.chunk_to_entity_ids.get(str(chunk_id), set()))
        return entities

    @staticmethod
    def _set_metrics(predicted: set[str], gold: set[str]) -> dict[str, Any]:
        true_positives = predicted & gold
        false_positives = predicted - gold
        false_negatives = gold - predicted
        precision = len(true_positives) / len(predicted) if predicted else 0.0
        recall = len(true_positives) / len(gold) if gold else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "true_positives": sorted(true_positives),
            "false_positives": sorted(false_positives),
            "false_negatives": sorted(false_negatives),
        }

    async def retrieval_quality(
        self,
        gold_entity_ids: Iterable[Any],
        retrieved_entity_ids: Iterable[Any] = (),
        retrieved_chunk_ids: Iterable[Any] = (),
        intermediate_entity_ids: Iterable[Any] = (),
        source_entity_ids: Iterable[Any] = (),
    ) -> dict[str, Any]:
        """Score nodes and chunk-derived entities as one retrieved entity set."""
        gold = {str(item) for item in gold_entity_ids}
        direct = {str(item) for item in retrieved_entity_ids}
        from_chunks = await self.entities_for_chunks(retrieved_chunk_ids)
        retrieved = direct | from_chunks
        ignored = {str(item) for item in intermediate_entity_ids} | {
            str(item) for item in source_entity_ids
        }
        # A gold answer is never removed even if it was also supplied as an
        # intermediate/source due to malformed benchmark metadata.
        evaluated = retrieved - (ignored - gold)
        metrics = self._set_metrics(evaluated, gold)
        metrics.update({
            "retrieved_entity_ids": sorted(retrieved),
            "evaluated_entity_ids": sorted(evaluated),
            "ignored_entity_ids": sorted(retrieved & ignored - gold),
            "direct_entity_ids": sorted(direct),
            "chunk_entity_ids": sorted(from_chunks),
        })
        return metrics

    async def retrieval_quality_for_question(
        self,
        question_id: Any,
        retrieved_entity_ids: Iterable[Any] = (),
        retrieved_chunk_ids: Iterable[Any] = (),
        intermediate_entity_ids: Iterable[Any] | None = None,
    ) -> dict[str, Any]:
        question_key = str(question_id)
        question = self.questions_by_id[question_key]
        gold = self._as_list(question.get("answer_entity_ids"))
        sources = self.source_entities_by_question.get(question_key, set())
        if not sources:
            sources = {str(item) for item in self._as_list(question.get("source_entity_id"))}
        intermediates = (
            self.intermediate_entities_by_question.get(question_key, set())
            if intermediate_entity_ids is None
            else intermediate_entity_ids
        )
        return await self.retrieval_quality(
            gold_entity_ids=gold,
            retrieved_entity_ids=retrieved_entity_ids,
            retrieved_chunk_ids=retrieved_chunk_ids,
            intermediate_entity_ids=intermediates,
            source_entity_ids=sources,
        )

    @classmethod
    async def token_overlap(cls, predicted: Iterable[Any], gold: Iterable[Any]) -> dict[str, float]:
        """Bag-of-token precision/recall/F1 across two answer-entity sets."""
        predicted_tokens = Counter(
            token
            for entity in predicted
            for token in cls.normalize_entity(entity).split()
        )
        gold_tokens = Counter(
            token
            for entity in gold
            for token in cls.normalize_entity(entity).split()
        )
        overlap = sum((predicted_tokens & gold_tokens).values())
        predicted_total = sum(predicted_tokens.values())
        gold_total = sum(gold_tokens.values())
        precision = overlap / predicted_total if predicted_total else 0.0
        recall = overlap / gold_total if gold_total else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        return {"precision": precision, "recall": recall, "f1": f1}

    async def extract_answer_entities(
        self,
        question: str,
        retrieved_context: Any,
    ) -> tuple[list[str], int, int, int]:
        """Use one shared prompt to extract answers supported by retrieved context."""
        if self.language_model is None:
            raise ValueError("A language model is required for answer-level evaluation")
        context = (
            retrieved_context
            if isinstance(retrieved_context, str)
            else json.dumps(retrieved_context, ensure_ascii=False)
        )
        prompt = f"""---Role---
You extract answer entities from retrieved context.

---Goal---
Answer the query using only the retrieved context and return the complete list of answer entities.

---Rules---
1) Do not use outside or pretrained knowledge.
2) Include every entity supported by the context that answers the query.
3) Do not include anchors, intermediate entities, explanations, or unsupported guesses.
4) Return an empty list when the context does not support an answer.

Query:
{question}

Retrieved context:
{context}
"""
        response, total_tokens, input_tokens, output_tokens = (
            await self.language_model.generate(
                prompt,
                structuredInput=ExtractedAnswerEntities,
            )
        )
        if isinstance(response, BaseModel):
            response = response.model_dump()
        parsed = ExtractedAnswerEntities.model_validate(response)
        return parsed.entities, total_tokens, input_tokens, output_tokens

    async def answer_quality(
        self,
        predicted_entities: Iterable[Any],
        gold_entities: Iterable[Any],
        retrieved_entities: Iterable[Any],
    ) -> dict[str, Any]:
        """Score answer-entity lists and report whether they are context-supported.

        Args:
            predicted_entities: Entity names produced by the answer model, for
                example ``["girl 6", "jungle fever"]``.
            gold_entities: Gold answer entity names in the same list format.
            retrieved_entities: Names of every entity present in the retrieved
                context. This is used only for grounding analysis; use
                :meth:`answer_quality_for_question` to derive these names from
                retrieved entity IDs and chunk IDs automatically.

        All three arguments must be iterables of complete entity-name strings.
        Passing one prose string would incorrectly treat it as an iterable of
        characters, so prose answers must first be converted to an entity list.
        """
        predicted = self._normalize_set(predicted_entities)
        gold = self._normalize_set(gold_entities)
        retrieved = self._normalize_set(retrieved_entities)
        metrics = self._set_metrics(predicted, gold)
        metrics["token_overlap"] = await self.token_overlap(predicted, gold)
        metrics.update({
            "predicted_entities": sorted(predicted),
            "gold_entities": sorted(gold),
            "retrieved_entities": sorted(retrieved),
            "correct_supported": sorted(predicted & gold & retrieved),
            "correct_unsupported": sorted((predicted & gold) - retrieved),
            "incorrect_grounded": sorted((predicted - gold) & retrieved),
            "incorrect_unsupported": sorted((predicted - gold) - retrieved),
        })
        correct = predicted & gold
        metrics["supported_correct_rate"] = (
            len(correct & retrieved) / len(correct) if correct else 0.0
        )
        return metrics

    async def answer_quality_from_context(
        self,
        question: str,
        retrieved_context: Any,
        gold_entities: Iterable[Any],
        retrieved_entities: Iterable[Any],
    ) -> dict[str, Any]:
        """Extract and score answers from context.

        ``retrieved_entities`` must contain entity names, not IDs. The LLM is
        required to return ``{"entities": ["entity name", ...]}`` through the
        :class:`ExtractedAnswerEntities` structured-output schema.
        """
        predicted, total_tokens, input_tokens, output_tokens = (
            await self.extract_answer_entities(question, retrieved_context)
        )
        metrics = await self.answer_quality(
            predicted,
            gold_entities,
            retrieved_entities,
        )
        metrics["tokens"] = total_tokens
        metrics["input_tokens"] = input_tokens
        metrics["output_tokens"] = output_tokens
        return metrics

    async def answer_quality_for_question(
        self,
        question_id: Any,
        retrieved_context: Any,
        retrieved_entity_ids: Iterable[Any] = (),
        retrieved_chunk_ids: Iterable[Any] = (),
    ) -> dict[str, Any]:
        """Run answer extraction and scoring using retrieval IDs.

        ``retrieved_entity_ids`` and ``retrieved_chunk_ids`` are ID lists. The
        evaluator resolves both to entity names before measuring grounding and
        compares the extracted ``list[str]`` against the question record's
        ``answer`` list.
        """
        question = self.questions_by_id[str(question_id)]
        retrieved_ids = {str(item) for item in retrieved_entity_ids}
        retrieved_ids.update(await self.entities_for_chunks(retrieved_chunk_ids))
        retrieved_names = [self.entity_id_to_name.get(item, item) for item in retrieved_ids]
        gold_names = self._as_list(question.get("answer", question.get("answers")))
        return await self.answer_quality_from_context(
            question=str(question.get("question", "")),
            retrieved_context=retrieved_context,
            gold_entities=gold_names,
            retrieved_entities=retrieved_names,
        )

    async def evaluate_single_question(
        self,
        question: str,
        gold_entity_ids: Iterable[Any],
        gold_entities: Iterable[Any],
        retrieved_context: Any,
        retrieved_entities: Iterable[Any] = (),
        retrieved_chunks: Iterable[Any] = (),
        retrieved_entity_ids: Iterable[Any] = (),
        retrieved_chunk_ids: Iterable[Any] = (),
        intermediate_entity_ids: Iterable[Any] = (),
        source_entity_ids: Iterable[Any] = (),
        predicted_answer_entities: Iterable[Any] | None = None,
    ) -> dict[str, Any]:
        """Evaluate one question without constructing or loading a benchmark.

        Natural architecture output can be passed through ``retrieved_entities``
        (entity names) and ``retrieved_chunks`` (complete chunk contents). These
        values are converted into canonical content-based IDs. Native methods
        may instead supply ``retrieved_entity_ids`` and ``retrieved_chunk_ids``.

        ``gold_entity_ids`` are used for retrieval scoring, while
        ``gold_entities`` are the corresponding human-readable answer names used
        for answer scoring. Valid source and intermediate IDs are ignored when
        counting retrieval false positives.

        When ``predicted_answer_entities`` is omitted, the configured shared LLM
        extracts them from ``retrieved_context``. Supplying a list evaluates an
        already-produced answer without making an LLM call.
        """
        natural_entity_names = list(retrieved_entities)
        natural_chunk_contents = list(retrieved_chunks)
        canonical_entity_ids = {str(item) for item in retrieved_entity_ids}
        canonical_chunk_ids = {str(item) for item in retrieved_chunk_ids}
        canonical_entity_ids.update(
            await self.transform_entity_names_to_ids(natural_entity_names)
        )
        canonical_chunk_ids.update(
            await self.transform_chunk_contents_to_ids(natural_chunk_contents)
        )

        retrieval = await self.retrieval_quality(
            gold_entity_ids=gold_entity_ids,
            retrieved_entity_ids=canonical_entity_ids,
            retrieved_chunk_ids=canonical_chunk_ids,
            intermediate_entity_ids=intermediate_entity_ids,
            source_entity_ids=source_entity_ids,
        )

        context_entity_ids = set(retrieval["retrieved_entity_ids"])
        context_entity_names = {
            self.entity_id_to_name.get(entity_id, entity_id)
            for entity_id in context_entity_ids
        }
        context_entity_names.update(str(name) for name in natural_entity_names)

        if predicted_answer_entities is None:
            answer = await self.answer_quality_from_context(
                question=question,
                retrieved_context=retrieved_context,
                gold_entities=gold_entities,
                retrieved_entities=context_entity_names,
            )
        else:
            answer = await self.answer_quality(
                predicted_entities=predicted_answer_entities,
                gold_entities=gold_entities,
                retrieved_entities=context_entity_names,
            )

        return {"retrieval": retrieval, "answer": answer}

    async def evaluate_architecture_results(
        self,
        results_json: str | Path | Sequence[Mapping[str, Any]],
    ) -> dict[str, Any]:
        """Evaluate one architecture's required per-question results JSON.

        ``results_json`` must follow the architecture-output contract documented
        in the class docstring. Entity names and chunk contents are converted to
        canonical IDs here before scoring.
        """
        records = self._load_json(results_json)
        if not isinstance(records, Sequence) or isinstance(records, (str, bytes, Mapping)):
            raise TypeError("Architecture results must be a JSON list of question records")

        required_fields = {"question_id", "retrieved_context"}
        per_question: list[dict[str, Any]] = []
        for position, record in enumerate(records):
            missing = required_fields - set(record)
            if missing:
                raise ValueError(
                    f"Architecture result at index {position} is missing: {sorted(missing)}"
                )
            question_id = str(record["question_id"])
            if question_id not in self.questions_by_id:
                raise KeyError(f"Unknown benchmark question_id: {question_id}")

            has_entity_names = "retrieved_entities" in record
            has_entity_ids = "retrieved_entity_ids" in record
            has_chunk_contents = "retrieved_chunks" in record
            has_chunk_ids = "retrieved_chunk_ids" in record
            if has_entity_names == has_entity_ids:
                raise ValueError(
                    f"Result {question_id} must provide exactly one of "
                    "retrieved_entities or retrieved_entity_ids"
                )
            if has_chunk_contents == has_chunk_ids:
                raise ValueError(
                    f"Result {question_id} must provide exactly one of "
                    "retrieved_chunks or retrieved_chunk_ids"
                )

            retrieved_entity_ids = (
                await self.transform_entity_names_to_ids(record["retrieved_entities"])
                if has_entity_names else list(record["retrieved_entity_ids"])
            )
            retrieved_chunk_ids = (
                await self.transform_chunk_contents_to_ids(record["retrieved_chunks"])
                if has_chunk_contents else list(record["retrieved_chunk_ids"])
            )

            retrieval = await self.retrieval_quality_for_question(
                question_id=question_id,
                retrieved_entity_ids=retrieved_entity_ids,
                retrieved_chunk_ids=retrieved_chunk_ids,
            )
            answer = await self.answer_quality_for_question(
                question_id=question_id,
                retrieved_context=record["retrieved_context"],
                retrieved_entity_ids=retrieved_entity_ids,
                retrieved_chunk_ids=retrieved_chunk_ids,
            )
            per_question.append({
                "question_id": question_id,
                "retrieval": retrieval,
                "answer": answer,
            })

        count = len(per_question)

        def mean(section: str, metric: str) -> float:
            return (
                sum(item[section][metric] for item in per_question) / count
                if count else 0.0
            )

        return {
            "count": count,
            "retrieval": {
                metric: mean("retrieval", metric)
                for metric in ("precision", "recall", "f1")
            },
            "answer": {
                metric: mean("answer", metric)
                for metric in ("precision", "recall", "f1")
            },
            "answer_token_overlap": {
                metric: (
                    sum(item["answer"]["token_overlap"][metric] for item in per_question) / count
                    if count else 0.0
                )
                for metric in ("precision", "recall", "f1")
            },
            "supported_correct_rate": mean("answer", "supported_correct_rate"),
            "tokens": sum(item["answer"].get("tokens", 0) for item in per_question),
            "input_tokens": sum(
                item["answer"].get("input_tokens", 0) for item in per_question
            ),
            "output_tokens": sum(
                item["answer"].get("output_tokens", 0) for item in per_question
            ),
            "per_question": per_question,
        }
