"""Map generations to the benchmark service evaluation and score contracts."""

from collections.abc import Mapping, Sequence
from typing import Any

from benchmark_runtime.backend import format_exception
from benchmark_runtime.protocols import EvaluationClientLike, ScoringClientLike
from benchmark_runtime.schemas import (
    EvalResult,
    EvalResultData,
    EvalStatus,
    GenerationResult,
    GenerationStatus,
    ScoreResult,
)


async def evaluate_generation(
    *,
    client: EvaluationClientLike,
    generation: GenerationResult | None,
    task_id: str,
    dataset: str | None,
) -> EvalResult:
    """Evaluate one normalized generation through the benchmark service."""
    if generation is None:
        return EvalResult(
            task_id=task_id,
            status=EvalStatus.GENERATION_ERROR,
            error="generation result missing",
        )
    if generation.status in (GenerationStatus.MAX_TIME, GenerationStatus.MAX_TURNS):
        return EvalResult(task_id=task_id, status=EvalStatus.DID_NOT_COMPLETE)
    if generation.status != GenerationStatus.SUCCESS:
        return EvalResult(
            task_id=task_id,
            status=EvalStatus.GENERATION_ERROR,
            error=generation.error,
        )

    try:
        raw = await client.evaluate_response(
            task_id=task_id,
            response=generation.data,
            dataset=dataset,
        )
        if raw is None:
            # `evaluate_response` returns `resp.json()` unvalidated, so an HTTP 200 whose
            # body is `null` arrives here as None. Recording that as EVALUATED makes a task
            # the service never graded indistinguishable from one that scored zero, and
            # `score_evaluations` then reports the run complete over it.
            return EvalResult(
                task_id=task_id,
                status=EvalStatus.ERROR,
                error="evaluation service returned no result",
            )
        data = EvalResultData.model_validate(raw)
        return EvalResult(task_id=task_id, status=EvalStatus.EVALUATED, result=data)
    except Exception as exc:
        return EvalResult(task_id=task_id, status=EvalStatus.ERROR, error=format_exception(exc))


async def score_evaluations(
    *,
    client: ScoringClientLike,
    evaluations: Mapping[str, EvalResult | None],
    task_ids: Sequence[str],
    dataset: str | None,
) -> ScoreResult:
    """Submit the complete task set to the benchmark service final scorer."""
    submitted: dict[str, Any] = {}
    missing = 0
    generation_errors = 0
    evaluation_errors = 0
    for task_id in task_ids:
        evaluation = evaluations.get(task_id)
        submitted[task_id] = (
            None
            if evaluation is None
            else evaluation.model_dump(mode="json") | {"error": None}
        )
        if evaluation is None:
            missing += 1
        elif evaluation.status == EvalStatus.GENERATION_ERROR:
            generation_errors += 1
        elif evaluation.status == EvalStatus.ERROR:
            evaluation_errors += 1

    complete = missing == 0 and generation_errors == 0 and evaluation_errors == 0
    response = await client.final_score(submitted, dataset=dataset)
    payload = response.model_dump()
    return ScoreResult(
        tasks_evaluated=payload.get("tasks_evaluated", []),
        final_score=float(payload.get("final_score", 0.0)),
        metadata=payload.get("metadata", {}),
        complete=complete,
    )


__all__ = ["evaluate_generation", "score_evaluations"]
