from typing import Any

from benchmark_runtime.grading import evaluate_generation, score_evaluations
from benchmark_runtime.schemas import (
    EvalResult,
    EvalStatus,
    GenerationResult,
    GenerationStatus,
)


class FakeEvaluationClient:
    """Returns a fixed payload and records what it was asked to evaluate."""

    def __init__(self, payload: Any = None) -> None:
        self.payload = payload
        self.calls: list[tuple[str, str, str | None]] = []

    async def evaluate_response(
        self, task_id: str, response: str, dataset: str | None = None
    ) -> Any:
        self.calls.append((task_id, response, dataset))
        return self.payload


class _RaisingEvaluationClient(FakeEvaluationClient):
    async def evaluate_response(
        self, task_id: str, response: str, dataset: str | None = None
    ) -> Any:
        raise RuntimeError("service unavailable")


class FakeFinalScoreResponse:
    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = payload

    def model_dump(self) -> dict[str, Any]:
        return self.payload


class FakeScoringClient:
    """Records the submitted mapping so the payload sent to the scorer can be asserted."""

    def __init__(self, payload: dict[str, Any] | None = None) -> None:
        self.payload = payload if payload is not None else {
            "tasks_evaluated": ["t1"],
            "final_score": 87.5,
            "metadata": {"mean_weighted_pass_percentage": 87.5},
        }
        self.submitted: dict[str, Any] | None = None
        self.dataset: str | None = None

    async def final_score(
        self, evaluation_results: dict[str, Any], dataset: str | None = None
    ) -> FakeFinalScoreResponse:
        self.submitted = evaluation_results
        self.dataset = dataset
        return FakeFinalScoreResponse(self.payload)


def _generation(status: GenerationStatus = GenerationStatus.SUCCESS, **kw: Any) -> GenerationResult:
    base: dict[str, Any] = dict(task_id="t1", status=status, data="the answer")
    base.update(kw)
    return GenerationResult(**base)


# --------------------------------------------------------------------- evaluate_generation


async def test_missing_generation_is_a_generation_error() -> None:
    result = await evaluate_generation(
        client=FakeEvaluationClient(), generation=None, task_id="t1", dataset=None
    )
    assert result.status == EvalStatus.GENERATION_ERROR
    assert result.error == "generation result missing"
    assert result.result is None


async def test_max_time_did_not_complete() -> None:
    result = await evaluate_generation(
        client=FakeEvaluationClient(),
        generation=_generation(GenerationStatus.MAX_TIME),
        task_id="t1",
        dataset=None,
    )
    assert result.status == EvalStatus.DID_NOT_COMPLETE


async def test_max_turns_did_not_complete() -> None:
    result = await evaluate_generation(
        client=FakeEvaluationClient(),
        generation=_generation(GenerationStatus.MAX_TURNS),
        task_id="t1",
        dataset=None,
    )
    assert result.status == EvalStatus.DID_NOT_COMPLETE


async def test_failed_generation_is_not_evaluated_and_keeps_its_error() -> None:
    client = FakeEvaluationClient(payload={"pass_percentage": 100.0})
    result = await evaluate_generation(
        client=client,
        generation=_generation(GenerationStatus.ERROR, error="agent crashed"),
        task_id="t1",
        dataset=None,
    )
    assert result.status == EvalStatus.GENERATION_ERROR
    assert result.error == "agent crashed"
    assert client.calls == [], "a failed generation must not be sent for evaluation"


async def test_successful_generation_is_evaluated_and_the_payload_is_parsed() -> None:
    client = FakeEvaluationClient(payload={"pass_percentage": 80.0, "eval_version": "v3"})
    result = await evaluate_generation(
        client=client, generation=_generation(), task_id="t1", dataset="ds"
    )
    assert result.status == EvalStatus.EVALUATED
    assert result.result is not None
    assert result.result.pass_percentage == 80.0
    assert result.result.eval_version == "v3"
    assert client.calls == [("t1", "the answer", "ds")]


async def test_sdk_weighted_key_is_backfilled() -> None:
    client = FakeEvaluationClient(payload={"pass_percentage_with_weight": 42.0})
    result = await evaluate_generation(
        client=client, generation=_generation(), task_id="t1", dataset=None
    )
    assert result.result is not None
    assert result.result.weighted_pass_percentage == 42.0


async def test_an_evaluation_that_returned_nothing_is_not_an_evaluation() -> None:
    """A task the service did not grade must not be recorded as graded.

    `BenchmarkServiceClient.evaluate_response` ends in `return resp.json()` with no
    validation, so an HTTP 200 whose body is `null` reaches this function as None. Recording
    that as EVALUATED makes a task nobody graded indistinguishable from one that scored zero,
    and `score_evaluations` then reports the run complete over it.
    """
    result = await evaluate_generation(
        client=FakeEvaluationClient(payload=None),
        generation=_generation(),
        task_id="t1",
        dataset=None,
    )
    assert result.status != EvalStatus.EVALUATED
    assert result.result is None
    # Named explicitly rather than just "some error": without the guard, validating None
    # raises and the broad `except` below reports ERROR too, so an assertion that only
    # checks for an error cannot tell the fix from its absence.
    assert result.error == "evaluation service returned no result"


async def test_an_exception_from_the_service_is_an_error_not_an_evaluation() -> None:
    result = await evaluate_generation(
        client=_RaisingEvaluationClient(), generation=_generation(), task_id="t1", dataset=None
    )
    assert result.status == EvalStatus.ERROR
    assert result.error is not None
    assert "service unavailable" in result.error


# --------------------------------------------------------------------- score_evaluations


def _evaluated(task_id: str = "t1") -> EvalResult:
    return EvalResult(task_id=task_id, status=EvalStatus.EVALUATED)


async def test_a_fully_evaluated_run_is_complete() -> None:
    client = FakeScoringClient()
    score = await score_evaluations(
        client=client, evaluations={"t1": _evaluated()}, task_ids=["t1"], dataset="ds"
    )
    assert score.complete is True
    assert score.final_score == 87.5
    assert score.tasks_evaluated == ["t1"]
    assert score.mean_weighted_pass_percentage == 87.5
    assert client.dataset == "ds"


async def test_a_missing_evaluation_makes_the_run_incomplete() -> None:
    client = FakeScoringClient()
    score = await score_evaluations(
        client=client, evaluations={}, task_ids=["t1"], dataset=None
    )
    assert score.complete is False
    assert client.submitted == {"t1": None}


async def test_a_generation_error_makes_the_run_incomplete() -> None:
    score = await score_evaluations(
        client=FakeScoringClient(),
        evaluations={"t1": EvalResult(task_id="t1", status=EvalStatus.GENERATION_ERROR)},
        task_ids=["t1"],
        dataset=None,
    )
    assert score.complete is False


async def test_an_evaluation_error_makes_the_run_incomplete() -> None:
    score = await score_evaluations(
        client=FakeScoringClient(),
        evaluations={"t1": EvalResult(task_id="t1", status=EvalStatus.ERROR, error="boom")},
        task_ids=["t1"],
        dataset=None,
    )
    assert score.complete is False


async def test_a_run_whose_task_was_never_graded_is_not_complete() -> None:
    """End to end for the null-payload path: the status evaluate_generation assigns must be
    one score_evaluations counts, or an ungraded task rides through as a complete run."""
    evaluation = await evaluate_generation(
        client=FakeEvaluationClient(payload=None),
        generation=_generation(),
        task_id="t1",
        dataset=None,
    )
    score = await score_evaluations(
        client=FakeScoringClient(), evaluations={"t1": evaluation}, task_ids=["t1"], dataset=None
    )
    assert score.complete is False


async def test_every_requested_task_is_submitted_even_when_absent() -> None:
    client = FakeScoringClient()
    await score_evaluations(
        client=client,
        evaluations={"t1": _evaluated()},
        task_ids=["t1", "t2"],
        dataset=None,
    )
    assert client.submitted is not None
    assert set(client.submitted) == {"t1", "t2"}
    assert client.submitted["t2"] is None


async def test_submitted_evaluations_carry_their_status_without_the_error_text() -> None:
    client = FakeScoringClient()
    await score_evaluations(
        client=client,
        evaluations={"t1": EvalResult(task_id="t1", status=EvalStatus.ERROR, error="stack trace")},
        task_ids=["t1"],
        dataset=None,
    )
    assert client.submitted is not None
    assert client.submitted["t1"]["status"] == EvalStatus.ERROR.value
    assert client.submitted["t1"]["error"] is None


async def test_the_score_is_read_from_the_service_payload() -> None:
    client = FakeScoringClient(
        payload={"tasks_evaluated": ["a", "b"], "final_score": 12.5, "metadata": {"k": "v"}}
    )
    score = await score_evaluations(
        client=client, evaluations={"a": _evaluated("a")}, task_ids=["a"], dataset=None
    )
    assert score.final_score == 12.5
    assert score.tasks_evaluated == ["a", "b"]
    assert score.metadata == {"k": "v"}


async def test_an_absent_final_score_is_reported_as_zero() -> None:
    """Characterisation, not endorsement.

    `BenchmarkServiceClient.final_score` validates into `FinalScoreResponse`, which requires
    `final_score`, so this default is unreachable through the shipped client. It is reachable
    for anything else satisfying `ScoringClientLike`, which `protocols.py` exists to allow,
    and there it turns a score the service never produced into a measured zero. Pinned here so
    the choice is visible and any change to it is deliberate.
    """
    client = FakeScoringClient(payload={"metadata": {}})
    score = await score_evaluations(
        client=client, evaluations={"t1": _evaluated()}, task_ids=["t1"], dataset=None
    )
    assert score.final_score == 0.0
    assert score.tasks_evaluated == []
