import json
from types import SimpleNamespace

from popgen.population.pipeline import AgentGenerationPipeline, RLAgentGenerationPipeline
from popgen.utils.exceptions import TrainingFailure
from popgen.population.types import Behavior, CodeSnippet


class _DummyGeneratedPolicy:
    def __init__(self, code: str):
        self.code = code


class _DummyEvalResult:
    def __init__(self):
        self.overall_metrics = {"adherence": 1.0}
        self.instance_metrics = [{"adherence": 1.0, "reasoning": "ok"}]
        self.metadata = {}


class _DummyPipeline(AgentGenerationPipeline):
    policy_type = "PolicyCode"

    def __init__(self, output_dir: str, fail_generate_n: int = 0, fail_rollout_n: int = 0):
        super().__init__(
            generator=None,
            evaluator=SimpleNamespace(),
            env_module=SimpleNamespace(),
            output_dir=output_dir,
        )
        self.fail_generate_n = fail_generate_n
        self.fail_rollout_n = fail_rollout_n
        self.generate_calls = 0
        self.rollout_calls = 0
        self.feedback_history: list[list[tuple[str, str]]] = []

    def _generate_policy(
        self,
        behavior: Behavior,
        feedbacks: list[tuple[str, str]],
        output_dir: str,
        num_workers: int = 1,
        **kwargs,
    ):
        del behavior, num_workers, kwargs
        self.generate_calls += 1
        self.feedback_history.append(list(feedbacks))
        if self.generate_calls <= self.fail_generate_n:
            raise RuntimeError(f"generation failed #{self.generate_calls}")
        with open(f"{output_dir}/policy.py", "w") as f:
            f.write(f"code-{self.generate_calls}")
        return _DummyGeneratedPolicy(code=f"code-{self.generate_calls}"), {"file_path": f"{output_dir}/policy.py"}, 0.01

    def _run_rollouts(
        self,
        behavior_dir: str,
        loading_kwargs: dict,
        n_rollouts: int,
        num_workers: int,
        seed: int = 0,
        **rollout_kwargs,
    ):
        del behavior_dir, loading_kwargs, n_rollouts, num_workers, seed, rollout_kwargs
        self.rollout_calls += 1
        if self.rollout_calls <= self.fail_rollout_n:
            return [{"error": "rollout crashed", "states": None, "images": None}], [0.1]
        return [{"error": None, "states": [{"t": 0}], "images": []}], [0.1]

    def _evaluate_policy(
        self,
        behavior: Behavior,
        rollouts: list[dict],
        behavior_dir: str,
        num_workers: int = 8,
        **kwargs,
    ):
        del behavior, rollouts, behavior_dir, num_workers, kwargs
        return _DummyEvalResult(), 0.01


class _EvalFailingPipeline(_DummyPipeline):
    def _evaluate_policy(
        self,
        behavior: Behavior,
        rollouts: list[dict],
        behavior_dir: str,
        num_workers: int = 8,
        **kwargs,
    ):
        del behavior, rollouts, behavior_dir, num_workers, kwargs
        raise RuntimeError("forced evaluation failure")


class _ExceptionReturningPipeline(_DummyPipeline):
    def _generate_policy(
        self,
        behavior: Behavior,
        feedbacks: list[tuple[str, str]],
        output_dir: str,
        num_workers: int = 1,
        **kwargs,
    ):
        del behavior, feedbacks, output_dir, num_workers, kwargs
        self.generate_calls += 1
        return RuntimeError("returned generation failure"), {"file_path": "unused.py"}, 0.01


def test_generation_exception_is_retried_with_feedback(tmp_path):
    pipeline = _DummyPipeline(output_dir=str(tmp_path), fail_generate_n=1, fail_rollout_n=0)
    behavior = Behavior(name="b0", description="test behavior")

    success, summary = pipeline(
        behavior=behavior,
        num_attempts=1,
        n_rollouts=1,
        n_generation_retries=1,
        num_workers=1,
    )

    assert success is True
    assert pipeline.generate_calls == 2
    assert summary["attempt_summaries"][0]["generation_retries"] == 1
    assert pipeline.feedback_history[0] == []
    assert "Generation or training failed" in pipeline.feedback_history[1][0][1]


def test_rollout_failure_triggers_generation_retry_with_rollout_feedback(tmp_path):
    pipeline = _DummyPipeline(output_dir=str(tmp_path), fail_generate_n=0, fail_rollout_n=1)
    behavior = Behavior(name="b1", description="test behavior")

    success, summary = pipeline(
        behavior=behavior,
        num_attempts=1,
        n_rollouts=1,
        n_generation_retries=1,
        num_workers=1,
    )

    assert success is True
    assert pipeline.generate_calls == 2
    assert pipeline.rollout_calls == 2
    assert summary["attempt_summaries"][0]["generation_retries"] == 1
    assert "Rollout execution failed" in pipeline.feedback_history[1][0][1]
    assert "Rollout #0:" in pipeline.feedback_history[1][0][1]
    assert "rollout crashed" in pipeline.feedback_history[1][0][1]
    assert not (tmp_path / "attempt_0" / "generation_retry_1").exists()
    assert (tmp_path / "attempt_0" / "policy.py").read_text() == "code-2"


def test_evaluation_exception_sets_zero_eval_latency(tmp_path):
    pipeline = _EvalFailingPipeline(output_dir=str(tmp_path), fail_generate_n=0, fail_rollout_n=0)
    behavior = Behavior(name="b_eval_fail", description="test behavior")

    success, summary = pipeline(
        behavior=behavior,
        num_attempts=1,
        n_rollouts=1,
        n_generation_retries=0,
        num_workers=1,
    )

    assert success is False
    assert summary["attempt_summaries"][0]["latencies"]["evaluation"] == 0.0
    latency_payload = json.loads((tmp_path / "attempt_0" / "latency.json").read_text())
    assert latency_payload["evaluation"] == 0.0


def test_rollout_latency_is_scalar_in_summary(tmp_path):
    pipeline = _DummyPipeline(output_dir=str(tmp_path), fail_generate_n=0, fail_rollout_n=0)
    behavior = Behavior(name="b_latency", description="test behavior")

    success, summary = pipeline(
        behavior=behavior,
        num_attempts=1,
        n_rollouts=1,
        n_generation_retries=0,
        num_workers=1,
    )

    assert success is True
    assert isinstance(summary["attempt_summaries"][0]["latencies"]["rollout"], float)
    assert summary["attempt_summaries"][0]["latencies"]["rollout"] == 0.1
    assert isinstance(summary["avg_latencies"]["rollout"], float)
    assert summary["avg_latencies"]["rollout"] == 0.1


def test_generate_and_rollout_defaults_do_not_share_feedback_state(tmp_path):
    pipeline = _DummyPipeline(output_dir=str(tmp_path), fail_generate_n=1, fail_rollout_n=0)
    behavior = Behavior(name="b_defaults", description="test behavior")

    first_attempt_dir = tmp_path / "attempt_0"
    second_attempt_dir = tmp_path / "attempt_1"
    first_attempt_dir.mkdir()
    second_attempt_dir.mkdir()

    pipeline.generate_and_rollout(
        behavior=behavior,
        attempt_dir=str(first_attempt_dir),
        n_generation_retries=1,
        n_rollouts=1,
        num_workers=1,
    )
    pipeline.generate_and_rollout(
        behavior=behavior,
        attempt_dir=str(second_attempt_dir),
        n_generation_retries=0,
        n_rollouts=1,
        num_workers=1,
    )

    assert pipeline.feedback_history[2] == []


def test_rl_generation_receives_env_reward_api_reference(tmp_path):
    class _RecordingRewardGenerator:
        def __init__(self):
            self.calls: list[dict] = []

        def generate(self, **kwargs):
            self.calls.append(kwargs)
            raise RuntimeError("forced generation failure")

    generator = _RecordingRewardGenerator()
    env_module = SimpleNamespace(
        ENVIRONMENT_DESCRIPTION="env",
        REWARD_EXAMPLE="example",
        REWARD_API_REFERENCE="reward api reference",
        train_fn=lambda *_args, **_kwargs: None,
    )
    pipeline = RLAgentGenerationPipeline(
        generator=generator,
        evaluator=SimpleNamespace(),
        env_module=env_module,
        output_dir=str(tmp_path),
    )

    behavior = Behavior(name="b2", description="test behavior")
    generated_item, _loading_kwargs, _latency = pipeline._generate_policy(
        behavior=behavior,
        feedbacks=[],
        output_dir=str(tmp_path),
    )

    assert isinstance(generated_item, Exception)
    assert len(generator.calls) == 1
    assert generator.calls[0]["reward_api_reference"] == "reward api reference"


def test_generate_returned_exception_is_retried_without_rollout(tmp_path):
    pipeline = _ExceptionReturningPipeline(output_dir=str(tmp_path), fail_generate_n=0, fail_rollout_n=0)
    behavior = Behavior(name="b_ret_exc", description="test behavior")

    generated_item, _rollouts, success, _gen_retry, _latencies, retry_info = pipeline.generate_and_rollout(
        behavior=behavior,
        attempt_dir=str(tmp_path),
        n_generation_retries=1,
        n_rollouts=1,
        num_workers=1,
    )

    assert isinstance(generated_item, Exception)
    assert success is False
    assert pipeline.generate_calls == 2
    assert pipeline.rollout_calls == 0
    # A plain returned Exception is a cheap failure: it must not be charged to
    # the training budget, and its text must survive for the next attempt.
    assert retry_info["training_failures"] == 0
    assert len(retry_info["feedbacks"]) == 2


def test_rl_generation_flags_missing_checkpoints_as_failure(tmp_path):
    class _StaticRewardGenerator:
        def generate(self, **kwargs):
            del kwargs
            return CodeSnippet(code="def compute_reward(self, agent):\n    return 0.0")

    generator = _StaticRewardGenerator()
    env_module = SimpleNamespace(
        ENVIRONMENT_DESCRIPTION="env",
        REWARD_EXAMPLE="example",
        REWARD_API_REFERENCE="reward api reference",
        train_fn=lambda *_args, **_kwargs: None,
    )
    pipeline = RLAgentGenerationPipeline(
        generator=generator,
        evaluator=SimpleNamespace(),
        env_module=env_module,
        output_dir=str(tmp_path),
    )

    behavior = Behavior(name="b_missing_ckpt", description="test behavior")
    generated_item, _loading_kwargs, _latency = pipeline._generate_policy(
        behavior=behavior,
        feedbacks=[],
        output_dir=str(tmp_path),
    )

    assert isinstance(generated_item, Exception)
    assert "No checkpoints produced" in str(generated_item)


class _TrainingFailurePipeline(_DummyPipeline):
    """Returns a TrainingFailure, as RLAgentGenerationPipeline does."""

    policy_type = "RewardFn"

    def __init__(self, *args, elapsed: float = 3600.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.elapsed = elapsed

    def _generate_policy(self, behavior, feedbacks, output_dir, num_workers=1, **kwargs):
        del behavior, feedbacks, output_dir, num_workers, kwargs
        self.generate_calls += 1
        return (
            TrainingFailure(RuntimeError("actor loss became nan"), elapsed=self.elapsed),
            {},
            {"reward_generation": 0.01, "training": 0.0},
        )


def test_slow_training_failure_ends_attempt_without_further_retrains(tmp_path):
    """A slow training failure must not consume the cheap retry budget.

    Each retrain costs tens of minutes, so exhausting n_generation_retries on
    them burned hours per attempt.
    """
    pipeline = _TrainingFailurePipeline(output_dir=str(tmp_path), elapsed=3600.0)

    generated_item, rollouts, success, _gen_retry, _latencies, retry_info = pipeline.generate_and_rollout(
        behavior=Behavior(name="b_slow_train", description="test behavior"),
        attempt_dir=str(tmp_path),
        n_generation_retries=5,
        n_rollouts=1,
        num_workers=1,
        n_training_retries=0,
    )

    assert isinstance(generated_item, TrainingFailure)
    assert success is False
    assert pipeline.generate_calls == 1, "a slow training failure must end the attempt immediately"
    assert pipeline.rollout_calls == 0
    assert retry_info["training_failures"] == 1
    assert rollouts == []


def test_fast_training_failure_stays_on_the_cheap_budget(tmp_path):
    """A crash in seconds (e.g. bad reward shape) is cheap and should retry."""
    pipeline = _TrainingFailurePipeline(output_dir=str(tmp_path), elapsed=1.0)

    _generated_item, _rollouts, success, _gen_retry, _latencies, retry_info = pipeline.generate_and_rollout(
        behavior=Behavior(name="b_fast_train", description="test behavior"),
        attempt_dir=str(tmp_path),
        n_generation_retries=2,
        n_rollouts=1,
        num_workers=1,
        n_training_retries=0,
        min_expensive_training_seconds=120.0,
    )

    assert success is False
    assert pipeline.generate_calls == 3, "fast failures keep the generation-retry budget"
    assert retry_info["training_failures"] == 0


def test_training_retry_budget_allows_configured_extra_retrains(tmp_path):
    pipeline = _TrainingFailurePipeline(output_dir=str(tmp_path), elapsed=3600.0)

    *_, retry_info = pipeline.generate_and_rollout(
        behavior=Behavior(name="b_budget", description="test behavior"),
        attempt_dir=str(tmp_path),
        n_generation_retries=5,
        n_rollouts=1,
        num_workers=1,
        n_training_retries=1,
    )

    assert pipeline.generate_calls == 2
    assert retry_info["training_failures"] == 2


def test_empty_rollouts_skip_evaluation_and_propagate_real_error(tmp_path):
    """No rollouts must not raise, and the next attempt must see the cause.

    Evaluating an empty rollout set used to raise ZeroDivisionError, and
    "Evaluation failed: division by zero" was the only feedback the next
    attempt received.
    """
    pipeline = _TrainingFailurePipeline(output_dir=str(tmp_path), elapsed=3600.0)

    success, summary = pipeline(
        behavior=Behavior(name="b_empty", description="test behavior"),
        num_attempts=2,
        n_rollouts=1,
        adherence_threshold=0.8,
        n_generation_retries=5,
        num_workers=1,
        n_training_retries=0,
    )

    assert success is False
    assert summary["n_attempts"] == 2
    assert all(a["adherence"] == 0.0 for a in summary["attempt_summaries"])

    # The second attempt's context must carry the training error, not a
    # division-by-zero artifact.
    with open(tmp_path / "attempt_1" / "feedback.json") as f:
        context = json.load(f)
    blob = json.dumps(context)
    assert "division by zero" not in blob
    assert "actor loss became nan" in blob
