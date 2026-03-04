import json
from types import SimpleNamespace

from popgen.population.pipeline import AgentGenerationPipeline, RLAgentGenerationPipeline
from popgen.population.types import Behavior


class _DummyGeneratedPolicy:
    def __init__(self, code: str):
        self.code = code


class _DummyEvalResult:
    def __init__(self):
        self.overall_metrics = {"adherence": 1.0}
        self.instance_metrics = [{"adherence": 1.0, "reasoning": "ok", "feedback": ""}]
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


def test_generate_with_retries_defaults_do_not_share_feedback_state(tmp_path):
    pipeline = _DummyPipeline(output_dir=str(tmp_path), fail_generate_n=1, fail_rollout_n=0)
    behavior = Behavior(name="b_defaults", description="test behavior")

    first_attempt_dir = tmp_path / "attempt_0"
    second_attempt_dir = tmp_path / "attempt_1"
    first_attempt_dir.mkdir()
    second_attempt_dir.mkdir()

    pipeline._generate_with_retries(
        behavior=behavior,
        attempt_dir=str(first_attempt_dir),
        n_generation_retries=1,
        n_rollouts=1,
        num_workers=1,
    )
    pipeline._generate_with_retries(
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
