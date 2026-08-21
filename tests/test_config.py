import pytest

from healthbench_harness.config import HarnessConfig, _openai_base_url


def test_openai_base_url_is_normalized() -> None:
    assert _openai_base_url("https://model.example/") == "https://model.example/v1"
    assert _openai_base_url("https://model.example/v1/") == "https://model.example/v1"


def test_config_reads_key_without_exposing_it(monkeypatch) -> None:
    monkeypatch.setenv("LUNIT_FM_API_KEY", "lunit-secret")
    config = HarnessConfig.from_env()
    assert config.lunit_api_key == "lunit-secret"
    assert config.l2_api_base == "https://model.hackathon.lunit.io/v1"
    assert config.l2_enable_thinking is True
    assert config.l2_max_tokens == 4096
    assert config.l2_retry_max_tokens == 8192
    assert config.l2_max_concurrency == 15
    assert config.mcp_max_concurrent_sessions == 8
    assert config.l2_repetition_penalty == 1.05
    assert config.l2_retry_repetition_penalty == 1.15
    assert config.enable_response_planning is False
    assert config.enable_answer_review is False


def test_config_reads_pipeline_feature_flags(monkeypatch) -> None:
    monkeypatch.setenv("LUNIT_FM_API_KEY", "lunit-secret")
    monkeypatch.setenv("ENABLE_RESPONSE_PLANNING", "false")
    monkeypatch.setenv("ENABLE_ANSWER_REVIEW", "true")

    config = HarnessConfig.from_env()

    assert config.enable_response_planning is False
    assert config.enable_answer_review is True


def test_config_can_explicitly_disable_l2_thinking(monkeypatch) -> None:
    monkeypatch.setenv("LUNIT_FM_API_KEY", "lunit-secret")
    monkeypatch.setenv("L2_ENABLE_THINKING", "false")

    config = HarnessConfig.from_env()

    assert config.l2_enable_thinking is False


def test_config_requires_key(monkeypatch) -> None:
    monkeypatch.delenv("LUNIT_FM_API_KEY", raising=False)
    try:
        HarnessConfig.from_env()
    except RuntimeError as error:
        assert "LUNIT_FM_API_KEY" in str(error)
    else:
        raise AssertionError("missing key should fail")


@pytest.mark.parametrize(
    "config",
    [
        HarnessConfig(generation_max_attempts=3),
        HarnessConfig(protocol_max_repairs=2),
        HarnessConfig(retrieval_max_attempts=3),
    ],
)
def test_config_rejects_retry_budgets_above_v1_limits(config: HarnessConfig) -> None:
    with pytest.raises(ValueError):
        config.validate()
