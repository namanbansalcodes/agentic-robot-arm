import pytest

from agent.llm import CacheMiss, LLMClient, VLMCall, VLMResponse


def _call(step=0, kind="plan"):
    return VLMCall(scene_id="h1_single", condition="agentic", seed=0,
                   step_index=step, call_kind=kind,
                   system="sys", text="hello", image_png=b"\x89PNG-fake",
                   tools=[{"type": "function", "name": "look"}])


def test_replay_miss_is_loud(tmp_path):
    client = LLMClient(mode="replay", cache_dir=tmp_path)
    with pytest.raises(CacheMiss):
        client.complete(_call())


def test_record_then_replay_returns_the_same_response(tmp_path):
    recorded = VLMResponse(text="ok", tool_calls=[{"name": "look", "args": {}}],
                           input_tokens=10, output_tokens=3, model="test-model")
    rec = LLMClient(mode="replay", cache_dir=tmp_path)
    rec.write_cache(_call(), recorded)

    client = LLMClient(mode="replay", cache_dir=tmp_path)
    got = client.complete(_call())
    assert got.text == "ok"
    assert got.tool_calls == [{"name": "look", "args": {}}]
    assert client.drift_count == 0


def test_changed_prompt_replays_but_counts_as_drift(tmp_path):
    rec = LLMClient(mode="replay", cache_dir=tmp_path)
    rec.write_cache(_call(), VLMResponse(text="ok", tool_calls=[], input_tokens=1,
                                         output_tokens=1, model="test-model"))
    client = LLMClient(mode="replay", cache_dir=tmp_path)
    changed = _call()
    changed.text = "a different prompt"
    got = client.complete(changed)
    assert got.text == "ok"
    assert client.drift_count == 1


def test_cost_accounting_uses_the_published_rates(tmp_path):
    client = LLMClient(mode="replay", cache_dir=tmp_path)
    client.input_tokens = 1_000_000
    client.output_tokens = 1_000_000
    assert client.cost_usd() == pytest.approx(12.0, rel=1e-6)


def test_cache_key_is_machine_stable_and_excludes_image_bytes(tmp_path):
    """Images are not bit-identical across machines. Keying on them would break
    `make judge` on a judge's laptop."""
    a = _call()
    b = _call()
    b.image_png = b"completely different bytes"
    assert a.cache_key() == b.cache_key()
    assert a.prompt_hash() != b.prompt_hash()


def test_distinct_steps_and_kinds_get_distinct_keys(tmp_path):
    keys = {_call(step=i, kind=k).cache_key()
            for i in range(3) for k in ("plan", "verify")}
    assert len(keys) == 6


def test_replay_never_silently_falls_back_to_live(tmp_path, monkeypatch):
    """A cache miss in replay mode must raise, never make a network call."""
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    client = LLMClient(mode="replay", cache_dir=tmp_path)
    with pytest.raises(CacheMiss):
        client.complete(_call(step=99))
