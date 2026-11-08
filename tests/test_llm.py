from cl_mapper.llm import DEFAULT_MODEL, assessment_chain, make_chat_model


def test_default_model_is_deepseek_v4_1_flash(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-not-a-real-key")
    llm = make_chat_model()
    assert DEFAULT_MODEL == "deepseek:deepseek-flash"
    assert type(llm).__name__ == "ChatDeepSeek"
    assert llm.model_name == "deepseek-flash"
    assert llm.temperature == 0


def test_openai_reasoning_model_gets_effort_not_temperature(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-not-a-real-key")
    llm = make_chat_model("openai:gpt-6-luna", reasoning_effort="high",
                          base_url="http://localhost:8000/v1")
    assert llm.reasoning_effort == "high" and llm.temperature is None
    assert llm.use_responses_api is False  # proxies serve Chat Completions only


def test_structured_output_uses_tool_calling(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-not-a-real-key")
    chain = assessment_chain(make_chat_model())
    assert "tools" in chain.steps[1].kwargs  # function calling, not response_format
