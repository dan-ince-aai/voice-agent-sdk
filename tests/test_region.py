from assembly_agent import Agent
from assembly_agent.endpoints import agents_base, llm_base


def test_default_region_is_us():
    a = Agent(name="X")
    assert a.region == "us"
    assert a.api_base == "https://agents.assemblyai.com/v1"
    assert a.llm_base_url == "https://llm-gateway.assemblyai.com/v1"
    assert a.gateway.base_url == "https://llm-gateway.assemblyai.com/v1"


def test_eu_switches_both_endpoints():
    a = Agent(name="X", region="eu")
    assert a.api_base == "https://agents.eu.assemblyai.com/v1"
    assert a.llm_base_url == "https://llm-gateway.eu.assemblyai.com/v1"
    assert a.gateway.base_url == "https://llm-gateway.eu.assemblyai.com/v1"


def test_region_from_env(monkeypatch):
    monkeypatch.setenv("ASSEMBLY_AGENT_REGION", "eu")
    a = Agent(name="X")
    assert a.region == "eu"
    assert a.api_base.startswith("https://agents.eu.")


def test_unknown_region_falls_back_to_us():
    assert agents_base("mars") == agents_base("us")
    assert llm_base(None) == llm_base("us")


def test_explicit_overrides_win():
    a = Agent(name="X", region="eu", llm_base_url="https://custom/v1", api_base="https://custom-agents/v1")
    assert a.llm_base_url == "https://custom/v1"
    assert a.api_base == "https://custom-agents/v1"
