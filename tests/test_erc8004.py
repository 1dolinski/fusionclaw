"""Tests for ERC-8004 integration — metadata generation, parsing, and claw wrapping."""

import asyncio
import json

from fusionclaw.claw import BaseClaw
from fusionclaw.erc8004 import (
    ERC8004_TYPE,
    ERC8004Claw,
    parse_erc8004_metadata,
    to_erc8004_metadata,
)
from fusionclaw.models import StateObject


class DummyClaw(BaseClaw):
    claw_id = "pricing_analyst"
    description = "Analyzes competitor pricing strategies and market positioning"

    async def run(self, input: str) -> StateObject:
        return StateObject(claw_id=self.claw_id, summary="test")


# --- Metadata Generation Tests ---


def test_to_erc8004_metadata_minimal():
    """Minimal claw produces valid ERC-8004 metadata."""
    claw = DummyClaw()
    meta = to_erc8004_metadata(claw)

    assert meta["type"] == ERC8004_TYPE
    assert meta["name"] == "pricing_analyst"
    assert meta["description"] == claw.description
    assert meta["active"] is True
    assert meta["services"] == []


def test_to_erc8004_metadata_with_mcp():
    """Claw with MCP endpoint gets proper service entry."""
    claw = DummyClaw()
    meta = to_erc8004_metadata(claw, mcp_endpoint="https://api.example.com/mcp")

    mcp_services = [s for s in meta["services"] if s["name"] == "MCP"]
    assert len(mcp_services) == 1
    assert mcp_services[0]["endpoint"] == "https://api.example.com/mcp"
    assert mcp_services[0]["version"] == "2025-06-18"


def test_to_erc8004_metadata_with_oasf():
    """Claw with OASF skills/domains gets proper service entry."""
    claw = DummyClaw()
    meta = to_erc8004_metadata(
        claw,
        oasf_skills=["analytical_skills/data_analysis", "natural_language_processing/summarization"],
        oasf_domains=["finance_and_business/finance"],
    )

    oasf_services = [s for s in meta["services"] if s["name"] == "OASF"]
    assert len(oasf_services) == 1
    assert len(oasf_services[0]["skills"]) == 2
    assert oasf_services[0]["domains"] == ["finance_and_business/finance"]
    assert oasf_services[0]["version"] == "0.8.0"


def test_to_erc8004_metadata_with_wallet():
    """Claw with wallet gets agentWallet service."""
    claw = DummyClaw()
    meta = to_erc8004_metadata(
        claw, wallet_endpoint="eip155:8453:0x742d35Cc6634C0532925a3b844Bc9e7595f0bEb7"
    )

    wallet_services = [s for s in meta["services"] if s["name"] == "agentWallet"]
    assert len(wallet_services) == 1
    assert "eip155:8453" in wallet_services[0]["endpoint"]


def test_to_erc8004_metadata_full():
    """Full metadata with all fields."""
    claw = DummyClaw()
    meta = to_erc8004_metadata(
        claw,
        image="ipfs://bafkreiabc123",
        mcp_endpoint="https://api.example.com/mcp",
        wallet_endpoint="eip155:1:0xABC",
        oasf_skills=["analytical_skills/data_analysis"],
        oasf_domains=["technology/blockchain"],
        extra_services=[{"name": "web", "endpoint": "https://myagent.com"}],
    )

    assert meta["image"] == "ipfs://bafkreiabc123"
    assert len(meta["services"]) == 4  # MCP, OASF, wallet, web
    service_names = [s["name"] for s in meta["services"]]
    assert "MCP" in service_names
    assert "OASF" in service_names
    assert "agentWallet" in service_names
    assert "web" in service_names


def test_to_erc8004_metadata_no_image():
    """Image field omitted when not provided."""
    claw = DummyClaw()
    meta = to_erc8004_metadata(claw)
    assert "image" not in meta


# --- Metadata Parsing Tests ---


SAMPLE_SDK_METADATA = {
    "type": ERC8004_TYPE,
    "name": "DataAnalyst Pro",
    "description": "Blockchain data analysis agent",
    "image": "ipfs://bafkreiabc",
    "active": True,
    "services": [
        {
            "name": "MCP",
            "endpoint": "https://api.example.com/mcp",
            "version": "2025-06-18",
        },
        {
            "name": "A2A",
            "endpoint": "https://agent.example/.well-known/agent-card.json",
            "version": "0.3.0",
        },
        {
            "name": "OASF",
            "endpoint": "https://github.com/agntcy/oasf/",
            "version": "0.8.0",
            "skills": ["analytical_skills/data_analysis/blockchain_analysis"],
            "domains": ["technology/blockchain"],
        },
        {
            "name": "agentWallet",
            "endpoint": "eip155:8453:0x742d35Cc6634C0532925a3b844Bc9e7595f0bEb7",
        },
    ],
    "registrations": [{"agentId": 241, "agentRegistry": "eip155:8453:0x8004a609"}],
    "supportedTrust": ["reputation", "crypto-economic"],
    "x402Support": True,
}


def test_parse_metadata_sdk_format():
    """Parse standard SDK-generated ERC-8004 metadata."""
    parsed = parse_erc8004_metadata(SAMPLE_SDK_METADATA)

    assert parsed["name"] == "DataAnalyst Pro"
    assert parsed["description"] == "Blockchain data analysis agent"
    assert parsed["active"] is True
    assert len(parsed["mcp_endpoints"]) == 1
    assert len(parsed["a2a_endpoints"]) == 1
    assert "blockchain_analysis" in parsed["oasf_skills"][0]
    assert "technology/blockchain" in parsed["oasf_domains"]
    assert len(parsed["wallets"]) == 1


def test_parse_metadata_legacy_endpoints_field():
    """Parse metadata using legacy 'endpoints' field name (pre-Jan 2026)."""
    legacy = {
        "type": ERC8004_TYPE,
        "name": "Legacy Agent",
        "description": "Uses old field name",
        "endpoints": [
            {"name": "MCP", "endpoint": "https://old.example.com/mcp", "version": "2025-06-18"},
        ],
    }
    parsed = parse_erc8004_metadata(legacy)
    assert len(parsed["mcp_endpoints"]) == 1
    assert parsed["mcp_endpoints"][0] == "https://old.example.com/mcp"


def test_parse_metadata_minimal():
    """Parse minimal metadata with no services."""
    minimal = {
        "type": ERC8004_TYPE,
        "name": "Test Agent",
        "description": "Minimal",
        "services": [],
    }
    parsed = parse_erc8004_metadata(minimal)
    assert parsed["name"] == "Test Agent"
    assert parsed["mcp_endpoints"] == []
    assert parsed["oasf_skills"] == []


def test_parse_metadata_missing_fields():
    """Parse metadata with missing optional fields."""
    sparse = {"type": ERC8004_TYPE}
    parsed = parse_erc8004_metadata(sparse)
    assert parsed["name"] == ""
    assert parsed["description"] == ""
    assert parsed["active"] is False


# --- ERC8004Claw Wrapper Tests ---


def test_erc8004_claw_init():
    """ERC8004Claw initializes from metadata."""
    claw = ERC8004Claw(SAMPLE_SDK_METADATA)
    assert claw.claw_id == "DataAnalyst Pro"
    assert "Blockchain" in claw.description


def test_erc8004_claw_run():
    """ERC8004Claw.run() produces a valid StateObject."""
    claw = ERC8004Claw(SAMPLE_SDK_METADATA)
    result = asyncio.get_event_loop().run_until_complete(claw.run("analyze something"))

    assert isinstance(result, StateObject)
    assert result.claw_id == "DataAnalyst Pro"
    assert result.token_count > 0
    assert len(result.key_facts) > 0

    # Should have skills, domains, mcp, a2a, wallet facts
    fact_keys = [f.key for f in result.key_facts]
    assert "skills" in fact_keys
    assert "domains" in fact_keys
    assert "mcp_endpoints" in fact_keys
    assert "wallets" in fact_keys


def test_erc8004_claw_raw_context_is_json():
    """raw_context should be the full metadata as JSON."""
    claw = ERC8004Claw(SAMPLE_SDK_METADATA)
    result = asyncio.get_event_loop().run_until_complete(claw.run("test"))
    parsed_back = json.loads(result.raw_context)
    assert parsed_back["name"] == "DataAnalyst Pro"


def test_erc8004_claw_minimal_metadata():
    """ERC8004Claw handles minimal metadata gracefully."""
    minimal = {
        "type": ERC8004_TYPE,
        "name": "Bare Agent",
        "description": "Nothing here",
        "services": [],
    }
    claw = ERC8004Claw(minimal)
    result = asyncio.get_event_loop().run_until_complete(claw.run("test"))
    assert result.claw_id == "Bare Agent"
    # Should have at least the 'active' fact
    assert len(result.key_facts) >= 1


# --- Round-Trip Test ---


def test_roundtrip_claw_to_metadata_to_claw():
    """Claw -> ERC-8004 metadata -> ERC8004Claw produces consistent identity."""
    original = DummyClaw()
    metadata = to_erc8004_metadata(
        original,
        mcp_endpoint="https://api.example.com/mcp",
        oasf_skills=["analytical_skills/data_analysis"],
    )

    # Wrap the metadata back as a claw
    wrapped = ERC8004Claw(metadata)
    assert wrapped.claw_id == original.claw_id
    assert wrapped.description == original.description

    # Run it and check state
    result = asyncio.get_event_loop().run_until_complete(wrapped.run("test"))
    fact_keys = [f.key for f in result.key_facts]
    assert "skills" in fact_keys
    assert "mcp_endpoints" in fact_keys
