"""ERC-8004 integration — register claws on-chain and wrap on-chain agents as claws.

Two-way bridge:
  1. Register FusionClaw claws as ERC-8004 agents (via RNWY API)
  2. Wrap ERC-8004 agents as FusionClaw claws (for context fusion)
"""

from __future__ import annotations

import json
import os
from typing import Any, Optional

import httpx

from .claw import BaseClaw
from .fuser import count_tokens
from .models import Fact, StateObject

# ERC-8004 metadata type identifier
ERC8004_TYPE = "https://eips.ethereum.org/EIPS/eip-8004#registration-v1"

# RNWY API base URL
RNWY_API_BASE = "https://rnwy.com/api"


def to_erc8004_metadata(
    claw: BaseClaw,
    *,
    image: str = "",
    mcp_endpoint: str = "",
    wallet_endpoint: str = "",
    oasf_skills: list[str] | None = None,
    oasf_domains: list[str] | None = None,
    extra_services: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Convert a FusionClaw claw to ERC-8004 agent metadata.

    Args:
        claw: The claw to convert.
        image: Agent avatar URL (IPFS, HTTPS, or data URI).
        mcp_endpoint: MCP server URL if the claw exposes one.
        wallet_endpoint: CAIP-10 wallet address for x402 payments.
        oasf_skills: OASF skill slugs (e.g., ["analytical_skills/data_analysis"]).
        oasf_domains: OASF domain slugs (e.g., ["technology/blockchain"]).
        extra_services: Additional service entries to include.

    Returns:
        ERC-8004 compliant metadata dict ready for registration.
    """
    services: list[dict[str, Any]] = []

    if mcp_endpoint:
        services.append({
            "name": "MCP",
            "endpoint": mcp_endpoint,
            "version": "2025-06-18",
        })

    if oasf_skills or oasf_domains:
        oasf_entry: dict[str, Any] = {
            "name": "OASF",
            "endpoint": "https://github.com/agntcy/oasf/",
            "version": "0.8.0",
        }
        if oasf_skills:
            oasf_entry["skills"] = oasf_skills
        if oasf_domains:
            oasf_entry["domains"] = oasf_domains
        services.append(oasf_entry)

    if wallet_endpoint:
        services.append({
            "name": "agentWallet",
            "endpoint": wallet_endpoint,
        })

    if extra_services:
        services.extend(extra_services)

    metadata: dict[str, Any] = {
        "type": ERC8004_TYPE,
        "name": claw.claw_id,
        "description": claw.description,
        "active": True,
        "services": services,
    }

    if image:
        metadata["image"] = image

    return metadata


async def register_claw(
    claw: BaseClaw,
    *,
    chain: str = "base",
    wallet_address: str,
    category: str = "ai",
    tags: list[str] | None = None,
    api_key: str | None = None,
    mcp_endpoint: str = "",
    oasf_skills: list[str] | None = None,
    oasf_domains: list[str] | None = None,
) -> dict[str, Any]:
    """Register a FusionClaw claw as an ERC-8004 agent via RNWY API.

    One POST. RNWY handles IPFS pinning, contract call, gas, and event parsing.

    Args:
        claw: The claw to register.
        chain: "base" or "ethereum".
        wallet_address: Owner wallet (0x...).
        category: Agent category.
        tags: Tags for discovery.
        api_key: RNWY API key (or set RNWY_API_KEY env var).
        mcp_endpoint: MCP server URL if exposed.
        oasf_skills: OASF skill slugs.
        oasf_domains: OASF domain slugs.

    Returns:
        RNWY API response with agent_id, did, ipfs_uri, tx_hash, etc.
    """
    key = api_key or os.environ.get("RNWY_API_KEY")
    if not key:
        raise ValueError("RNWY API key required. Pass api_key or set RNWY_API_KEY env var.")

    payload = {
        "name": claw.claw_id,
        "description": claw.description,
        "chain": chain,
        "wallet_address": wallet_address,
        "bio": claw.description,
        "category": category,
        "tags": tags or ["fusionclaw", "context-fusion"],
    }

    async with httpx.AsyncClient() as client:
        resp = await client.post(
            f"{RNWY_API_BASE}/create-agent",
            json=payload,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {key}",
            },
            timeout=30.0,
        )
        resp.raise_for_status()
        return resp.json()


def parse_erc8004_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    """Parse ERC-8004 agent metadata into a normalized structure.

    Handles both 'services' (new) and 'endpoints' (legacy) field names.

    Returns:
        Normalized dict with name, description, services, skills, domains, etc.
    """
    services = metadata.get("services") or metadata.get("endpoints") or []

    mcp_endpoints = []
    a2a_endpoints = []
    oasf_skills: list[str] = []
    oasf_domains: list[str] = []
    wallets = []

    for svc in services:
        svc_name = svc.get("name", "").upper()
        if svc_name == "MCP":
            mcp_endpoints.append(svc.get("endpoint", ""))
        elif svc_name == "A2A":
            a2a_endpoints.append(svc.get("endpoint", ""))
        elif svc_name == "OASF":
            oasf_skills.extend(svc.get("skills", []))
            oasf_domains.extend(svc.get("domains", []))
        elif svc_name == "AGENTWALLET":
            wallets.append(svc.get("endpoint", ""))

    return {
        "name": metadata.get("name", ""),
        "description": metadata.get("description", ""),
        "image": metadata.get("image", ""),
        "active": metadata.get("active", False),
        "mcp_endpoints": mcp_endpoints,
        "a2a_endpoints": a2a_endpoints,
        "oasf_skills": oasf_skills,
        "oasf_domains": oasf_domains,
        "wallets": wallets,
        "raw_metadata": metadata,
    }


class ERC8004Claw(BaseClaw):
    """Wraps an ERC-8004 registered agent as a FusionClaw claw.

    This lets you pull any on-chain agent into a fusion pipeline.
    The claw's context comes from the agent's metadata and capabilities.

    For agents with MCP endpoints, a future version will call the
    endpoint to get live context. For now, it uses the metadata itself.
    """

    def __init__(self, metadata: dict[str, Any]):
        parsed = parse_erc8004_metadata(metadata)
        self.claw_id = parsed["name"] or "erc8004_agent"
        self.description = parsed["description"] or "ERC-8004 registered agent"
        self._parsed = parsed
        self._raw_metadata = metadata

    async def run(self, input: str) -> StateObject:
        """Export the agent's metadata as a StateObject for fusion.

        Currently uses static metadata. Future: call MCP endpoint
        for live context based on the input query.
        """
        facts = []

        if self._parsed["oasf_skills"]:
            facts.append(Fact(key="skills", value=", ".join(self._parsed["oasf_skills"])))
        if self._parsed["oasf_domains"]:
            facts.append(Fact(key="domains", value=", ".join(self._parsed["oasf_domains"])))
        if self._parsed["mcp_endpoints"]:
            facts.append(Fact(key="mcp_endpoints", value=", ".join(self._parsed["mcp_endpoints"])))
        if self._parsed["a2a_endpoints"]:
            facts.append(Fact(key="a2a_endpoints", value=", ".join(self._parsed["a2a_endpoints"])))
        if self._parsed["wallets"]:
            facts.append(Fact(key="wallets", value=", ".join(self._parsed["wallets"])))
        facts.append(Fact(key="active", value=str(self._parsed["active"])))

        raw_context = json.dumps(self._raw_metadata, indent=2)

        return StateObject(
            claw_id=self.claw_id,
            summary=self.description,
            key_facts=facts,
            raw_context=raw_context,
            token_count=count_tokens(raw_context),
        )
