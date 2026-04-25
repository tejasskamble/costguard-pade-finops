"""OPA client with deterministic inline fallback."""
from __future__ import annotations

import logging
from typing import Any, Dict, Mapping

import httpx

from config import settings
from .policy_engine import (
    VALID_DECISIONS,
    build_policy_input,
    evaluate_inline_policy,
    normalize_policy_bundle,
)
from runtime_hardening import retry_async

logger = logging.getLogger(__name__)


def _inline_policy(crs: float, warn: float, auto: float, block: float) -> str:
    policy_input = build_policy_input(
        {"crs": crs},
        {},
        normalize_policy_bundle(
            None,
            warn_threshold=warn,
            auto_optimise_threshold=auto,
            block_threshold=block,
        ),
    )
    return evaluate_inline_policy(policy_input)["decision"]


async def evaluate_policy(
    metrics: Mapping[str, Any],
    context: Mapping[str, Any],
    policy_bundle: Mapping[str, Any],
) -> Dict[str, Any]:
    payload = build_policy_input(metrics, context, policy_bundle)
    try:
        async def _post_opa() -> httpx.Response:
            async with httpx.AsyncClient(timeout=2.0) as client:
                return await client.post(settings.OPA_URL, json={"input": payload})

        resp = await retry_async(
            _post_opa,
            attempts=2,
            delay=0.25,
            exceptions=(httpx.ConnectError, httpx.TimeoutException, httpx.RemoteProtocolError),
            logger=logger,
            label="OPA policy evaluation",
        )
        if resp.status_code == 200:
            data = resp.json().get("result", {})
            decision = data.get("decision")
            if decision in VALID_DECISIONS:
                data.setdefault("reasons", [])
                data.setdefault("matched_rules", [])
                data.setdefault("actions", [])
                data["policy_source"] = "opa"
                return data
    except (httpx.ConnectError, httpx.TimeoutException) as exc:
        logger.debug("OPA unavailable (%s); using inline fallback", type(exc).__name__)
    except Exception as exc:
        logger.warning("OPA call failed (%s); using inline fallback", type(exc).__name__)

    inline = evaluate_inline_policy(payload)
    inline["policy_source"] = "inline"
    return inline


async def evaluate_policy_with_custom_thresholds(crs: float, warn: float, auto: float, block: float) -> str:
    return _inline_policy(crs, warn, auto, block)
