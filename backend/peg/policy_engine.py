"""Shared policy configuration and inline evaluation helpers."""
from __future__ import annotations

import json
from copy import deepcopy
from typing import Any, Dict, Mapping, MutableMapping, Optional


VALID_DECISIONS = ("ALLOW", "WARN", "AUTO_OPTIMISE", "BLOCK")
DECISION_RANK = {decision: index for index, decision in enumerate(VALID_DECISIONS)}

DEFAULT_POLICY_BUNDLE: Dict[str, Any] = {
    "version": "v17.0",
    "thresholds": {
        "warn_threshold": 0.50,
        "auto_optimise_threshold": 0.75,
        "block_threshold": 0.90,
    },
    "rules": {
        "protected_branches": ["main", "release", "production"],
        "sensitive_stages": ["security_scan", "deploy_staging", "deploy_prod"],
        "block_pr_prod_deploys": True,
        "require_core_team_for_sensitive_stages": True,
        "stage_cost_ceiling_usd": {
            "build": 0.05,
            "integration_test": 0.08,
            "security_scan": 0.04,
            "docker_build": 0.09,
            "deploy_staging": 0.06,
            "deploy_prod": 0.08,
        },
    },
}

DEFAULT_STAGE_ACTIONS: Dict[str, str] = {
    "build": "Enable dependency and layer caching for the build stage.",
    "integration_test": "Reduce integration-test parallelism and scope test selection.",
    "security_scan": "Scope security scans to changed components for the next run.",
    "docker_build": "Reuse cached Docker layers and compress image artifacts.",
    "deploy_staging": "Move staging deployment to a cheaper runner profile.",
    "deploy_prod": "Require protected-branch deploy approval before production release.",
}


def _merge_dict(base: MutableMapping[str, Any], incoming: Mapping[str, Any]) -> MutableMapping[str, Any]:
    for key, value in incoming.items():
        if isinstance(value, Mapping) and isinstance(base.get(key), MutableMapping):
            _merge_dict(base[key], value)  # type: ignore[index]
        else:
            base[key] = value
    return base


def _coerce_policy_bundle(raw_policy_bundle: Any) -> Dict[str, Any]:
    if raw_policy_bundle is None:
        return {}
    if isinstance(raw_policy_bundle, str):
        try:
            return json.loads(raw_policy_bundle)
        except Exception:
            return {}
    if isinstance(raw_policy_bundle, Mapping):
        return dict(raw_policy_bundle)
    return {}


def normalize_policy_bundle(
    policy_bundle: Optional[Mapping[str, Any]] = None,
    *,
    warn_threshold: Optional[float] = None,
    auto_optimise_threshold: Optional[float] = None,
    block_threshold: Optional[float] = None,
) -> Dict[str, Any]:
    bundle = deepcopy(DEFAULT_POLICY_BUNDLE)
    if policy_bundle:
        _merge_dict(bundle, dict(policy_bundle))
    thresholds = bundle.setdefault("thresholds", {})
    if warn_threshold is not None:
        thresholds["warn_threshold"] = float(warn_threshold)
    if auto_optimise_threshold is not None:
        thresholds["auto_optimise_threshold"] = float(auto_optimise_threshold)
    if block_threshold is not None:
        thresholds["block_threshold"] = float(block_threshold)
    return bundle


def extract_policy_bundle(row: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    if not row:
        return normalize_policy_bundle()
    payload = dict(row)
    return normalize_policy_bundle(
        _coerce_policy_bundle(payload.get("policy_bundle")),
        warn_threshold=float(payload.get("warn_threshold", DEFAULT_POLICY_BUNDLE["thresholds"]["warn_threshold"])),
        auto_optimise_threshold=float(
            payload.get("auto_optimise_threshold", DEFAULT_POLICY_BUNDLE["thresholds"]["auto_optimise_threshold"])
        ),
        block_threshold=float(payload.get("block_threshold", DEFAULT_POLICY_BUNDLE["thresholds"]["block_threshold"])),
    )


def make_policy_response(row: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    bundle = extract_policy_bundle(row)
    thresholds = bundle["thresholds"]
    response = {
        "warn_threshold": thresholds["warn_threshold"],
        "auto_optimise_threshold": thresholds["auto_optimise_threshold"],
        "block_threshold": thresholds["block_threshold"],
        "policy_bundle": bundle,
    }
    if row:
        for key, value in dict(row).items():
            if key not in response and key != "policy_bundle":
                response[key] = value
    return response


def build_policy_input(metrics: Mapping[str, Any], context: Mapping[str, Any], policy_bundle: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "metrics": dict(metrics),
        "context": dict(context),
        "policy_config": normalize_policy_bundle(policy_bundle),
    }


def _append_unique(items, value: Optional[str]) -> None:
    if value and value not in items:
        items.append(value)


def _promote(result: Dict[str, Any], decision: str, reason: str, matched_rule: str, action: Optional[str] = None) -> None:
    if DECISION_RANK[decision] > DECISION_RANK[result["decision"]]:
        result["decision"] = decision
    _append_unique(result["reasons"], reason)
    _append_unique(result["matched_rules"], matched_rule)
    _append_unique(result["actions"], action)


def evaluate_inline_policy(policy_input: Mapping[str, Any]) -> Dict[str, Any]:
    bundle = normalize_policy_bundle(policy_input.get("policy_config"))  # type: ignore[arg-type]
    thresholds = bundle["thresholds"]
    rules = bundle.get("rules", {})
    metrics = dict(policy_input.get("metrics", {}))
    context = dict(policy_input.get("context", {}))

    stage_name = str(context.get("stage_name", "") or "")
    branch = str(context.get("branch", "") or "").lower()
    is_pr = bool(context.get("gh_is_pr", False))
    is_core_team = bool(context.get("gh_by_core_team_member", False))
    sensitive_stages = {str(item) for item in rules.get("sensitive_stages", [])}
    protected_branches = {str(item).lower() for item in rules.get("protected_branches", [])}
    stage_cost_ceilings = dict(rules.get("stage_cost_ceiling_usd", {}))

    crs = float(metrics.get("crs", 0.0) or 0.0)
    billed_cost = float(metrics.get("billed_cost", 0.0) or 0.0)

    result = {
        "decision": "ALLOW",
        "reasons": [],
        "matched_rules": [],
        "actions": [],
    }

    if rules.get("block_pr_prod_deploys", True) and stage_name == "deploy_prod" and is_pr:
        _promote(
            result,
            "BLOCK",
            "Production deployment is blocked for pull-request runs.",
            "block_pr_prod_deploys",
            "Reroute production deployment to a post-merge protected branch pipeline.",
        )

    if (
        rules.get("require_core_team_for_sensitive_stages", True)
        and stage_name in sensitive_stages
        and branch in protected_branches
        and not is_core_team
    ):
        _promote(
            result,
            "BLOCK",
            "Sensitive stages on protected branches require a core-team member.",
            "require_core_team_for_sensitive_stages",
            "Require core-team approval or rerun from an authorized branch.",
        )

    ceiling = stage_cost_ceilings.get(stage_name)
    if ceiling is not None and billed_cost > float(ceiling):
        _promote(
            result,
            "AUTO_OPTIMISE" if stage_name in sensitive_stages else "WARN",
            f"Stage cost ${billed_cost:.4f} exceeded the configured ceiling of ${float(ceiling):.4f}.",
            "stage_cost_ceiling_usd",
            DEFAULT_STAGE_ACTIONS.get(stage_name, "Apply the recommended cost-optimization action before retrying."),
        )

    if crs >= float(thresholds["block_threshold"]):
        _promote(
            result,
            "BLOCK",
            f"CRS {crs:.3f} exceeded the BLOCK threshold {float(thresholds['block_threshold']):.2f}.",
            "block_threshold",
            DEFAULT_STAGE_ACTIONS.get(stage_name, "Block the current run and review the anomaly before rerun."),
        )
    elif crs >= float(thresholds["auto_optimise_threshold"]):
        _promote(
            result,
            "AUTO_OPTIMISE",
            f"CRS {crs:.3f} exceeded the AUTO_OPTIMISE threshold {float(thresholds['auto_optimise_threshold']):.2f}.",
            "auto_optimise_threshold",
            DEFAULT_STAGE_ACTIONS.get(stage_name, "Apply the recommended optimization automatically."),
        )
    elif crs >= float(thresholds["warn_threshold"]):
        _promote(
            result,
            "WARN",
            f"CRS {crs:.3f} exceeded the WARN threshold {float(thresholds['warn_threshold']):.2f}.",
            "warn_threshold",
            "Notify the owning team and review the cost anomaly.",
        )

    if not result["reasons"]:
        result["reasons"] = ["All policy checks passed for the supplied metrics and context."]
        result["matched_rules"] = ["allow_default"]
        result["actions"] = ["Continue the pipeline run."]

    return result
