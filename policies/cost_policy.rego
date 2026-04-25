package costguard

default result := {
  "decision": "ALLOW",
  "reasons": ["All policy checks passed for the supplied metrics and context."],
  "matched_rules": ["allow_default"],
  "actions": ["Continue the pipeline run."]
}

thresholds := object.get(object.get(input, "policy_config", {}), "thresholds", {
  "warn_threshold": 0.50,
  "auto_optimise_threshold": 0.75,
  "block_threshold": 0.90,
})

rules_cfg := object.get(object.get(input, "policy_config", {}), "rules", {})
metrics := object.get(input, "metrics", {})
context := object.get(input, "context", {})

stage_name := object.get(context, "stage_name", "")
branch_name := lower(object.get(context, "branch", ""))
is_pr := object.get(context, "gh_is_pr", false)
is_core_team := object.get(context, "gh_by_core_team_member", false)
crs := to_number(object.get(metrics, "crs", 0.0))
billed_cost := to_number(object.get(metrics, "billed_cost", 0.0))

protected_branches := {lower(branch) | branch := object.get(rules_cfg, "protected_branches", [])[_]}
sensitive_stages := {stage | stage := object.get(rules_cfg, "sensitive_stages", [])[_]}
stage_cost_ceiling_usd := object.get(rules_cfg, "stage_cost_ceiling_usd", {})

cost_ceiling := object.get(stage_cost_ceiling_usd, stage_name, -1)
stage_cost_exceeded {
  cost_ceiling >= 0
  billed_cost > cost_ceiling
}

block_pr_prod_deploys {
  object.get(rules_cfg, "block_pr_prod_deploys", true)
  stage_name == "deploy_prod"
  is_pr
}

block_sensitive_branch {
  object.get(rules_cfg, "require_core_team_for_sensitive_stages", true)
  sensitive_stages[stage_name]
  protected_branches[branch_name]
  not is_core_team
}

block_threshold_hit {
  crs >= to_number(object.get(thresholds, "block_threshold", 0.90))
}

auto_threshold_hit {
  crs >= to_number(object.get(thresholds, "auto_optimise_threshold", 0.75))
  not block_threshold_hit
}

warn_threshold_hit {
  crs >= to_number(object.get(thresholds, "warn_threshold", 0.50))
  not auto_threshold_hit
  not block_threshold_hit
}

final_decision := "BLOCK" {
  block_pr_prod_deploys
} else := "BLOCK" {
  block_sensitive_branch
} else := "BLOCK" {
  block_threshold_hit
} else := "AUTO_OPTIMISE" {
  stage_cost_exceeded
  sensitive_stages[stage_name]
} else := "AUTO_OPTIMISE" {
  auto_threshold_hit
} else := "WARN" {
  stage_cost_exceeded
} else := "WARN" {
  warn_threshold_hit
} else := "ALLOW"

reason_set[reason] {
  block_pr_prod_deploys
  reason := "Production deployment is blocked for pull-request runs."
}
reason_set[reason] {
  block_sensitive_branch
  reason := "Sensitive stages on protected branches require a core-team member."
}
reason_set[reason] {
  stage_cost_exceeded
  reason := sprintf("Stage cost $%.4f exceeded the configured ceiling of $%.4f.", [billed_cost, cost_ceiling])
}
reason_set[reason] {
  block_threshold_hit
  reason := sprintf("CRS %.3f exceeded the BLOCK threshold %.2f.", [crs, to_number(object.get(thresholds, "block_threshold", 0.90))])
}
reason_set[reason] {
  auto_threshold_hit
  reason := sprintf("CRS %.3f exceeded the AUTO_OPTIMISE threshold %.2f.", [crs, to_number(object.get(thresholds, "auto_optimise_threshold", 0.75))])
}
reason_set[reason] {
  warn_threshold_hit
  reason := sprintf("CRS %.3f exceeded the WARN threshold %.2f.", [crs, to_number(object.get(thresholds, "warn_threshold", 0.50))])
}

matched_rule_set["block_pr_prod_deploys"] { block_pr_prod_deploys }
matched_rule_set["require_core_team_for_sensitive_stages"] { block_sensitive_branch }
matched_rule_set["stage_cost_ceiling_usd"] { stage_cost_exceeded }
matched_rule_set["block_threshold"] { block_threshold_hit }
matched_rule_set["auto_optimise_threshold"] { auto_threshold_hit }
matched_rule_set["warn_threshold"] { warn_threshold_hit }

default_stage_action := "Apply the recommended cost-optimization action before retrying."
stage_action := "Enable dependency and layer caching for the build stage." { stage_name == "build" }
else := "Reduce integration-test parallelism and scope test selection." { stage_name == "integration_test" }
else := "Scope security scans to changed components for the next run." { stage_name == "security_scan" }
else := "Reuse cached Docker layers and compress image artifacts." { stage_name == "docker_build" }
else := "Move staging deployment to a cheaper runner profile." { stage_name == "deploy_staging" }
else := "Require protected-branch deploy approval before production release." { stage_name == "deploy_prod" }
else := default_stage_action

action_set["Reroute production deployment to a post-merge protected branch pipeline."] {
  block_pr_prod_deploys
}
action_set["Require core-team approval or rerun from an authorized branch."] {
  block_sensitive_branch
}
action_set[stage_action] { stage_cost_exceeded }
action_set[stage_action] { block_threshold_hit }
action_set[stage_action] { auto_threshold_hit }
action_set["Notify the owning team and review the cost anomaly."] { warn_threshold_hit }

computed_reasons := sort([reason | reason := reason_set[_]])
computed_matched_rules := sort([rule | rule := matched_rule_set[_]])
computed_actions := sort([action | action := action_set[_]])

result := {
  "decision": final_decision,
  "reasons": reasons,
  "matched_rules": matched_rules,
  "actions": actions,
} {
  reasons := computed_reasons
  matched_rules := computed_matched_rules
  actions := computed_actions
  count(reasons) > 0
}
