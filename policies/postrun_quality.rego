package costguard

default postrun_result := {
  "decision": "PASS",
  "reasons": ["All post-run quality checks passed."],
  "thresholds": {
    "min_ensemble_f1": 0.80,
  },
  "source": "opa",
}

postrun_summary := object.get(input, "summary", {})
postrun_aggregate_summary := object.get(input, "aggregate_summary", {})
postrun_domains := object.get(postrun_aggregate_summary, "domains", {})
postrun_thresholds := object.get(input, "thresholds", {})
postrun_min_ensemble_f1 := to_number(object.get(postrun_thresholds, "min_ensemble_f1", 0.80))

postrun_has_incomplete_trials {
  total_trials := to_number(object.get(postrun_summary, "total_trials", 0))
  completed_trials := to_number(object.get(postrun_summary, "completed_trials", 0))
  total_trials > 0
  completed_trials < total_trials
}

postrun_low_f1_domains[domain] {
  domain_data := postrun_domains[domain]
  ens := object.get(domain_data, "ens", {})
  f1_at_opt := object.get(ens, "f1_at_opt", {})
  mean_score := to_number(object.get(f1_at_opt, "mean", -1))
  mean_score >= 0
  mean_score < postrun_min_ensemble_f1
}

postrun_decision := "BLOCK" {
  postrun_has_incomplete_trials
} else := "WARN" {
  count(postrun_low_f1_domains) > 0
} else := "PASS"

postrun_reason_set["Incomplete trials detected in post-run summary."] {
  postrun_has_incomplete_trials
}
postrun_reason_set[reason] {
  domain := postrun_low_f1_domains[_]
  reason := sprintf(
    "Domain '%s' ensemble F1 mean is below gate %.4f.",
    [domain, postrun_min_ensemble_f1],
  )
}

postrun_computed_reasons := sort([r | r := postrun_reason_set[_]])

postrun_result := {
  "decision": postrun_decision,
  "reasons": reasons,
  "thresholds": {
    "min_ensemble_f1": postrun_min_ensemble_f1,
  },
  "source": "opa",
} {
  reasons := postrun_computed_reasons
  count(reasons) > 0
}
