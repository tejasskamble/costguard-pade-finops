-- CostGuard v17.0 policy-governance upgrade

ALTER TABLE policy_config
ADD COLUMN IF NOT EXISTS policy_bundle JSONB DEFAULT '{}'::jsonb;

UPDATE policy_config
SET policy_bundle = jsonb_build_object(
    'version', 'v17.0',
    'thresholds', jsonb_build_object(
        'warn_threshold', warn_threshold,
        'auto_optimise_threshold', auto_optimise_threshold,
        'block_threshold', block_threshold
    ),
    'rules', jsonb_build_object(
        'protected_branches', jsonb_build_array('main', 'release', 'production'),
        'sensitive_stages', jsonb_build_array('security_scan', 'deploy_staging', 'deploy_prod'),
        'block_pr_prod_deploys', true,
        'require_core_team_for_sensitive_stages', true,
        'stage_cost_ceiling_usd', jsonb_build_object(
            'build', 0.05,
            'integration_test', 0.08,
            'security_scan', 0.04,
            'docker_build', 0.09,
            'deploy_staging', 0.06,
            'deploy_prod', 0.08
        )
    )
)
WHERE COALESCE(policy_bundle, '{}'::jsonb) = '{}'::jsonb;
