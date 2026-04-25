"""Policy governance regression tests for CostGuard v17.0."""
import os
import sys
from unittest.mock import AsyncMock, patch

import pytest

from tests.conftest import FAKE_POLICY, FAKE_USER, FAKE_VIEWER_USER, MockRecord

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))

from peg.opa_client import _inline_policy, evaluate_policy
from peg.policy_engine import build_policy_input, evaluate_inline_policy, extract_policy_bundle


class TestInlinePolicy:
    def test_threshold_compatibility_helper_remains_backward_compatible(self):
        assert _inline_policy(0.30, 0.50, 0.75, 0.90) == 'ALLOW'
        assert _inline_policy(0.60, 0.50, 0.75, 0.90) == 'WARN'
        assert _inline_policy(0.80, 0.50, 0.75, 0.90) == 'AUTO_OPTIMISE'
        assert _inline_policy(0.95, 0.50, 0.75, 0.90) == 'BLOCK'

    def test_structured_inline_policy_blocks_pr_prod_deploys(self):
        payload = build_policy_input(
            {'crs': 0.40, 'billed_cost': 0.01},
            {
                'run_id': 'tt_1',
                'stage_name': 'deploy_prod',
                'branch': 'main',
                'domain': 'real',
                'gh_is_pr': True,
                'gh_by_core_team_member': False,
            },
            extract_policy_bundle(FAKE_POLICY),
        )
        result = evaluate_inline_policy(payload)
        assert result['decision'] == 'BLOCK'
        assert 'block_pr_prod_deploys' in result['matched_rules']
        assert any('pull-request' in reason.lower() for reason in result['reasons'])
        assert result['actions']

    def test_structured_inline_policy_warns_on_cost_ceiling(self):
        payload = build_policy_input(
            {'crs': 0.20, 'billed_cost': 0.20},
            {
                'run_id': 'bb_1',
                'stage_name': 'build',
                'branch': 'feature/demo',
                'domain': 'bitbrains',
                'gh_is_pr': False,
                'gh_by_core_team_member': True,
            },
            extract_policy_bundle(FAKE_POLICY),
        )
        result = evaluate_inline_policy(payload)
        assert result['decision'] == 'WARN'
        assert 'stage_cost_ceiling_usd' in result['matched_rules']


class TestOPAClientFallback:
    @pytest.mark.asyncio
    async def test_opa_unreachable_uses_inline_fallback(self):
        class BrokenAsyncClient:
            def __init__(self, *args, **kwargs):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, tb):
                return False

            async def post(self, *args, **kwargs):
                import httpx

                raise httpx.ConnectError('offline')

        with patch('peg.opa_client.httpx.AsyncClient', BrokenAsyncClient):
            result = await evaluate_policy(
                metrics={'crs': 0.92, 'billed_cost': 0.02},
                context={
                    'run_id': 'tt_2',
                    'stage_name': 'integration_test',
                    'branch': 'main',
                    'domain': 'real',
                    'gh_is_pr': False,
                    'gh_by_core_team_member': True,
                },
                policy_bundle=extract_policy_bundle(FAKE_POLICY),
            )
        assert result['decision'] == 'BLOCK'
        assert result['policy_source'] == 'inline'
        assert result['reasons']


class TestPolicyApi:
    @pytest.mark.asyncio
    async def test_get_policy_returns_structured_bundle(self, test_app, mock_conn_and_pool, valid_jwt_headers):
        conn, _ = mock_conn_and_pool
        conn.fetchrow.side_effect = [
            MockRecord(FAKE_USER),
            MockRecord(FAKE_POLICY),
        ]

        resp = await test_app.get('/api/policy', headers=valid_jwt_headers)
        assert resp.status_code == 200
        body = resp.json()
        assert 'policy_bundle' in body
        assert body['policy_bundle']['version'] == 'v17.0'
        assert body['policy_bundle']['thresholds']['warn_threshold'] == pytest.approx(0.50)

    @pytest.mark.asyncio
    async def test_put_policy_accepts_governance_bundle(self, test_app, mock_conn_and_pool, valid_jwt_headers):
        conn, _ = mock_conn_and_pool
        conn.fetchrow.return_value = MockRecord(FAKE_USER)
        conn.execute.return_value = 'UPDATE 1'

        resp = await test_app.put(
            '/api/policy',
            headers=valid_jwt_headers,
            json={
                'warn_threshold': 0.55,
                'auto_optimise_threshold': 0.78,
                'block_threshold': 0.93,
                'policy_bundle': {
                    'version': 'v17.0',
                    'rules': {
                        'protected_branches': ['main', 'release'],
                        'sensitive_stages': ['deploy_prod'],
                        'block_pr_prod_deploys': True,
                        'require_core_team_for_sensitive_stages': True,
                        'stage_cost_ceiling_usd': {'deploy_prod': 0.10},
                    },
                },
            },
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body['policy_bundle']['thresholds']['warn_threshold'] == pytest.approx(0.55)
        assert 'deploy_prod' in body['policy_bundle']['rules']['sensitive_stages']

    @pytest.mark.asyncio
    async def test_peg_decision_returns_governance_metadata(self, test_app, mock_conn_and_pool, valid_jwt_headers):
        conn, _ = mock_conn_and_pool
        conn.fetchrow.side_effect = [
            MockRecord(FAKE_USER),
            MockRecord(FAKE_POLICY),
            MockRecord({'email': 'owner@costguard.dev'}),
        ]
        conn.execute.return_value = 'UPDATE 1'

        with patch('peg.router.send_slack_alert', AsyncMock()), \
             patch('peg.router.send_email_alert', AsyncMock()), \
             patch('pade.inference.generate_anomaly_recommendation', AsyncMock(return_value='Use a protected-branch deploy gate.')):
            resp = await test_app.post(
                '/api/decision',
                headers=valid_jwt_headers,
                json={
                    'run_id': 'tt_3',
                    'stage_name': 'deploy_prod',
                    'crs': 0.35,
                    'billed_cost': 0.01,
                    'duration_seconds': 120.0,
                    'latency_p95': 1000.0,
                    'executor_type': 'github_actions',
                    'branch': 'main',
                    'domain': 'real',
                    'gh_is_pr': True,
                    'gh_by_core_team_member': False,
                    'attribution_snapshot': {},
                },
            )
        assert resp.status_code == 200
        body = resp.json()
        assert body['decision'] == 'BLOCK'
        assert body['reasons']
        assert body['matched_rules']
        assert body['actions']
        assert body['policy_source'] in {'inline', 'opa'}

    @pytest.mark.asyncio
    async def test_policy_routes_require_authentication(self, test_app):
        get_resp = await test_app.get('/api/policy')
        put_resp = await test_app.put(
            '/api/policy',
            json={
                'warn_threshold': 0.55,
                'auto_optimise_threshold': 0.78,
                'block_threshold': 0.93,
                'policy_bundle': {},
            },
        )
        decision_resp = await test_app.post(
            '/api/decision',
            json={
                'run_id': 'tt_3',
                'stage_name': 'deploy_prod',
                'crs': 0.35,
                'billed_cost': 0.01,
                'duration_seconds': 120.0,
                'latency_p95': 1000.0,
                'executor_type': 'github_actions',
                'branch': 'main',
                'domain': 'real',
                'gh_is_pr': True,
                'gh_by_core_team_member': False,
                'attribution_snapshot': {},
            },
        )
        assert get_resp.status_code == 401
        assert put_resp.status_code == 401
        assert decision_resp.status_code == 401

    @pytest.mark.asyncio
    async def test_policy_routes_require_admin_role(self, test_app, mock_conn_and_pool, valid_jwt_headers):
        conn, _ = mock_conn_and_pool
        conn.fetchrow.return_value = MockRecord(FAKE_VIEWER_USER)

        get_resp = await test_app.get('/api/policy', headers=valid_jwt_headers)
        put_resp = await test_app.put(
            '/api/policy',
            headers=valid_jwt_headers,
            json={
                'warn_threshold': 0.55,
                'auto_optimise_threshold': 0.78,
                'block_threshold': 0.93,
                'policy_bundle': {},
            },
        )
        decision_resp = await test_app.post(
            '/api/decision',
            headers=valid_jwt_headers,
            json={
                'run_id': 'tt_3',
                'stage_name': 'deploy_prod',
                'crs': 0.35,
                'billed_cost': 0.01,
                'duration_seconds': 120.0,
                'latency_p95': 1000.0,
                'executor_type': 'github_actions',
                'branch': 'main',
                'domain': 'real',
                'gh_is_pr': True,
                'gh_by_core_team_member': False,
                'attribution_snapshot': {},
            },
        )
        assert get_resp.status_code == 403
        assert put_resp.status_code == 403
        assert decision_resp.status_code == 403
