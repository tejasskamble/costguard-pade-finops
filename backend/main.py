"""
backend/main.py - CostGuard v17.0 Enterprise Edition.

Registers the active FastAPI routers, applies required database migrations,
and boots the enterprise services used by the canonical 3-domain CostGuard stack.
"""
import asyncio
import logging
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

from api.alerts import router as alerts_router
from api.attribution import router as attribution_router
from api.auth import router as auth_router
from api.budget import router as budget_router
from api.continual import router as continual_router
from api.ingest import pcam_attribution_loop, router as ingest_router
from api.jobs import get_worker_state, router as jobs_router, start_worker
from api.metrics import metrics_middleware, router as metrics_router
from api.notifications import router as notif_router
from api.pade_training import router as pade_train_router
from api.policy import router as policy_router
from api.postrun import router as postrun_router
from api.providers import router as providers_router
from api.support import router as support_router
from api.users import router as users_router
from cache import invalidate_all
from config import settings
from database import close_pool, create_pool
from lcqi.router import limiter, router as lcqi_router
from pade.inference import (
    bootstrap_gat_checkpoint,
    get_pade_runtime_status,
    router as pade_router,
)
from peg.router import router as peg_router
from runtime_hardening import (
    configure_backend_runtime,
    install_asyncio_exception_handler,
    install_backend_exception_hooks,
    retry_async,
    safe_create_task,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s - %(message)s',
)
logger = logging.getLogger(__name__)
configure_backend_runtime()
install_backend_exception_hooks(logger)

MIGRATIONS_DIR = Path(__file__).parent.parent / 'migrations'
MIGRATIONS = [
    '001_init_focus_schema.sql',
    '002_add_notify_trigger.sql',
    '003_add_query_history.sql',
    '004_new_features.sql',
    '005_policy_governance.sql',
    '006_ieee_postrun_artifacts.sql',
    '007_postrun_continual_readiness.sql',
]


async def _run_migration_file(pool, filename: str) -> None:
    """Execute a single SQL migration file."""
    path = MIGRATIONS_DIR / filename
    if not path.exists():
        logger.warning('Migration file not found: %s', path)
        return
    sql = path.read_text(encoding='utf-8')
    async with pool.acquire() as conn:
        try:
            await conn.execute(sql)
            logger.info('Migration applied: %s', filename)
        except Exception as exc:
            logger.exception('Migration failed for %s: %s', filename, exc)
            raise


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.db = None
    app.state.pcam_task = None
    app.state.job_task = None
    loop = asyncio.get_running_loop()
    install_asyncio_exception_handler(loop, logger)

    settings.validate_runtime_requirements()

    try:
        logger.info('Creating DB connection pool...')
        pool = await retry_async(
            create_pool,
            attempts=3,
            delay=1.0,
            logger=logger,
            label='database pool creation',
        )
        app.state.db = pool

        logger.info('Running database migrations...')
        for migration in MIGRATIONS:
            await _run_migration_file(pool, migration)
    except Exception as exc:
        logger.exception('Database startup degraded; API will continue without a live pool: %s', exc)
        pool = None

    try:
        logger.info('Bootstrapping PADE checkpoint...')
        bootstrap_gat_checkpoint()
    except Exception as exc:
        logger.exception('PADE checkpoint bootstrap failed; API continuing in degraded mode: %s', exc)

    logger.info('Clearing caches...')
    invalidate_all()

    if pool is not None:
        logger.info('Starting PCAM attribution loop...')
        app.state.pcam_task = safe_create_task(
            pcam_attribution_loop(pool),
            logger=logger,
            label='PCAM attribution loop',
        )

        logger.info('Starting background job worker...')
        try:
            app.state.job_task = start_worker(pool)
        except Exception as exc:
            logger.exception('Background job worker failed to start: %s', exc)

    logger.info('CostGuard v17.0 Enterprise API ready.')
    yield

    for task_name, task_attr in [('PCAM loop', 'pcam_task'), ('Job worker', 'job_task')]:
        task = getattr(app.state, task_attr, None)
        if task:
            logger.info('Cancelling %s...', task_name)
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

    logger.info('Closing DB pool...')
    await close_pool(getattr(app.state, 'db', None))
    logger.info('Shutdown complete.')


app = FastAPI(
    title='CostGuard Enterprise API',
    description=(
        'Automated cloud cost monitoring, anomaly detection, and FinOps intelligence.\n\n'
        'Active v17.0 capabilities:\n'
        '- Canonical 3-domain PADE pipeline: Synthetic -> TravisTorrent -> BitBrains\n'
        '- Structured OPA governance with inline parity fallback\n'
        '- Unified IEEE analytics, BWT tracking, and paper-ready exports\n'
        '- Enterprise support, users, notifications, and training orchestration\n'
    ),
    version='17.0.0-enterprise',
    lifespan=lifespan,
    docs_url='/docs',
    redoc_url='/redoc',
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)

app.middleware('http')(metrics_middleware)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


@app.exception_handler(Exception)
async def unhandled_exception_handler(request, exc: Exception):
    logger.exception('Unhandled request error on %s %s: %s', request.method, request.url.path, exc)
    return JSONResponse(
        status_code=500,
        content={
            'detail': 'Internal server error',
            'service': 'costguard-api',
            'status': 'degraded',
        },
    )

app.include_router(auth_router)
app.include_router(ingest_router)
app.include_router(metrics_router)
app.include_router(attribution_router)
app.include_router(alerts_router)
app.include_router(policy_router)
app.include_router(peg_router)
app.include_router(lcqi_router)
app.include_router(pade_router)
app.include_router(budget_router)
app.include_router(jobs_router)
app.include_router(providers_router)
app.include_router(support_router)
app.include_router(users_router)
app.include_router(notif_router)
app.include_router(pade_train_router)
app.include_router(postrun_router)
app.include_router(continual_router)


@app.get('/health', tags=['health'])
async def health():
    """Health check endpoint."""
    def _task_state(task) -> str:
        if task is None:
            return 'stopped'
        done_attr = getattr(task, 'done', None)
        if callable(done_attr):
            try:
                done_value = done_attr()
                if isinstance(done_value, bool):
                    return 'stopped' if done_value else 'running'
            except Exception:
                logger.debug('Health task probe failed for %s', task, exc_info=True)
        return 'running'

    db_pool = getattr(app.state, 'db', None)
    db_status = 'unavailable'
    if db_pool is not None:
        try:
            async with db_pool.acquire() as conn:
                await conn.execute('SELECT 1')
            db_status = 'ready'
        except Exception as exc:
            logger.warning('Health DB probe failed: %s', exc)
            db_status = 'degraded'

    pade_status = get_pade_runtime_status()
    worker_state = get_worker_state()
    pcam_task = getattr(app.state, 'pcam_task', None)
    job_task = getattr(app.state, 'job_task', None)

    components = {
        'database': db_status,
        'pade': pade_status.get('status', 'degraded'),
        'worker': 'running' if worker_state.get('running') else 'stopped',
        'pcam_loop': _task_state(pcam_task),
        'job_task': _task_state(job_task),
    }
    overall_ok = components['database'] == 'ready' and pade_status.get('inference_ready', False)
    return {
        'status': 'ok' if overall_ok else 'degraded',
        'service': 'costguard-api',
        'version': '17.0.0-enterprise',
        'domains': ['synthetic', 'travistorrent', 'bitbrains'],
        'governance': 'opa+inline-fallback',
        'database': db_status,
        'components': components,
        'worker': worker_state,
        'pade': pade_status,
    }


if __name__ == '__main__':
    import uvicorn

    uvicorn.run(
        'main:app',
        host='0.0.0.0',
        port=settings.API_PORT,
        log_level='info',
    )
