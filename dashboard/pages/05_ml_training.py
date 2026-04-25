"""CostGuard ML Training Lab dashboard page."""
import time

import pandas as pd
import streamlit as st

from utils.api_client import (
    get_baseline_report,
    get_training_history,
    get_training_status,
    is_authenticated,
    log_page_visit,
    start_pade_training,
)

if not is_authenticated():
    st.error('Please login to access this page.')
    st.stop()

log_page_visit('05_ml_training')

st.markdown(
    """
<style>
.cg-page-header{background:linear-gradient(135deg,rgba(255,107,53,.08) 0%,rgba(44,62,122,.06) 100%);
  border:1px solid rgba(255,107,53,.15);border-radius:16px;padding:24px 28px;margin-bottom:24px;}
.cg-page-header h1{font-family:'Syne',sans-serif;font-size:1.9rem;font-weight:800;color:#fff;margin:0;}
.cg-page-header p{color:#6B7A99;margin:6px 0 0;}
.job-card{background:#0D1B2E;border:1px solid rgba(255,107,53,.15);border-radius:12px;padding:16px 20px;margin:8px 0;}
</style>
""",
    unsafe_allow_html=True,
)

st.markdown(
    """
<div class="cg-page-header">
  <h1>ML Training Lab</h1>
  <p>Train the canonical CostGuard v17.0 PADE stack for the 3-domain IEEE workflow: Synthetic -> TravisTorrent -> BitBrains.</p>
</div>
""",
    unsafe_allow_html=True,
)

tab1, tab2, tab3 = st.tabs(['Train Model', 'Training History', 'Baseline Comparison'])

with tab1:
    col1, col2 = st.columns([1, 1])
    with col1:
        st.markdown('#### Training Configuration')
        mode = st.selectbox(
            'Training Mode',
            ['full', 'lstm', 'gat', 'baseline'],
            format_func=lambda value: {
                'full': 'Full (LSTM + GAT + Ensemble)',
                'lstm': 'LSTM Only',
                'gat': 'GAT Only',
                'baseline': 'Baseline Comparison',
            }[value],
        )
        synth_mode = st.selectbox(
            'Synthetic Data Mode',
            ['enhanced', 'legacy'],
            format_func=lambda value: {
                'enhanced': 'TT+BB Calibrated v17.0',
                'legacy': 'Legacy Seed-42 Compatibility',
            }[value],
        )
        epochs = st.slider('Training Epochs', 5, 150, 30, 5)
        n_rows = st.number_input('Synthetic Rows', 1000, 100000, 10000, 1000)

    with col2:
        st.markdown('#### Model Architecture')
        st.markdown(
            """
        <div style="background:#111D35;border-radius:12px;padding:20px;">
          <div style="color:#FF6B35;font-weight:700;margin-bottom:12px;">PADE v17.0 Stack</div>
          <div style="color:#B0BDD0;font-size:.85rem;line-height:1.8;">
            <strong style="color:#E8F0FE;">C4 BahdanauBiLSTM</strong> - residual 3-layer temporal anomaly model<br>
            <strong style="color:#E8F0FE;">C5 GATv2Pipeline</strong> - graph attention model with mean+max pooling<br>
            <strong style="color:#E8F0FE;">Ensemble</strong> - validation-tuned F1@opt reporting on the locked test split<br>
            <strong style="color:#E8F0FE;">Lifelong Order</strong> - D0 Synthetic -> L1 TravisTorrent -> L2 BitBrains<br>
            <strong style="color:#E8F0FE;">Governance</strong> - OPA + inline policy parity for cost-control actions
          </div>
        </div>
        """,
            unsafe_allow_html=True,
        )

        est = {'full': '~15 min', 'lstm': '~5 min', 'gat': '~8 min', 'baseline': '~3 min'}
        st.info(f'Estimated duration: **{est[mode]}**')

    st.markdown('---')
    if st.button('Start Training', use_container_width=True):
        config = {
            'mode': mode,
            'synth_mode': synth_mode,
            'epochs': epochs,
            'n_synthetic_rows': int(n_rows),
        }
        with st.spinner('Launching training job...'):
            result = start_pade_training(config)

        if result and result.get('job_id'):
            job_id = result['job_id']
            st.success(f"Job #{job_id} launched. Estimated: {result.get('estimated_duration_minutes', '?')} min")
            st.session_state['active_job_id'] = job_id

    if 'active_job_id' in st.session_state:
        job_id = st.session_state['active_job_id']
        st.markdown(f'#### Live Progress - Job #{job_id}')
        status_placeholder = st.empty()
        prog_placeholder = st.empty()

        with st.spinner(f'Polling job #{job_id}...'):
            for _ in range(60):
                status = get_training_status(job_id)
                if status:
                    prog = status.get('progress', 0)
                    job_status = status.get('status', 'running')
                    prog_placeholder.progress(prog / 100, text=f'Progress: {prog}%')
                    status_placeholder.markdown(
                        f"""
                    <div class="job-card">
                      <div style="display:flex;gap:20px;">
                        <div><span style="color:#6B7A99;font-size:.8rem;">STATUS</span>
                          <div style="color:#FF6B35;font-weight:700;">{job_status.upper()}</div></div>
                        <div><span style="color:#6B7A99;font-size:.8rem;">PROGRESS</span>
                          <div style="color:#E8F0FE;font-weight:700;">{prog}%</div></div>
                        <div><span style="color:#6B7A99;font-size:.8rem;">MODE</span>
                          <div style="color:#E8F0FE;font-weight:700;">{status.get('job_type', '-').upper()}</div></div>
                      </div>
                    </div>""",
                        unsafe_allow_html=True,
                    )
                    if job_status in ('done', 'failed'):
                        if job_status == 'done':
                            st.success('Training completed successfully.')
                        else:
                            result_json = status.get('result_json', {})
                            err = result_json.get('error', 'Unknown') if isinstance(result_json, dict) else str(result_json)
                            st.error(f'Training failed: {err}')
                        del st.session_state['active_job_id']
                        break
                time.sleep(3)

with tab2:
    st.markdown('#### Training Job History')
    if st.button('Refresh History'):
        st.rerun()
    with st.spinner('Loading history...'):
        history = get_training_history()
    if history:
        df = pd.DataFrame(history)
        if 'started_at' in df.columns:
            df['started_at'] = pd.to_datetime(df['started_at']).dt.strftime('%Y-%m-%d %H:%M')
        if 'duration_seconds' in df.columns:
            df['duration'] = df['duration_seconds'].apply(
                lambda value: f"{int(value)//60}m {int(value)%60}s" if value else '-'
            )
        display_cols = ['id', 'job_type', 'status', 'progress', 'started_at', 'duration']
        display_cols = [column for column in display_cols if column in df.columns]
        st.dataframe(df[display_cols], use_container_width=True, hide_index=True)
    else:
        st.info('No training jobs found. Start your first training run above.')

with tab3:
    st.markdown('#### PADE vs Baseline Model Comparison')
    st.markdown(
        """
    <div style="background:#111D35;border-radius:12px;padding:16px 20px;margin-bottom:16px;">
      <div style="color:#B0BDD0;font-size:.88rem;line-height:1.6;">
        Run a <strong style="color:#FF6B35;">baseline</strong> training job to compare PADE scores
        against classical detectors while preserving the canonical 3-domain protocol.
      </div>
    </div>
    """,
        unsafe_allow_html=True,
    )

    with st.spinner('Loading baseline report...'):
        report = get_baseline_report()

    if report and 'result' in report:
        result_data = report['result']
        st.success(f"Baseline computed at {report.get('computed_at', 'N/A')}")
        if isinstance(result_data, dict):
            st.json(result_data)
    else:
        st.info("No baseline results found. Start a 'Baseline Comparison' training job.")
        if st.button('Run Baseline Now'):
            result = start_pade_training({'mode': 'baseline'})
            if result:
                st.session_state['active_job_id'] = result.get('job_id')
                st.success(f"Baseline job #{result.get('job_id')} started")
                st.rerun()
