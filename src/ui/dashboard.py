"""
Ashva Master Observability Dashboard
Unified observability suite providing multi-tab inspection for Data, Alpha Factory, Trading, and System.
"""

import streamlit as st
import pandas as pd
from src.ui.data_access import UIDataAccess

st.set_page_config(
    page_title="Ashva Observability Hub",
    page_icon="🐎",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize Data Access Layer
def get_dal() -> UIDataAccess:
    return UIDataAccess()

@st.cache_data(ttl=5)
def load_trading_data():
    dal = get_dal()
    df_replay = dal.get_trading_state("REPLAY")
    df_paper = dal.get_trading_state("PAPER")
    df_live = dal.get_trading_state("LIVE")
    df_diagnostics = dal.get_replay_diagnostics()
    return df_replay, df_paper, df_live, df_diagnostics


# =============================================================================
# TAB 1: DATA OBSERVABILITY COMPONENT
# =============================================================================

def render_data_observability(dal: UIDataAccess):
    st.title("Data Lake Observability")
    st.caption("Authoritative inspection of raw market data, timeframe coverage, 540-day research horizon, and data hygiene.")

    # 1. System Overview Metrics
    overview = dal.get_data_overview()
    quality = dal.get_data_quality_summary()

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Universe Symbols", f"{overview['total_symbols']} Equities")
    col2.metric("Total OHLCV Bars", f"{overview['total_bars']:,}")
    col3.metric("Available Timeframes", f"{len(overview['available_timeframes'])} TFs", help=", ".join(overview['available_timeframes']))
    col4.metric("Research Horizon (540d)", f"{quality['symbols_with_540d_coverage']}/{overview['total_symbols']} Symbols", help="Symbols with >= 540 calendar days of 15m data")
    col5.metric("Data Quality Status", quality['quality_status'])

    st.markdown("---")

    subtab_matrix, subtab_detail, subtab_hygiene, subtab_ingestion, subtab_alpha_conn = st.tabs([
        "📊 Coverage Matrix",
        "🔍 Symbol Deep Dive",
        "🛡️ Quality & Hygiene Audit",
        "📥 Ingestion Telemetry",
        "🔗 Data → Alpha Mapping"
    ])

    # --- SUBTAB 1: COVERAGE MATRIX ---
    with subtab_matrix:
        st.subheader("Symbol / Timeframe Coverage Matrix")
        st.caption("Point-in-time bar counts and 540-day horizon compliance across DuckDB & Apache Parquet.")

        df_matrix = dal.get_coverage_matrix()
        if not df_matrix.empty:
            search_sym = st.text_input("Filter by Symbol Name", placeholder="e.g. INFY, TCS, RELIANCE...", key="search_matrix")
            df_disp = df_matrix
            if search_sym:
                df_disp = df_disp[df_disp["symbol"].str.contains(search_sym.upper(), case=False, na=False)]
            
            st.dataframe(df_disp, use_container_width=True, hide_index=True)
        else:
            st.warning("No market data discovered in DataLake.")

    # --- SUBTAB 2: SYMBOL DEEP DIVE ---
    with subtab_detail:
        st.subheader("Symbol Detail & Timeframe Inspector")
        symbols = dal.get_symbol_list()
        
        if symbols:
            selected_sym = st.selectbox("Select Symbol for Point-in-Time Inspection", symbols, index=0, key="select_data_symbol")
            detail = dal.get_symbol_detail(selected_sym)

            if detail:
                c1, c2, c3 = st.columns(3)
                c1.metric("Selected Symbol", detail["symbol"])
                c1.write(f"**Primary Storage**: {detail['data_source']}")
                
                q_metrics = detail["quality_metrics"]
                c2.metric("Duplicate Bars", q_metrics["duplicate_bars"])
                c2.metric("Invalid OHLC Bars", q_metrics["invalid_ohlc_bars"])
                
                c3.metric("Out-of-Market-Hours Bars", q_metrics["out_of_market_hours_bars"])
                c3.write(f"**Holiday Calendar Audit**: `{q_metrics['missing_bars_calendar_audit']}`")

                st.write("#### Available Timeframes Breakdown")
                df_tf_detail = pd.DataFrame(detail["timeframes_detail"])
                st.dataframe(df_tf_detail, use_container_width=True, hide_index=True)
            else:
                st.info(f"No detail available for {selected_sym}.")
        else:
            st.warning("No symbols available in DataLake.")

    # --- SUBTAB 3: QUALITY & HYGIENE AUDIT ---
    with subtab_hygiene:
        st.subheader("Repository-Wide Data Quality Audit")
        st.caption("Automated structural sanity checks executed across all stored time series in DuckDB.")

        q1, q2, q3, q4 = st.columns(4)
        q1.metric("Total Bars Audited", f"{quality['total_bars_audited']:,}")
        q2.metric("Duplicate Timestamps", quality['duplicate_bars'])
        q3.metric("Invalid Price/Vol Bars", quality['invalid_ohlc_bars'])
        q4.metric("Out-of-Hours Bars", quality['out_of_hours_intraday_bars'])

        st.info(f"**Audit Timestamp**: {quality['last_audit_time']} | **Result**: {quality['quality_status']}")
        
        st.markdown("""
        **Institutional Data Quality Verification Rules**:
        1. **Zero Duplicate Timestamps**: Stored series contains unique `(symbol, timeframe, timestamp)` composite keys.
        2. **Valid OHLC Bounds**: Strict validation ($High \\ge Low$, $Open > 0$, $Close > 0$, $Volume \\ge 0$).
        3. **Strict Intraday Hours**: Intraday bars must strictly fall between 09:15:00 and 15:30:00 IST.
        4. **540-Day Research Horizon**: Lookback availability satisfies the 18-month statistical robustness threshold.
        """)

    # --- SUBTAB 4: INGESTION TELEMETRY ---
    with subtab_ingestion:
        st.subheader("Ingestion Status & Log Telemetry")
        st.caption("Observability around historical ingestion jobs, active sessions, and data storage files.")

        c_ing1, c_ing2 = st.columns(2)
        c_ing1.write(f"**Data Lake File**: `{overview['db_path']}`")
        c_ing1.write(f"**Database Size**: `{overview['db_size_mb']} MB`")
        c_ing1.write(f"**Last File Modification**: `{overview['last_updated']}`")
        c_ing1.write(f"**Earliest Global Timestamp**: `{overview['earliest_timestamp']}`")
        c_ing1.write(f"**Latest Global Timestamp**: `{overview['latest_timestamp']}`")

        c_ing2.write("**Ingestion Provider**: Angel One SmartAPI / DuckDB Historical Store")
        c_ing2.write("**Parquet Mirror Directory**: `data_lake/parquet/`")
        c_ing2.write("**Incremental Ingestion Engine**: `scripts/ingest_nifty50_to_today.py` (Available)")

        st.write("#### Discovered Session Logs (`logs/`)")
        df_logs = dal.get_ingestion_log_summary()
        if not df_logs.empty:
            st.dataframe(df_logs, use_container_width=True, hide_index=True)
        else:
            st.info("No session log files discovered in logs/ directory.")

    # --- SUBTAB 5: DATA -> ALPHA CONNECTION ---
    with subtab_alpha_conn:
        st.subheader("Data Lake → Alpha Factory Requirements Mapping")
        st.caption("Bridges research hypotheses requirements with live Data Lake availability.")

        df_conn = dal.get_alpha_data_connection()
        if not df_conn.empty:
            search_alpha = st.text_input("Filter by Alpha ID or Mechanism", placeholder="e.g. ALPHA_86, ALPHA_70...", key="search_alpha_conn")
            df_disp_conn = df_conn
            if search_alpha:
                df_disp_conn = df_disp_conn[
                    df_disp_conn["Alpha ID"].str.contains(search_alpha, case=False, na=False) |
                    df_disp_conn["Mechanism"].str.contains(search_alpha, case=False, na=False)
                ]
            st.dataframe(df_disp_conn, use_container_width=True, hide_index=True)
        else:
            st.info("No Alpha mechanisms found in AlphaKnowledgeMap.")


# =============================================================================
# TAB 2: ALPHA FACTORY OBSERVABILITY COMPONENT
# =============================================================================

def render_alpha_factory(dal: UIDataAccess):
    st.title("Alpha Factory Observability")
    st.caption("Authoritative governance, empirical audit records, qualification gates, and statistical lifecycle tracking.")

    # 1. Section 1: Factory Summary KPI Metrics
    summary = dal.get_alpha_factory_summary()
    kpi1, kpi2, kpi3, kpi4, kpi5, kpi6 = st.columns(6)
    kpi1.metric("Total Alphas", summary["total_alphas"])
    kpi2.metric("Tested", f"{summary['tested']} / {summary['total_alphas']}")
    kpi3.metric("PROVEN (Capital)", summary["proven"])
    kpi4.metric("FAILED (Rejected)", summary["failed"])
    kpi5.metric("UNCERTAIN (Watchlist)", summary["uncertain"])
    kpi6.metric("UNEXPLORED", summary["unexplored"])

    st.markdown("---")

    # 2. Section 2: Master Alpha Registry Table with Filters
    st.subheader("Master Alpha Registry")
    
    col_f1, col_f2, col_f3, col_f4 = st.columns(4)
    with col_f1:
        status_filter = st.selectbox("Filter by Status", ["ALL", "PROVEN", "FAILED", "UNCERTAIN", "UNEXPLORED"], index=0, key="filter_alpha_status")
    with col_f2:
        tested_filter = st.selectbox("Tested / Untested", ["ALL", "YES", "NO"], index=0, key="filter_alpha_tested")
    with col_f3:
        category_filter = st.selectbox("Category Filter", ["ALL", "MOMENTUM", "VOLATILITY_EXPANSION", "OPENING_AUCTION", "ORDER_FLOW_IMBALANCE", "STATISTICAL_REVERSION"], index=0, key="filter_alpha_category")
    with col_f4:
        search_query = st.text_input("Search Alpha ID / Name / Rationale", placeholder="e.g. alpha_86, DOUBLE_INSIDE, ORB...", key="search_alpha_registry")

    df_registry = dal.get_alpha_registry_table()

    if not df_registry.empty:
        df_disp = df_registry.copy()
        
        # Apply filters
        if status_filter != "ALL":
            df_disp = df_disp[df_disp["status"] == status_filter]
        if tested_filter != "ALL":
            df_disp = df_disp[df_disp["tested"] == tested_filter]
        if category_filter != "ALL":
            df_disp = df_disp[df_disp["category"].str.contains(category_filter, case=False, na=False)]
        if search_query:
            df_disp = df_disp[
                df_disp["alpha_id"].str.contains(search_query, case=False, na=False) |
                df_disp["name"].str.contains(search_query, case=False, na=False) |
                df_disp["category"].str.contains(search_query, case=False, na=False)
            ]

        # Display Registry Table
        st.dataframe(
            df_disp,
            use_container_width=True,
            hide_index=True,
            column_config={
                "alpha_id": st.column_config.TextColumn("Alpha ID", width="small"),
                "name": st.column_config.TextColumn("Strategy Name", width="medium"),
                "status": st.column_config.TextColumn("Status", width="small"),
                "sharpe": st.column_config.TextColumn("Sharpe (IS)", width="small"),
                "net_profit_factor": st.column_config.TextColumn("Net PF", width="small"),
                "oos_sharpe": st.column_config.TextColumn("OOS Sharpe", width="small"),
                "max_drawdown_pct": st.column_config.TextColumn("Max DD", width="small"),
                "positive_symbols": st.column_config.TextColumn("Positive Assets", width="medium"),
                "last_tested": st.column_config.TextColumn("Last Tested", width="medium"),
            }
        )
    else:
        st.warning("No Alpha strategies discovered in registry.")

    st.markdown("---")

    # 3. Section 3-12: Alpha Deep Dive Inspector
    st.subheader("Alpha Hypothesis Deep Dive & Audit Inspector")
    
    alpha_options = list(df_registry["alpha_id"].values) if not df_registry.empty else ["alpha_86"]
    selected_alpha = st.selectbox("Select Alpha ID for Full Quantitative Audit", alpha_options, index=0, key="select_alpha_detail")

    detail = dal.get_alpha_detail(selected_alpha)

    if detail:
        # Header Badge & Metadata
        st.markdown(f"### `{detail['alpha_id'].upper()}`: {detail['name']} — Status: `{detail['status']}`")
        st.caption(f"Category: **{detail['category']}** | Timeframe: **{detail['timeframe']}** | Version: **{detail['version']}** | Tested: **{'YES' if detail['is_tested'] else 'NO'}**")

        detail_tabs = st.tabs([
            "🎯 Hypothesis & Rationale",
            "🛡️ Qualification Gates",
            "📈 Performance Metrics",
            "🏢 Symbol-Level Audit",
            "🔬 Research Evidence & 540d",
            "📜 Test History Journal"
        ])

        # SUBTAB 1: HYPOTHESIS & RATIONALE
        with detail_tabs[0]:
            st.markdown("#### Economic Rationale & Mechanism")
            st.write(detail["hypothesis"])
            
            st.markdown("#### Market Mechanism Description")
            st.info(detail["mechanism"])

            col_p1, col_p2 = st.columns(2)
            with col_p1:
                st.markdown("#### Entry & Holding Rules")
                st.write(f"**Entry Window**: `{detail['entry_window']}`")
                st.write(f"**Holding Concept**: `{detail['holding_concept']}`")
                st.write(f"**Target Universe**: `{', '.join(detail['target_instruments'][:8])}...`")
            with col_p2:
                st.markdown("#### Parameter Specifications")
                if detail["parameters"]:
                    st.json(detail["parameters"])
                else:
                    st.write("No parameters defined (Default baseline).")

        # SUBTAB 2: QUALIFICATION GATES
        with detail_tabs[1]:
            st.markdown("#### Institutional Qualification Gates Evaluation")
            gates = detail["qualification_gates"]
            
            g_col1, g_col2, g_col3, g_col4 = st.columns(4)
            
            g1 = gates.get("gate_1_dsr", {})
            g_col1.metric(g1.get("name", "DSR Test"), g1.get("value", "N/A"), delta="PASS" if g1.get("passed") else "FAIL", delta_color="normal" if g1.get("passed") else "inverse")
            g_col1.caption(f"Hurdle: {g1.get('threshold', 'N/A')}")

            g2 = gates.get("gate_2_cpcv", {})
            g_col2.metric(g2.get("name", "CPCV OOS Quality"), g2.get("value", "N/A"), delta="PASS" if g2.get("passed") else "FAIL", delta_color="normal" if g2.get("passed") else "inverse")
            g_col2.caption(f"Hurdle: {g2.get('threshold', 'N/A')}")

            g3 = gates.get("gate_3_mc_tail", {})
            g_col3.metric(g3.get("name", "Monte Carlo 5000 DD"), g3.get("value", "N/A"), delta="PASS" if g3.get("passed") else "FAIL", delta_color="normal" if g3.get("passed") else "inverse")
            g_col3.caption(f"Tolerance: {g3.get('threshold', 'N/A')}")

            g4 = gates.get("gate_4_net_pf", {})
            g_col4.metric(g4.get("name", "Post-Tax Net PF"), g4.get("value", "N/A"), delta="PASS" if g4.get("passed") else "FAIL", delta_color="normal" if g4.get("passed") else "inverse")
            g_col4.caption(f"Hurdle: {g4.get('threshold', 'N/A')}")

            st.markdown("#### Status Explanation & Institutional Rationale")
            st.success(detail["explanations"]["status_reason"]) if detail["status"] == "PROVEN" else st.error(detail["explanations"]["status_reason"])

            if detail["explanations"]["failure_lessons"] != "N/A":
                st.warning(f"**Failure Lessons & Empirical Observations**: {detail['explanations']['failure_lessons']}")
            if detail["explanations"]["known_limitations"]:
                st.write(f"**Known Regime Limitations**: {detail['explanations']['known_limitations']}")

        # SUBTAB 3: PERFORMANCE METRICS
        with detail_tabs[2]:
            st.markdown("#### Persisted Research & In-Sample Metrics")
            m = detail["metrics"]
            
            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("In-Sample Sharpe", f"{m.get('in_sample_sharpe', 0.0):+.2f}")
            m2.metric("CPCV OOS Sharpe", f"{m.get('cpcv_oos_sharpe', 0.0):+.2f}")
            m3.metric("Net Profit Factor", f"{m.get('net_profit_factor', 0.0):.2f}")
            m4.metric("DSR p-value", f"{m.get('deflated_sharpe_p_value', 1.0):.4f}")
            m5.metric("Monte Carlo 95% DD", f"{m.get('monte_carlo_95_max_dd_pct', 0.0):.2f}%")

            st.markdown("#### Out-Of-Sample (OOS) Baseline Accounting")
            oos1, oos2, oos3 = st.columns(3)
            oos1.write(f"**540d Net PnL (INR)**: `Rs {m.get('pnl_540d_inr', 'NOT AVAILABLE')}`")
            oos2.write(f"**OOS Validated Trades**: `{m.get('oos_trades', 'NOT AVAILABLE')}`")
            oos3.write(f"**OOS Net PnL (INR)**: `Rs {m.get('oos_pnl_inr', 'NOT AVAILABLE')}`")

        # SUBTAB 4: SYMBOL PERFORMANCE
        with detail_tabs[3]:
            st.markdown("#### Cross-Sectional Asset Performance & Data Status")
            sym_df = pd.DataFrame(detail["symbol_performance"])
            if not sym_df.empty:
                st.dataframe(sym_df, use_container_width=True, hide_index=True)
            else:
                st.info("No symbol breakdown available.")

        # SUBTAB 5: RESEARCH EVIDENCE & 540D CEILING AUDIT
        with detail_tabs[4]:
            st.markdown("#### 540-Day Research Horizon Compliance Audit")
            d_read = detail["data_readiness"]
            
            st.write(f"**Target Timeframe**: `{d_read.get('timeframe', '15m')}`")
            st.write(f"**Data Lake Universe Coverage**: `{d_read.get('symbols_ready', 0)} / {d_read.get('symbols_total', 0)} symbols qualified`")
            st.write(f"**18-Month Lookback Compliance**: `{d_read.get('horizon_compliance', 'NOT AVAILABLE')}`")
            
            st.info("""
            **Ashva Research Lookback Rule**:
            The quantitative factory strictly enforces an 18-month / 540-day empirical window to ensure statistical significance 
            across multiple market regimes while discarding stale pre-structural market data.
            """)

        # SUBTAB 6: TEST HISTORY JOURNAL
        with detail_tabs[5]:
            st.markdown("#### Chronological Experiment Trial Ledger")
            hist = detail["test_history"]
            if hist:
                df_hist = pd.DataFrame(hist)
                st.dataframe(
                    df_hist[["experiment_id", "timestamp", "status", "in_sample_sharpe", "cpcv_oos_sharpe", "net_profit_factor", "git_commit_sha"]],
                    use_container_width=True,
                    hide_index=True
                )
            else:
                st.info("No recorded trial history found in SQLite experiment ledger.")
    else:
        st.info("Select an alpha from the dropdown above to view its quantitative dossier.")


# =============================================================================
# TAB 3: TRADING OBSERVABILITY COMPONENT (PRESERVED)
# =============================================================================

def render_trading_observability(df_replay, df_paper, df_live, df_diagnostics):
    st.title("Trading & Execution Observability")
    st.caption("Observability around the TradingEngine lifecycle across REPLAY, PAPER, and LIVE modes.")
    
    tab_replay, tab_paper, tab_live = st.tabs(["REPLAY", "PAPER", "LIVE"])
    
    with tab_replay:
        st.subheader("Replay Diagnostics")
        if not df_diagnostics.empty:
            st.dataframe(df_diagnostics, use_container_width=True)
            
            st.write("### Drop-off Analysis")
            for _, row in df_diagnostics.iterrows():
                st.write(f"**{row['alpha_id']}**: Bars: {row['bars_received']} -> Generate: {row['generate_signals_calls']} -> Raw: {row['raw_signals']} -> Alloc: {row['allocator_rejected']} dropped -> Risk: {row['risk_rejected']} dropped -> Final: {row['final_trades']}")
        else:
            st.info("No replay diagnostic data available.")
            
        st.subheader("Replay Trades")
        if not df_replay.empty:
            st.dataframe(df_replay, use_container_width=True)
        else:
            st.info("No REPLAY trades found.")

    with tab_paper:
        st.subheader("Paper Trades")
        if not df_paper.empty:
            st.dataframe(df_paper, use_container_width=True)
        else:
            st.info("No PAPER trades found.")

    with tab_live:
        st.subheader("Live Trades")
        if not df_live.empty:
            st.dataframe(df_live, use_container_width=True)
        else:
            st.info("No LIVE trades found.")


# =============================================================================
# TAB 4: SYSTEM OBSERVABILITY (PLACEHOLDER FOR PHASE 4)
# =============================================================================

def render_system_observability():
    st.title("System Observability")
    st.caption("System health, resource monitoring, environment integrity, and broker connectivity.")
    st.info("System Observability module will be integrated in Phase 4.")


# =============================================================================
# MASTER NAVIGATION ENTRY POINT
# =============================================================================

def main():
    dal = get_dal()
    df_replay, df_paper, df_live, df_diagnostics = load_trading_data()

    # Top-Level Unified Tabs
    tab_data, tab_alpha, tab_trading, tab_system = st.tabs([
        "DATA",
        "ALPHA FACTORY",
        "TRADING",
        "SYSTEM"
    ])

    with tab_data:
        render_data_observability(dal)

    with tab_alpha:
        render_alpha_factory(dal)

    with tab_trading:
        render_trading_observability(df_replay, df_paper, df_live, df_diagnostics)

    with tab_system:
        render_system_observability()


if __name__ == "__main__":
    main()
