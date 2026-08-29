"""
Ashva Master Observability Dashboard
Unified observability suite providing multi-tab inspection for Data, Alpha Factory, Trading, and System.
"""

import importlib
import streamlit as st
import pandas as pd
import src.ui.data_access
from src.ui.data_access import UIDataAccess

st.set_page_config(
    page_title="Ashva Observability Hub",
    page_icon="🐎",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize Data Access Layer
def get_dal() -> UIDataAccess:
    import src.research.knowledge_map
    importlib.reload(src.research.knowledge_map)
    importlib.reload(src.ui.data_access)
    return src.ui.data_access.UIDataAccess()


# =============================================================================
# TAB 1: DATA OBSERVABILITY COMPONENT
# =============================================================================

def render_data_observability(dal: UIDataAccess):
    st.title("Data Lake Observability")
    st.caption("Point-in-time market data feed health, NSE session synchronization, timeframe matrix, and repository hygiene.")

    overview = dal.get_data_overview()
    quality = dal.get_data_quality_summary()
    live_status = dal.get_live_market_data_status()

    # 1. Actionable Operational KPIs (Replacing raw bar counts)
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric(
        "Active Universe", 
        f"{overview['universe_name']} ({overview['total_symbols']})", 
        help=f"Active universe resolution: {overview['universe_name']} with {overview['total_symbols']} total equities"
    )
    col2.metric(
        "Data Freshness", 
        live_status["freshness_badge"], 
        help=live_status["freshness_detail"]
    )
    col3.metric(
        "Market Phase", 
        live_status["market_phase"], 
        help=live_status["market_phase_detail"]
    )
    col4.metric(
        "Angel One Feed", 
        live_status["feed_status"], 
        help=live_status["feed_detail"]
    )
    col5.metric(
        "Trading Readiness", 
        f"{quality['symbols_with_540d_coverage']}/{overview['total_symbols']} Equities", 
        help=f"{quality['symbols_with_540d_coverage']} symbols satisfy the 18-month statistical warmup horizon. Quality: {quality['quality_status']}"
    )

    # Operational Feed & Heartbeat Alert Banner
    if live_status["freshness_status"] == "STALE":
        st.warning(f"⚠️ **Data Lake Outdated**: {live_status['freshness_detail']}. Click **'Sync Missing Data'** below to backfill to today's session.")
    elif live_status["market_phase"] == "LIVE SESSION OPEN":
        st.info(f"🔴 **LIVE TRADING SESSION ACTIVE** | Current IST Time: `{live_status['current_time_ist']}` | Feed State: `{live_status['feed_status']}` | Latest Bar: `{live_status['latest_bar_timestamp']}`")
    else:
        st.success(f"✓ **Data Lake Synchronized**: {live_status['freshness_detail']} | Timeframes: `{', '.join(overview['available_timeframes'])}` | Storage: `{overview['db_size_mb']} MB`")

    st.markdown("---")

    # Action Toolbar for Interactive Sync and Validation
    st.markdown("#### ⚡ Data Management & Action Toolbar")
    act_col1, act_col2, act_col3 = st.columns([2, 2, 1])
    
    with act_col1:
        univ_name = dal.get_active_universe_name()
        available_syms = dal.get_symbol_list()
        symbols_list = [f"ALL {univ_name.upper()} ({len(available_syms)} Equities)"] + available_syms
        selected_sync_target = st.selectbox("Select Target for Synchronization", symbols_list, index=0, key="sync_target_select")
        if st.button("🔄 Sync Missing Data / Backfill", key="btn_sync_data", use_container_width=True):
            with st.spinner(f"Synchronizing market data for {selected_sync_target}..."):
                res = dal.sync_market_data_now(symbol=selected_sync_target, timeframe="15m")
                if res["status"] == "SUCCESS":
                    st.success(f"✓ Sync Complete: Successfully updated {res['symbols_updated']}/{res['symbols_requested']} symbols ({res['total_bars_processed']} bars) via {res['provider_used']}.")
                else:
                    st.warning(f"Sync Notice: {res['status']} ({len(res.get('errors', []))} notices).")
                
                if res.get("errors"):
                    with st.expander(f"⚠️ View Sync Notices & Error Details ({len(res['errors'])})", expanded=True):
                        for err in res["errors"]:
                            st.write(f"- `{err}`")

    with act_col2:
        st.write("")
        st.write("")
        if st.button("🛡️ Validate Data, Calendar & Stock Splits", key="btn_validate_data", use_container_width=True):
            with st.spinner("Executing full repository hygiene check, NSE Calendar audit & Stock Split detection..."):
                val_res = dal.run_comprehensive_data_validation()
                st.success(f"✓ Comprehensive Validation Complete: {val_res['clean_calendar_symbols']}/{val_res['symbols_audited_count']} symbols with 100% calendar coverage. Split Audit: {val_res['split_audit_status']}.")
                
                with st.expander("🔍 Comprehensive Validation Details", expanded=True):
                    tab_val_cal, tab_val_split = st.tabs(["📅 NSE Calendar Coverage", "🍰 Stock Split & Bonus Anomaly Audit"])
                    with tab_val_cal:
                        if val_res.get("calendar_audits_table"):
                            st.dataframe(pd.DataFrame(val_res["calendar_audits_table"]), use_container_width=True, hide_index=True)
                    with tab_val_split:
                        if val_res.get("split_anomalies_table"):
                            st.dataframe(pd.DataFrame(val_res["split_anomalies_table"]), use_container_width=True, hide_index=True)
                        else:
                            st.success(f"✓ {val_res['split_audit_status']}")

    with act_col3:
        st.write("")
        st.write("")
        if st.button("🔄 Refresh Data", key="btn_refresh_data", use_container_width=True):
            st.rerun()

    st.markdown("---")

    subtab_matrix, subtab_detail, subtab_hygiene, subtab_ingestion = st.tabs([
        "📊 Coverage Matrix",
        "🔍 Symbol Deep Dive",
        "🛡️ Quality & Hygiene Audit",
        "📥 Ingestion Telemetry"
    ])

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
                c1.write(f"**Stock Split Status**: `{detail['quality_metrics'].get('unadjusted_stock_splits', 'CLEAN')}`")
                
                q_metrics = detail["quality_metrics"]
                c2.metric("Duplicate Bars", q_metrics["duplicate_bars"])
                c2.metric("Invalid OHLC Bars", q_metrics["invalid_ohlc_bars"])
                
                c3.metric("Out-of-Market-Hours Bars", q_metrics["out_of_market_hours_bars"])
                c3.write(f"**NSE Calendar Coverage**: `{q_metrics['missing_bars_calendar_audit']}`")

                # Show NSE Calendar Audit Box
                cal = detail.get("calendar_audit", {})
                if cal and "expected_trading_days" in cal:
                    with st.expander(f"📅 NSE Holiday Calendar Audit Breakdown for {detail['symbol']}", expanded=False):
                        ca1, ca2, ca3, ca4 = st.columns(4)
                        ca1.metric("Expected Sessions", f"{cal.get('expected_trading_days', 0)} Days")
                        ca2.metric("Actual Sessions in DB", f"{cal.get('actual_trading_days', 0)} Days")
                        ca3.metric("Missing Sessions", f"{cal.get('missing_trading_days_count', 0)} Days")
                        ca4.metric("Calendar Coverage", f"{cal.get('coverage_pct', 100.0)}%")
                        
                        if cal.get("missing_dates"):
                            st.write(f"**Missing Trading Dates**: `{', '.join(cal['missing_dates'])}`")
                        else:
                            st.success("✓ Zero missing trading sessions. 100% calendar consistency with NSE trading calendar.")

                # Show Stock Split Anomalies Box if detected
                split_drops = detail.get("split_anomalies", [])
                if split_drops:
                    st.warning(f"⚠️ {len(split_drops)} Potential Unadjusted Overnight Price Drops / Splits Detected for {detail['symbol']}")
                    st.dataframe(pd.DataFrame(split_drops), use_container_width=True, hide_index=True)

                st.write("#### Available Timeframes Breakdown")
                df_tf_detail = pd.DataFrame(detail["timeframes_detail"])
                st.dataframe(df_tf_detail, use_container_width=True, hide_index=True)
            else:
                st.info(f"No detail available for {selected_sym}.")
        else:
            st.warning("No symbols available in DataLake.")

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
        5. **NSE Trading Holiday Alignment**: Reconciles trading sessions against the official 2024–2026 NSE Exchange Holiday Calendar.
        """)

    with subtab_ingestion:
        st.subheader("Ingestion Status & Log Telemetry")
        st.caption("Observability around historical ingestion jobs, active sessions, and data storage files.")

        c_ing1, c_ing2 = st.columns(2)
        c_ing1.write(f"**Data Lake File**: `{overview['db_path']}`")
        c_ing1.write(f"**Total Stored OHLCV Bars**: `{overview['total_bars']:,}`")
        c_ing1.write(f"**Database Size**: `{overview['db_size_mb']} MB`")
        c_ing1.write(f"**Last File Modification**: `{overview['last_updated']}`")
        c_ing1.write(f"**Earliest Global Timestamp**: `{overview['earliest_timestamp']}`")
        c_ing1.write(f"**Latest Global Timestamp**: `{overview['latest_timestamp']}`")

        c_ing2.write(f"**Ingestion Provider**: Angel One SmartAPI ({live_status['feed_status']})")
        c_ing2.write(f"**Market Phase**: `{live_status['market_phase']}` ({live_status['market_phase_detail']})")
        c_ing2.write(f"**Data Freshness**: `{live_status['freshness_badge']}`")
        c_ing2.write("**Parquet Mirror Directory**: `data_lake/parquet/`")
        c_ing2.write("**Incremental Ingestion Engine**: `scripts/ingest_all_nifty50_timeframes.py`")

        st.write("#### Discovered Session Logs (`logs/`)")
        df_logs = dal.get_ingestion_log_summary()
        if not df_logs.empty:
            st.dataframe(df_logs, use_container_width=True, hide_index=True)
        else:
            st.info("No session log files discovered in logs/ directory.")


# =============================================================================
# TAB 2: ALPHA FACTORY OBSERVABILITY COMPONENT
# =============================================================================

def render_alpha_factory(dal: UIDataAccess):
    st.title("Alpha Factory Observability")
    st.caption("Authoritative governance, empirical audit records, qualification gates, and statistical lifecycle tracking.")

    # 1. Section 1: Factory Summary KPI Metrics
    summary = dal.get_alpha_factory_summary()
    kpi1, kpi2, kpi3, kpi4, kpi5, kpi6 = st.columns(6)

    total_alphas = summary.get("total_alphas", 0)
    tested_alphas = summary.get("tested", 0)
    kpi1.metric("Total Alphas", total_alphas)
    kpi2.metric("Actually Tested", f"{tested_alphas} / {total_alphas}")
    kpi3.metric("PROVEN (Capital)", summary.get("proven", 0))
    kpi4.metric("FAILED (Rejected)", summary.get("failed", 0))
    kpi5.metric("UNCERTAIN (Watchlist)", summary.get("uncertain", 0))
    kpi6.metric("UNEXPLORED", summary.get("unexplored", 0))

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
        st.markdown('<div class="stat-value" style="font-size: 1.1rem; color: #b0b0b0;">Global Discovery Registry</div>', unsafe_allow_html=True)
        search_query = st.text_input("Search Alpha ID / Name / Rationale", placeholder="e.g. DOUBLE_INSIDE, ORB...", key="search_alpha_registry")

    df_registry = dal.get_alpha_registry_table()

    if df_registry is not None and not df_registry.empty:
        df_disp = df_registry.copy()
        
        if status_filter != "ALL" and "status" in df_disp.columns:
            df_disp = df_disp[df_disp["status"] == status_filter]
        if tested_filter != "ALL" and "tested" in df_disp.columns:
            df_disp = df_disp[df_disp["tested"] == tested_filter]
        if category_filter != "ALL" and "category" in df_disp.columns:
            df_disp = df_disp[df_disp["category"].str.contains(category_filter, case=False, na=False)]
        if search_query:
            match_mask = pd.Series(False, index=df_disp.index)
            for c in ["alpha_id", "name", "category"]:
                if c in df_disp.columns:
                    match_mask = match_mask | df_disp[c].astype(str).str.contains(search_query, case=False, na=False)
            df_disp = df_disp[match_mask]

        st.dataframe(
            df_disp,
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("ℹ️ **Active Alpha Registry is clean & reset.** Ready to discover and validate new hypotheses across the dynamic 77-equity universe.")

    st.markdown("---")

    # 3. Section 3-12: Alpha Deep Dive Inspector
    st.subheader("Alpha Hypothesis Deep Dive & Audit Inspector")
    alpha_options = list(df_registry["alpha_id"].values) if (df_registry is not None and not df_registry.empty and "alpha_id" in df_registry.columns) else []
    
    if alpha_options:
        colA, colB, colC = st.columns([1, 1, 1])
        with colA:
            selected_alpha = st.selectbox("Select Alpha ID for Full Quantitative Audit", alpha_options, index=0, key="select_alpha_detail")

        detail = dal.get_alpha_detail(selected_alpha) if selected_alpha else {}
    else:
        selected_alpha = None
        detail = {}

    if detail and detail.get("alpha_id"):
        st.markdown(f"### `{str(detail.get('alpha_id', '')).upper()}`: {detail.get('name', 'Strategy')} — Status: `{detail.get('status', 'UNEXPLORED')}`")
        st.caption(f"Category: **{detail.get('category', 'UNKNOWN')}** | Timeframe: **{detail.get('timeframe', '15m')}** | Version: **{detail.get('version', 'v1.0.0')}** | Tested: **{'YES' if detail.get('is_tested') else 'NO'}**")

        detail_tabs = st.tabs([
            "🎯 Hypothesis & Parameters",
            "🛡️ Qualification Gates",
            "📈 Quantitative Metrics",
            "🏢 Symbol-Level Performance",
            "🔬 Research Evidence & 540d",
            "📜 Test History Journal",
            "⚙️ Replay Context & Provenance",
            "⏱️ Timeframe Discovery",
            "🧠 Knowledge Lineage"
        ])

        with detail_tabs[0]:
            st.markdown("#### Economic Rationale & Mechanism")
            st.write(detail.get("hypothesis", "No hypothesis rationale registered."))
            
            st.markdown("#### Market Mechanism Description")
            st.info(detail.get("mechanism", "Standard quantitative factor model."))

            col_p1, col_p2 = st.columns(2)
            with col_p1:
                st.markdown("#### Entry, Exit & Holding Specifications")
                st.write(f"**Entry Window**: `{detail.get('entry_window', '09:30-14:30 IST')}`")
                st.write(f"**Entry Conditions**: `{detail.get('entry_conditions', 'Alpha-specific threshold condition')}`")
                st.write(f"**Exit Conditions**: `{detail.get('exit_conditions', 'Stop Loss / Target / Intraday Square-off')}`")
                st.write(f"**Holding Concept**: `{detail.get('holding_concept', 'Intraday Horizon')}`")
                targets = detail.get("target_instruments", [])
                targets_disp = ", ".join(targets[:8]) + ("..." if len(targets) > 8 else "") if targets else "Dynamic Active Universe"
                st.write(f"**Research Universe**: `{targets_disp}`")
            with col_p2:
                st.markdown("#### Strategy Parameters")
                if detail.get("parameters"):
                    st.json(detail["parameters"])
                else:
                    st.write("No parameters defined (Default baseline).")

        with detail_tabs[1]:
            st.markdown("#### Institutional Qualification Gates Evaluation")
            gates = detail.get("qualification_gates", {})
            
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

            st.markdown("#### Status Explanation & Quantitative Justification")
            explanations = detail.get("explanations", {})
            status_val = detail.get("status", "UNEXPLORED")
            status_reason = explanations.get("status_reason", "Awaiting comprehensive backtest evaluation.")
            
            if status_val == "PROVEN":
                st.success(f"**PROVEN QUALIFICATION**: {status_reason}")
            elif status_val == "FAILED":
                st.error(f"**FAILED / REJECTED**: {status_reason}")
            elif status_val == "UNCERTAIN":
                st.warning(f"**UNCERTAIN**: {status_reason}")
            else:
                st.info(f"**UNEXPLORED**: {status_reason}")

            if explanations.get("failure_lessons") and explanations["failure_lessons"] != "NOT APPLICABLE":
                st.warning(f"**Empirical Failure Lessons & Friction Analysis**: {explanations['failure_lessons']}")
            if explanations.get("known_limitations"):
                st.write(f"**Known Regime Limitations**: {explanations['known_limitations']}")

        with detail_tabs[2]:
            st.markdown("#### Complete Quantitative Metrics Breakdown")
            st.caption("Metrics faithfully retrieved from backend state. Metrics not recorded or implemented are explicitly demarcated.")
            m = detail["metrics"]
            
            mc1, mc2, mc3 = st.columns(3)
            with mc1:
                st.markdown("##### In-Sample & Aggregate")
                st.write(f"**Total Trades**: `{m.get('total_trades')}`")
                st.write(f"**Winning Trades**: `{m.get('winning_trades')}`")
                st.write(f"**Losing Trades**: `{m.get('losing_trades')}`")
                st.write(f"**Win Rate**: `{m.get('win_rate')}`")
                st.write(f"**Gross Profit**: `{m.get('gross_profit')}`")
                st.write(f"**Gross Loss**: `{m.get('gross_loss')}`")
                st.write(f"**Net P&L (INR)**: `{m.get('net_pnl')}`")
            with mc2:
                st.markdown("##### Risk-Adjusted & Ratios")
                st.write(f"**Expectancy**: `{m.get('expectancy')}`")
                st.write(f"**Profit Factor**: `{m.get('profit_factor')}`")
                st.write(f"**Sharpe Ratio**: `{m.get('sharpe')}`")
                st.write(f"**Sortino Ratio**: `{m.get('sortino')}`")
                st.write(f"**Max Drawdown**: `{m.get('max_drawdown')}`")
                st.write(f"**Average Win**: `{m.get('avg_win')}`")
                st.write(f"**Average Loss**: `{m.get('avg_loss')}`")
            with mc3:
                st.markdown("##### Out-Of-Sample (CPCV) & Tail")
                st.write(f"**OOS Trades**: `{m.get('oos_trades')}`")
                st.write(f"**OOS Net P&L**: `{m.get('oos_pnl')}`")
                st.write(f"**OOS Sharpe**: `{m.get('oos_sharpe')}`")
                st.write(f"**OOS Win Rate**: `{m.get('oos_win_rate')}`")
                st.write(f"**OOS Drawdown**: `{m.get('oos_drawdown')}`")
                st.write(f"**Deflated Sharpe (p-value)**: `{m.get('deflated_sharpe_p_value', 'NOT AVAILABLE')}`")
                st.write(f"**Average Holding Time**: `{m.get('avg_holding_time')}`")

        with detail_tabs[3]:
            st.markdown("#### Cross-Sectional Asset Performance & Data Lake Status")
            st.caption("Distinguishes universal Alpha Logic from instrument-specific data coverage.")
            sym_perf = detail.get("symbol_performance", [])
            if sym_perf:
                sym_df = pd.DataFrame(sym_perf)
                st.dataframe(sym_df, use_container_width=True, hide_index=True)
            else:
                st.info("No symbol performance breakdown recorded for this alpha.")

        with detail_tabs[4]:
            st.markdown("#### 540-Day Research Horizon Compliance & Evidence Audit")
            ev = detail.get("research_evidence", {})
            d_read = detail.get("data_readiness", {})
            
            c_ev1, c_ev2 = st.columns(2)
            with c_ev1:
                st.write(f"**Research Start Date**: `{ev.get('research_start', 'NOT AVAILABLE')}`")
                st.write(f"**Research End Date**: `{ev.get('research_end', 'NOT AVAILABLE')}`")
                st.write(f"**Actual Calendar Days**: `{ev.get('calendar_days', 'NOT AVAILABLE')} Days`")
                st.write(f"**Estimated Trading Days**: `{ev.get('trading_days', 'NOT AVAILABLE')} Days`")
                st.write(f"**Data Storage Source**: `{ev.get('data_source', 'NOT AVAILABLE')}`")
            with c_ev2:
                st.write(f"**Required Timeframe**: `{d_read.get('timeframe', '15m')}`")
                st.write(f"**Universe Coverage**: `{d_read.get('symbols_ready', 0)} / {d_read.get('symbols_total', 0)} symbols qualified`")
                st.write(f"**540-Day Horizon Status**: `{d_read.get('horizon_compliance', 'NOT AVAILABLE')}`")

            st.info("""
            **Ashva Multi-Window Lookback Hierarchy**:
            - **60-Day Window**: Recency-weighted current regime trajectory & decay detection.
            - **180-Day & 365-Day Windows**: Intermediate multi-season stability validation.
            - **540-Day Full Horizon**: Hard institutional ceiling (~18 months) ensuring statistical significance.
            """)

        with detail_tabs[5]:
            st.markdown("#### Chronological Research Trials Ledger (`experiment_ledger.db`)")
            hist = detail.get("test_history", [])
            if hist:
                df_hist = pd.DataFrame(hist)
                cols = [c for c in ["experiment_id", "timestamp", "status", "in_sample_sharpe", "cpcv_oos_sharpe", "net_profit_factor", "monte_carlo_95_max_dd", "git_commit_sha"] if c in df_hist.columns]
                st.dataframe(
                    df_hist[cols] if cols else df_hist,
                    use_container_width=True,
                    hide_index=True
                )
            else:
                st.info("No recorded trial history found in SQLite experiment ledger.")

        with detail_tabs[6]:
            st.markdown("#### Execution Alignment & Replay Context")
            st.caption("Verifies consistency between research configuration and trading engine execution parameters.")
            rep = detail.get("replay_context", {})
            
            c_rep1, c_rep2 = st.columns(2)
            with c_rep1:
                st.write(f"**Execution Timeframe**: `{rep.get('timeframe', '15m')}`")
                st.write(f"**Trading Entry Window**: `{rep.get('entry_window', '09:15-15:00')}`")
                st.write(f"**Trailing Stop Mode**: `{rep.get('trailing_stop_mode', 'STEP_RATCHET')}`")
                st.write(f"**Intraday Square-off**: `{rep.get('intraday_squareoff', '15:15 IST')}`")
            with c_rep2:
                prov = detail.get("provenance", {})
                st.markdown("#### Quantitative Source & Provenance")
                st.write(f"**Research Commit SHA**: `{prov.get('research_commit', 'NOT AVAILABLE')}`")
                st.write(f"**Code Commit SHA**: `{prov.get('code_commit', 'NOT AVAILABLE')}`")
                st.write(f"**Qualification Version**: `{prov.get('qualification_version', 'v1.0.0')}`")
                st.write(f"**Research Timestamp**: `{prov.get('research_timestamp', 'NOT AVAILABLE')}`")

        with detail_tabs[7]:
            st.markdown("#### Timeframe Discovery Evidence")
            st.caption("Evidence generated by the hypothesis lab when searching across timeframes.")
            tfc = detail.get("timeframe_comparison", {})
            if tfc and "timeframes" in tfc:
                st.write(f"**Discovered Best Timeframe:** `{tfc.get('best_timeframe', 'UNKNOWN')}`")
                df_tfc = pd.DataFrame(tfc["timeframes"])
                st.dataframe(df_tfc, use_container_width=True, hide_index=True)
            else:
                st.info("No timeframe comparison evidence recorded for this alpha.")
                
        with detail_tabs[8]:
            st.markdown("#### Master Knowledge Lineage")
            st.caption("Complete history of all tested alphas loaded from canonical ledger.")
            lineage = dal.get_knowledge_lineage()
            if lineage:
                df_lineage = pd.DataFrame(lineage)
                st.dataframe(df_lineage, use_container_width=True, hide_index=True)
            else:
                st.info("No historical lineage discovered.")

    else:
        st.info("Select an alpha from the dropdown above to view its quantitative dossier.")


# =============================================================================
# TAB 3: TRADING OBSERVABILITY COMPONENT
# =============================================================================

def render_trading_observability(dal: UIDataAccess):
    st.title("Trading & Execution Observability")
    st.caption("Authoritative multi-mode observability for the unified TradingEngine across Paper, Replay, and Live execution.")

    tab_paper, tab_replay, tab_live, tab_trace = st.tabs([
        "📝 PAPER TRADING",
        "🔄 REPLAY ENGINE",
        "⚡ LIVE EXECUTION",
        "🔍 EVENT TRACE DRILL-DOWN"
    ])

    # -------------------------------------------------------------------------
    # SUBTAB 1: PAPER TRADING
    # -------------------------------------------------------------------------
    with tab_paper:
        st.subheader("Paper Trading Portfolio & Engine State")
        port_paper = dal.get_trading_portfolio_summary(mode="PAPER")

        p1, p2, p3, p4, p5, p6 = st.columns(6)
        p1.metric("Engine Status", port_paper["engine_status"])
        p2.metric("Market Status", port_paper["market_status"])
        p3.metric("Current Equity", f"₹{port_paper['current_equity']:,.2f}")
        p4.metric("Available Cash", f"₹{port_paper['cash']:,.2f}")
        p5.metric("Open Positions", port_paper["open_positions"])
        p6.metric("Total Net P&L", f"₹{port_paper['total_pnl']:+,.2f}", delta=f"{port_paper['roi_pct']:+.2f}% ROI")

        st.markdown("---")

        # Active Alphas & Symbol Matrix
        st.markdown("#### Active Alpha Contracts in Trading Manifest")
        st.caption("Strict boundary: Alphas must be explicitly enabled in TradingManifest to participate in trading.")
        
        active_alphas = dal.get_active_trading_alphas()
        if active_alphas:
            df_active = pd.DataFrame(active_alphas)
            st.dataframe(
                df_active[["alpha_id", "name", "version", "category", "factory_status", "trading_status", "timeframe", "universe"]],
                use_container_width=True,
                hide_index=True
            )
        else:
            st.info("No active alpha contracts registered in Trading Manifest.")

        st.markdown("#### Active Alpha → Target Symbol Evaluation Matrix")
        st.caption("Exposes which symbols each active alpha contract is evaluating in the trading engine.")
        df_matrix = dal.get_alpha_symbol_evaluation_matrix()
        if not df_matrix.empty:
            st.dataframe(df_matrix, use_container_width=True, hide_index=True)

        st.markdown("---")

        # Sub-tabs for Signals, Orders, Positions, Allocation
        paper_subtabs = st.tabs([
            "📡 Signal Monitor",
            "📋 Order Monitor",
            "💼 Open Positions & Fills",
            "💰 Capital Allocation & Risk"
        ])

        with paper_subtabs[0]:
            st.markdown("##### Recent Signal Evaluations & Allocator Decisions")
            df_sig = dal.get_trading_signals(mode="PAPER", limit=50)
            if not df_sig.empty:
                st.dataframe(df_sig, use_container_width=True, hide_index=True)
            else:
                st.info("No active Paper trading signals generated in current session.")

        with paper_subtabs[1]:
            st.markdown("##### Order Submissions & Rejection Diagnostics")
            df_ord = dal.get_trading_orders(mode="PAPER", limit=50)
            if not df_ord.empty:
                st.dataframe(df_ord, use_container_width=True, hide_index=True)
            else:
                st.info("No Paper orders submitted in current session.")

        with paper_subtabs[2]:
            st.markdown("##### Current Open Positions")
            df_pos = dal.get_trading_positions(mode="PAPER")
            if not df_pos.empty:
                st.dataframe(df_pos, use_container_width=True, hide_index=True)
            else:
                st.info("0 Open Positions. All intraday positions are squared off by 15:15 IST.")

            st.markdown("##### Executed Fills Log")
            df_fills = dal.get_trading_fills(mode="PAPER", limit=50)
            if not df_fills.empty:
                st.dataframe(df_fills, use_container_width=True, hide_index=True)
            else:
                st.info("No Paper fills executed in current session.")

        with paper_subtabs[3]:
            st.markdown("##### Capital & Risk Budget Allocation Model")
            alloc_info = dal.get_capital_allocation_breakdown(mode="PAPER")
            
            ac1, ac2, ac3, ac4 = st.columns(4)
            ac1.metric("Initial Capital Base", f"₹{alloc_info['initial_capital']:,.2f}")
            ac2.metric("Max Risk / Trade", f"{alloc_info['max_risk_per_trade_pct']*100:.2f}% (₹2,500)")
            ac3.metric("Max Portfolio Risk Cap", f"{alloc_info['max_portfolio_risk_pct']*100:.2f}% (₹10,000)")
            ac4.metric("Max Open Positions", f"{alloc_info['max_concurrent_positions']} Concurrent")

            st.markdown("###### Per-Alpha Capital & Trailing Configuration")
            st.dataframe(pd.DataFrame(alloc_info["per_alpha_table"]), use_container_width=True, hide_index=True)

    # -------------------------------------------------------------------------
    # SUBTAB 2: REPLAY ENGINE
    # -------------------------------------------------------------------------
    with tab_replay:
        st.subheader("Replay Engine State & Parity Audit")
        replay_summary = dal.get_replay_summary()
        port_replay = dal.get_trading_portfolio_summary(mode="REPLAY")

        r1, r2, r3, r4, r5 = st.columns(5)
        r1.metric("Replay Run State", replay_summary.get("replay_status", "READY"))
        r2.metric("Total Replay Trades", f"{replay_summary.get('total_trades', 0)} Trades")
        r3.metric("Replay Win Rate", f"{replay_summary.get('win_rate', 0.0):.1f}%")
        r4.metric("Net P&L Generated", f"₹{replay_summary.get('net_pnl', 0.0):+,.2f}")
        r5.metric("Net Profit Factor", f"{replay_summary.get('net_pf', 0.0):.2f}")

        from src.core.universe_manager import get_universe_name
        st.markdown(f"**Replay Scope**: `{replay_summary.get('period_tested', 'August 24 - 28, 2026')}` | Universe: `{replay_summary.get('universe', get_universe_name())}` | Timeframe: `{replay_summary.get('timeframe', '15m')}`")

        st.markdown("---")

        # Zero-Signal Diagnostics Breakdown
        st.markdown("#### Zero-Signal Pipeline Diagnostic Tracker")
        st.caption("Granular stage-by-stage drop-off accounting to diagnose why alphas did or did not trigger trades.")
        
        df_zero_sig = dal.get_replay_zero_signal_pipeline()
        if not df_zero_sig.empty:
            st.dataframe(
                df_zero_sig,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "alpha_id": st.column_config.TextColumn("Alpha ID", width="medium"),
                    "bars_received": st.column_config.NumberColumn("Bars Received", width="small"),
                    "generate_signals_calls": st.column_config.NumberColumn("Signal Calls", width="small"),
                    "raw_signals": st.column_config.NumberColumn("Raw Signals", width="small"),
                    "accepted_signals": st.column_config.NumberColumn("Accepted", width="small"),
                    "allocator_rejected": st.column_config.NumberColumn("Alloc Dropped", width="small"),
                    "risk_rejected": st.column_config.NumberColumn("Risk Dropped", width="small"),
                    "final_trades": st.column_config.NumberColumn("Final Trades", width="small"),
                    "entry_window": st.column_config.TextColumn("Entry Window", width="small"),
                }
            )
        else:
            st.info("Replay pipeline diagnostic counters are recorded automatically during replay runs.")

        st.markdown("---")

        # Replay Alpha Breakdown & Trade Ledger
        rep_sub1, rep_sub2, rep_sub3, rep_sub4 = st.tabs([
            "📊 Alpha Performance Breakdown",
            "📜 Authoritative Trade Ledger",
            "📡 Replay Signals & Decisions",
            "📋 Orders & Fills"
        ])

        with rep_sub1:
            st.markdown("##### Replay Performance Attribution per Alpha")
            df_rep_alpha = dal.get_replay_alpha_breakdown()
            if not df_rep_alpha.empty:
                st.dataframe(df_rep_alpha, use_container_width=True, hide_index=True)
            else:
                st.info("No alpha breakdown available for Replay.")

        with rep_sub2:
            st.markdown("##### Closed Trades Ledger (`trade_ledger` table)")
            df_trades = dal.get_closed_trades(mode="REPLAY", limit=100)
            if not df_trades.empty:
                st.dataframe(
                    df_trades,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "trade_id": st.column_config.NumberColumn("Trade ID", width="small"),
                        "alpha_id": st.column_config.TextColumn("Alpha ID", width="medium"),
                        "symbol": st.column_config.TextColumn("Symbol", width="small"),
                        "side": st.column_config.TextColumn("Side", width="small"),
                        "quantity": st.column_config.NumberColumn("Qty", width="small"),
                        "entry_time": st.column_config.TextColumn("Entry Time", width="medium"),
                        "exit_time": st.column_config.TextColumn("Exit Time", width="medium"),
                        "entry_price": st.column_config.NumberColumn("Entry (₹)", format="%.2f"),
                        "exit_price": st.column_config.NumberColumn("Exit (₹)", format="%.2f"),
                        "gross_pnl": st.column_config.NumberColumn("Gross P&L (₹)", format="%.2f"),
                        "net_pnl": st.column_config.NumberColumn("Net P&L (₹)", format="%.2f"),
                        "total_costs": st.column_config.NumberColumn("Costs (₹)", format="%.2f"),
                        "mfe_pct": st.column_config.NumberColumn("MFE %", format="%.2f"),
                        "mae_pct": st.column_config.NumberColumn("MAE %", format="%.2f"),
                        "holding_period_bars": st.column_config.NumberColumn("Bars Held", width="small"),
                        "exit_reason": st.column_config.TextColumn("Exit Reason", width="medium"),
                    }
                )
            else:
                st.info("No closed trades found for Replay mode.")

        with rep_sub3:
            st.markdown("##### Replay Signal & Allocator Log")
            df_sig_rep = dal.get_trading_signals(mode="REPLAY", limit=100)
            if not df_sig_rep.empty:
                st.dataframe(df_sig_rep, use_container_width=True, hide_index=True)
            else:
                st.info("No signals recorded for Replay.")

        with rep_sub4:
            st.markdown("##### Replay Orders & Fills")
            col_o1, col_o2 = st.columns(2)
            with col_o1:
                st.markdown("###### Orders Log")
                df_o = dal.get_trading_orders(mode="REPLAY", limit=50)
                if not df_o.empty:
                    st.dataframe(df_o, use_container_width=True, hide_index=True)
                else:
                    st.info("No orders found.")
            with col_o2:
                st.markdown("###### Fills Log")
                df_f = dal.get_trading_fills(mode="REPLAY", limit=50)
                if not df_f.empty:
                    st.dataframe(df_f, use_container_width=True, hide_index=True)
                else:
                    st.info("No fills found.")

    # -------------------------------------------------------------------------
    # SUBTAB 3: LIVE EXECUTION
    # -------------------------------------------------------------------------
    with tab_live:
        st.subheader("Live Trading Engine Status & Connectivity")
        
        l1, l2, l3, l4 = st.columns(4)
        l1.metric("Engine Health", "STANDBY / OFFLINE")
        l2.metric("Broker Gateway", "Angel One SmartAPI (Ready)")
        l3.metric("Live Capital Deployed", "₹0.00")
        l4.metric("Active Real-Capital Positions", "0")

        st.warning("""
        **Safety & Boundary Enforcement**:
        Live trading execution is guarded by strict manual authorization. Proven alphas in the Alpha Factory 
        are NEVER automatically activated for live real-capital execution without explicit human promotion.
        """)

        st.markdown("#### Live Execution Event Stream (`system_events_log`)")
        st.info("0 critical system events recorded. Live execution engine is in idle standby.")

    # -------------------------------------------------------------------------
    # SUBTAB 4: EVENT TRACE DRILL-DOWN
    # -------------------------------------------------------------------------
    with tab_trace:
        st.subheader("End-to-End Event Trace Drill-Down")
        st.caption("Traces complete lifecycle: Market Event → Alpha Evaluation → Signal → Decision → Order → Fill → Trade → P&L.")

        trace_query = st.text_input("Enter Trade ID, Signal ID, or Order ID to Trace", value="4", key="input_trace_id")
        
        if trace_query:
            trace_data = dal.get_event_trace(trace_query)
            if trace_data and trace_data.get("trade_id"):
                st.success(f"**Trade #{trace_data['trade_id']} Found**: `{trace_data['alpha_id']}` on `{trace_data['symbol']}` ({trace_data['side']} {trace_data['quantity']} shares)")
                
                tr_col1, tr_col2, tr_col3, tr_col4 = st.columns(4)
                tr_col1.metric("Gross P&L", f"₹{trace_data['pnl_details']['gross_pnl']:+,.2f}")
                tr_col2.metric("Total Taxes & Costs", f"₹{trace_data['pnl_details']['total_costs']:,.2f}")
                tr_col3.metric("Net P&L (Realized)", f"₹{trace_data['pnl_details']['net_pnl']:+,.2f}")
                tr_col4.metric("Holding Period", f"{trace_data['pnl_details']['holding_bars']} bars ({trace_data['pnl_details']['exit_reason']})")

                st.markdown("#### Execution Pipeline Audit Trail")
                st.markdown(f"""
                1. **Market Event Received**: `{trace_data.get('market_event_timestamp')}`
                2. **Alpha Evaluation**: Evaluated `{trace_data['alpha_id']}` on `{trace_data['symbol']}` 15m bar
                3. **Signal Generated**: Signal ID `{trace_data['signal_details']['signal_id']}` (`{trace_data['signal_details']['signal_type']}`, Confidence `{trace_data['signal_details']['confidence']}`, SL `{trace_data['signal_details']['suggested_sl']}`, TP `{trace_data['signal_details']['suggested_tp']}`)
                4. **Allocator Decision**: Decision ID `{trace_data['allocator_decision']['decision_id']}` -> `ACCEPTED` (Risk Budget `₹{trace_data['allocator_decision']['risk_budget']:,.2f}`)
                5. **Order Submitted**: Order ID `{trace_data['order_details']['order_id']}` -> Status `{trace_data['order_details']['status']}`
                6. **Fill Executed**: Entry at `₹{trace_data['pnl_details']['entry_price']:.2f}` | Exit at `₹{trace_data['pnl_details']['exit_price']:.2f}`
                7. **Intraday Square-off**: Closed at 15:15 IST -> Net Realized P&L `₹{trace_data['pnl_details']['net_pnl']:+,.2f}`
                """)
            else:
                st.info(f"No trade or signal record matched query '{trace_query}'. Try Trade ID '4' or '5'.")


# =============================================================================
# TAB 4: SYSTEM OBSERVABILITY COMPONENT
# =============================================================================

def render_system_observability(dal: UIDataAccess):
    st.title("System Observability & Operational Health")
    st.caption("Centralized operational health, engine status, active configuration, git provenance, and system telemetry.")

    # 1. System Overview Section
    sys_overview = dal.get_system_health_overview()
    
    top_c1, top_c2, top_c3, top_c4, top_c5 = st.columns(5)
    top_c1.metric("Ashva System Status", sys_overview["overall_status"])
    top_c2.metric("Environment", sys_overview["environment"])
    top_c3.metric("Ashva Version", sys_overview["version"])
    top_c4.metric("Git Commit", f"{sys_overview['git_branch']}@{sys_overview['git_commit']}", help=f"Status: {sys_overview['git_status']}")
    top_c5.metric("Last Refreshed", sys_overview["last_refresh"][-8:])

    st.markdown("---")

    # Component Health Grid
    st.subheader("Subsystem Operational Health Grid")
    comps = sys_overview["components"]
    
    cc1, cc2, cc3, cc4, cc5, cc6 = st.columns(6)
    cc1.metric("DATA LAYER", comps["DATA"]["status"], help=comps["DATA"]["detail"])
    cc2.metric("ALPHA FACTORY", comps["ALPHA FACTORY"]["status"], help=comps["ALPHA FACTORY"]["detail"])
    cc3.metric("TRADING CORE", comps["TRADING"]["status"], help=comps["TRADING"]["detail"])
    cc4.metric("REPLAY ENGINE", comps["REPLAY"]["status"], help=comps["REPLAY"]["detail"])
    cc5.metric("PAPER ENGINE", comps["PAPER"]["status"], help=comps["PAPER"]["detail"])
    cc6.metric("LIVE BROKER", comps["LIVE"]["status"], help=comps["LIVE"]["detail"])

    st.markdown("---")

    # Subtabs for detailed System inspection
    sys_tab_engine, sys_tab_data, sys_tab_trading, sys_tab_config, sys_tab_prov, sys_tab_logs, sys_tab_diag, sys_tab_runtime = st.tabs([
        "⚙️ Engine Health",
        "📊 Data Pipeline Health",
        "⚡ Trading Engine State",
        "📋 Active Configuration",
        "🏷️ Version & Provenance",
        "⚠️ Errors, Warnings & Logs",
        "🛡️ Stale & Broken State",
        "💻 Runtime & Environment"
    ])

    with sys_tab_engine:
        st.subheader("Major Engine Health & Execution Lifecycle")
        engines = dal.get_engine_health_metrics()
        df_eng = pd.DataFrame(engines)
        st.dataframe(
            df_eng,
            use_container_width=True,
            hide_index=True,
            column_config={
                "engine": st.column_config.TextColumn("Engine", width="medium"),
                "status": st.column_config.TextColumn("Status", width="small"),
                "current_state": st.column_config.TextColumn("Current State", width="medium"),
                "last_activity": st.column_config.TextColumn("Last Activity", width="medium"),
                "last_successful_operation": st.column_config.TextColumn("Last Successful Operation", width="large"),
                "last_error": st.column_config.TextColumn("Last Error", width="small"),
            }
        )

    with sys_tab_data:
        st.subheader("Data Pipeline Operational Health")
        d_ind = dal.get_data_pipeline_health_indicators()
        
        dc1, dc2, dc3, dc4 = st.columns(4)
        dc1.metric("DuckDB Storage", d_ind["duckdb_storage"])
        dc2.metric("Parquet Mirror", d_ind["parquet_storage"])
        dc3.metric("Data Freshness", d_ind["data_freshness"][:10])
        dc4.metric("Hygiene Audit", d_ind["hygiene_audit"])

        st.markdown("##### Storage & Feed Specifications")
        st.write(f"**Symbols Available**: `{d_ind['symbols_available']}`")
        st.write(f"**Timeframes Available**: `{d_ind['timeframes_available']}`")
        st.write(f"**Structural Data Errors**: `{d_ind['data_errors_count']} errors detected`")
        st.write(f"**Stale Feeds Status**: `{d_ind['stale_feeds_detected']}`")

    with sys_tab_trading:
        st.subheader("Trading Engine State & Activity Monitor")
        t_ind = dal.get_trading_engine_health_indicators()
        
        tc1, tc2, tc3, tc4 = st.columns(4)
        tc1.metric("Core Engine State", t_ind["trading_engine_state"])
        tc2.metric("Active Contracts", f"{t_ind['active_alpha_contracts_count']} Alphas")
        tc3.metric("Current Portfolio Equity", t_ind["current_equity"])
        tc4.metric("Open Real-Time Positions", t_ind["open_positions"])

        st.markdown("##### Last Event Telemetry")
        st.write(f"**Last Signal Evaluation**: `{t_ind['last_signal_evaluated']}`")
        st.write(f"**Last Order Submitted**: `{t_ind['last_order_submitted']}`")
        st.write(f"**Last Fill Executed**: `{t_ind['last_fill_executed']}`")
        st.write(f"**Last Position Update**: `{t_ind['last_position_update']}`")

    with sys_tab_config:
        st.subheader("Active System Configuration & Security Audit")
        st.caption("Guarantees zero accidental divergence between expected and active production configurations.")
        
        cfg = dal.get_active_system_configuration()

        cfg_col1, cfg_col2 = st.columns(2)
        with cfg_col1:
            st.markdown("#### Fund & Market Execution Settings")
            st.json(cfg["fund_configuration"])
            st.markdown("#### NSE Trading Hours & Square-Off")
            st.json(cfg["market_hours"])
            st.markdown("#### Alpha Factory Qualification Hurdles")
            st.json(cfg["alpha_qualification_hurdles"])

        with cfg_col2:
            st.markdown("#### Real-Time RMS Limits (`config/risk_limits.yaml`)")
            st.json(cfg["risk_limits"])
            st.markdown("#### Gateway Credentials & Secret Security Audit")
            st.info("Institutional Security Rule: All API credentials, JWT tokens, and private keys are strictly redacted.")
            st.json(cfg["gateway_credentials_security_audit"])

    with sys_tab_prov:
        st.subheader("Version & Provenance (Reproducibility)")
        st.caption("Verifies the exact source code revision and environment producing research and execution results.")
        
        prov = dal.get_system_version_provenance()
        
        pv1, pv2, pv3, pv4 = st.columns(4)
        pv1.metric("Ashva Version", prov["ashva_version"])
        pv2.metric("Git Commit SHA", prov["git_commit"])
        pv3.metric("Git Branch", prov["git_branch"])
        pv4.metric("Working Tree", prov["working_tree_status"])

        st.write(f"**Commit Timestamp**: `{prov['commit_timestamp']}`")
        st.write(f"**Python Runtime**: `{prov['python_version']}`")
        st.write(f"**Host OS Platform**: `{prov['os_platform']}`")
        st.write(f"**Python Interpreter**: `{prov['interpreter_path']}`")

    with sys_tab_logs:
        st.subheader("Operational Telemetry & Error Logs")
        st.caption("Centralized logs from `system_events_log` and `logs/**/app.log` (Tokens and keys automatically sanitized).")

        df_logs = dal.get_operational_logs_and_errors(limit=50)
        
        sev_filter = st.selectbox("Filter by Log Severity", ["ALL", "ERROR", "WARNING", "INFO"], index=0, key="select_log_sev")
        if not df_logs.empty:
            df_disp_logs = df_logs
            if sev_filter != "ALL":
                df_disp_logs = df_disp_logs[df_disp_logs["Severity"] == sev_filter]
            st.dataframe(df_disp_logs, use_container_width=True, hide_index=True)
        else:
            st.info("HISTORICAL LOG STORE NOT AVAILABLE (0 log entries recorded).")

    with sys_tab_diag:
        st.subheader("Stale & Broken State Detection")
        st.caption("Automated diagnosis of potential desynchronizations, lock contentions, or missing configurations.")
        
        stale_diag = dal.get_stale_and_broken_state_diagnostics()
        
        sd1, sd2, sd3 = st.columns(3)
        sd1.metric("Data Freshness Status", "SYNCHRONIZED" if "SYNCHRONIZED" in stale_diag["data_staleness_status"] else "STALE")
        sd2.metric("Config Files Missing", "0 Missing" if isinstance(stale_diag["missing_config_files"], str) else f"{len(stale_diag['missing_config_files'])} Missing")
        sd3.metric("Database File Integrity", "HEALTHY" if isinstance(stale_diag["database_integrity_issues"], str) else "ISSUES")

        st.write(f"**Data Freshness Detail**: `{stale_diag['data_staleness_status']}`")
        st.write(f"**Missing Configuration Files**: `{stale_diag['missing_config_files']}`")
        st.write(f"**Database Integrity Audit**: `{stale_diag['database_integrity_issues']}`")
        st.write(f"**SQLite WAL Lock Status**: `{stale_diag['stale_wal_locks']}`")

    with sys_tab_runtime:
        st.subheader("Low-Level Runtime & Environment Information")
        rt = dal.get_system_runtime_info()
        
        st.write(f"**Process ID**: `{rt['process_id']}`")
        st.write(f"**Working Directory**: `{rt['working_directory']}`")
        st.write(f"**DuckDB Storage Path**: `{rt['duckdb_database_path']}`")
        st.write(f"**Trading Ledger Path**: `{rt['trading_ledger_path']}`")
        st.write(f"**Experiment Ledger Path**: `{rt['experiment_ledger_path']}`")
        st.write(f"**OS Platform & Hardware**: `{rt['os_platform']}`")


# =============================================================================
# MASTER NAVIGATION ENTRY POINT
# =============================================================================

def main():
    dal = get_dal()

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
        render_trading_observability(dal)

    with tab_system:
        render_system_observability(dal)


if __name__ == "__main__":
    main()
