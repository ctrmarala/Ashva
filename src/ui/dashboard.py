"""
Ashva Master Observability Dashboard
Unified observability suite providing multi-tab inspection for Data, Alpha Factory, Trading, and System.
"""

import streamlit as st
import pandas as pd
from src.ui.data_access import UIDataAccess

st.set_page_config(
    page_title="Ashva Observability Hub",
    page_icon="??",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize Data Access Layer
@st.cache_resource
def get_dal():
    return UIDataAccess()

@st.cache_data(ttl=5)
def load_alpha_trading_data():
    dal = get_dal()
    df_alphas = dal.get_alpha_registry_summary()
    df_replay = dal.get_trading_state("REPLAY")
    df_paper = dal.get_trading_state("PAPER")
    df_live = dal.get_trading_state("LIVE")
    df_diagnostics = dal.get_replay_diagnostics()
    return df_alphas, df_replay, df_paper, df_live, df_diagnostics


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

    # 2. Sub-sections via tabs or clean sections
    subtab_matrix, subtab_detail, subtab_hygiene, subtab_ingestion, subtab_alpha_conn = st.tabs([
        "?? Coverage Matrix",
        "?? Symbol Deep Dive",
        "??? Quality & Hygiene Audit",
        "?? Ingestion Telemetry",
        "?? Data ? Alpha Mapping"
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
            selected_sym = st.selectbox("Select Symbol for Point-in-Time Inspection", symbols, index=0)
            detail = dal.get_symbol_detail(selected_sym)

            if detail:
                c1, c2, c3 = st.columns(3)
                c1.metric("Selected Symbol", detail["symbol"])
                c1.write(f"**Primary Storage**: {detail['data_source']}")
                
                # Quality breakdown for this symbol
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
        2. **Valid OHLC Bounds**: Strict validation ($High \ge Low$, $Open > 0$, $Close > 0$, $Volume \ge 0$).
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
        st.subheader("Data Lake ? Alpha Factory Requirements Mapping")
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
# TAB 2: EXISTING ALPHA FACTORY COMPONENT (PRESERVED)
# =============================================================================

def render_alpha_factory(df_alphas: pd.DataFrame):
    st.title("Alpha Factory Registry")
    st.caption("Observability over the autonomous hypothesis laboratory, research trials, and capital candidates.")
    
    # 1. Summary Metrics
    col1, col2, col3, col4 = st.columns(4)
    total = len(df_alphas) if not df_alphas.empty else 0
    proven = len(df_alphas[df_alphas["dynamic_status"] == "CAPITAL_CANDIDATE"]) if "dynamic_status" in df_alphas.columns else 0
    failed = len(df_alphas[df_alphas["status"] == "EXPLORED_FAILED"]) if "status" in df_alphas.columns else 0
    
    col1.metric("Total Alphas", total)
    col2.metric("Proven / Capital Candidate", proven)
    col3.metric("Failed / Rejected", failed)
    
    if df_alphas.empty:
        st.warning("No alphas found in registry.")
        return

    # 2. DataFrame with search
    search = st.text_input("Search Alpha ID", key="search_alpha_factory")
    df_disp = df_alphas
    if search:
        df_disp = df_disp[df_disp["alpha_id"].str.contains(search, case=False, na=False)]
        
    st.dataframe(df_disp, use_container_width=True)


# =============================================================================
# TAB 3: EXISTING TRADING OBSERVABILITY COMPONENT (PRESERVED)
# =============================================================================

def render_trading_observability(df_replay, df_paper, df_live, df_diagnostics):
    st.title("Trading & Execution Observability")
    st.caption("Observability around the TradingEngine lifecycle across REPLAY, PAPER, and LIVE modes.")
    
    tab_replay, tab_paper, tab_live = st.tabs(["REPLAY", "PAPER", "LIVE"])
    
    with tab_replay:
        st.subheader("Replay Diagnostics")
        if not df_diagnostics.empty:
            st.dataframe(df_diagnostics, use_container_width=True)
            
            # Highlight drops
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
    df_alphas, df_replay, df_paper, df_live, df_diagnostics = load_alpha_trading_data()

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
        render_alpha_factory(df_alphas)

    with tab_trading:
        render_trading_observability(df_replay, df_paper, df_live, df_diagnostics)

    with tab_system:
        render_system_observability()


if __name__ == "__main__":
    main()
