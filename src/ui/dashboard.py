import streamlit as st
import pandas as pd
from src.ui.data_access import UIDataAccess

st.set_page_config(page_title="Ashva Observability", layout="wide")

@st.cache_data(ttl=5)
def load_data():
    dal = UIDataAccess()
    df_alphas = dal.get_alpha_registry_summary()
    df_replay = dal.get_trading_state("REPLAY")
    df_paper = dal.get_trading_state("PAPER")
    df_live = dal.get_trading_state("LIVE")
    df_diagnostics = dal.get_replay_diagnostics()
    return df_alphas, df_replay, df_paper, df_live, df_diagnostics

def render_alpha_factory(df_alphas: pd.DataFrame):
    st.header("Alpha Factory Registry")
    
    # 1. Summary Metrics
    col1, col2, col3, col4 = st.columns(4)
    total = len(df_alphas) if not df_alphas.empty else 0
    proven = len(df_alphas[df_alphas["dynamic_status"] == "CAPITAL_CANDIDATE"]) if "dynamic_status" in df_alphas.columns else 0
    # In knowledge_map it uses "status", but dynamic uses "dynamic_status"
    failed = len(df_alphas[df_alphas["status"] == "EXPLORED_FAILED"]) if "status" in df_alphas.columns else 0
    
    col1.metric("Total Alphas", total)
    col2.metric("Proven / Capital Candidate", proven)
    col3.metric("Failed / Rejected", failed)
    
    if df_alphas.empty:
        st.warning("No alphas found in registry.")
        return

    # 2. DataFrame with search
    search = st.text_input("Search Alpha ID")
    df_disp = df_alphas
    if search:
        df_disp = df_disp[df_disp["alpha_id"].str.contains(search, case=False, na=False)]
        
    st.dataframe(df_disp, use_container_width=True)

def render_trading_observability(df_replay, df_paper, df_live, df_diagnostics):
    st.header("Trading Observability")
    
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

def main():
    st.title("Ashva Quantitative Engine")
    
    df_alphas, df_replay, df_paper, df_live, df_diagnostics = load_data()
    
    tab1, tab2 = st.tabs(["Alpha Factory", "Trading & Execution"])
    
    with tab1:
        render_alpha_factory(df_alphas)
        
    with tab2:
        render_trading_observability(df_replay, df_paper, df_live, df_diagnostics)

if __name__ == "__main__":
    main()
