import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from openai import OpenAI
from scipy.stats import norm
import numpy as np
import json
import re
import graphviz
from datetime import datetime

# --- 1. CORE ENGINE CONFIGURATION ---
st.set_page_config(page_title="LSS Master Consultant", layout="wide", page_icon="🏥")

# --- 2. AI CLIENT SETUP ---
if "local_model_id" not in st.session_state:
    # UPDATED: Common Qwen 2.5 format for LM Studio
    st.session_state.local_model_id = "qwen2.5-coder-7b-instruct"

try:
    # Try Streamlit Secrets first (for Cloud deployment)
    api_key = st.secrets["OPENAI_API_KEY"]
    client = OpenAI(api_key=api_key)
    AI_MODEL = "gpt-3.5-turbo" 
except Exception:
    # Fallback to Local LM Studio Server
    client = OpenAI(base_url="http://127.0.0.1:1234/v1", api_key="lm-studio")
    AI_MODEL = st.session_state.local_model_id

# --- 3. GLOBAL STATE MANAGEMENT ---
state_defaults = {
    "project_title": "Process Improvement Initiative",
    "charter": {"bus_case": "", "problem": "", "goal": "", "scope": "", "out_of_scope": "", "financials": ""},
    "team": [{"role": "Project Champion", "name": ""}, {"role": "Black Belt / Lead", "name": ""}],
    "sipoc": {"suppliers": "", "inputs": "", "process": "", "outputs": "", "customers": ""},
    "fishbone": {
        "People": [], "Methods": [], "Machines": [], 
        "Materials": [], "Environment": [], "Measurement": []
    },
    "verified_causes": [], 
    "process_steps": [], 
    "facility_name": "City General Hospital",
    "dept_name": "ER",
    "project_status": "Green",
    "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M"),
    "ai_recommendations": {"charter": "", "measure": "", "analyze": "", "improve": "", "control": ""},
    "chat_history": [],
    "uploaded_df": None 
}

for key, val in state_defaults.items():
    if key not in st.session_state:
        st.session_state[key] = val

# --- 4. CORE ANALYTICS & VISUALIZATION ---

def generate_flow_diag(steps, title):
    dot = graphviz.Digraph(comment=title)
    dot.attr(rankdir='LR', size='12,5', bgcolor='transparent')
    waste_colors = {
        "Value Added (VA)": "#C8E6C9", "Transportation": "#FFCCBC", 
        "Inventory": "#D1C4E9", "Motion": "#F0F4C3", "Waiting": "#FFECB3", 
        "Overproduction": "#B3E5FC", "Overprocessing": "#CFD8DC", "Defects": "#FFCDD2"
    }
    for i, step in enumerate(steps):
        w_type = step.get('waste_type', "Value Added (VA)")
        color = waste_colors.get(w_type, "#FFFFFF")
        shape = "rectangle" if w_type == "Value Added (VA)" else "octagon"
        label = f"{step.get('name')}\n({step.get('time')}m)\n[{w_type}]"
        dot.node(str(i), label, style="filled", fillcolor=color, shape=shape)
        if i > 0: 
            w = step.get('wait', 0)
            dot.edge(str(i-1), str(i), label=f"Wait: {w}m" if w > 0 else "")
    return dot

def render_fishbone(fishbone_data, problem):
    dot = graphviz.Digraph(comment='Fishbone')
    dot.attr(rankdir='RL', size='10,6', bgcolor='transparent')
    dot.node('Effect', f"PROBLEM:\n{problem[:60]}...", shape='rectangle', style='filled', fillcolor='#E74C3C', fontcolor='white')
    for i, (cat, causes) in enumerate(fishbone_data.items()):
        cat_id = f"cat_{i}"
        dot.node(cat_id, cat, shape='ellipse', style='filled', fillcolor='#AED6F1')
        dot.edge(cat_id, 'Effect', penwidth='3')
        for j, cause in enumerate(causes):
            cause_id = f"c_{i}_{j}"
            dot.node(cause_id, cause, shape='none', fontsize='10')
            dot.edge(cause_id, cat_id)
    return dot

def get_ai_consultant_advice(section_name, context_data):
    try:
        system_prompt = (
            "You are a Senior Lean Six Sigma Master Black Belt. "
            "NEVER provide Python code or technical installation steps. "
            "Provide executive-level insights on waste, variation, and ROI. "
            "Speak directly to a business stakeholder."
        )
        user_msg = f"Review this {section_name} data: {context_data}. Provide 3-4 strategic recommendations."
        resp = client.chat.completions.create(
            model=AI_MODEL,
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_msg}]
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"🚨 AI Connection Error: {str(e)}"

# --- 5. SIDEBAR ---
with st.sidebar:
    st.title("💼 Consulting Suite")
    
    with st.expander("⚙️ AI Model Configuration", expanded=True):
        st.warning("Match this ID to the name in LM Studio's top bar.")
        st.session_state.local_model_id = st.text_input("Local Model ID:", value=st.session_state.local_model_id)

    st.subheader("🚦 Project Pulse")
    status_map = {"Green": "🟢 On Track", "Amber": "🟡 Delayed / Flagged", "Red": "🔴 At Risk / Critical"}
    st.session_state.project_status = st.selectbox(
        "Update Status:", options=list(status_map.keys()), format_func=lambda x: status_map[x], 
        index=list(status_map.keys()).index(st.session_state.project_status)
    )
    
    st.markdown("---")
    st.session_state.facility_name = st.text_input("Healthcare Facility:", st.session_state.facility_name)
    st.session_state.dept_name = st.selectbox("Department:", ["ER", "Radiology", "Pharmacy", "OR", "Lab", "Inpatient"],
                                              index=["ER", "Radiology", "Pharmacy", "OR", "Lab", "Inpatient"].index(st.session_state.dept_name))
    
    st.markdown("---")
    load_project = st.file_uploader("Load Project Archive (.json)", type=["json"])
    if load_project:
        try:
            p_data = json.load(load_project)
            for k, v in p_data.items():
                if k == "uploaded_df" and v is not None:
                    st.session_state[k] = pd.DataFrame(v)
                else:
                    st.session_state[k] = v
            st.success("Archive Loaded!")
        except: st.error("Invalid file.")
    
    save_payload = {}
    for k in state_defaults.keys():
        val = st.session_state[k]
        if isinstance(val, pd.DataFrame):
            save_payload[k] = val.to_dict(orient='records')
        else:
            save_payload[k] = val

    st.download_button("📥 Export Case Study", data=json.dumps(save_payload), file_name=f"LSS_Project.json")
    
    if st.button("🗑️ Purge All Case Data"):
        st.session_state.clear()
        st.rerun()

# --- 6. MAIN INTERFACE ---
st.title(f"🏥 {st.session_state.facility_name}: Lean Six Sigma Engine")
st.subheader(f"Project: {st.session_state.project_title}")

tabs = st.tabs(["📋 DEFINE (Charter)", "📊 MEASURE (Data)", "📈 ANALYZE (Visuals)", "🔍 IMPROVE (Root Cause)", "💬 AI CONSULTANT CHAT", "📄 EXECUTIVE SUMMARY"])

# --- TAB 1: DEFINE ---
with tabs[0]:
    st.header("Phase 1: Opportunity Definition")
    st.session_state.project_title = st.text_input("Project Name / Title:", st.session_state.project_title)
    
    col_a, col_b = st.columns(2)
    with col_a:
        with st.container(border=True):
            st.subheader("Project Charter")
            st.session_state.charter["bus_case"] = st.text_area("Business Case (ROI Focus):", st.session_state.charter["bus_case"])
            st.session_state.charter["problem"] = st.text_area("Problem Statement:", st.session_state.charter["problem"])
            st.session_state.charter["goal"] = st.text_area("SMART Goal:", st.session_state.charter["goal"])

    with col_b:
        with st.container(border=True):
            st.subheader("Project Team")
            for i, member in enumerate(st.session_state.team):
                c1, c2 = st.columns([1, 2])
                member["role"] = c1.text_input(f"Role {i}", member["role"], key=f"r_{i}")
                member["name"] = c2.text_input(f"Name {i}", member["name"], key=f"n_{i}")
            if st.button("➕ Add Team Member"):
                st.session_state.team.append({"role": "", "name": ""}); st.rerun()

    with st.container(border=True):
        st.subheader("Project Boundaries")
        s1, s2, s3 = st.columns(3)
        st.session_state.charter["scope"] = s1.text_area("In-Scope:", st.session_state.charter["scope"])
        st.session_state.charter["out_of_scope"] = s2.text_area("Out-of-Scope:", st.session_state.charter["out_of_scope"])
        st.session_state.charter["financials"] = s3.text_area("Financials:", st.session_state.charter["financials"])

    st.subheader("SIPOC (High-Level Value Chain)")
    sc = st.columns(5)
    sipoc_keys = ["suppliers", "inputs", "process", "outputs", "customers"]
    for i, key in enumerate(sipoc_keys):
        st.session_state.sipoc[key] = sc[i].text_area(key.capitalize(), st.session_state.sipoc[key], key=f"sipoc_{key}")

    if st.button("🧙‍♂️ Get AI Charter Draft"):
        st.session_state.ai_recommendations["charter"] = get_ai_consultant_advice("Charter & SIPOC", {"charter": st.session_state.charter, "sipoc": st.session_state.sipoc})
        st.rerun()
    if st.session_state.ai_recommendations["charter"]:
        st.info(st.session_state.ai_recommendations["charter"])

# --- TAB 2: MEASURE ---
with tabs[1]:
    st.header("Phase 2: Data Baseline & Pareto Analysis")
    u_file = st.file_uploader("Upload Baseline CSV/XLSX", type=["csv", "xlsx"], key="measure_upload")
    
    if u_file:
        df = pd.read_csv(u_file) if u_file.name.endswith('.csv') else pd.read_excel(u_file)
        st.session_state.uploaded_df = df
        st.success(f"✅ Data Locked!")

    if st.session_state.uploaded_df is not None:
        working_df = st.session_state.uploaded_df
        st.subheader("Pareto Analysis")
        p_col1, p_col2 = st.columns([1, 2])
        with p_col1:
            all_cols = [str(c) for c in working_df.columns]
            target_cat = st.selectbox("Select Category Column:", ["-- Select --"] + all_cols)
        
        if target_cat != "-- Select --":
            raw_data = working_df[target_cat].astype(str).str.strip()
            clean_data = raw_data[raw_data.str.lower() != 'nan']
            if not clean_data.empty:
                counts = clean_data.value_counts().reset_index()
                counts.columns = ["Category", "Count"]
                counts = counts.sort_values(by="Count", ascending=False)
                counts["Cum %"] = (counts["Count"].cumsum() / counts["Count"].sum()) * 100
                with p_col2:
                    fig = go.Figure()
                    fig.add_trace(go.Bar(x=counts["Category"], y=counts["Count"], name="Frequency"))
                    fig.add_trace(go.Scatter(x=counts["Category"], y=counts["Cum %"], name="Cumulative %", yaxis="y2", line=dict(color="red")))
                    fig.update_layout(yaxis2=dict(overlaying="y", side="right", range=[0, 110]), template="plotly_white")
                    st.plotly_chart(fig, use_container_width=True)

        if st.button("🧙‍♂️ Get AI Data Analysis"):
            st.session_state.ai_recommendations["measure"] = get_ai_consultant_advice("Baseline Data", working_df.describe().to_json())
            st.rerun()
    if st.session_state.ai_recommendations["measure"]:
        st.info(st.session_state.ai_recommendations["measure"])

# --- TAB 3: ANALYZE ---
with tabs[2]:
    st.header("Phase 3: Statistical Analysis")
    if st.session_state.uploaded_df is not None:
        df = st.session_state.uploaded_df
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

        if num_cols:
            viz_type = st.radio("Select Analysis Tool:", ["Control Chart", "Box Plot", "Histogram"], horizontal=True)
            if viz_type == "Control Chart":
                col = st.selectbox("Select Numeric Metric:", num_cols, key="cc_sel")
                mean, std = df[col].mean(), df[col].std()
                fig = px.line(df, y=col, title=f"Stability Chart: {col}")
                fig.add_hline(y=mean, line_color="green")
                fig.add_hline(y=mean+(3*std), line_dash="dot", line_color="red")
                fig.add_hline(y=mean-(3*std), line_dash="dot", line_color="red")
                st.plotly_chart(fig, use_container_width=True)
            elif viz_type == "Box Plot" and cat_cols:
                c_col = st.selectbox("Group By:", cat_cols, key="box_cat")
                v_col = st.selectbox("Value:", num_cols, key="box_val")
                st.plotly_chart(px.box(df, x=c_col, y=v_col, color=c_col), use_container_width=True)
            elif viz_type == "Histogram":
                h_col = st.selectbox("Select Metric:", num_cols, key="hist_sel")
                st.plotly_chart(px.histogram(df, x=h_col, marginal="box"), use_container_width=True)
    else: st.warning("Upload data in the MEASURE tab first.")

# --- TAB 4: IMPROVE ---
with tabs[3]:
    st.header("Phase 4: Root Cause & Optimization")
    st.subheader("Fishbone (Ishikawa) Diagram")
    f_cols = st.columns(3)
    for i, cat in enumerate(st.session_state.fishbone.keys()):
        with f_cols[i % 3]:
            nc = st.text_input(f"New {cat} Cause:", key=f"f_{cat}")
            if st.button(f"Add {cat}", key=f"b_{cat}") and nc:
                st.session_state.fishbone[cat].append(nc); st.rerun()
    if any(st.session_state.fishbone.values()):
        st.graphviz_chart(render_fishbone(st.session_state.fishbone, st.session_state.charter["problem"]))

    st.markdown("---")
    st.subheader("Value Stream Map (TIMWOOD)")
    c1, c2, c3, c4 = st.columns([2, 1, 1, 2])
    sn = c1.text_input("Step Name:")
    stm = c2.number_input("Cycle Time (m):", min_value=0)
    sw = c3.number_input("Wait Time (m):", min_value=0)
    wt = c4.selectbox("Waste Classification:", ["Value Added (VA)", "Transportation", "Inventory", "Motion", "Waiting", "Overproduction", "Overprocessing", "Defects"])
    if st.button("➕ Add Process Step"):
        st.session_state.process_steps.append({"name": sn, "time": stm, "wait": sw, "waste_type": wt})
        st.rerun()

    if st.session_state.process_steps:
        st.graphviz_chart(generate_flow_diag(st.session_state.process_steps, "Current Flow"))
        if st.button("🧙‍♂️ Get AI Optimization Advice"):
            improve_context = {"problem": st.session_state.charter["problem"], "fishbone": st.session_state.fishbone, "steps": st.session_state.process_steps}
            st.session_state.ai_recommendations["improve"] = get_ai_consultant_advice("Fishbone & VSM", improve_context)
            st.rerun()
            
    if st.session_state.ai_recommendations["improve"]:
        st.success(st.session_state.ai_recommendations["improve"])

# --- TAB 5: AI CONSULTANT CHAT ---
with tabs[4]:
    st.header("🎓 Senior Master Black Belt Consultant")
    chat_container = st.container(height=500)
    for msg in st.session_state.chat_history:
        with chat_container.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("Ask about waste drivers..."):
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        with chat_container.chat_message("user"):
            st.markdown(prompt)

        with chat_container.chat_message("assistant"):
            data_context = "No data uploaded."
            if st.session_state.uploaded_df is not None:
                data_context = f"Columns: {st.session_state.uploaded_df.columns.tolist()}"

            SYSTEM_PERSONA = "You are an Elite Lean Six Sigma Master Black Belt. NEVER provide code."
            full_prompt = f"Problem: {st.session_state.charter['problem']}\nContext: {data_context}\nQuestion: {prompt}"
            
            try:
                response = client.chat.completions.create(
                    model=AI_MODEL,
                    messages=[{"role": "system", "content": SYSTEM_PERSONA}, {"role": "user", "content": full_prompt}]
                )
                answer = response.choices[0].message.content
                st.markdown(answer)
                st.session_state.chat_history.append({"role": "assistant", "content": answer})
            except Exception as e: st.error(f"Chat Error: {str(e)}")

# --- TAB 6: SUMMARY ---
with tabs[5]:
    st.header("📄 Executive Summary")
    st.write(f"**Facility:** {st.session_state.facility_name} | **Project:** {st.session_state.project_title}")
    for phase, rec in st.session_state.ai_recommendations.items():
        if rec:
            with st.expander(f"Strategic Insight: {phase.upper()}"):
                st.markdown(rec)
