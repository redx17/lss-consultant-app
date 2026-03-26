import streamlit as st
import pandas as pd
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
client = OpenAI(base_url="http://127.0.0.1:1234/v1", api_key="lm-studio")

# --- 2. GLOBAL STATE MANAGEMENT ---
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
    "project_status": "Green",
    "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M"),
    "ai_recommendations": {"charter": "", "measure": "", "analyze": "", "improve": "", "control": ""}
}

for key, val in state_defaults.items():
    if key not in st.session_state:
        st.session_state[key] = val

# --- 3. CORE ANALYTICS & VISUALIZATION ---

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
        models = client.models.list()
        active_model = models.data[0].id if models.data else "model-identifier"
        system_prompt = (
            "You are a Senior Lean Six Sigma Master Black Belt Consultant specializing in Healthcare. "
            "Critically analyze the provided data for waste, risk, and ROI. Reference specific user data."
        )
        user_msg = f"Review this {section_name} data: {context_data}. Provide 3-4 strategic recommendations."
        resp = client.chat.completions.create(
            model=active_model,
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_msg}]
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"AI connection error: {str(e)}"

# --- 4. SIDEBAR (Restored Project Pulse) ---
with st.sidebar:
    st.title("💼 Consulting Suite")
    
    # RESTORED: Project Pulse Status logic from Image_0fa446
    st.subheader("🚦 Project Pulse")
    status_map = {"Green": "🟢 On Track", "Amber": "🟡 Delayed / Flagged", "Red": "🔴 At Risk / Critical"}
    st.session_state.project_status = st.selectbox(
        "Update Status:", 
        options=list(status_map.keys()), 
        format_func=lambda x: status_map[x], 
        index=list(status_map.keys()).index(st.session_state.project_status)
    )
    
    # Status messaging based on selection
    if st.session_state.project_status == "Green":
        st.success("Project is meeting all milestones.")
    elif st.session_state.project_status == "Amber":
        st.warning("Project requires attention.")
    else:
        st.error("Critical bottlenecks detected.")

    st.markdown("---")
    st.session_state.facility_name = st.text_input("Healthcare Facility:", st.session_state.facility_name)
    st.session_state.dept_name = st.selectbox("Department:", ["ER", "Radiology", "Pharmacy", "OR", "Lab", "Inpatient"])
    
    st.markdown("---")
    load_project = st.file_uploader("Load Project Archive (.json)", type=["json"])
    if load_project:
        try:
            p_data = json.load(load_project)
            for k, v in p_data.items(): st.session_state[k] = v
            st.success("Archive Loaded!")
        except: st.error("Invalid file.")
    
    save_payload = {k: st.session_state[k] for k in state_defaults.keys()}
    st.download_button("📥 Export Full Case Study", data=json.dumps(save_payload), file_name=f"LSS_{st.session_state.project_title.replace(' ', '_')}.json")
    
    if st.button("🗑️ Purge All Case Data"):
        st.session_state.clear()
        st.rerun()

# --- 5. MAIN INTERFACE ---
st.title(f"🏥 {st.session_state.facility_name}: Lean Six Sigma Engine")
st.subheader(f"Project: {st.session_state.project_title}")

tabs = st.tabs(["📋 DEFINE (Charter)", "📊 MEASURE (Data)", "🔍 ANALYZE/IMPROVE", "🚀 CONTROL/EVOLVE", "📄 EXECUTIVE SUMMARY"])

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

    if st.button("🧙‍♂️ Get AI Consultant Draft (Define)"):
        st.session_state.ai_recommendations["charter"] = get_ai_consultant_advice("Charter & SIPOC", {"charter": st.session_state.charter, "sipoc": st.session_state.sipoc})
        st.rerun()
    if st.session_state.ai_recommendations["charter"]:
        st.info(st.session_state.ai_recommendations["charter"])

# --- TAB 2: MEASURE ---
with tabs[1]:
    st.header("Phase 2: Data Baseline & Sigma Analysis")
    u_file = st.file_uploader("Upload Baseline CSV/XLSX", type=["csv", "xlsx"])
    if u_file:
        df = pd.read_csv(u_file) if u_file.name.endswith('.csv') else pd.read_excel(u_file)
        st.dataframe(df.head())
        if st.button("🧙‍♂️ Get AI Data Advice"):
            st.session_state.ai_recommendations["measure"] = get_ai_consultant_advice("Baseline Data", df.describe().to_json())
            st.rerun()

# --- TAB 3: ANALYZE/IMPROVE ---
with tabs[2]:
    st.header("Phase 3 & 4: Root Cause & Optimization")
    
    # Fishbone Section
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
    
    # VSM & TIMWOOD Section
    st.subheader("Value Stream Map & Waste Identification")
    with st.container(border=True):
        c1, c2, c3, c4 = st.columns([2, 1, 1, 2])
        sn = c1.text_input("Step Name:")
        stm = c2.number_input("Cycle Time (m):", min_value=0)
        sw = c3.number_input("Wait Time (m):", min_value=0)
        wt = c4.selectbox("Classification (TIMWOOD):", ["Value Added (VA)", "Transportation", "Inventory", "Motion", "Waiting", "Overproduction", "Overprocessing", "Defects"])
        
        if st.button("➕ Add Process Step"):
            st.session_state.process_steps.append({"name": sn, "time": stm, "wait": sw, "waste_type": wt})
            st.rerun()

    if st.session_state.process_steps:
        st.graphviz_chart(generate_flow_diag(st.session_state.process_steps, "Current State Flow"))
        
        # WASTE SUMMARY TABLE
        st.subheader("Waste Analysis Summary")
        w_df = pd.DataFrame(st.session_state.process_steps)
        summary = w_df.groupby("waste_type")[["time", "wait"]].sum()
        st.table(summary)

    if st.button("🧙‍♂️ Get AI Optimization Advice"):
        st.session_state.ai_recommendations["analyze"] = get_ai_consultant_advice("Fishbone & VSM Waste", {"fishbone": st.session_state.fishbone, "vsm": st.session_state.process_steps})
        st.rerun()

# --- TAB 5: SUMMARY ---
with tabs[4]:
    st.header("📄 Executive Summary")
    st.write(f"**Facility:** {st.session_state.facility_name} | **Project:** {st.session_state.project_title}")
    for phase, rec in st.session_state.ai_recommendations.items():
        if rec:
            with st.expander(f"AI Consultant Advice: {phase.upper()}"):
                st.markdown(rec)