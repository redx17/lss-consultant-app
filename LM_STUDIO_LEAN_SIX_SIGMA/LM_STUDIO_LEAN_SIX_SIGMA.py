from __future__ import annotations

import json
from datetime import datetime

import reflex as rx

from .core import (
    control_chart_payload,
    descriptive_stats_payload,
    get_ai_chat_response,
    get_ai_consultant_advice,
    load_tabular_data,
    pareto_payload,
    preview_payload,
)


SURFACE = "#f4f7fb"
PANEL = "#ffffff"
PANEL_ALT = "#f8fafc"
SIDEBAR = "#0f172a"
PRIMARY = "#0f4c81"
PRIMARY_SOFT = "#dbeafe"
ACCENT = "#0ea5e9"
TEXT = "#0f172a"
TEXT_MUTED = "#475569"
TEXT_SOFT = "#64748b"
BORDER = "#d9e2ec"
SUCCESS = "#166534"
WARNING = "#9a3412"
DANGER = "#b91c1c"
GRADIENT = "linear-gradient(135deg, #0f4c81 0%, #0ea5e9 100%)"

PHASES = ["Define", "Measure", "Analyze", "Improve", "Consult", "Summary"]
WASTE_TYPES = [
    "Value Added (VA)",
    "Transportation",
    "Inventory",
    "Motion",
    "Waiting",
    "Overproduction",
    "Overprocessing",
    "Defects",
]
FISHBONE_CATEGORIES = ["People", "Methods", "Machines", "Materials", "Environment", "Measurement"]

DEFAULT_TEAM = [
    {"role": "Project Champion", "name": "Dana Brooks"},
    {"role": "Black Belt Lead", "name": "Marcus Chen"},
    {"role": "Process Owner", "name": "Elena Ruiz"},
]
DEFAULT_FISHBONE = {category: [] for category in FISHBONE_CATEGORIES}
DEFAULT_PREVIEW_ROWS = [
    {"c1": "2026-03-30 14:22", "c2": "BT-9902", "c3": "Calibration Drift", "c4": "Fail"},
    {"c1": "2026-03-30 14:18", "c2": "BT-9901", "c3": "Material Purity", "c4": "Pass"},
]
DEFAULT_PARETO = [
    {"label": "Machine Calibration", "count": 542, "percent": "42%", "height": "100%"},
    {"label": "Material Impurity", "count": 382, "percent": "30%", "height": "70%"},
    {"label": "Operator Error", "count": 241, "percent": "19%", "height": "45%"},
]
DEFAULT_CONTROL = [
    {"line_height": "54px", "high": False},
    {"line_height": "62px", "high": False},
    {"line_height": "78px", "high": True},
    {"line_height": "70px", "high": False},
    {"line_height": "88px", "high": True},
    {"line_height": "82px", "high": False},
]


class State(rx.State):
    active_phase: str = "Define"
    project_title: str = ""
    facility_name: str = ""
    department_name: str = ""
    project_status: str = "Green"
    last_saved_at: str = "Not saved yet"
    workspace_message: str = "Workspace ready."
    ai_model_id: str = "qwen2.5-coder-7b-instruct"

    business_case: str = ""
    problem_statement: str = ""
    goal_statement: str = ""
    in_scope: str = ""
    out_of_scope: str = ""
    financial_target: str = ""
    sipoc_suppliers: str = ""
    sipoc_inputs: str = ""
    sipoc_process: str = ""
    sipoc_outputs: str = ""
    sipoc_customers: str = ""
    team: list[dict[str, str]] = DEFAULT_TEAM
    define_guidance: str = "Use the charter to lock the financial baseline and approval criteria."

    uploaded_file_name: str = "Demo baseline dataset"
    uploaded_records: list[dict] = []
    data_columns: list[str] = []
    numeric_columns: list[str] = []
    categorical_columns: list[str] = []
    selected_pareto_column: str = ""
    selected_metric_column: str = ""
    pareto_data: list[dict] = DEFAULT_PARETO
    preview_header_1: str = "Timestamp"
    preview_header_2: str = "Batch ID"
    preview_header_3: str = "Defect Type"
    preview_header_4: str = "Status"
    preview_rows: list[dict[str, str]] = DEFAULT_PREVIEW_ROWS
    active_samples: str = "1204"
    process_stability: str = "84.2%"
    measure_status: str = "Load demo data or upload CSV/XLSX to compute live analysis."
    measure_guidance: str = "Pareto and preview will update from the uploaded dataset."

    control_chart_points: list[dict] = DEFAULT_CONTROL
    control_mean: str = "142.10"
    control_ucl: str = "185.20"
    control_lcl: str = "98.90"
    metric_min: str = "112.00"
    metric_max: str = "158.00"
    metric_std: str = "14.37"
    analyze_note: str = ""
    analyze_guidance: str = "Validate measurement reliability before changing process settings."

    fishbone: dict[str, list[str]] = DEFAULT_FISHBONE
    new_people_cause: str = ""
    new_methods_cause: str = ""
    new_machines_cause: str = ""
    new_materials_cause: str = ""
    new_environment_cause: str = ""
    new_measurement_cause: str = ""
    process_steps: list[dict[str, str | int]] = []
    step_name: str = ""
    step_time: int = 0
    step_wait: int = 0
    step_waste: str = "Value Added (VA)"
    improve_note: str = ""
    improve_guidance: str = "Sequence actions by expected impact, implementation effort, and change risk."

    chat_history: list[dict[str, str]] = []
    chat_input_text: str = ""

    def set_active_phase(self, phase: str):
        self.active_phase = phase

    def set_project_title(self, value: str): self.project_title = value
    def set_facility_name(self, value: str): self.facility_name = value
    def set_department_name(self, value: str): self.department_name = value
    def set_business_case(self, value: str): self.business_case = value
    def set_problem_statement(self, value: str): self.problem_statement = value
    def set_goal_statement(self, value: str): self.goal_statement = value
    def set_in_scope(self, value: str): self.in_scope = value
    def set_out_of_scope(self, value: str): self.out_of_scope = value
    def set_financial_target(self, value: str): self.financial_target = value
    def set_sipoc_suppliers(self, value: str): self.sipoc_suppliers = value
    def set_sipoc_inputs(self, value: str): self.sipoc_inputs = value
    def set_sipoc_process(self, value: str): self.sipoc_process = value
    def set_sipoc_outputs(self, value: str): self.sipoc_outputs = value
    def set_sipoc_customers(self, value: str): self.sipoc_customers = value
    def set_analyze_note(self, value: str): self.analyze_note = value
    def set_improve_note(self, value: str): self.improve_note = value
    def set_chat_input_text(self, value: str): self.chat_input_text = value
    def set_step_name(self, value: str): self.step_name = value
    def set_step_time(self, value): self.step_time = int(value)
    def set_step_wait(self, value): self.step_wait = int(value)
    def set_step_waste(self, value: str): self.step_waste = value
    def set_new_people_cause(self, value: str): self.new_people_cause = value
    def set_new_methods_cause(self, value: str): self.new_methods_cause = value
    def set_new_machines_cause(self, value: str): self.new_machines_cause = value
    def set_new_materials_cause(self, value: str): self.new_materials_cause = value
    def set_new_environment_cause(self, value: str): self.new_environment_cause = value
    def set_new_measurement_cause(self, value: str): self.new_measurement_cause = value

    def set_project_status(self, status: str):
        self.project_status = status
        self.workspace_message = f"Project pulse updated to {status}."

    def save_workspace(self):
        self.last_saved_at = datetime.now().strftime("%Y-%m-%d %H:%M")
        self.workspace_message = f"Workspace saved at {self.last_saved_at}."

    def _metric_chip(self, current: str, fallback: str) -> str:
        return current if current else fallback

    def load_demo_data(self):
        self.uploaded_file_name = "demo_baseline.csv"
        self.active_samples = "1204"
        self.process_stability = "84.2%"
        self.pareto_data = DEFAULT_PARETO
        self.control_chart_points = DEFAULT_CONTROL
        self.control_mean = "142.10"
        self.control_ucl = "185.20"
        self.control_lcl = "98.90"
        self.metric_min = "112.00"
        self.metric_max = "158.00"
        self.metric_std = "14.37"
        self.measure_status = "Loaded demo baseline data."
        self.measure_guidance = "Demo analysis loaded. Upload your own file to replace it."

    def add_team_member(self):
        self.team.append({"role": "New Role", "name": ""})

    def update_team_role(self, index: int, value: str):
        if 0 <= index < len(self.team):
            self.team[index]["role"] = value
            self.team = [dict(item) for item in self.team]

    def update_team_name(self, index: int, value: str):
        if 0 <= index < len(self.team):
            self.team[index]["name"] = value
            self.team = [dict(item) for item in self.team]

    def set_selected_pareto_column(self, value: str):
        self.selected_pareto_column = value
        self._recompute_from_records()

    def set_selected_metric_column(self, value: str):
        self.selected_metric_column = value
        self._recompute_from_records()

    def _recompute_from_records(self):
        if not self.uploaded_records:
            return
        import pandas as pd

        df = pd.DataFrame(self.uploaded_records)
        if self.selected_pareto_column:
            self.pareto_data = pareto_payload(df, self.selected_pareto_column)
        if self.selected_metric_column:
            control = control_chart_payload(df, self.selected_metric_column)
            stats = descriptive_stats_payload(df, self.selected_metric_column)
            self.control_chart_points = control["points"] or DEFAULT_CONTROL
            self.control_mean = control["mean"] or self.control_mean
            self.control_ucl = control["ucl"] or self.control_ucl
            self.control_lcl = control["lcl"] or self.control_lcl
            self.metric_min = stats["min"] or self.metric_min
            self.metric_max = stats["max"] or self.metric_max
            self.metric_std = stats["std"] or self.metric_std

    async def handle_upload(self, files: list[rx.UploadFile]):
        if not files:
            self.measure_status = "No file selected."
            return
        file = files[0]
        self.uploaded_file_name = file.filename
        try:
            raw = await file.read()
            df = load_tabular_data(raw, file.filename)
            payload = preview_payload(df)
            self.uploaded_records = payload["records"]
            self.data_columns = payload["columns"]
            self.numeric_columns = payload["numeric_columns"]
            self.categorical_columns = payload["categorical_columns"]
            headers = (self.data_columns + ["Column 1", "Column 2", "Column 3", "Column 4"])[:4]
            self.preview_header_1, self.preview_header_2, self.preview_header_3, self.preview_header_4 = headers
            self.preview_rows = [
                {
                    "c1": str(row.get(headers[0], "")),
                    "c2": str(row.get(headers[1], "")),
                    "c3": str(row.get(headers[2], "")),
                    "c4": str(row.get(headers[3], "")),
                }
                for row in payload["records"]
            ] or DEFAULT_PREVIEW_ROWS
            self.active_samples = str(payload["row_count"])
            self.process_stability = "Loaded"
            self.selected_pareto_column = self.categorical_columns[0] if self.categorical_columns else (self.data_columns[0] if self.data_columns else "")
            self.selected_metric_column = self.numeric_columns[0] if self.numeric_columns else ""
            self._recompute_from_records()
            self.measure_status = f"Loaded {file.filename} with {self.active_samples} rows."
            self.measure_guidance = "Uploaded dataset is now driving the Measure and Analyze views."
        except Exception as exc:
            self.measure_status = f"Upload error: {exc}"

    def add_cause(self, category: str):
        field_name = f"new_{category.lower()}_cause"
        value = getattr(self, field_name, "").strip()
        if value:
            self.fishbone[category].append(value)
            self.fishbone = {key: list(items) for key, items in self.fishbone.items()}
            setattr(self, field_name, "")

    def add_process_step(self):
        if not self.step_name.strip():
            return
        self.process_steps.append(
            {
                "name": self.step_name.strip(),
                "time": int(self.step_time),
                "wait": int(self.step_wait),
                "waste_type": self.step_waste,
            }
        )
        self.process_steps = [dict(item) for item in self.process_steps]
        self.step_name = ""
        self.step_time = 0
        self.step_wait = 0
        self.step_waste = "Value Added (VA)"

    def get_define_ai_advice(self):
        context = {
            "project_title": self.project_title,
            "charter": {
                "business_case": self.business_case,
                "problem_statement": self.problem_statement,
                "goal_statement": self.goal_statement,
                "in_scope": self.in_scope,
                "out_of_scope": self.out_of_scope,
                "financial_target": self.financial_target,
            },
            "sipoc": {
                "suppliers": self.sipoc_suppliers,
                "inputs": self.sipoc_inputs,
                "process": self.sipoc_process,
                "outputs": self.sipoc_outputs,
                "customers": self.sipoc_customers,
            },
        }
        self.define_guidance = get_ai_consultant_advice("Charter & SIPOC", context, self.ai_model_id)

    def get_measure_ai_advice(self):
        context = {
            "file_name": self.uploaded_file_name,
            "row_count": self.active_samples,
            "pareto_data": self.pareto_data,
            "selected_pareto_column": self.selected_pareto_column,
        }
        self.measure_guidance = get_ai_consultant_advice("Baseline Data", context, self.ai_model_id)

    def get_analyze_ai_advice(self):
        context = {
            "selected_metric": self.selected_metric_column,
            "mean": self.control_mean,
            "ucl": self.control_ucl,
            "lcl": self.control_lcl,
            "note": self.analyze_note,
        }
        self.analyze_guidance = get_ai_consultant_advice("Analyze Phase", context, self.ai_model_id)

    def get_improve_ai_advice(self):
        context = {
            "problem_statement": self.problem_statement,
            "fishbone": self.fishbone,
            "process_steps": self.process_steps,
            "improve_note": self.improve_note,
        }
        self.improve_guidance = get_ai_consultant_advice("Improve Phase", context, self.ai_model_id)

    async def send_chat(self):
        prompt = self.chat_input_text.strip()
        if not prompt:
            return
        self.chat_history.append({"role": "user", "content": prompt})
        self.chat_input_text = ""
        reply = get_ai_chat_response(
            prompt=prompt,
            problem_statement=self.problem_statement,
            uploaded_columns=self.data_columns,
            model_id=self.ai_model_id,
        )
        self.chat_history.append({"role": "assistant", "content": reply})


def card(*children, **style):
    base = {
        "background": PANEL,
        "border": f"1px solid {BORDER}",
        "border_radius": "18px",
        "padding": "1.25rem",
        "box_shadow": "0 10px 24px rgba(15, 23, 42, 0.05)",
        "width": "100%",
    }
    base.update(style)
    return rx.box(rx.vstack(*children, spacing="4", align_items="stretch"), **base)


def section_title(title: str, subtitle: str):
    return rx.vstack(
        rx.heading(title, size="5", color=TEXT),
        rx.text(subtitle, color=TEXT_SOFT, font_size="0.9rem"),
        spacing="1",
        align_items="start",
    )


def chip(text: rx.Var | str, bg: str = PRIMARY_SOFT, color: str = PRIMARY):
    return rx.box(
        rx.text(text, font_size="0.74rem", font_weight="600", color=color),
        background=bg,
        padding="0.38rem 0.7rem",
        border_radius="999px",
        width="fit-content",
    )


def side_button(label: str):
    active = State.active_phase == label
    return rx.button(
        label,
        on_click=lambda: State.set_active_phase(label),
        justify="start",
        width="100%",
        border="none",
        border_radius="12px",
        padding="0.85rem 1rem",
        background=rx.cond(active, "rgba(255,255,255,0.12)", "transparent"),
        color="white",
        font_weight=rx.cond(active, "700", "500"),
        _hover={"background": "rgba(255,255,255,0.12)"},
    )


def top_tab(label: str):
    active = State.active_phase == label
    return rx.button(
        label,
        on_click=lambda: State.set_active_phase(label),
        background="transparent",
        color=rx.cond(active, PRIMARY, TEXT_MUTED),
        border="none",
        border_bottom=rx.cond(active, f"2px solid {PRIMARY}", "2px solid transparent"),
        border_radius="0",
        padding="0.85rem 0.1rem",
        font_weight=rx.cond(active, "700", "500"),
    )


def editor(label: str, value: rx.Var, on_change, height: str = "140px"):
    return rx.vstack(
        rx.text(label, color=TEXT_SOFT, font_size="0.78rem", font_weight="700", text_transform="uppercase"),
        rx.text_area(
            value=value,
            on_change=on_change,
            min_height=height,
            background=PANEL_ALT,
            border=f"1px solid {BORDER}",
            color=TEXT,
            placeholder="Enter information...",
            style={"color": TEXT, "::placeholder": {"color": TEXT_SOFT}},
        ),
        spacing="2",
        align_items="stretch",
    )


def stat_tile(label: str, value: rx.Var | str, note: rx.Var | str):
    return rx.box(
        rx.vstack(
            rx.text(label, color=TEXT_SOFT, font_size="0.76rem", font_weight="700", text_transform="uppercase"),
            rx.heading(value, size="6", color=TEXT),
            rx.text(note, color=TEXT_MUTED, font_size="0.85rem"),
            spacing="1",
            align_items="start",
        ),
        background=PANEL_ALT,
        border=f"1px solid {BORDER}",
        border_radius="16px",
        padding="1rem",
    )


def team_row(item: rx.Var, index: int):
    return rx.grid(
        rx.input(value=item["role"], on_change=lambda value: State.update_team_role(index, value), background=PANEL_ALT, color=TEXT, placeholder="Role", style={"color": TEXT, "::placeholder": {"color": TEXT_SOFT}}),
        rx.input(value=item["name"], on_change=lambda value: State.update_team_name(index, value), background=PANEL_ALT, color=TEXT, placeholder="Name", style={"color": TEXT, "::placeholder": {"color": TEXT_SOFT}}),
        columns=rx.breakpoints(initial="1", md="2"),
        spacing="3",
        width="100%",
    )


def pareto_bar(item: rx.Var):
    return rx.vstack(
        rx.text(item["count"], font_weight="700", color=TEXT, font_size="0.8rem"),
        rx.box(width="100%", height=item["height"], min_height="32px", background=GRADIENT, border_radius="12px 12px 4px 4px"),
        rx.text(item["label"], color=TEXT_MUTED, text_align="center", font_size="0.75rem"),
        rx.text(item["percent"], color=PRIMARY, font_size="0.72rem", font_weight="700"),
        justify="end",
        align_items="center",
        height="100%",
        spacing="2",
    )


def preview_row(row: rx.Var):
    return rx.table.row(
        rx.table.cell(row["c1"]),
        rx.table.cell(row["c2"]),
        rx.table.cell(row["c3"]),
        rx.table.cell(row["c4"]),
    )


def control_point(point: rx.Var):
    return rx.vstack(
        rx.box(width="10px", height="10px", border_radius="999px", background=rx.cond(point["high"], ACCENT, PRIMARY)),
        rx.box(width="2px", height=point["line_height"], background="#cbd5e1"),
        justify="end",
        align_items="center",
        height="100%",
        spacing="2",
    )


def cause_block(category: str, value: rx.Var, setter, action):
    return card(
        rx.text(category, font_weight="700", color=TEXT),
        rx.input(value=value, on_change=setter, placeholder=f"Add a {category.lower()} cause", background=PANEL_ALT, color=TEXT, style={"color": TEXT, "::placeholder": {"color": TEXT_SOFT}}),
        rx.button(f"Add {category}", on_click=action, background=PRIMARY, color="white"),
        padding="1rem",
    )


def process_step_row(step: rx.Var):
    return rx.hstack(
        chip(step["waste_type"], bg="#e2e8f0", color=TEXT),
        rx.text(step["name"], font_weight="600", color=TEXT),
        rx.spacer(),
        rx.hstack(
            rx.text(step["time"], color=TEXT_MUTED),
            rx.text("m cycle", color=TEXT_MUTED),
            spacing="1",
            align="center",
        ),
        rx.hstack(
            rx.text(step["wait"], color=TEXT_MUTED),
            rx.text("m wait", color=TEXT_MUTED),
            spacing="1",
            align="center",
        ),
        width="100%",
        padding="0.85rem 1rem",
        border=f"1px solid {BORDER}",
        border_radius="14px",
        background=PANEL_ALT,
    )


def chat_message(msg: rx.Var):
    return rx.hstack(
        rx.cond(msg["role"] == "user", rx.spacer(), rx.box()),
        rx.box(
            rx.text(msg["content"], color=TEXT, white_space="pre-wrap"),
            background=rx.cond(msg["role"] == "user", PRIMARY_SOFT, PANEL_ALT),
            border=f"1px solid {BORDER}",
            border_radius="16px",
            padding="0.9rem 1rem",
            max_width="760px",
        ),
        rx.cond(msg["role"] == "user", rx.box(), rx.spacer()),
        width="100%",
    )


def define_view():
    return rx.vstack(
        rx.grid(
            card(
                section_title("Project Charter", "Editable define-phase workspace linked to your current charter logic."),
                rx.input(value=State.project_title, on_change=State.set_project_title, background=PANEL_ALT, color=TEXT, placeholder="Project title", size="3", style={"color": TEXT, "::placeholder": {"color": TEXT_SOFT}}),
                rx.grid(
                    editor("Business Case", State.business_case, State.set_business_case),
                    editor("Problem Statement", State.problem_statement, State.set_problem_statement),
                    editor("Goal Statement", State.goal_statement, State.set_goal_statement),
                    columns=rx.breakpoints(initial="1", lg="3"),
                    spacing="4",
                ),
                rx.grid(
                    editor("In Scope", State.in_scope, State.set_in_scope, "110px"),
                    editor("Out of Scope", State.out_of_scope, State.set_out_of_scope, "110px"),
                    editor("Financial Target", State.financial_target, State.set_financial_target, "110px"),
                    columns=rx.breakpoints(initial="1", lg="3"),
                    spacing="4",
                ),
                rx.hstack(
                    rx.button("Use Current AI Logic", on_click=State.get_define_ai_advice, background=PRIMARY, color="white"),
                    chip(State.define_guidance, bg="#eff6ff", color=PRIMARY),
                    spacing="3",
                    wrap="wrap",
                ),
                grid_column=rx.breakpoints(initial="1 / -1", xl="1 / 9"),
            ),
            card(
                section_title("Project Pulse", "Operational settings"),
                rx.select(["Green", "Amber", "Red"], value=State.project_status, on_change=State.set_project_status),
                rx.input(value=State.facility_name, on_change=State.set_facility_name, placeholder="Facility", background=PANEL_ALT, color=TEXT, style={"color": TEXT, "::placeholder": {"color": TEXT_SOFT}}),
                rx.input(value=State.department_name, on_change=State.set_department_name, placeholder="Department", background=PANEL_ALT, color=TEXT, style={"color": TEXT, "::placeholder": {"color": TEXT_SOFT}}),
                stat_tile("Status", State.project_status, State.workspace_message),
                grid_column=rx.breakpoints(initial="1 / -1", xl="9 / 13"),
            ),
            columns=rx.breakpoints(initial="1", xl="12"),
            spacing="6",
            width="100%",
        ),
        card(
            section_title("SIPOC", "High-level value chain"),
            rx.grid(
                editor("Suppliers", State.sipoc_suppliers, State.set_sipoc_suppliers, "110px"),
                editor("Inputs", State.sipoc_inputs, State.set_sipoc_inputs, "110px"),
                editor("Process", State.sipoc_process, State.set_sipoc_process, "110px"),
                editor("Outputs", State.sipoc_outputs, State.set_sipoc_outputs, "110px"),
                editor("Customers", State.sipoc_customers, State.set_sipoc_customers, "110px"),
                columns=rx.breakpoints(initial="1", md="2", xl="5"),
                spacing="4",
            ),
        ),
        card(
            section_title("Team", "Project ownership"),
            rx.foreach(State.team, lambda item, index: team_row(item, index)),
            rx.button("Add Team Member", on_click=State.add_team_member, variant="outline"),
        ),
        spacing="6",
        width="100%",
    )


def measure_view():
    return rx.vstack(
        rx.grid(
            card(
                section_title("Dataset Intake", "Upload CSV or XLSX to drive the Measure and Analyze phases."),
                rx.upload(
                    rx.vstack(
                        rx.heading("Upload baseline data", size="5", color=TEXT),
                        rx.text("The uploaded dataset now feeds preview, Pareto, and analysis panels.", color=TEXT_MUTED, text_align="center"),
                        rx.foreach(rx.selected_files("measure_upload"), lambda f: chip(f, bg="#eff6ff", color=PRIMARY)),
                        spacing="3",
                        align_items="center",
                    ),
                    id="measure_upload",
                    border=f"2px dashed {BORDER}",
                    background=PANEL_ALT,
                    border_radius="18px",
                    min_height="220px",
                    padding="2rem",
                    width="100%",
                ),
                rx.hstack(
                    rx.button("Load Demo Data", on_click=State.load_demo_data, variant="outline"),
                    rx.button("Process Upload", on_click=lambda: State.handle_upload(rx.upload_files(upload_id="measure_upload")), background=PRIMARY, color="white"),
                    spacing="3",
                ),
                chip(State.measure_status, bg="#eff6ff", color=PRIMARY),
                grid_column=rx.breakpoints(initial="1 / -1", xl="1 / 9"),
            ),
            rx.vstack(
                stat_tile("Active Samples", State.active_samples, State.uploaded_file_name),
                stat_tile("Process Stability", State.process_stability, State.measure_guidance),
                spacing="4",
                grid_column=rx.breakpoints(initial="1 / -1", xl="9 / 13"),
                width="100%",
            ),
            columns=rx.breakpoints(initial="1", xl="12"),
            spacing="6",
            width="100%",
        ),
        rx.grid(
            card(
                section_title("Pareto Analysis", "Uses the same category-count logic as your existing app."),
                rx.cond(
                    State.data_columns != [],
                    rx.select(State.data_columns, value=State.selected_pareto_column, on_change=State.set_selected_pareto_column, width=rx.breakpoints(initial="100%", md="320px")),
                    rx.text("Upload data to unlock live category selection.", color=TEXT_MUTED),
                ),
                rx.hstack(rx.foreach(State.pareto_data, pareto_bar), align="end", justify="between", height="280px", width="100%", spacing="5"),
                grid_column=rx.breakpoints(initial="1 / -1", xl="1 / 9"),
            ),
            card(
                section_title("Measure Guidance", "Current codebase AI hooks"),
                chip(State.measure_guidance, bg="#eff6ff", color=PRIMARY),
                rx.button("Use Current AI Logic", on_click=State.get_measure_ai_advice, background=PRIMARY, color="white"),
                grid_column=rx.breakpoints(initial="1 / -1", xl="9 / 13"),
            ),
            columns=rx.breakpoints(initial="1", xl="12"),
            spacing="6",
            width="100%",
        ),
        card(
            section_title("Data Preview", "First four columns from the uploaded file"),
            rx.table.root(
                rx.table.header(
                    rx.table.row(
                        rx.table.column_header_cell(State.preview_header_1),
                        rx.table.column_header_cell(State.preview_header_2),
                        rx.table.column_header_cell(State.preview_header_3),
                        rx.table.column_header_cell(State.preview_header_4),
                    )
                ),
                rx.table.body(rx.foreach(State.preview_rows, preview_row)),
                width="100%",
            ),
        ),
        spacing="6",
        width="100%",
    )


def analyze_view():
    return rx.vstack(
        rx.grid(
            card(
                section_title("Statistical Analysis", "Uploaded numeric columns now drive this panel."),
                rx.cond(
                    State.numeric_columns != [],
                    rx.select(State.numeric_columns, value=State.selected_metric_column, on_change=State.set_selected_metric_column, width=rx.breakpoints(initial="100%", md="320px")),
                    rx.text("Upload data with numeric columns to unlock analysis.", color=TEXT_MUTED),
                ),
                rx.hstack(
                    chip("Alpha 0.05", bg="#ecfeff", color=ACCENT),
                    chip(State.active_samples, bg="#eff6ff", color=PRIMARY),
                    spacing="3",
                ),
                rx.box(
                    rx.vstack(
                        rx.box(height="1px", background="#cbd5e1", width="100%"),
                        rx.box(height="1px", background="#cbd5e1", width="100%"),
                        rx.box(height="2px", background="#94a3b8", width="100%"),
                        rx.box(height="1px", background="#cbd5e1", width="100%"),
                        rx.box(height="1px", background="#cbd5e1", width="100%"),
                        position="absolute",
                        inset="1.25rem",
                        justify="between",
                    ),
                    rx.hstack(
                        rx.foreach(State.control_chart_points, control_point),
                        align="end",
                        justify="between",
                        width="100%",
                        height="100%",
                        padding="1.5rem",
                        position="relative",
                        z_index="1",
                    ),
                    position="relative",
                    min_height="320px",
                    background=PANEL_ALT,
                    border=f"1px solid {BORDER}",
                    border_radius="18px",
                    overflow="hidden",
                ),
                grid_column=rx.breakpoints(initial="1 / -1", xl="1 / 8"),
            ),
            card(
                section_title("Metric Readout", "Live descriptive statistics"),
                stat_tile("Mean", State.control_mean, "Selected metric average"),
                stat_tile("UCL", State.control_ucl, "Upper control limit"),
                stat_tile("LCL", State.control_lcl, "Lower control limit"),
                chip("Min: " + State.metric_min, bg="#f8fafc", color=TEXT),
                chip("Max: " + State.metric_max, bg="#f8fafc", color=TEXT),
                chip("Std Dev: " + State.metric_std, bg="#f8fafc", color=TEXT),
                editor("Current Interpretation", State.analyze_note, State.set_analyze_note, "150px"),
                rx.button("Use Current AI Logic", on_click=State.get_analyze_ai_advice, background=PRIMARY, color="white"),
                chip(State.analyze_guidance, bg="#eff6ff", color=PRIMARY),
                grid_column=rx.breakpoints(initial="1 / -1", xl="8 / 13"),
            ),
            columns=rx.breakpoints(initial="1", xl="12"),
            spacing="6",
            width="100%",
        ),
        rx.grid(
            card(
                section_title("Pareto Chart", "Root-cause ranking carried into Analyze for prioritization."),
                rx.cond(
                    State.data_columns != [],
                    rx.select(
                        State.data_columns,
                        value=State.selected_pareto_column,
                        on_change=State.set_selected_pareto_column,
                        width=rx.breakpoints(initial="100%", md="320px"),
                    ),
                    rx.text("Upload data to unlock live Pareto analysis.", color=TEXT_MUTED),
                ),
                rx.hstack(
                    rx.foreach(State.pareto_data, pareto_bar),
                    align="end",
                    justify="between",
                    height="280px",
                    width="100%",
                    spacing="5",
                ),
                chip("Prioritize the tallest bars first for hypothesis testing.", bg="#f8fafc", color=TEXT),
                grid_column=rx.breakpoints(initial="1 / -1", xl="1 / 9"),
            ),
            card(
                section_title("Analysis Focus", "Use Pareto plus control limits to narrow the investigation."),
                chip("Selected category: " + State.selected_pareto_column, bg="#eff6ff", color=PRIMARY),
                chip("Selected metric: " + State.selected_metric_column, bg="#ecfeff", color=ACCENT),
                rx.text(
                    "The Pareto chart highlights where defects concentrate, while the control chart shows whether the chosen metric is behaving predictably over time.",
                    color=TEXT_MUTED,
                    font_size="0.95rem",
                ),
                grid_column=rx.breakpoints(initial="1 / -1", xl="9 / 13"),
            ),
            columns=rx.breakpoints(initial="1", xl="12"),
            spacing="6",
            width="100%",
        ),
        spacing="6",
        width="100%",
    )


def improve_view():
    return rx.vstack(
        rx.grid(
            card(
                section_title("Fishbone Workspace", "Add causes by category using your existing root-cause structure."),
                rx.grid(
                    cause_block("People", State.new_people_cause, State.set_new_people_cause, lambda: State.add_cause("People")),
                    cause_block("Methods", State.new_methods_cause, State.set_new_methods_cause, lambda: State.add_cause("Methods")),
                    cause_block("Machines", State.new_machines_cause, State.set_new_machines_cause, lambda: State.add_cause("Machines")),
                    cause_block("Materials", State.new_materials_cause, State.set_new_materials_cause, lambda: State.add_cause("Materials")),
                    cause_block("Environment", State.new_environment_cause, State.set_new_environment_cause, lambda: State.add_cause("Environment")),
                    cause_block("Measurement", State.new_measurement_cause, State.set_new_measurement_cause, lambda: State.add_cause("Measurement")),
                    columns=rx.breakpoints(initial="1", md="2", xl="3"),
                    spacing="4",
                ),
                grid_column=rx.breakpoints(initial="1 / -1", xl="1 / 9"),
            ),
            card(
                section_title("Current Cause Map", "Visible root-cause inventory"),
                rx.foreach(
                    list(DEFAULT_FISHBONE.keys()),
                    lambda category: rx.box(
                        rx.vstack(
                            rx.text(category, font_weight="700", color=TEXT),
                            rx.cond(
                                State.fishbone[category] != [],
                                rx.foreach(State.fishbone[category], lambda item: chip(item, bg="#f8fafc", color=TEXT)),
                                rx.text("No causes added yet.", color=TEXT_SOFT),
                            ),
                            spacing="2",
                            align_items="start",
                        ),
                        background=PANEL_ALT,
                        border=f"1px solid {BORDER}",
                        border_radius="14px",
                        padding="0.9rem",
                    ),
                ),
                grid_column=rx.breakpoints(initial="1 / -1", xl="9 / 13"),
            ),
            columns=rx.breakpoints(initial="1", xl="12"),
            spacing="6",
            width="100%",
        ),
        card(
            section_title("Value Stream Map Inputs", "TIMWOOD process-step capture from your current workflow."),
            rx.grid(
                rx.input(value=State.step_name, on_change=State.set_step_name, placeholder="Step name", background=PANEL_ALT, color=TEXT, style={"color": TEXT, "::placeholder": {"color": TEXT_SOFT}}),
                rx.input(value=State.step_time, on_change=State.set_step_time, type="number", background=PANEL_ALT, color=TEXT, placeholder="Cycle time", style={"color": TEXT, "::placeholder": {"color": TEXT_SOFT}}),
                rx.input(value=State.step_wait, on_change=State.set_step_wait, type="number", background=PANEL_ALT, color=TEXT, placeholder="Wait time", style={"color": TEXT, "::placeholder": {"color": TEXT_SOFT}}),
                rx.select(WASTE_TYPES, value=State.step_waste, on_change=State.set_step_waste),
                columns=rx.breakpoints(initial="1", md="2", xl="4"),
                spacing="3",
            ),
            rx.button("Add Process Step", on_click=State.add_process_step, background=PRIMARY, color="white"),
            rx.foreach(State.process_steps, process_step_row),
            editor("Implementation Note", State.improve_note, State.set_improve_note, "130px"),
            rx.button("Use Current AI Logic", on_click=State.get_improve_ai_advice, background=PRIMARY, color="white"),
            chip(State.improve_guidance, bg="#eff6ff", color=PRIMARY),
        ),
        spacing="6",
        width="100%",
    )


def consult_view():
    return card(
        section_title("AI Consultant", "Business-facing Lean Six Sigma chat linked to the current codebase AI flow."),
        rx.box(
            rx.vstack(rx.foreach(State.chat_history, chat_message), spacing="3", width="100%"),
            background=PANEL_ALT,
            border=f"1px solid {BORDER}",
            border_radius="18px",
            padding="1rem",
            min_height="360px",
        ),
        rx.hstack(
            rx.input(
                value=State.chat_input_text,
                on_change=State.set_chat_input_text,
                placeholder="Ask about waste drivers, variation, ROI, or next actions...",
                background=PANEL_ALT,
                color=TEXT,
                style={"color": TEXT, "::placeholder": {"color": TEXT_SOFT}},
                width="100%",
            ),
            rx.button("Send", on_click=State.send_chat, background=PRIMARY, color="white"),
            spacing="3",
            width="100%",
        ),
    )


def summary_view():
    return rx.vstack(
        rx.grid(
            stat_tile("Project", State.project_title, State.facility_name),
            stat_tile("Status", State.project_status, State.department_name),
            stat_tile("Last Saved", State.last_saved_at, State.workspace_message),
            columns=rx.breakpoints(initial="1", md="3"),
            spacing="4",
            width="100%",
        ),
        card(
            section_title("Executive Summary", "Current state across DMAIC"),
            rx.vstack(
                chip("Define: " + State.define_guidance, bg="#eff6ff", color=PRIMARY),
                chip("Measure: " + State.measure_guidance, bg="#eff6ff", color=PRIMARY),
                chip("Analyze: " + State.analyze_guidance, bg="#eff6ff", color=PRIMARY),
                chip("Improve: " + State.improve_guidance, bg="#eff6ff", color=PRIMARY),
                spacing="3",
                align_items="start",
            ),
        ),
        spacing="6",
        width="100%",
    )


def phase_view():
    return rx.match(
        State.active_phase,
        ("Define", define_view()),
        ("Measure", measure_view()),
        ("Analyze", analyze_view()),
        ("Improve", improve_view()),
        ("Consult", consult_view()),
        ("Summary", summary_view()),
        define_view(),
    )


def sidebar():
    return rx.vstack(
        rx.vstack(
            rx.box(width="42px", height="42px", border_radius="12px", background=GRADIENT),
            rx.heading("Lean Six Sigma Consultant", size="6", color="white"),
            rx.text("Business Performance Workspace", color="#cbd5e1", font_size="0.8rem"),
            spacing="1",
            align_items="start",
        ),
        rx.vstack(*[side_button(phase) for phase in PHASES], width="100%", spacing="2"),
        card(
            rx.text("Workspace Controls", font_weight="700", color=TEXT),
            rx.select(["Green", "Amber", "Red"], value=State.project_status, on_change=State.set_project_status, width="100%"),
            rx.button("Save Snapshot", on_click=State.save_workspace, background=PRIMARY, color="white", width="100%"),
            rx.text(State.workspace_message, color=TEXT_MUTED, font_size="0.84rem"),
            rx.text("Last saved: " + State.last_saved_at, color=TEXT_SOFT, font_size="0.78rem"),
            padding="1rem",
        ),
        rx.spacer(),
        width="290px",
        min_height="100vh",
        background=SIDEBAR,
        padding="1.25rem",
        spacing="5",
        position="sticky",
        top="0",
    )


def topbar():
    return rx.hstack(
        rx.vstack(
            chip("DMAIC", bg="#e0f2fe", color=PRIMARY),
            rx.heading(State.project_title, size="7", color=TEXT),
            rx.text("Modern business-professional interface linked to the current Lean Six Sigma codebase.", color=TEXT_SOFT),
            spacing="1",
            align_items="start",
        ),
        rx.spacer(),
        rx.hstack(*[top_tab(phase) for phase in PHASES], spacing="4"),
        spacing="6",
        width="100%",
        padding="1rem 1.25rem",
        background="rgba(255,255,255,0.92)",
        backdrop_filter="blur(14px)",
        border=f"1px solid {BORDER}",
        border_radius="18px",
        position="sticky",
        top="1rem",
        z_index="10",
    )


def index():
    return rx.box(
        rx.hstack(
            sidebar(),
            rx.box(
                rx.vstack(topbar(), phase_view(), spacing="6", align_items="stretch", width="100%"),
                flex="1",
                padding="1rem 1.2rem 2rem 1.2rem",
            ),
            width="100%",
            align="start",
            spacing="0",
        ),
        background=SURFACE,
        min_height="100vh",
        width="100%",
    )


app = rx.App(
    style={
        "font_family": "'Segoe UI', 'Inter', sans-serif",
        "color": TEXT,
        "background": SURFACE,
    }
)
app.add_page(index)
