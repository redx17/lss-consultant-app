from __future__ import annotations

import io
import json
from typing import Any

import graphviz
import pandas as pd
from openai import OpenAI


def _clean_value(value: Any) -> Any:
    if pd.isna(value):
        return ""
    if hasattr(value, "isoformat"):
        try:
            return value.isoformat()
        except Exception:
            return str(value)
    if isinstance(value, (int, float, str, bool)):
        return value
    return str(value)


def dataframe_to_records(df: pd.DataFrame) -> list[dict[str, Any]]:
    clean_df = df.copy()
    for column in clean_df.columns:
        clean_df[column] = clean_df[column].map(_clean_value)
    return clean_df.to_dict(orient="records")


def load_tabular_data(raw_bytes: bytes, filename: str) -> pd.DataFrame:
    lower_name = filename.lower()
    buffer = io.BytesIO(raw_bytes)
    if lower_name.endswith(".csv"):
        return pd.read_csv(buffer)
    if lower_name.endswith(".xlsx") or lower_name.endswith(".xls"):
        return pd.read_excel(buffer)
    raise ValueError("Only CSV and Excel files are supported.")


def preview_payload(df: pd.DataFrame, row_limit: int = 5) -> dict[str, Any]:
    columns = [str(col) for col in df.columns.tolist()]
    records = dataframe_to_records(df.head(row_limit))
    numeric_columns = [str(col) for col in df.select_dtypes(include="number").columns.tolist()]
    categorical_columns = [str(col) for col in df.columns.tolist() if str(col) not in numeric_columns]
    return {
        "columns": columns,
        "records": records,
        "row_count": int(len(df.index)),
        "numeric_columns": numeric_columns,
        "categorical_columns": categorical_columns,
    }


def pareto_payload(df: pd.DataFrame, category_column: str) -> list[dict[str, Any]]:
    if category_column not in df.columns:
        return []
    raw_data = df[category_column].astype(str).str.strip()
    clean_data = raw_data[(raw_data != "") & (raw_data.str.lower() != "nan")]
    if clean_data.empty:
        return []
    counts = clean_data.value_counts().reset_index()
    counts.columns = ["label", "count"]
    counts = counts.sort_values(by="count", ascending=False).head(6).reset_index(drop=True)
    total = max(int(counts["count"].sum()), 1)
    max_count = max(int(counts["count"].max()), 1)
    payload: list[dict[str, Any]] = []
    cumulative = 0
    for _, row in counts.iterrows():
        count = int(row["count"])
        cumulative += count
        payload.append(
            {
                "label": str(row["label"]),
                "count": count,
                "percent": f"{round((count / total) * 100):.0f}%",
                "cum_percent": round((cumulative / total) * 100, 1),
                "height": f"{max(12, round((count / max_count) * 100))}%",
            }
        )
    return payload


def control_chart_payload(df: pd.DataFrame, metric_column: str) -> dict[str, Any]:
    if metric_column not in df.columns:
        return {"points": [], "mean": "", "ucl": "", "lcl": ""}

    series = pd.to_numeric(df[metric_column], errors="coerce").dropna()
    if series.empty:
        return {"points": [], "mean": "", "ucl": "", "lcl": ""}

    mean = float(series.mean())
    std = float(series.std()) if len(series) > 1 else 0.0
    ucl = mean + (3 * std)
    lcl = mean - (3 * std)
    sampled = series.head(12).tolist()
    minimum = min(sampled)
    maximum = max(sampled)
    span = max(maximum - minimum, 1e-9)
    points = []
    for value in sampled:
        normalized = (value - minimum) / span
        points.append(
            {
                "line_height": f"{max(40, round(40 + normalized * 90))}px",
                "high": value > mean,
            }
        )
    return {
        "points": points,
        "mean": f"{mean:.2f}",
        "ucl": f"{ucl:.2f}",
        "lcl": f"{lcl:.2f}",
    }


def descriptive_stats_payload(df: pd.DataFrame, metric_column: str) -> dict[str, str]:
    if metric_column not in df.columns:
        return {"min": "", "max": "", "mean": "", "std": ""}
    series = pd.to_numeric(df[metric_column], errors="coerce").dropna()
    if series.empty:
        return {"min": "", "max": "", "mean": "", "std": ""}
    std = float(series.std()) if len(series) > 1 else 0.0
    return {
        "min": f"{float(series.min()):.2f}",
        "max": f"{float(series.max()):.2f}",
        "mean": f"{float(series.mean()):.2f}",
        "std": f"{std:.2f}",
    }


def generate_flow_dot(steps: list[dict[str, Any]], title: str) -> str:
    dot = graphviz.Digraph(comment=title)
    dot.attr(rankdir="LR", size="12,5", bgcolor="transparent")
    waste_colors = {
        "Value Added (VA)": "#C8E6C9",
        "Transportation": "#FFCCBC",
        "Inventory": "#D1C4E9",
        "Motion": "#F0F4C3",
        "Waiting": "#FFECB3",
        "Overproduction": "#B3E5FC",
        "Overprocessing": "#CFD8DC",
        "Defects": "#FFCDD2",
    }
    for idx, step in enumerate(steps):
        waste_type = step.get("waste_type", "Value Added (VA)")
        color = waste_colors.get(waste_type, "#FFFFFF")
        shape = "rectangle" if waste_type == "Value Added (VA)" else "octagon"
        label = f"{step.get('name', 'Step')}\n({step.get('time', 0)}m)\n[{waste_type}]"
        dot.node(str(idx), label, style="filled", fillcolor=color, shape=shape)
        if idx > 0:
            wait_time = step.get("wait", 0)
            dot.edge(str(idx - 1), str(idx), label=f"Wait: {wait_time}m" if wait_time else "")
    return dot.source


def render_fishbone_dot(fishbone_data: dict[str, list[str]], problem: str) -> str:
    dot = graphviz.Digraph(comment="Fishbone")
    dot.attr(rankdir="RL", size="10,6", bgcolor="transparent")
    effect = problem[:60] + ("..." if len(problem) > 60 else "")
    dot.node("Effect", f"PROBLEM:\n{effect or 'No problem statement'}", shape="rectangle", style="filled", fillcolor="#E74C3C", fontcolor="white")
    for idx, (category, causes) in enumerate(fishbone_data.items()):
        category_id = f"cat_{idx}"
        dot.node(category_id, category, shape="ellipse", style="filled", fillcolor="#AED6F1")
        dot.edge(category_id, "Effect", penwidth="3")
        for cause_idx, cause in enumerate(causes):
            cause_id = f"c_{idx}_{cause_idx}"
            dot.node(cause_id, cause, shape="none", fontsize="10")
            dot.edge(cause_id, category_id)
    return dot.source


def get_ai_consultant_advice(section_name: str, context_data: Any, model_id: str = "qwen2.5-coder-7b-instruct") -> str:
    try:
        try:
            from streamlit import secrets  # type: ignore

            client = OpenAI(api_key=secrets["OPENAI_API_KEY"])
            model = "gpt-3.5-turbo"
        except Exception:
            client = OpenAI(base_url="http://127.0.0.1:1234/v1", api_key="lm-studio")
            model = model_id

        system_prompt = (
            "You are a Senior Lean Six Sigma Master Black Belt. "
            "Never provide code. Give executive-level guidance on waste, variation, and ROI."
        )
        user_message = (
            f"Review this {section_name} data and provide 3-4 strategic recommendations:\n"
            f"{json.dumps(context_data, default=str)[:12000]}"
        )
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
        )
        return response.choices[0].message.content.strip()
    except Exception as exc:
        return f"AI connection error: {exc}"


def get_ai_chat_response(
    prompt: str,
    problem_statement: str,
    uploaded_columns: list[str] | None = None,
    model_id: str = "qwen2.5-coder-7b-instruct",
) -> str:
    try:
        try:
            from streamlit import secrets  # type: ignore

            client = OpenAI(api_key=secrets["OPENAI_API_KEY"])
            model = "gpt-3.5-turbo"
        except Exception:
            client = OpenAI(base_url="http://127.0.0.1:1234/v1", api_key="lm-studio")
            model = model_id

        system_prompt = (
            "You are an elite Lean Six Sigma Master Black Belt. "
            "Never provide code. Respond in a concise, business-professional style "
            "focused on waste, variation, root cause, and ROI."
        )
        context = {
            "problem_statement": problem_statement,
            "uploaded_columns": uploaded_columns or [],
            "question": prompt,
        }
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": json.dumps(context)},
            ],
        )
        return response.choices[0].message.content.strip()
    except Exception as exc:
        return f"Chat error: {exc}"
