"""Reusable Plotly chart functions for the EV Load Forecasting Streamlit app."""
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

TRAIN_COLOR = "#4C8BF5"
FORECAST_COLOR = "#F4511E"
ACTUAL_COLOR = "#34A853"
CI_80_COLOR = "rgba(244,81,30,0.15)"
CI_95_COLOR = "rgba(244,81,30,0.08)"
SPLIT_COLOR = "rgba(255,200,0,0.3)"


def energy_timeseries(
    df: pd.DataFrame,
    train_cutoff: pd.Timestamp,
    show_outliers: bool = False,
) -> go.Figure:
    """Interactive energy time series with train/test shading and optional outlier overlay.

    Args:
        df: DataFrame with 'timestamp' and 'energy' columns; optionally 'energy_outlier'.
        train_cutoff: Timestamp dividing train and test sets.
        show_outliers: If True, overlay outlier points in red.

    Returns:
        Plotly Figure.
    """
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df["timestamp"],
        y=df["energy"],
        mode="lines",
        name="Energy (kWh)",
        line=dict(color=TRAIN_COLOR, width=1),
    ))

    if show_outliers and "energy_outlier" in df.columns:
        outliers = df[df["energy_outlier"] == 1]
        fig.add_trace(go.Scatter(
            x=outliers["timestamp"],
            y=outliers["energy"],
            mode="markers",
            name="Outlier",
            marker=dict(color="red", size=4, symbol="x"),
        ))

    # Shade test region
    x_max = df["timestamp"].max()
    fig.add_vrect(
        x0=train_cutoff, x1=x_max,
        fillcolor=SPLIT_COLOR, layer="below", line_width=0,
        annotation_text="Test period", annotation_position="top left",
        annotation_font_size=11,
    )
    fig.add_vline(x=train_cutoff, line_dash="dash", line_color="orange", line_width=1.5)

    fig.update_layout(
        title="EV Charging Load — Hourly Energy (kWh)",
        xaxis_title="Date",
        yaxis_title="Energy (kWh)",
        hovermode="x unified",
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=420,
    )
    return fig


def hourly_seasonality(df: pd.DataFrame) -> go.Figure:
    """Average energy by hour of day.

    Args:
        df: DataFrame with 'hour' and 'energy' columns.
    """
    avg = df.groupby("hour")["energy"].mean().reset_index()
    fig = go.Figure(go.Bar(
        x=avg["hour"],
        y=avg["energy"],
        marker_color=TRAIN_COLOR,
        name="Avg energy",
    ))
    fig.update_layout(
        title="Average Energy by Hour of Day",
        xaxis_title="Hour",
        yaxis_title="Avg Energy (kWh)",
        template="plotly_white",
        height=340,
        xaxis=dict(tickmode="linear", dtick=2),
    )
    return fig


def weekly_seasonality(df: pd.DataFrame) -> go.Figure:
    """Average energy by day of week.

    Args:
        df: DataFrame with 'weekday' (0=Mon) and 'energy' columns.
    """
    day_labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    avg = df.groupby("weekday")["energy"].mean().reset_index()
    avg["day_label"] = avg["weekday"].map(lambda d: day_labels[d] if d < 7 else str(d))

    fig = go.Figure(go.Bar(
        x=avg["day_label"],
        y=avg["energy"],
        marker_color=TRAIN_COLOR,
        name="Avg energy",
    ))
    fig.update_layout(
        title="Average Energy by Day of Week",
        xaxis_title="Day",
        yaxis_title="Avg Energy (kWh)",
        template="plotly_white",
        height=340,
    )
    return fig


def correlation_bar(df: pd.DataFrame, target_col: str = "energy") -> go.Figure:
    """Pearson correlation of all numeric features with the target column.

    Args:
        df: Feature DataFrame including target column.
        target_col: Name of the target column.
    """
    numeric = df.select_dtypes(include="number")
    corr = numeric.corr()[target_col].drop(target_col).sort_values()

    colors = [FORECAST_COLOR if v < 0 else ACTUAL_COLOR for v in corr.values]

    fig = go.Figure(go.Bar(
        x=corr.values,
        y=corr.index,
        orientation="h",
        marker_color=colors,
    ))
    fig.update_layout(
        title=f"Feature Correlation with {target_col}",
        xaxis_title="Pearson r",
        template="plotly_white",
        height=max(400, len(corr) * 20),
        margin=dict(l=160),
    )
    return fig


def model_comparison_bar(summary_df: pd.DataFrame) -> go.Figure:
    """Grouped bar chart comparing RMSE and MAE across model families.

    Args:
        summary_df: DataFrame with columns model_family, rmse, mae.
    """
    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="RMSE",
        x=summary_df["model_family"],
        y=summary_df["rmse"],
        marker_color=TRAIN_COLOR,
    ))
    fig.add_trace(go.Bar(
        name="MAE",
        x=summary_df["model_family"],
        y=summary_df["mae"],
        marker_color=ACTUAL_COLOR,
    ))
    fig.update_layout(
        barmode="group",
        title="Model Comparison — Best Run per Family",
        xaxis_title="Model",
        yaxis_title="Error (kWh)",
        template="plotly_white",
        height=400,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def forecast_chart(fc_df: pd.DataFrame, ci_level: int = 80) -> go.Figure:
    """Actual vs forecast line chart with confidence interval ribbon.

    Args:
        fc_df: DataFrame with columns: timestamp, yhat, y (optional),
               yhat_lower_80, yhat_upper_80, yhat_lower_95, yhat_upper_95 (optional).
        ci_level: Which CI level to display: 80 or 95 (or 0 for none).

    Returns:
        Plotly Figure.
    """
    fig = go.Figure()

    lower_col = f"yhat_lower_{ci_level}"
    upper_col = f"yhat_upper_{ci_level}"

    # CI ribbon
    if ci_level and lower_col in fc_df.columns and upper_col in fc_df.columns:
        fill_color = CI_80_COLOR if ci_level == 80 else CI_95_COLOR
        fig.add_trace(go.Scatter(
            x=pd.concat([fc_df["timestamp"], fc_df["timestamp"].iloc[::-1]]),
            y=pd.concat([fc_df[upper_col], fc_df[lower_col].iloc[::-1]]),
            fill="toself",
            fillcolor=fill_color,
            line=dict(color="rgba(255,255,255,0)"),
            name=f"{ci_level}% CI",
            showlegend=True,
        ))

    # Actuals
    if "y" in fc_df.columns and fc_df["y"].notna().any():
        fig.add_trace(go.Scatter(
            x=fc_df["timestamp"],
            y=fc_df["y"],
            mode="lines",
            name="Actual",
            line=dict(color=ACTUAL_COLOR, width=2),
        ))

    # Forecast
    fig.add_trace(go.Scatter(
        x=fc_df["timestamp"],
        y=fc_df["yhat"],
        mode="lines",
        name="Forecast",
        line=dict(color=FORECAST_COLOR, width=2, dash="dash"),
    ))

    fig.update_layout(
        title="Recursive Forecast vs Actual",
        xaxis_title="Timestamp",
        yaxis_title="Energy (kWh)",
        hovermode="x unified",
        template="plotly_white",
        height=450,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig
