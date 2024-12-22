import streamlit as st
import pandas as pd
import numpy as np
import math
import plotly.graph_objects as go

# Configure page for wide layout
st.set_page_config(page_title="Transmission Line Power Flow Simulator", layout="wide")

# ===============================
# 1) CASE PARAMETERS & DATAFRAME
# ===============================
case_parameters = {
    "10 kV Overhead Line":  {"R": 0.30, "X": 0.45, "C": 20e-9,   "length": 10,  "base_voltage": 10},
    "10 kV Cable":          {"R": 0.20, "X": 0.20, "C": 300e-9,  "length": 5,   "base_voltage": 10},
    "110 kV Overhead Line": {"R": 0.07, "X": 0.35, "C": 15e-9,   "length": 50,  "base_voltage": 110},
    "110 kV Cable":         {"R": 0.055,"X": 0.15, "C": 200e-9,  "length": 20,  "base_voltage": 110},
    "400 kV Overhead Line": {"R": 0.015,"X": 0.30, "C": 5e-9,    "length": 200, "base_voltage": 400},
    "400 kV Cable":         {"R": 0.015,"X": 0.10, "C": 150e-9,  "length": 50,  "base_voltage": 400},
}

df_cases = pd.DataFrame([
    {
        "Line Type": k,
        "R (Ω/km)": v["R"],
        "X (Ω/km)": v["X"],
        "C (F/km)": v["C"],
        "Length (km)": v["length"],
        "Base Voltage (kV)": v["base_voltage"]
    }
    for k, v in case_parameters.items()
])
# Remove the default numeric index entirely
df_cases.reset_index(drop=True, inplace=True)


# ===============================
# 2) POWER FLOW CALCULATION
# ===============================
@st.cache_data
def calculate_power_flow(Vs, Vr, R, X, C, delta_deg, base_voltage, length, freq=50):
    """
    Returns (S_s, S_i, S_o, S_r, S_loss) as complex MVA,
    plus Q_s, Q_r in MVAr.
    Q_s from S_s -> S_s + jQ_s,
    Q_r from S_o -> S_o + jQ_r.
    """
    delta_rad = np.deg2rad(delta_deg)
    Z = (R + 1j*X)*length
    if abs(Z) < 1e-12:
        return (np.nan,)*5 + (np.nan, np.nan)

    V_s_actual = Vs*base_voltage
    V_r_actual = Vr*base_voltage

    # Receiving-end phasor
    V_r_phasor = V_r_actual * np.exp(-1j * delta_rad)
    I_line = (V_s_actual - V_r_phasor) / Z

    # Apparent power at sending end
    S_i = V_s_actual * np.conjugate(I_line)

    # Q_s
    Q_s = 2.0 * math.pi * freq * C * length * (V_s_actual**2)
    S_s = S_i - 1j*Q_s

    # S_loss
    S_loss = I_line * np.conjugate(I_line)*Z

    # S_o
    S_o = S_i - S_loss

    # Q_r
    Q_r = 2.0 * math.pi * freq * C * length * (V_r_actual**2)
    S_r = S_o + 1j*Q_r

    return S_s, S_i, S_o, S_r, S_loss, Q_s, Q_r


# ===============================
# 3) PLOTTING
# ===============================
def plot_power_vectors(S_s, S_i, S_o, S_r, S_loss, Q_s, Q_r):
    """Vector diagram with S_s, S_i, S_o, S_r from origin, plus Q_s, Q_r, S_loss."""
    fig = go.Figure()

    def to_xy(z):
        return (z.real, z.imag)

    S_s_xy = to_xy(S_s)
    S_i_xy = to_xy(S_i)
    S_o_xy = to_xy(S_o)
    S_r_xy = to_xy(S_r)

    # S_loss from S_o
    S_loss_xy = (S_o_xy[0] + S_loss.real, S_o_xy[1] + S_loss.imag)
    # Q_s from S_s
    Q_s_xy = (S_s_xy[0], S_s_xy[1] + Q_s)
    # Q_r from S_o
    Q_r_xy = (S_o_xy[0], S_o_xy[1] + Q_r)

    all_pts = [
        (0, 0), S_s_xy, S_i_xy, S_o_xy, S_r_xy, S_loss_xy, Q_s_xy, Q_r_xy
    ]
    coords = [c for p in all_pts for c in p]
    if any(np.isnan(c) or not np.isfinite(c) for c in coords):
        fig.update_layout(title="NaN or Inf. Check parameters!")
        return fig

    def add_arrow(tail, tip, color, label):
        x0, y0 = tail
        x1, y1 = tip
        fig.add_annotation(
            x=x1, y=y1, xref="x", yref="y",
            ax=x0, ay=y0, axref="x", ayref="y",
            arrowhead=3, arrowsize=1.5, arrowwidth=2, arrowcolor=color,
            text="", showarrow=True
        )
        mx, my = 0.5*(x0+x1), 0.5*(y0+y1)
        dx, dy = (x1 - x0), (y1 - y0)
        length = np.hypot(dx, dy)
        if length > 1e-12:
            nx = -dy/length
            ny = dx/length
            offset = 0.025 * length
            ox = mx + offset*nx
            oy = my + offset*ny
        else:
            ox, oy = mx, my

        fig.add_annotation(
            x=ox, y=oy, xref="x", yref="y",
            text=label, showarrow=False,
            font=dict(color=color, size=14),
            xanchor="center", yanchor="middle"
        )

    # Vectors from origin
    add_arrow((0,0), S_s_xy, "red",    "S_s")
    add_arrow((0,0), S_i_xy, "green",  "S_i")
    add_arrow((0,0), S_o_xy, "blue",   "S_o")
    add_arrow((0,0), S_r_xy, "purple", "S_r")

    # S_loss from S_o
    add_arrow(S_o_xy, S_loss_xy, "magenta", "S_loss")
    # Q_s
    add_arrow(S_s_xy, Q_s_xy, "orange", "Q_s")
    # Q_r
    add_arrow(S_o_xy, Q_r_xy, "cyan",   "Q_r")

    xs = [pt[0] for pt in all_pts]
    ys = [pt[1] for pt in all_pts]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    span_x = (max_x - min_x) if max_x!=min_x else 1
    span_y = (max_y - min_y) if max_y!=min_y else 1
    margin_x = 0.1*span_x
    margin_y = 0.1*span_y

    fig.update_layout(
        title="Vector Diagram",
        xaxis=dict(range=[min_x-margin_x, max_x+margin_x], zeroline=True),
        yaxis=dict(range=[min_y-margin_y, max_y+margin_y], zeroline=True),
        width=700, height=500
    )
    return fig


def plot_bar_chart(S_s, S_i, S_o, S_r, S_loss):
    labels = ["S_s", "S_i", "S_o", "S_r", "S_loss"]
    reals = [S_s.real, S_i.real, S_o.real, S_r.real, S_loss.real]
    imags = [S_s.imag, S_i.imag, S_o.imag, S_r.imag, S_loss.imag]

    fig = go.Figure(data=[
        go.Bar(name="Active (MW)", x=labels, y=reals),
        go.Bar(name="Reactive (MVAr)", x=labels, y=imags)
    ])
    fig.update_layout(
        barmode='group',
        title="Active & Reactive Power",
        xaxis_title="Power Component",
        yaxis_title="Magnitude",
        width=700, height=400
    )
    return fig


# ===============================
# 4) STREAMLIT APP
# ===============================
def main():
    st.title("Transmission Line Power Flow Simulator")

    # Row 1: diagram (left), case table (right)
    row1_col1, row1_col2 = st.columns([1,1.3])
    with row1_col1:
        st.image(
            "https://i.postimg.cc/FKStDhY9/Frame-2-1.png",
            caption="Transmission Line Pi Section",
            width=300
        )
    with row1_col2:
        st.subheader("Line Cases")
        st.dataframe(df_cases.style.format(precision=3).hide_index(), height=220)

    # Row 2: narrower col with case selector + sliders, bigger col with vector diagram
    row2_col1, row2_col2 = st.columns([0.8,2])
    with row2_col1:
        st.markdown("### Select Case & Sliders")
        case_name = st.selectbox("Select line case", list(case_parameters.keys()))
        cinfo = case_parameters[case_name]

        Vs = st.slider("Vs (p.u.)", 0.8, 1.2, 1.0, 0.01)
        Vr = st.slider("Vr (p.u.)", 0.8, 1.2, 0.95, 0.01)
        delta_deg = st.slider("Delta (°)", -60, 60, 10, 1)
        length_km = st.slider("Line Length (km)", 1.0, 300.0, float(cinfo["length"]), 1.0)
        R_ohm_km = st.slider("R (Ω/km)", 0.0001, 0.5, float(cinfo["R"]), 0.01)
        X_ohm_km = st.slider("X (Ω/km)", 0.0001, 0.5, float(cinfo["X"]), 0.01)
        C_nF_km = st.slider("C (nF/km)", 0.1, 400.0, float(cinfo["C"]*1e9), 10.0)

    with row2_col2:
        C_f = C_nF_km * 1e-9
        S_s, S_i, S_o, S_r, S_loss, Q_s, Q_r = calculate_power_flow(
            Vs, Vr, R_ohm_km, X_ohm_km, C_f, delta_deg, cinfo["base_voltage"], length_km
        )

        fig_vector = plot_power_vectors(S_s, S_i, S_o, S_r, S_loss, Q_s, Q_r)
        st.plotly_chart(fig_vector, use_container_width=True)

    # Row 3: results table (left), bar chart (right)
    row3_col1, row3_col2 = st.columns([1,1])
    with row3_col1:
        st.subheader("Power Flow Results (MVA = MW + jMVAr)")
        def fmt_cplx(z):
            if np.isnan(z.real) or not np.isfinite(z.real):
                return "NaN"
            return f"{z.real:.2f} + j{z.imag:.2f}"

        def fmt_float(f):
            return f"{f:.2f}" if np.isfinite(f) else "NaN"

        data = {
            "Name": ["S_s","S_i","S_o","S_r","S_loss","Q_s (MVAr)","Q_r (MVAr)"],
            "Value":[
                fmt_cplx(S_s), fmt_cplx(S_i), fmt_cplx(S_o),
                fmt_cplx(S_r), fmt_cplx(S_loss),
                fmt_float(Q_s), fmt_float(Q_r)
            ]
        }
        df_res = pd.DataFrame(data)
        st.table(df_res)

    with row3_col2:
        fig_bar = plot_bar_chart(S_s, S_i, S_o, S_r, S_loss)
        st.plotly_chart(fig_bar, use_container_width=True)


if __name__ == "__main__":
    main()
