import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import math

# ===============================
# 1) CASE PARAMETERS & DATAFRAME
# ===============================
case_parameters = {
    "10 kV Overhead Line":  {"R": 0.30,  "X": 0.45, "C": 20e-9,   "length": 10,  "base_voltage": 10},
    "10 kV Cable":          {"R": 0.20,  "X": 0.20, "C": 300e-9,  "length": 5,   "base_voltage": 10},
    "110 kV Overhead Line": {"R": 0.07,  "X": 0.35, "C": 15e-9,   "length": 50,  "base_voltage": 110},
    "110 kV Cable":         {"R": 0.055, "X": 0.15, "C": 200e-9,  "length": 20,  "base_voltage": 110},
    "400 kV Overhead Line": {"R": 0.015, "X": 0.30, "C": 5e-9,    "length": 200, "base_voltage": 400},
    "400 kV Cable":         {"R": 0.015, "X": 0.10, "C": 150e-9,  "length": 50,  "base_voltage": 400},
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

# ==================================
# 2) POWER FLOW CALCULATION
# ==================================
@st.cache_data
def calculate_power_flow(Vs, Vr, R, X, C, delta_deg, base_voltage, length, freq=50):
    """
    Returns:
      S_s, S_i, S_o, S_loss, S_r  (complex MVA),
      Q_s, Q_r (floats, MVAr).
    """
    delta_rad = np.deg2rad(delta_deg)
    Z = (R + 1j*X)*length
    if abs(Z) < 1e-12:
        return (np.nan,)*5 + (np.nan, np.nan)

    V_s_actual = Vs*base_voltage
    V_r_actual = Vr*base_voltage

    # Receiving-end phasor
    V_r_phasor = V_r_actual * np.exp(-1j*delta_rad)

    # Current in kA
    I_line = (V_s_actual - V_r_phasor)/Z

    # S_i
    S_i = V_s_actual*np.conjugate(I_line)

    # Q_s
    Q_s = 2*math.pi*freq*C*length*(V_s_actual**2)
    S_s = S_i - 1j*Q_s

    # Losses
    S_loss = I_line*np.conjugate(I_line)*Z

    # S_o
    S_o = S_i - S_loss

    # Q_r
    Q_r = 2*math.pi*freq*C*length*(V_r_actual**2)
    S_r = S_o + 1j*Q_r

    return S_s, S_i, S_o, S_loss, S_r, Q_s, Q_r


# =================================================
# 3) PLOTLY VECTORS - Q_s from S_s, Q_r from S_r
# =================================================
def plot_power_vectors(S_s, S_i, S_o, S_loss, S_r, Q_s, Q_r):
    """
    Draw arrowed vectors:
      S_s, S_i, S_o, S_r from origin,
      S_loss from S_o->(S_o+S_loss),
      Q_s from S_s->(S_s + j Q_s),
      Q_r from S_r->(S_r + j Q_r).
    Labels in the middle, offset so text doesn't overlap the arrow line.
    """
    fig = go.Figure()

    # Check if any are invalid
    # We'll build a list of the endpoints:
    def pair(z): return [z.real, z.imag]

    S_s_end = pair(S_s)
    S_i_end = pair(S_i)
    S_o_end = pair(S_o)
    S_loss_end = [S_o_end[0] + S_loss.real, S_o_end[1] + S_loss.imag]
    S_r_end = pair(S_r)
    # Q_s from S_s => S_s + jQ_s
    Q_s_end = [S_s_end[0], S_s_end[1] + Q_s]
    # Q_r from S_r => S_r + jQ_r
    Q_r_end = [S_r_end[0], S_r_end[1] + Q_r]

    all_points = [
        [0,0], S_s_end, S_i_end, S_o_end, S_r_end, S_loss_end, Q_s_end, Q_r_end
    ]
    # Flatten
    coords = []
    for p in all_points:
        coords.extend(p)
    if any(np.isnan(c) or not np.isfinite(c) for c in coords):
        fig.update_layout(title="Invalid flows (NaN or Inf). Check parameters!")
        return fig

    # We'll define a helper to add a vector from (x0,y0) to (x1,y1)
    def add_vector(x0, y0, x1, y1, color, label):
        # 1) The arrow from tail->tip, no label
        fig.add_annotation(
            x=x1, y=y1, xref="x", yref="y",
            ax=x0, ay=y0, axref="x", ayref="y",
            arrowhead=3, arrowsize=1.5, arrowwidth=2, arrowcolor=color,
            text="", showarrow=True
        )
        # 2) A label in the midpoint, offset perpendicular
        mx = 0.5*(x0 + x1)
        my = 0.5*(y0 + y1)
        dx = x1 - x0
        dy = y1 - y0
        length = np.hypot(dx, dy)
        if length > 1e-12:
            # unit normal is (-dy, dx) or (dy, -dx). We'll pick one
            # for consistent offset to the "left" of vector
            nx = -dy/length
            ny = dx/length
            # e.g. offset by 5% of vector length
            offset = 0.05*length
            ox = mx + offset*nx
            oy = my + offset*ny
        else:
            # degenerate vector, no offset
            ox, oy = mx, my

        fig.add_annotation(
            x=ox, y=oy, xref="x", yref="y",
            text=label, showarrow=False,
            font=dict(color=color, size=14),
            xanchor="center", yanchor="middle"
        )

    # 1) S_s, S_i, S_o, S_r from origin
    add_vector(0, 0, S_s_end[0], S_s_end[1], "red", "S_s")
    add_vector(0, 0, S_i_end[0], S_i_end[1], "green", "S_i")
    add_vector(0, 0, S_o_end[0], S_o_end[1], "blue", "S_o")
    add_vector(0, 0, S_r_end[0], S_r_end[1], "purple", "S_r")

    # 2) S_loss from S_o => S_o + S_loss
    add_vector(S_o_end[0], S_o_end[1], S_loss_end[0], S_loss_end[1], "magenta", "S_loss")

    # 3) Q_s from S_s => S_s + jQ_s
    add_vector(S_s_end[0], S_s_end[1], Q_s_end[0], Q_s_end[1], "orange", "Q_s")

    # 4) Q_r from S_r => S_r + jQ_r
    add_vector(S_r_end[0], S_r_end[1], Q_r_end[0], Q_r_end[1], "cyan", "Q_r")

    # Auto-range
    all_x = [p[0] for p in all_points]
    all_y = [p[1] for p in all_points]
    min_x, max_x = min(all_x), max(all_x)
    min_y, max_y = min(all_y), max(all_y)

    margin_x = 0.1*(max_x - min_x if max_x!=min_x else 1)
    margin_y = 0.1*(max_y - min_y if max_y!=min_y else 1)
    x_range = [min_x - margin_x, max_x + margin_x]
    y_range = [min_y - margin_y, max_y + margin_y]

    fig.update_layout(
        title="Power Vectors with Q_s, Q_r from S_s, S_r + Perp-Offset Labels",
        xaxis=dict(range=x_range, zeroline=True),
        yaxis=dict(range=y_range, zeroline=True),
        width=800, height=500
    )
    return fig


def plot_bar_chart(S_s, S_i, S_o, S_loss, S_r):
    labels = ["S_s", "S_i", "S_o", "S_loss", "S_r"]
    real_vals = [S_s.real, S_i.real, S_o.real, S_loss.real, S_r.real]
    imag_vals = [S_s.imag, S_i.imag, S_o.imag, S_loss.imag, S_r.imag]

    fig = go.Figure(data=[
        go.Bar(name='Active (MW)', x=labels, y=real_vals),
        go.Bar(name='Reactive (MVAr)', x=labels, y=imag_vals)
    ])
    fig.update_layout(
        barmode='group',
        title="Active & Reactive Power",
        xaxis_title="Power Component",
        yaxis_title="Magnitude",
        width=800,
        height=400
    )
    return fig


# ===============================
# 4) STREAMLIT APP
# ===============================
def main():
    st.title("Q_s from S_s, Q_r from S_r, with Perp-Offset Labels")
    st.write("""
        In this version, Q_s is drawn from the tip of S_s, Q_r from the tip of S_r.
        We also offset the label perpendicular to each vector's midpoint so it doesn't cross the line.
    """)

    st.image(
        "https://i.postimg.cc/FKStDhY9/Frame-2-1.png",
        caption="Power Flow on a Transmission Line (Pi Model)",
        width=500
    )

    st.subheader("Line Cases")
    st.table(df_cases)

    # Sliders
    selected_case_name = st.selectbox("Select a line case", list(case_parameters.keys()))
    scase = case_parameters[selected_case_name]

    col1, col2, col3 = st.columns(3)
    with col1:
        Vs = st.slider("Vs (p.u.)", 0.8, 1.2, 1.0, 0.01)
        Vr = st.slider("Vr (p.u.)", 0.8, 1.2, 0.95, 0.01)
    with col2:
        delta_deg = st.slider("Delta (°)", -60, 60, 10, 1)
        length_km = st.slider("Line Length (km)", 1.0, 300.0, float(scase["length"]), 1.0)
    with col3:
        R_ohm_km = st.slider("R (Ω/km)", 0.0001, 0.5, float(scase["R"]), 0.01)
        X_ohm_km = st.slider("X (Ω/km)", 0.0001, 0.5, float(scase["X"]), 0.01)
        C_nF_km = st.slider("C (nF/km)", 0.1, 400.0, float(scase["C"]*1e9), 10.0)
    C_f = C_nF_km*1e-9

    # Calculate
    S_s, S_i, S_o, S_loss, S_r, Q_s, Q_r = calculate_power_flow(
        Vs, Vr, R_ohm_km, X_ohm_km, C_f, delta_deg, scase["base_voltage"], length_km
    )

    def fmt_complex(z):
        if np.isnan(z.real) or not np.isfinite(z.real):
            return "NaN"
        return f"{z.real:.2f} + j{z.imag:.2f}"

    def fmt_float(x):
        return f"{x:.2f}" if np.isfinite(x) else "NaN"

    # Build table: (Name, Value)
    results_data = {
        "Name": [
            "S_s", "S_i", "S_o", "S_loss", "S_r",
            "Q_s (MVAr)", "Q_r (MVAr)"
        ],
        "Value": [
            fmt_complex(S_s),
            fmt_complex(S_i),
            fmt_complex(S_o),
            fmt_complex(S_loss),
            fmt_complex(S_r),
            fmt_float(Q_s),
            fmt_float(Q_r),
        ]
    }
    df_results = pd.DataFrame(results_data)

    st.subheader("Power Flow Results (MVA = MW + jMVAr)")
    st.table(df_results)

    fig_vector = plot_power_vectors(S_s, S_i, S_o, S_loss, S_r, Q_s, Q_r)
    st.plotly_chart(fig_vector, use_container_width=True)

    fig_bar = plot_bar_chart(S_s, S_i, S_o, S_loss, S_r)
    st.plotly_chart(fig_bar, use_container_width=True)


if __name__ == "__main__":
    main()
