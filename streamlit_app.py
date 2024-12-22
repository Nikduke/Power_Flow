import streamlit as st
import pandas as pd
import numpy as np
import math
import plotly.graph_objects as go

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

# ===============================
# 2) POWER FLOW CALCULATION
# ===============================
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

    # Actual voltages
    V_s_actual = Vs*base_voltage
    V_r_actual = Vr*base_voltage

    # Receiving-end phasor
    V_r_phasor = V_r_actual * np.exp(-1j*delta_rad)

    # Current
    I_line = (V_s_actual - V_r_phasor)/Z

    S_i = V_s_actual*np.conjugate(I_line)

    # Q_s from line's C, at sending end
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


# ===============================
# 3) PLOTLY VECTORS
# ===============================
def plot_power_vectors(S_s, S_i, S_o, S_loss, S_r, Q_s, Q_r):
    """
    Draw arrowed vectors from:
      - Origin to S_s, S_i, S_o, S_r,
      - S_o to (S_o + S_loss),
      - S_s to (S_s + j Q_s),
      - S_r to (S_r + j Q_r).

    Each vector has 2 annotations:
      (1) arrow (no text),
      (2) label at midpoint (offset).
    """
    fig = go.Figure()

    def to_xy(z):
        return (z.real, z.imag)

    # Endpoints
    S_s_xy = to_xy(S_s)
    S_i_xy = to_xy(S_i)
    S_o_xy = to_xy(S_o)
    S_r_xy = to_xy(S_r)

    S_loss_xy = (S_o_xy[0] + S_loss.real, S_o_xy[1] + S_loss.imag)
    Q_s_xy = (S_s_xy[0], S_s_xy[1] + Q_s)   # from tip of S_s, up/down by Q_s
    Q_r_xy = (S_r_xy[0], S_r_xy[1] + Q_r)   # from tip of S_r, up/down by Q_r

    # Collect for autoscale
    all_points = [
        (0, 0), S_s_xy, S_i_xy, S_o_xy, S_r_xy,
        S_loss_xy, Q_s_xy, Q_r_xy
    ]
    coords = []
    for (x, y) in all_points:
        coords.append(x)
        coords.append(y)
    if any(np.isnan(c) or not np.isfinite(c) for c in coords):
        fig.update_layout(title="NaN or Inf in results - check sliders!")
        return fig

    # Helper
    def add_vector(tail, tip, color, label):
        (x0, y0) = tail
        (x1, y1) = tip

        # 1) arrow
        fig.add_annotation(
            x=x1, y=y1, xref="x", yref="y",
            ax=x0, ay=y0, axref="x", ayref="y",
            arrowhead=3, arrowsize=1.5, arrowwidth=2, arrowcolor=color,
            showarrow=True, text=""
        )

        # 2) label at midpoint, offset perpendicular
        mx = 0.5*(x0+x1)
        my = 0.5*(y0+y1)
        dx = x1 - x0
        dy = y1 - y0
        length = np.hypot(dx, dy)
        if length > 1e-12:
            # Perp direction
            nx = -dy/length
            ny = dx/length
            offset = 0.05*length
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

    # Now draw
    add_vector((0,0), S_s_xy, "red", "S_s")
    add_vector((0,0), S_i_xy, "green", "S_i")
    add_vector((0,0), S_o_xy, "blue", "S_o")
    add_vector((0,0), S_r_xy, "purple", "S_r")

    add_vector(S_o_xy, S_loss_xy, "magenta", "S_loss")
    add_vector(S_s_xy, Q_s_xy, "orange", "Q_s")
    add_vector(S_r_xy, Q_r_xy, "cyan", "Q_r")

    # Autoscale
    xvals = [p[0] for p in all_points]
    yvals = [p[1] for p in all_points]
    min_x, max_x = min(xvals), max(xvals)
    min_y, max_y = min(yvals), max(yvals)
    margin_x = 0.1*(max_x - min_x if max_x!=min_x else 1)
    margin_y = 0.1*(max_y - min_y if max_y!=min_y else 1)

    fig.update_layout(
        title="Flow vectors with Q_s, Q_r from S_s, S_r (Label offset)",
        xaxis=dict(range=[min_x - margin_x, max_x + margin_x], zeroline=True),
        yaxis=dict(range=[min_y - margin_y, max_y + margin_y], zeroline=True),
        width=800, height=500
    )
    return fig


def plot_bar_chart(S_s, S_i, S_o, S_loss, S_r):
    labels = ["S_s", "S_i", "S_o", "S_loss", "S_r"]
    reals = [S_s.real, S_i.real, S_o.real, S_loss.real, S_r.real]
    imags = [S_s.imag, S_i.imag, S_o.imag, S_loss.imag, S_r.imag]

    fig = go.Figure(data=[
        go.Bar(name='Active (MW)', x=labels, y=reals),
        go.Bar(name='Reactive (MVAr)', x=labels, y=imags)
    ])
    fig.update_layout(
        barmode='group',
        title="Active & Reactive Power",
        xaxis_title="Power Component",
        yaxis_title="Magnitude",
        width=800, height=400
    )
    return fig


# ===============================
# 4) STREAMLIT APP
# ===============================
def main():
    st.title("Correct Q_s, Q_r from S_s, S_r + Perp Offset Labels")
    st.write(
        """
        - Q_s arrow starts at the tip of S_s.
        - Q_r arrow starts at the tip of S_r.
        - All labels are offset so they don't cross the vector lines.
        """
    )

    st.image(
        "https://i.postimg.cc/FKStDhY9/Frame-2-1.png",
        caption="Power Flow on a Transmission Line (Pi Model)",
        width=500
    )

    st.subheader("Line Cases")
    st.table(df_cases)

    # Sliders
    case_name = st.selectbox("Select a line case", list(case_parameters.keys()))
    cparams = case_parameters[case_name]

    col1, col2, col3 = st.columns(3)
    with col1:
        Vs = st.slider("Vs (p.u.)", 0.8, 1.2, 1.0, 0.01)
        Vr = st.slider("Vr (p.u.)", 0.8, 1.2, 0.95, 0.01)
    with col2:
        delta_deg = st.slider("Delta (°)", -60, 60, 10, 1)
        length_km = st.slider("Line Length (km)", 1.0, 300.0, float(cparams["length"]), 1.0)
    with col3:
        R_ohm_km = st.slider("R (Ω/km)", 0.0001, 0.5, float(cparams["R"]), 0.01)
        X_ohm_km = st.slider("X (Ω/km)", 0.0001, 0.5, float(cparams["X"]), 0.01)
        C_nF_km = st.slider("C (nF/km)", 0.1, 400.0, float(cparams["C"]*1e9), 10.0)
    C_f = C_nF_km*1e-9

    S_s, S_i, S_o, S_loss, S_r, Q_s, Q_r = calculate_power_flow(
        Vs, Vr, R_ohm_km, X_ohm_km, C_f, delta_deg,
        cparams["base_voltage"], length_km
    )

    def fmt_complex(z):
        if np.isnan(z.real) or not np.isfinite(z.real):
            return "NaN"
        return f"{z.real:.2f} + j{z.imag:.2f}"

    def fmt_float(x):
        return f"{x:.2f}" if np.isfinite(x) else "NaN"

    data = {
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
            fmt_float(Q_r)
        ]
    }
    df_results = pd.DataFrame(data)

    st.subheader("Results (MVA = MW + jMVAr), plus Q_s and Q_r in MVAr")
    st.table(df_results)

    # Plot
    fig_vectors = plot_power_vectors(S_s, S_i, S_o, S_loss, S_r, Q_s, Q_r)
    st.plotly_chart(fig_vectors, use_container_width=True)

    fig_bars = plot_bar_chart(S_s, S_i, S_o, S_loss, S_r)
    st.plotly_chart(fig_bars, use_container_width=True)


if __name__ == "__main__":
    main()
