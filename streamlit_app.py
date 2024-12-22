import streamlit as st
import pandas as pd
import numpy as np
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
def calculate_power_flow(Vs, Vr, R, X, C, delta_deg, base_voltage, length, frequency=50):
    """
    Calculates power-flow for a single transmission line in MVA = MW + jMVAr.

    Vs, Vr: per-unit sending & receiving voltages
    R, X, C: per-km line parameters
    delta_deg: angle (deg) at receiving end w.r.t sending
    base_voltage: nominal line voltage in kV
    length: line length in km
    frequency: system frequency in Hz
    """
    delta_rad = np.deg2rad(delta_deg)

    # Build total line impedance
    Z = (R + 1j * X) * length

    # Handle near-zero impedance (avoid division by zero)
    if abs(Z) < 1e-12:
        # Return NaNs so we can detect invalid flows
        return (np.nan,)*5

    # Convert per-unit voltages to actual kV
    V_s_actual = Vs * base_voltage
    V_r_actual = Vr * base_voltage

    # Receiving-end phasor with angle shift
    V_r_phasor = V_r_actual * np.exp(-1j * delta_rad)

    # Current in kA
    I_line = (V_s_actual - V_r_phasor) / Z

    # Apparent power at sending end (S_i)
    S_i = V_s_actual * np.conjugate(I_line)  # MVA

    # Reactive power from line capacitance at sending end
    Q_s = 2 * np.pi * frequency * C * length * (V_s_actual**2)  # MVAr
    S_s = S_i - 1j * Q_s  # S_s = S_i - jQ_s

    # Line losses
    S_loss = I_line * np.conjugate(I_line) * Z  # MW + jMVAr

    # Power before shunt at receiving end
    S_o = S_i - S_loss

    # Reactive power from line capacitance at receiving end
    Q_r = 2 * np.pi * frequency * C * length * (V_r_actual**2)
    S_r = S_o + 1j * Q_r

    return S_s, S_i, S_o, S_loss, S_r

# ===============================
# 3) PLOTLY: VECTORS WITH ARROWS
# ===============================
def plot_power_vectors(S_s, S_i, S_o, S_loss, S_r):
    """
    Creates a Plotly figure that draws ARROWED lines for:
    - S_s, S_i, S_o, S_r from origin
    - S_loss from S_o to S_o + S_loss
    using Plotly annotations' arrowhead functionality.
    """

    fig = go.Figure()

    # Step A: Check for invalid flows
    coords = [
        S_s.real, S_s.imag,
        S_i.real, S_i.imag,
        S_o.real, S_o.imag,
        S_r.real, S_r.imag,
        (S_o.real + S_loss.real),
        (S_o.imag + S_loss.imag)
    ]
    # If any are NaN or Inf, skip
    if any(np.isnan(c) or not np.isfinite(c) for c in coords):
        fig.update_layout(title="Invalid (NaN or Inf) flow results")
        return fig

    # Step B: Build dynamic autoscale
    reals = [
        0, S_s.real, S_i.real, S_o.real, S_r.real,
        S_o.real + S_loss.real
    ]
    imags = [
        0, S_s.imag, S_i.imag, S_o.imag, S_r.imag,
        S_o.imag + S_loss.imag
    ]
    min_x, max_x = min(reals), max(reals)
    min_y, max_y = min(imags), max(imags)

    # 10% margin
    margin_x = 0.1 * (max_x - min_x if max_x != min_x else 1)
    margin_y = 0.1 * (max_y - min_y if max_y != min_y else 1)
    x_range = [min_x - margin_x, max_x + margin_x]
    y_range = [min_y - margin_y, max_y + margin_y]

    # Helper: draw an arrow with an annotation
    def add_arrow(x0, y0, x1, y1, color, label):
        """
        Creates an annotation that draws an arrow from (x0,y0) to (x1,y1).
        arrowhead=3 => triangle arrow
        """
        fig.add_annotation(
            x=x1, y=y1,             # arrow tip
            xref="x", yref="y",
            ax=x0, ay=y0,           # arrow tail
            axref="x", ayref="y",
            arrowhead=3,            # style of arrow
            arrowsize=1.5,
            arrowwidth=2,
            arrowcolor=color,
            text=label,
            showarrow=True,
            font=dict(color=color, size=14),
        )

    # Step C: Draw each arrow
    #  - from origin for S_s, S_i, S_o, S_r
    add_arrow(0, 0, S_s.real, S_s.imag, "red", "S_s")
    add_arrow(0, 0, S_i.real, S_i.imag, "green", "S_i")
    add_arrow(0, 0, S_o.real, S_o.imag, "blue", "S_o")
    add_arrow(0, 0, S_r.real, S_r.imag, "purple", "S_r")

    #  - from S_o to S_o + S_loss
    add_arrow(
        S_o.real, S_o.imag,
        S_o.real + S_loss.real,
        S_o.imag + S_loss.imag,
        "magenta",
        "S_loss"
    )

    # Step D: layout
    fig.update_layout(
        title="Power Flow Vectors (MVA) with Arrowheads",
        xaxis=dict(range=x_range, zeroline=True),
        yaxis=dict(range=y_range, zeroline=True),
        width=700,
        height=500
    )

    return fig


def plot_bar_chart(S_s, S_i, S_o, S_loss, S_r):
    labels = ["S_s", "S_i", "S_o", "S_loss", "S_r"]
    real_vals = [S_s.real, S_i.real, S_o.real, S_loss.real, S_r.real]
    imag_vals = [S_s.imag, S_i.imag, S_o.imag, S_loss.imag, S_r.imag]

    fig = go.Figure(data=[
        go.Bar(name='Active Power (MW)', x=labels, y=real_vals),
        go.Bar(name='Reactive Power (MVAr)', x=labels, y=imag_vals)
    ])
    fig.update_layout(
        barmode='group',
        title="Active & Reactive Power (MW / MVAr)",
        xaxis_title="Power Component",
        yaxis_title="Magnitude",
        width=700,
        height=400
    )
    return fig

# ===============================
# 4) STREAMLIT APP
# ===============================
def main():
    st.title("Power Flow Simulator for Transmission Lines")
    st.write(
        """
        This app calculates and visualizes power flows (in MVA) for a single transmission line 
        using arrows for each vector. Move the sliders and see the results update in real time!
        """
    )

    # Show your original line diagram
    st.image(
        "https://i.postimg.cc/FKStDhY9/Frame-2-1.png",
        caption="Power Flow on a Transmission Line (Pi Model)",
        width=500
    )

    # Table of line cases
    st.subheader("Line Cases")
    st.dataframe(df_cases)

    st.subheader("Parameters & Sliders")

    # 1) Choose the case
    selected_case_name = st.selectbox("Select a line case", list(case_parameters.keys()))
    selected_case = case_parameters[selected_case_name]

    # 2) Sliders
    col1, col2, col3 = st.columns(3)
    with col1:
        Vs = st.slider("Vs (p.u.)", 0.8, 1.2, 1.0, 0.01)
        Vr = st.slider("Vr (p.u.)", 0.8, 1.2, 0.95, 0.01)
    with col2:
        delta_deg = st.slider("Angle (°)", -60, 60, 10, 1)
        length_km = st.slider("Line Length (km)", 1.0, 300.0, float(selected_case["length"]), 1.0)
    with col3:
        R_ohm_km = st.slider("R (Ω/km)", 0.0001, 0.5, float(selected_case["R"]), 0.01)
        X_ohm_km = st.slider("X (Ω/km)", 0.0001, 0.5, float(selected_case["X"]), 0.01)
        C_nF_km = st.slider("C (nF/km)", 0.1, 400.0, selected_case["C"]*1e9, 10.0)

    # Convert from nF to F
    C_f = C_nF_km * 1e-9

    # 3) Calculate
    S_s, S_i, S_o, S_loss, S_r = calculate_power_flow(
        Vs, Vr,
        R_ohm_km, X_ohm_km, C_f,
        delta_deg,
        selected_case["base_voltage"],
        length_km
    )

    # 4) Display results
    st.subheader("Power Flow Results (MW + j MVAr)")
    colA, colB, colC, colD, colE = st.columns(5)

    def fmt(c):
        # If invalid
        if np.isnan(c.real) or not np.isfinite(c.real):
            return "NaN"
        return f"{c.real:.2f} + j{c.imag:.2f}"

    colA.metric("S_s", fmt(S_s))
    colB.metric("S_i", fmt(S_i))
    colC.metric("S_o", fmt(S_o))
    colD.metric("S_loss", fmt(S_loss))
    colE.metric("S_r", fmt(S_r))

    # 5) Vector plot with arrowheads
    fig_vector = plot_power_vectors(S_s, S_i, S_o, S_loss, S_r)
    st.plotly_chart(fig_vector, use_container_width=True)

    # 6) Bar chart
    fig_bar = plot_bar_chart(S_s, S_i, S_o, S_loss, S_r)
    st.plotly_chart(fig_bar, use_container_width=True)


if __name__ == "__main__":
    main()
