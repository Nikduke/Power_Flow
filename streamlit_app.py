import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# ===============================
# 1) LINE CASE PARAMETERS & TABLE
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

# ===================================
# 2) POWER FLOW CALCULATION (CACHED)
# ===================================
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
    
    # Safeguard: if R & X & length are effectively zero, skip
    Z = (R + 1j*X) * length
    if abs(Z) < 1e-9:  # near-zero impedance => invalid
        return (np.nan,)*5  # tuple of five NaNs

    # Convert from p.u. to actual kV
    V_s_actual = Vs * base_voltage
    V_r_actual = Vr * base_voltage

    # Represent Vr with phase shift
    V_r_phasor = V_r_actual * np.exp(-1j * delta_rad)

    # Line current in kA (kV / ohms = kA)
    I_line = (V_s_actual - V_r_phasor) / Z

    # Apparent power at sending end, S_i (MVA)
    S_i = V_s_actual * np.conjugate(I_line)

    # Reactive power from line capacitance at sending end
    Q_s = 2 * np.pi * frequency * C * length * (V_s_actual ** 2)  # MVAr
    S_s = S_i - 1j * Q_s

    # Losses (I^2 * Z)
    S_loss = I_line * np.conjugate(I_line) * Z  # MW + jMVAr

    # Power before shunt at receiving end
    S_o = S_i - S_loss

    # Reactive power from line capacitance at receiving end
    Q_r = 2 * np.pi * frequency * C * length * (V_r_actual ** 2)  # MVAr
    S_r = S_o + 1j * Q_r

    return S_s, S_i, S_o, S_loss, S_r

# ==================================
# 3) ADVANCED PLOTLY VECTORS
# ==================================
def plot_power_vectors(S_s, S_i, S_o, S_loss, S_r):
    """
    Creates a Plotly figure that draws line shapes with arrowheads
    for each vector: S_s, S_i, S_o, S_r, plus S_loss from S_o to S_o+S_loss.

    Uses dynamic autoscaling to avoid overlap.
    """
    fig = go.Figure()

    # Helper to check for NaN or Inf
    def is_invalid(c):
        return (np.isnan(c) or not np.isfinite(c))

    # If ANY component is invalid, return an empty figure with an error message
    coords = [
        S_s.real, S_s.imag, S_i.real, S_i.imag,
        S_o.real, S_o.imag, S_r.real, S_r.imag,
        (S_o.real + S_loss.real), (S_o.imag + S_loss.imag)
    ]
    if any(is_invalid(c) for c in coords):
        fig.update_layout(title="Invalid Calculation (NaN / Inf)")
        return fig

    # Collect endpoints for autoscale
    reals = [0, S_s.real, S_i.real, S_o.real, S_r.real,
             S_o.real + S_loss.real]
    imags = [0, S_s.imag, S_i.imag, S_o.imag, S_r.imag,
             S_o.imag + S_loss.imag]

    min_x, max_x = min(reals), max(reals)
    min_y, max_y = min(imags), max(imags)
    margin_x = 0.1 * (max_x - min_x if max_x != min_x else 1)
    margin_y = 0.1 * (max_y - min_y if max_y != min_y else 1)
    x_range = [min_x - margin_x, max_x + margin_x]
    y_range = [min_y - margin_y, max_y + margin_y]

    def add_arrow(x0, y0, x1, y1, color, label):
        """Add a line shape arrow from (x0,y0) to (x1,y1)."""
        fig.add_shape(
            type="line",
            x0=x0, y0=y0, x1=x1, y1=y1,
            xref="x", yref="y",
            line=dict(color=color, width=3),
            arrowhead=3,
            arrowsize=1.5,
            opacity=0.9
        )
        # Label
        fig.add_annotation(
            x=x1, y=y1,
            text=label,
            showarrow=False,
            xanchor="left",
            yanchor="bottom",
            font=dict(color=color, size=14),
            opacity=0.9
        )

    # Arrows from origin
    add_arrow(0, 0, S_s.real, S_s.imag, "red",    "S_s")
    add_arrow(0, 0, S_i.real, S_i.imag, "green",  "S_i")
    add_arrow(0, 0, S_o.real, S_o.imag, "blue",   "S_o")
    add_arrow(0, 0, S_r.real, S_r.imag, "purple", "S_r")

    # Arrow for losses
    add_arrow(
        S_o.real, S_o.imag,
        S_o.real + S_loss.real, S_o.imag + S_loss.imag,
        "magenta", "S_loss"
    )

    fig.update_layout(
        title="Power Flow Vectors (MVA)",
        xaxis=dict(range=x_range, zeroline=True),
        yaxis=dict(range=y_range, zeroline=True),
        width=700, height=500
    )
    return fig

def plot_bar_chart(S_s, S_i, S_o, S_loss, S_r):
    """
    Creates a grouped bar chart for real (MW) and imaginary (MVAr) components.
    """
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

# =========================
# 4) STREAMLIT APP
# =========================
def main():
    st.title("Transmission Lines")
    st.write(
        """
        This app calculates and visualizes the power flows (in MVA) for a single 
        transmission line under different parameters and line types. 
        Adjust the sliders to see **real-time** changes in the power-flow vectors.
        """
    )

    # Display the line diagram from Postimages
    st.image(
        "https://i.postimg.cc/FKStDhY9/Frame-2-1.png",
        caption="Power Flow on a Transmission Line (Pi Model)",
        width=500
    )

    # Display table of line cases
    st.subheader("Line Cases Table")
    st.table(df_cases)

    st.subheader("Line & Operating Parameters")

    # 1) Select a line case
    selected_case_name = st.selectbox("Select a line case", list(case_parameters.keys()))
    selected_case = case_parameters[selected_case_name]

    # 2) Sliders for user inputs
    col1, col2, col3 = st.columns(3)
    with col1:
        Vs = st.slider("Vs (p.u.)", min_value=0.8, max_value=1.2, value=1.00, step=0.01)
        Vr = st.slider("Vr (p.u.)", min_value=0.8, max_value=1.2, value=0.95, step=0.01)
    with col2:
        delta_deg = st.slider("Delta (°)", min_value=-60, max_value=60, value=10, step=1)
        length_km = st.slider("Line Length (km)", min_value=1.0, max_value=300.0,
                              value=float(selected_case["length"]), step=1.0)
    with col3:
        # Give R & X small lower bounds so they're never fully zero
        R_ohm_km = st.slider("R (Ω/km)", min_value=0.0001, max_value=0.5,
                             value=selected_case["R"], step=0.01)
        X_ohm_km = st.slider("X (Ω/km)", min_value=0.0001, max_value=0.5,
                             value=selected_case["X"], step=0.01)
        C_nF_km = st.slider("C (nF/km)", min_value=0.1, max_value=400.0,
                            value=selected_case["C"]*1e9, step=10.0)

    # Convert C from nF to F
    C_f = C_nF_km * 1e-9

    # 3) Calculate power flow
    S_s, S_i, S_o, S_loss, S_r = calculate_power_flow(
        Vs, Vr, R_ohm_km, X_ohm_km, C_f, delta_deg,
        selected_case["base_voltage"], length_km
    )

    # 4) Display results
    st.subheader("Power Flow Results (in MVA = MW + jMVAr)")
    colA, colB, colC, colD, colE = st.columns(5)
    def format_complex(c):
        if np.isnan(c.real) or not np.isfinite(c.real):
            return "NaN"
        return f"{c.real:.1f} + j{c.imag:.1f}"

    colA.metric("S_s", format_complex(S_s))
    colB.metric("S_i", format_complex(S_i))
    colC.metric("S_o", format_complex(S_o))
    colD.metric("S_loss", format_complex(S_loss))
    colE.metric("S_r", format_complex(S_r))

    # 5) Plot the vectors with advanced approach
    fig_vector = plot_power_vectors(S_s, S_i, S_o, S_loss, S_r)
    st.plotly_chart(fig_vector, use_container_width=True)

    # 6) Plot bar chart
    fig_bar = plot_bar_chart(S_s, S_i, S_o, S_loss, S_r)
    st.plotly_chart(fig_bar, use_container_width=True)


if __name__ == "__main__":
    main()
