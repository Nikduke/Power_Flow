import streamlit as st
import numpy as np
import plotly.graph_objects as go

# ========== CASE PARAMETERS ==========
case_parameters = {
    "10 kV Overhead Line":  {"R": 0.30,  "X": 0.45, "C": 20e-9,   "length": 10,  "base_voltage": 10},
    "10 kV Cable":          {"R": 0.20,  "X": 0.20, "C": 300e-9,  "length": 5,   "base_voltage": 10},
    "110 kV Overhead Line": {"R": 0.07,  "X": 0.35, "C": 15e-9,   "length": 50,  "base_voltage": 110},
    "110 kV Cable":         {"R": 0.055, "X": 0.15, "C": 200e-9,  "length": 20,  "base_voltage": 110},
    "400 kV Overhead Line": {"R": 0.015, "X": 0.30, "C": 5e-9,    "length": 200, "base_voltage": 400},
    "400 kV Cable":         {"R": 0.015, "X": 0.10, "C": 150e-9,  "length": 50,  "base_voltage": 400},
}

# ========== POWER FLOW CALCULATION ==========
def calculate_power_flow(Vs, Vr, R, X, C, delta_deg, base_voltage, length, frequency=50):
    """
    Vs, Vr: sending & receiving end voltages (p.u.)
    R, X, C: per-km line parameters
    delta_deg: angle in degrees (receiving-end w.r.t. sending-end)
    base_voltage: nominal kV level (e.g. 10, 110, 400)
    length: line length in km
    frequency: system frequency in Hz
    """
    delta_rad = np.deg2rad(delta_deg)
    
    # Impedance in ohms for the entire line
    Z = (R + 1j*X) * length

    # Convert from p.u. to actual kV
    V_s_actual = Vs * base_voltage
    V_r_actual = Vr * base_voltage

    # Current (kA)
    #   Vr has an angle delta, so Vr * exp(-j*delta_rad) is used in the original approach
    #   but we can represent it directly as complex if we prefer. For simplicity:
    V_r_phasor = V_r_actual * np.exp(-1j * delta_rad)  # “minus” if we consider receiving end lag/lead
    I_line = (V_s_actual - V_r_phasor) / Z

    # Apparent power at sending end (kV * kA = MVA)
    S_i = V_s_actual * np.conjugate(I_line)

    # Reactive power from line capacitance at sending end
    Q_s = 2 * np.pi * frequency * C * length * (V_s_actual ** 2)  # MVAr
    S_s = S_i - 1j * Q_s  # MVA

    # Losses
    S_loss = I_line * np.conjugate(I_line) * Z  # MW + jMVAr

    # Power before shunt at receiving end
    S_o = S_i - S_loss

    # Reactive power from line capacitance at receiving end
    Q_r = 2 * np.pi * frequency * C * length * (V_r_actual ** 2)  # MVAr
    S_r = S_o + 1j * Q_r

    return S_s, S_i, S_o, S_loss, S_r


# ========== PLOTTING HELPER FUNCTIONS ==========

def plot_power_vectors(S_s, S_i, S_o, S_loss, S_r):
    """
    Creates a 2D vector diagram (Real vs. Imag part) using Plotly.
    """
    fig = go.Figure()

    # Common arrow style
    arrow_style = dict(
        xref="x", yref="y",
        showarrow=True,
        arrowhead=3,
        arrowsize=1,
        arrowwidth=2
    )

    def add_arrow(fig, end_complex, color, name):
        """Draw an arrow from (0,0) to end_complex in the complex plane."""
        fig.add_annotation(
            x=end_complex.real, 
            y=end_complex.imag,
            ax=0, ay=0,
            arrowcolor=color,
            text=name,
            **arrow_style
        )

    # Vectors from origin
    add_arrow(fig, S_s, "red", "S_s")
    add_arrow(fig, S_i, "green", "S_i")
    add_arrow(fig, S_o, "blue", "S_o")
    add_arrow(fig, S_r, "purple", "S_r")

    # Power loss arrow: from S_o to S_o + S_loss
    fig.add_annotation(
        x=S_o.real + S_loss.real,
        y=S_o.imag + S_loss.imag,
        ax=S_o.real,
        ay=S_o.imag,
        arrowcolor="magenta",
        text="S_loss",
        **arrow_style
    )

    # Format the axes
    fig.update_layout(
        title="Power Flow Vectors (MVA)",
        xaxis_title="Real Power (MW)",
        yaxis_title="Reactive Power (MVAr)",
        xaxis=dict(zeroline=True, range=[-50, 150]),
        yaxis=dict(zeroline=True, range=[-50, 150]),
        width=700,
        height=500
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

# ========== STREAMLIT APP ==========

def main():
    st.title("Power Flow Simulator for Transmission Lines")
    st.write(
        """
        This app calculates and visualizes the power flows (in MVA) for a single transmission line 
        under different parameters and line types. Adjust the sliders below or select a line case.
        """
    )

    # 1) Select case
    selected_case_name = st.selectbox("Select a line case", list(case_parameters.keys()))
    selected_case = case_parameters[selected_case_name]

    # 2) Sliders for user inputs
    st.subheader("Line & Operating Parameters")
    col1, col2, col3 = st.columns(3)
    with col1:
        Vs = st.slider("Vs (p.u.)", min_value=0.8, max_value=1.2, value=1.0, step=0.01)
        Vr = st.slider("Vr (p.u.)", min_value=0.8, max_value=1.2, value=0.95, step=0.01)
    with col2:
        delta_deg = st.slider("Delta (°)", min_value=-60, max_value=60, value=10, step=1)
        length_km = st.slider("Line Length (km)", min_value=1, max_value=300, value=selected_case["length"], step=1)
    with col3:
        R_ohm_km = st.slider("R (Ω/km)", min_value=0.001, max_value=0.5, value=selected_case["R"], step=0.001)
        X_ohm_km = st.slider("X (Ω/km)", min_value=0.001, max_value=0.5, value=selected_case["X"], step=0.001)
        C_nF_km = st.slider("C (nF/km)", min_value=0.1, max_value=400.0, value=selected_case["C"]*1e9, step=1.0)

    # Convert C back to F
    C_f = C_nF_km * 1e-9

    # 3) Calculate
    S_s, S_i, S_o, S_loss, S_r = calculate_power_flow(
        Vs, Vr,
        R_ohm_km, X_ohm_km, C_f,
        delta_deg,
        selected_case["base_voltage"],
        length_km
    )

    # 4) Display numeric results
    st.subheader("Power Flow Results (in MVA = MW + jMVAr)")
    colA, colB, colC, colD, colE = st.columns(5)
    colA.metric("S_s", f"{S_s.real:.2f} + j{S_s.imag:.2f}")
    colB.metric("S_i", f"{S_i.real:.2f} + j{S_i.imag:.2f}")
    colC.metric("S_o", f"{S_o.real:.2f} + j{S_o.imag:.2f}")
    colD.metric("S_loss", f"{S_loss.real:.2f} + j{S_loss.imag:.2f}")
    colE.metric("S_r", f"{S_r.real:.2f} + j{S_r.imag:.2f}")

    # 5) Plot vector diagram
    fig_vector = plot_power_vectors(S_s, S_i, S_o, S_loss, S_r)
    st.plotly_chart(fig_vector, use_container_width=True)

    # 6) Plot bar chart
    fig_bar = plot_bar_chart(S_s, S_i, S_o, S_loss, S_r)
    st.plotly_chart(fig_bar, use_container_width=True)


if __name__ == "__main__":
    main()
