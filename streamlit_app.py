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

# ===========================================
# 2) POWER FLOW CALCULATION (returns Q_s, Q_r)
# ===========================================
@st.cache_data
def calculate_power_flow(
    Vs, Vr, R, X, C, delta_deg, base_voltage, length, frequency=50
):
    """
    Calculates power-flow for a single transmission line in MVA = MW + jMVAr.
    Also returns Q_s, Q_r separately (in MVAr).

    Returns:
      S_s, S_i, S_o, S_loss, S_r (complex, MVA),
      Q_s, Q_r (float, MVAr)
    """
    delta_rad = np.deg2rad(delta_deg)
    Z = (R + 1j * X) * length

    # if the line is too small => invalid flows
    if abs(Z) < 1e-12:
        return (np.nan,)*5 + (np.nan, np.nan)

    # Convert from p.u. to kV
    V_s_actual = Vs * base_voltage
    V_r_actual = Vr * base_voltage

    # Receiving phasor
    V_r_phasor = V_r_actual * np.exp(-1j * delta_rad)

    # Current
    I_line = (V_s_actual - V_r_phasor) / Z

    # S_i
    S_i = V_s_actual * np.conjugate(I_line)  # MVA

    # Q_s
    # reactive power at sending end due to line capacitance (MVAr)
    Q_s = 2 * np.pi * frequency * C * length * (V_s_actual**2)
    S_s = S_i - 1j * Q_s

    # Losses
    S_loss = I_line * np.conjugate(I_line) * Z

    # S_o
    S_o = S_i - S_loss

    # Q_r
    Q_r = 2 * np.pi * frequency * C * length * (V_r_actual**2)
    S_r = S_o + 1j * Q_r

    return S_s, S_i, S_o, S_loss, S_r, Q_s, Q_r


# ===============================
# 3) PLOTTING WITH ANNOTATIONS
# ===============================
def plot_power_vectors(S_s, S_i, S_o, S_loss, S_r):
    """
    Draw arrowed vectors (S_s, S_i, S_o, S_r) from origin,
    plus S_loss from S_o to (S_o + S_loss).
    """
    fig = go.Figure()

    # Check invalid
    coords = [
        S_s.real, S_s.imag, S_i.real, S_i.imag,
        S_o.real, S_o.imag, S_r.real, S_r.imag,
        (S_o.real + S_loss.real), (S_o.imag + S_loss.imag)
    ]
    if any(np.isnan(c) or not np.isfinite(c) for c in coords):
        fig.update_layout(title="Invalid flows (NaN or Inf). Check your parameters!")
        return fig

    # Auto-scale
    reals = [0, S_s.real, S_i.real, S_o.real, S_r.real, S_o.real + S_loss.real]
    imags = [0, S_s.imag, S_i.imag, S_o.imag, S_r.imag, S_o.imag + S_loss.imag]

    min_x, max_x = min(reals), max(reals)
    min_y, max_y = min(imags), max(imags)
    margin_x = 0.1 * (max_x - min_x if max_x != min_x else 1)
    margin_y = 0.1 * (max_y - min_y if max_y != min_y else 1)
    x_range = [min_x - margin_x, max_x + margin_x]
    y_range = [min_y - margin_y, max_y + margin_y]

    def add_arrow(x0, y0, x1, y1, color, label):
        fig.add_annotation(
            x=x1, y=y1, xref="x", yref="y",
            ax=x0, ay=y0, axref="x", ayref="y",
            showarrow=True,
            arrowhead=3,   # triangle arrow
            arrowsize=1.5,
            arrowwidth=2,
            arrowcolor=color,
            text=label,
            font=dict(color=color, size=14),
        )

    # Arrows from origin
    add_arrow(0, 0, S_s.real, S_s.imag, "red", "S_s")
    add_arrow(0, 0, S_i.real, S_i.imag, "green", "S_i")
    add_arrow(0, 0, S_o.real, S_o.imag, "blue", "S_o")
    add_arrow(0, 0, S_r.real, S_r.imag, "purple", "S_r")

    # Arrow from S_o => S_o + S_loss
    add_arrow(
        S_o.real, S_o.imag,
        S_o.real + S_loss.real,
        S_o.imag + S_loss.imag,
        "magenta", "S_loss"
    )

    fig.update_layout(
        title="Power Flow Vectors (MVA)",
        xaxis=dict(range=x_range, zeroline=True),
        yaxis=dict(range=y_range, zeroline=True),
        width=800,
        height=500
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
    st.title("Transmission Lines with Q_s & Q_r and a Table")
    st.write(
        """
        This version shows Q_s and Q_r in addition to the main power flows, and 
        displays results in a single table for easy reading.
        """
    )

    # Show original line diagram
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
        delta_deg = st.slider("Angle (°)", -60, 60, 10, 1)
        length_km = st.slider("Line Length (km)", 1.0, 300.0, float(scase["length"]), 1.0)
    with col3:
        R_ohm_km = st.slider("R (Ω/km)", 0.0001, 0.5, float(scase["R"]), 0.01)
        X_ohm_km = st.slider("X (Ω/km)", 0.0001, 0.5, float(scase["X"]), 0.01)
        C_nF_km = st.slider("C (nF/km)", 0.1, 400.0, float(scase["C"]*1e9), 10.0)

    # Convert nF -> F
    C_f = C_nF_km * 1e-9

    # Calculate
    S_s, S_i, S_o, S_loss, S_r, Q_s, Q_r = calculate_power_flow(
        Vs, Vr, R_ohm_km, X_ohm_km, C_f, delta_deg,
        scase["base_voltage"], length_km
    )

    # Build a small DataFrame for final results
    # We'll show the real+imag parts of S_x plus Q_s, Q_r
    def fmt_complex(c):
        if np.isnan(c.real) or not np.isfinite(c.real):
            return "NaN"
        return f"{c.real:.2f} + j{c.imag:.2f}"

    # Rows:
    #  "S_s", "S_i", "S_o", "S_loss", "S_r", "Q_s", "Q_r"
    table_data = {
        "Component": [
            "S_s", "S_i", "S_o", "S_loss", "S_r", "Q_s (MVAr)", "Q_r (MVAr)"
        ],
        "Value": [
            fmt_complex(S_s),
            fmt_complex(S_i),
            fmt_complex(S_o),
            fmt_complex(S_loss),
            fmt_complex(S_r),
            f"{Q_s:.2f}" if np.isfinite(Q_s) else "NaN",
            f"{Q_r:.2f}" if np.isfinite(Q_r) else "NaN",
        ]
    }
    df_results = pd.DataFrame(table_data)

    st.subheader("Power Flow Results (MVA = MW + jMVAr)")
    st.table(df_results)

    # Plot vectors
    fig_vector = plot_power_vectors(S_s, S_i, S_o, S_loss, S_r)
    st.plotly_chart(fig_vector, use_container_width=True)

    # Plot bar chart
    fig_bar = plot_bar_chart(S_s, S_i, S_o, S_loss, S_r)
    st.plotly_chart(fig_bar, use_container_width=True)


if __name__ == "__main__":
    main()
