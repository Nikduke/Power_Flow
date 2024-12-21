import streamlit as st
import numpy as np

# ----------------------------------
# Power Flow Calculation Function
# ----------------------------------
def calculate_power_flow(Vs, Vr, R, X, C, delta, base_voltage, length):
    """
    Calculates power flow on a line pi-section based on user inputs.

    Parameters:
    -----------
    Vs : float
        Sending-end voltage (pu).
    Vr : float
        Receiving-end voltage (pu).
    R : float
        Resistance (ohms per km).
    X : float
        Reactance (ohms per km).
    C : float
        Capacitance (F per km).
    delta : float
        Phase angle difference (degrees).
    base_voltage : float
        System base voltage (kV).
    length : float
        Line length (km).

    Returns:
    --------
    S_s, S_i, S_o, S_loss, S_r, Q_s, Q_r : complex or float
        Various power flow values in MVA (real + j*imag).
        Q_s and Q_r in MVAr (floats).
    """
    frequency = 50  # Hz
    # Convert angle to radians
    delta_rad = np.deg2rad(delta)

    # Compute line impedance for the entire length
    Z = (R + 1j * X) * length  # total ohms

    # Actual (kV) from per-unit
    V_s_actual = Vs * base_voltage
    V_r_actual = Vr * base_voltage

    # Line current
    #   I = (Vs - Vr * e^{-j delta}) / Z
    #   note: Vr has angle delta (we assume delta is the difference)
    #         here we do Vr * exp(-j delta)
    I_line = (V_s_actual - (V_r_actual * np.exp(-1j * delta_rad))) / Z

    # Power after "shunt" at sending side: S_i = Vs * I*
    #   (here: kV * kA = MVA, but keep consistent with complex math)
    S_i = V_s_actual * np.conjugate(I_line)

    # Reactive power from line capacitance at sending end (MVAr)
    #   Q_s = w * C * (Vs^2)
    #   w = 2*pi*f
    Q_s = 2 * np.pi * frequency * C * length * (V_s_actual**2)

    # Total sending end power (S_s)
    #   S_s = S_i - j Q_s
    #   (capacitance injects reactive, so we subtract j*Q_s)
    S_s = S_i - 1j * Q_s

    # Power losses in line: S_loss = I^2 * Z
    S_loss = I_line * np.conjugate(I_line) * Z

    # Power "before shunt" at receiving side: S_o = S_i - S_loss
    S_o = S_i - S_loss

    # Reactive power from line capacitance at receiving end (MVAr)
    Q_r = 2 * np.pi * frequency * C * length * (V_r_actual**2)

    # Receiving end power (S_r)
    #   S_r = S_o + j Q_r
    S_r = S_o + 1j * Q_r

    return S_s, S_i, S_o, S_loss, S_r, Q_s, Q_r

# ------------------------
# Streamlit App Layout
# ------------------------
st.title("Power Flow Simulator for Transmission Lines")

st.markdown("""
This is a simple Streamlit app that calculates power flow 
(approximate line PI section model) based on user inputs.
""")

# Sliders (per-unit, ohms per km, etc.)
Vs = st.slider("Sending Voltage (pu)", min_value=0.80, max_value=1.20, value=1.00, step=0.01)
Vr = st.slider("Receiving Voltage (pu)", min_value=0.80, max_value=1.20, value=0.95, step=0.01)

R = st.slider("Line Resistance (Ω/km)", min_value=0.001, max_value=0.4, value=0.30, step=0.0001)
X = st.slider("Line Reactance (Ω/km)", min_value=0.001, max_value=0.5, value=0.45, step=0.0001)

C_nF = st.slider("Line Capacitance (nF/km)", min_value=0.1, max_value=400.0, value=20.0, step=1.0)
C = C_nF * 1e-9  # Convert nF to F

delta = st.slider("Phase Angle (degrees)", min_value=-60, max_value=60, value=10, step=1)
length = st.slider("Line Length (km)", min_value=1, max_value=200, value=10, step=1)

# Choose base voltage from dropdown
base_voltage = st.selectbox("Voltage Level (kV)", [10, 110, 400], index=0)

# ------------
# Computation
# ------------
S_s, S_i, S_o, S_loss, S_r, Q_s, Q_r = calculate_power_flow(
    Vs, Vr, R, X, C, delta, base_voltage, length
)

# ------------
# Display Results
# ------------
st.subheader("Results (Approximate)")

# Because S_s, S_i, etc. are complex, we can extract real and imag parts:
st.write(f"**Sending End Power (S_s)** = {S_s.real:.2f} MW + j{S_s.imag:.2f} MVAr")
st.write(f"**Power after Shunt (S_i)** = {S_i.real:.2f} MW + j{S_i.imag:.2f} MVAr")
st.write(f"**Power before Shunt (S_o)** = {S_o.real:.2f} MW + j{S_o.imag:.2f} MVAr")
st.write(f"**Receiving End Power (S_r)** = {S_r.real:.2f} MW + j{S_r.imag:.2f} MVAr")
st.write(f"**Power Losses (S_loss)** = {S_loss.real:.2f} MW + j{S_loss.imag:.2f} MVAr")

st.write(f"**Reactive Power at Sending End (Q_s)** = {Q_s:.2f} MVAr")
st.write(f"**Reactive Power at Receiving End (Q_r)** = {Q_r:.2f} MVAr")


st.markdown("---")
st.markdown("Designed with ❤️ using [Streamlit](https://streamlit.io).")
