import streamlit as st
import numpy as np

# The same function you used:
def calculate_power_flow(Vs, Vr, R, X, C, delta, base_voltage, length):
    delta_rad = np.deg2rad(delta)  # Convert delta to radians
    Z = (R + 1j * X) * length  # Impedance in Ω
    V_s_actual = Vs * base_voltage  # Convert Vs to kV
    V_r_actual = Vr * base_voltage  # Convert Vr to kV
    I_line = (V_s_actual - V_r_actual * np.exp(-1j * delta_rad)) / Z

    # ...and so forth...

    return S_s, S_i, S_o, S_loss, S_r, Q_s, Q_r

# Build the Streamlit UI:
st.title("Power Flow Simulator")

# Streamlit sliders for user input
Vs = st.slider("Vs (pu)", min_value=0.8, max_value=1.2, value=1.00, step=0.01)
Vr = st.slider("Vr (pu)", min_value=0.8, max_value=1.2, value=0.95, step=0.01)
R = st.slider("R (Ω/km)", min_value=0.001, max_value=0.4, value=0.3, step=0.0001)
X = st.slider("X (Ω/km)", min_value=0.001, max_value=0.5, value=0.45, step=0.0001)
C_nF = st.slider("C (nF/km)", min_value=0.1, max_value=400.0, value=20.0, step=1.0)
delta = st.slider("delta (deg)", min_value=-60, max_value=60, value=10, step=1)
length = st.slider("Line Length (km)", min_value=1, max_value=200, value=10, step=1)
base_voltage = st.selectbox("Voltage Level (kV)", options=[10, 110, 400], index=0)

# Convert nF to F
C = C_nF * 1e-9

# Calculate the results
S_s, S_i, S_o, S_loss, S_r, Q_s, Q_r = calculate_power_flow(Vs, Vr, R, X, C, delta, base_voltage, length)

# Display the results
st.write("### Results")
st.write(f"Sending End Power, S_s = {S_s.real:.2f} MW + j{S_s.imag:.2f} MVAr")
st.write(f"Power after Shunt, S_i = {S_i.real:.2f} MW + j{S_i.imag:.2f} MVAr")
st.write(f"Power before Shunt, S_o = {S_o.real:.2f} MW + j{S_o.imag:.2f} MVAr")
st.write(f"Power Losses, S_loss = {S_loss.real:.2f} MW + j{S_loss.imag:.2f} MVAr")
st.write(f"Receiving End Power, S_r = {S_r.real:.2f} MW + j{S_r.imag:.2f} MVAr")

st.write(f"Reactive Power at Sending End, Q_s = {Q_s:.2f} MVAr")
st.write(f"Reactive Power at Receiving End, Q_r = {Q_r:.2f} MVAr")
