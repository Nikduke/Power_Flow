import streamlit as st
import numpy as np
import pandas as pd
import requests
from io import BytesIO
from PIL import Image

# Bokeh 2.4.3
from bokeh.plotting import figure
from bokeh.models import Arrow, VeeHead, ColumnDataSource
from bokeh.models.tools import HoverTool

# ---------------------------
# Data for Different Line Cases
# ---------------------------
case_parameters = [
    {
        "name": "10 kV Overhead Line",
        "Voltage_Level": "10 kV",
        "Type": "Overhead Line",
        "R": 0.30,
        "X": 0.45,
        "C": 20e-9,
        "length": 10,
        "base_voltage": 10
    },
    {
        "name": "10 kV Cable",
        "Voltage_Level": "10 kV",
        "Type": "Cable",
        "R": 0.20,
        "X": 0.20,
        "C": 300e-9,
        "length": 5,
        "base_voltage": 10
    },
    {
        "name": "110 kV Overhead Line",
        "Voltage_Level": "110 kV",
        "Type": "Overhead Line",
        "R": 0.07,
        "X": 0.35,
        "C": 15e-9,
        "length": 50,
        "base_voltage": 110
    },
    {
        "name": "110 kV Cable",
        "Voltage_Level": "110 kV",
        "Type": "Cable",
        "R": 0.055,
        "X": 0.15,
        "C": 200e-9,
        "length": 20,
        "base_voltage": 110
    },
    {
        "name": "400 kV Overhead Line",
        "Voltage_Level": "400 kV",
        "Type": "Overhead Line",
        "R": 0.015,
        "X": 0.30,
        "C": 5e-9,
        "length": 200,
        "base_voltage": 400
    },
    {
        "name": "400 kV Cable",
        "Voltage_Level": "400 kV",
        "Type": "Cable",
        "R": 0.015,
        "X": 0.10,
        "C": 150e-9,
        "length": 50,
        "base_voltage": 400
    },
]


# ---------------------------
# Power Flow Calculation
# ---------------------------
def calculate_power_flow(Vs, Vr, R, X, C, delta_deg, base_voltage, length):
    """
    Calculates approximate power flow using a line pi-section model.
    Returns (S_s, S_i, S_o, S_loss, S_r, Q_s, Q_r).
    """
    frequency = 50  # Hz
    delta_rad = np.deg2rad(delta_deg)

    # Total line impedance
    Z = (R + 1j * X) * length

    # Convert per-unit voltages to actual kV
    V_s_kV = Vs * base_voltage
    V_r_kV = Vr * base_voltage

    # Receiving end voltage as a complex (angle = -delta)
    V_r_complex = V_r_kV * np.exp(-1j * delta_rad)

    # Current
    I_line = (V_s_kV - V_r_complex) / Z

    # S_i: sending side, after shunt
    S_i = V_s_kV * np.conjugate(I_line)

    # Reactive power at sending end
    Q_s = 2 * np.pi * frequency * C * length * (V_s_kV ** 2)
    S_s = S_i - 1j * Q_s

    # Power losses
    S_loss = I_line * np.conjugate(I_line) * Z

    # Power before shunt at receiving end
    S_o = S_i - S_loss

    # Reactive power at receiving end
    Q_r = 2 * np.pi * frequency * C * length * (V_r_kV ** 2)
    S_r = S_o + 1j * Q_r

    return S_s, S_i, S_o, S_loss, S_r, Q_s, Q_r


# ---------------------------
# Helpers: Create Bokeh Plots
# ---------------------------
def create_vector_figure(S_s, S_i, S_o, S_r, S_loss):
    """
    Bokeh figure with arrows for power flow vectors.
    Use single-element arrays in ColumnDataSource so lengths match.
    """
    source = ColumnDataSource(data=dict(
        # S_s: from (0,0) -> (S_s.real, S_s.imag)
        s_s_x0=[0], s_s_y0=[0],
        s_s_x1=[S_s.real], s_s_y1=[S_s.imag],

        # S_i: from (0,0) -> (S_i.real, S_i.imag)
        s_i_x0=[0], s_i_y0=[0],
        s_i_x1=[S_i.real], s_i_y1=[S_i.imag],

        # S_o: from (0,0) -> (S_o.real, S_o.imag)
        s_o_x0=[0], s_o_y0=[0],
        s_o_x1=[S_o.real], s_o_y1=[S_o.imag],

        # S_r: from (0,0) -> (S_r.real, S_r.imag)
        s_r_x0=[0], s_r_y0=[0],
        s_r_x1=[S_r.real], s_r_y1=[S_r.imag],

        # S_loss: from S_o to S_o + S_loss
        loss_x0=[S_o.real],
        loss_y0=[S_o.imag],
        loss_x1=[S_o.real + S_loss.real],
        loss_y1=[S_o.imag + S_loss.imag],
    ))

    p = figure(title="Power Flow Vectors",
               x_axis_label="Real Power (MW)",
               y_axis_label="Reactive Power (MVAr)",
               width=600, height=450)
    arrow_head = VeeHead(size=10)

    # Draw arrows
    p.add_layout(Arrow(end=arrow_head,
                       x_start='s_s_x0', y_start='s_s_y0',
                       x_end='s_s_x1',   y_end='s_s_y1',
                       source=source, line_width=3, line_color="red"))
    p.add_layout(Arrow(end=arrow_head,
                       x_start='s_i_x0', y_start='s_i_y0',
                       x_end='s_i_x1',   y_end='s_i_y1',
                       source=source, line_width=3, line_color="green"))
    p.add_layout(Arrow(end=arrow_head,
                       x_start='s_o_x0', y_start='s_o_y0',
                       x_end='s_o_x1',   y_end='s_o_y1',
                       source=source, line_width=3, line_color="blue"))
    p.add_layout(Arrow(end=arrow_head,
                       x_start='s_r_x0', y_start='s_r_y0',
                       x_end='s_r_x1',   y_end='s_r_y1',
                       source=source, line_width=3, line_color="purple"))
    p.add_layout(Arrow(end=arrow_head,
                       x_start='loss_x0', y_start='loss_y0',
                       x_end='loss_x1',   y_end='loss_y1',
                       source=source, line_width=3, line_color="magenta"))

    p.add_tools(HoverTool())
    return p


def create_bar_chart(S_s, S_i, S_o, S_r, S_loss):
    """Bokeh bar chart of active/reactive power."""
    labels = ["S_s", "S_i", "S_o", "S_r", "S_loss"]
    active_values = [S_s.real, S_i.real, S_o.real, S_r.real, S_loss.real]
    reactive_values = [S_s.imag, S_i.imag, S_o.imag, S_r.imag, S_loss.imag]

    source_bar = ColumnDataSource(data=dict(
        labels=labels,
        active=active_values,
        reactive=reactive_values
    ))

    p_bar = figure(x_range=labels, title="Active vs Reactive Power",
                   width=500, height=450, toolbar_location=None, tools="")

    x_vals = list(range(len(labels)))
    width = 0.3

    # Active bars
    p_bar.vbar(x=[x - 0.15 for x in x_vals],
               top='active', width=width,
               source=source_bar, color="#4a69bd", legend_label="Active (MW)")
    # Reactive bars
    p_bar.vbar(x=[x + 0.15 for x in x_vals],
               top='reactive', width=width,
               source=source_bar, color="#e55039", legend_label="Reactive (MVAr)")

    p_bar.xaxis.ticker = x_vals
    p_bar.xaxis.major_label_overrides = {i: lbl for i, lbl in enumerate(labels)}
    p_bar.yaxis.axis_label = "Power"
    p_bar.legend.location = "top_left"

    return p_bar


# ---------------------------
# Streamlit UI
# ---------------------------
st.set_page_config(layout="wide")
st.title("Power Flow Simulator")

# Show table of line cases & diagram
col_img, col_tbl = st.columns([1, 2])

with col_img:
    st.subheader("Line Diagram (PI Section)")
    url = "https://i.postimg.cc/FKStDhY9/Frame-2-1.png"
    try:
        r = requests.get(url, timeout=5)
        im = Image.open(BytesIO(r.content))
        st.image(im, width=300)
    except:
        st.write("Diagram not available.")

with col_tbl:
    st.subheader("Example Cases")
    df_table = pd.DataFrame({
        "Voltage_Level": [c["Voltage_Level"] for c in case_parameters],
        "Type": [c["Type"] for c in case_parameters],
        "R (Ω/km)": [c["R"] for c in case_parameters],
        "X (Ω/km)": [c["X"] for c in case_parameters],
        "C (nF/km)": [f"{c['C']*1e9:.1f}" for c in case_parameters],
        "Length (km)": [c["length"] for c in case_parameters],
    })
    st.dataframe(df_table, use_container_width=True)


# Select a default case
case_names = [c["name"] for c in case_parameters]
selected_case = st.selectbox("Select a line/cable case:", case_names)
default_case = next(c for c in case_parameters if c["name"] == selected_case)

st.markdown("---")
col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    Vs = st.slider("Vs (pu)", 0.8, 1.2, 1.0, 0.01)
    Vr = st.slider("Vr (pu)", 0.8, 1.2, 0.95, 0.01)

with col2:
    R = st.number_input("R (Ω/km)", 0.0001, 1.0, float(default_case["R"]), 0.0001)
    X = st.number_input("X (Ω/km)", 0.0001, 2.0, float(default_case["X"]), 0.0001)

with col3:
    default_c_nF = default_case["C"] * 1e9
    c_nF = st.number_input("C (nF/km)", 0.1, 1000.0, float(default_c_nF), 1.0)
    C = c_nF * 1e-9

with col4:
    delta_deg = st.slider("Delta (°)", -60, 60, 10, 1)

with col5:
    length = st.number_input("Line Length (km)", 1.0, 1000.0, float(default_case["length"]), 1.0)
    base_voltage = st.number_input("Base Voltage (kV)", 1.0, 1000.0, float(default_case["base_voltage"]), 1.0)

# Perform calculation
S_s, S_i, S_o, S_loss, S_r, Q_s, Q_r = calculate_power_flow(
    Vs, Vr, R, X, C, delta_deg, base_voltage, length
)

# Show plots
col_p1, col_p2 = st.columns(2)
with col_p1:
    st.subheader("Power Vectors")
    fig_vec = create_vector_figure(S_s, S_i, S_o, S_r, S_loss)
    st.bokeh_chart(fig_vec, use_container_width=True)

with col_p2:
    st.subheader("Active vs Reactive")
    fig_bar = create_bar_chart(S_s, S_i, S_o, S_r, S_loss)
    st.bokeh_chart(fig_bar, use_container_width=True)

# Display numeric results
st.markdown("---")
st.subheader("Results")

cA, cB, cC = st.columns(3)
with cA:
    st.write("**Sending End (S_s):**")
    st.write(f"{S_s.real:.2f} MW + j{S_s.imag:.2f} MVAr")

    st.write("**After Shunt (S_i):**")
    st.write(f"{S_i.real:.2f} MW + j{S_i.imag:.2f} MVAr")

with cB:
    st.write("**Before Shunt (S_o):**")
    st.write(f"{S_o.real:.2f} MW + j{S_o.imag:.2f} MVAr")

    st.write("**Receiving End (S_r):**")
    st.write(f"{S_r.real:.2f} MW + j{S_r.imag:.2f} MVAr")

with cC:
    st.write("**Line Loss (S_loss):**")
    st.write(f"{S_loss.real:.2f} MW + j{S_loss.imag:.2f} MVAr")

    st.write(f"**Q_s** (MVAr at Sending): {Q_s:.2f}")
    st.write(f"**Q_r** (MVAr at Receiving): {Q_r:.2f}")

st.markdown("---")
st.caption("Using Bokeh 2.4.3 + NumPy 1.24+ to avoid np.bool8 errors.")


#
# That’s all!
#
