import streamlit as st
import numpy as np
from bokeh.plotting import figure
from bokeh.models import Arrow, VeeHead, ColumnDataSource
from bokeh.models import Legend
from bokeh.models.tools import HoverTool
from PIL import Image
import requests
from io import BytesIO

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
    Calculates power flow on a line pi-section based on user inputs.

    Returns:
      S_s, S_i, S_o, S_loss, S_r, Q_s, Q_r
      (All complex except Q_s and Q_r are floats)
    """
    frequency = 50  # Hz
    delta_rad = np.deg2rad(delta_deg)

    # Total impedance for the line length
    Z = (R + 1j * X) * length

    # Convert from per-unit to actual kV
    V_s_actual = Vs * base_voltage
    # The receiving end has magnitude Vr * base_voltage with angle delta (approx)
    V_r_actual = Vr * base_voltage

    # Complex receiving voltage: let delta be angle difference (V_r lags or leads)
    #   So Vr = Vr * e^{-j delta}
    #   negative sign because you used (V_s - V_r e^{-j delta}) in your original code
    V_r_complex = V_r_actual * np.exp(-1j * delta_rad)

    # Line current
    I_line = (V_s_actual - V_r_complex) / Z  # kA, but dimensionally it's consistent for MVA

    # Power after Shunt from sending side (S_i)
    # S_i = Vs * conj(I_line)
    S_i = V_s_actual * np.conjugate(I_line)

    # Reactive power at sending end due to line capacitance
    Q_s = 2 * np.pi * frequency * C * length * (V_s_actual ** 2)  # MVAr
    # Total sending end power
    S_s = S_i - 1j * Q_s

    # Losses: S_loss = I^2 * Z
    # (I^2 = I * conj(I))
    S_loss = I_line * np.conjugate(I_line) * Z

    # Power before shunt at receiving side
    S_o = S_i - S_loss

    # Reactive power at receiving end due to line capacitance
    Q_r = 2 * np.pi * frequency * C * length * (V_r_actual ** 2)
    # Receiving end power
    S_r = S_o + 1j * Q_r

    return S_s, S_i, S_o, S_loss, S_r, Q_s, Q_r

# ---------------------------
# Helper to Create Vector Plot
# ---------------------------
def create_vector_figure(S_s, S_i, S_o, S_r, S_loss):
    """
    Create a Bokeh figure showing arrows for S_s, S_i, S_o, S_r, and S_loss.
    """

    # Prepare data for ColumnDataSource
    source = ColumnDataSource(data=dict(
        S_s_real=[0, S_s.real],
        S_s_imag=[0, S_s.imag],
        S_i_real=[0, S_i.real],
        S_i_imag=[0, S_i.imag],
        S_o_real=[0, S_o.real],
        S_o_imag=[0, S_o.imag],
        S_r_real=[0, S_r.real],
        S_r_imag=[0, S_r.imag],
        S_loss_start_real=[S_o.real],
        S_loss_start_imag=[S_o.imag],
        S_loss_end_real=[S_o.real + S_loss.real],
        S_loss_end_imag=[S_o.imag + S_loss.imag],
    ))

    p = figure(
        title="Power Flow Vectors",
        x_axis_label="Real Power (MW)",
        y_axis_label="Reactive Power (MVAr)",
        width=600,
        height=450,
        toolbar_location="above",
    )

    arrow_head = VeeHead(size=10)

    # Arrows from origin
    p.add_layout(Arrow(end=arrow_head, x_start=0, y_start=0,
                       x_end="S_s_real", y_end="S_s_imag",
                       source=source, line_width=3, line_color="red", name="S_s"))
    p.add_layout(Arrow(end=arrow_head, x_start=0, y_start=0,
                       x_end="S_i_real", y_end="S_i_imag",
                       source=source, line_width=3, line_color="green", name="S_i"))
    p.add_layout(Arrow(end=arrow_head, x_start=0, y_start=0,
                       x_end="S_o_real", y_end="S_o_imag",
                       source=source, line_width=3, line_color="blue", name="S_o"))
    p.add_layout(Arrow(end=arrow_head, x_start=0, y_start=0,
                       x_end="S_r_real", y_end="S_r_imag",
                       source=source, line_width=3, line_color="purple", name="S_r"))

    # Arrow for power loss, from S_o to S_o + S_loss
    p.add_layout(Arrow(end=arrow_head,
                       x_start="S_loss_start_real",
                       y_start="S_loss_start_imag",
                       x_end="S_loss_end_real",
                       y_end="S_loss_end_imag",
                       source=source, line_width=3, line_color="magenta", name="S_loss"))

    # Legend-like approach: We'll just manually create lines (invisible) so we can show them in a Legend
    # But simpler approach: we can do p.legend.visible = True if we had labeled glyphs
    legend = Legend(items=[], location="top_right")
    p.add_tools(HoverTool())
    p.add_layout(legend, 'right')

    return p

# ---------------------------
# Helper to Create Bar Chart
# ---------------------------
def create_bar_chart(S_s, S_i, S_o, S_r, S_loss):
    """
    Create a Bokeh bar plot for Active and Reactive powers
    """
    bar_labels = ["S_s", "S_i", "S_o", "S_r", "S_loss"]
    bar_active_values = [S_s.real, S_i.real, S_o.real, S_r.real, S_loss.real]
    bar_reactive_values = [S_s.imag, S_i.imag, S_o.imag, S_r.imag, S_loss.imag]

    source_bar = ColumnDataSource(data=dict(
        labels=bar_labels,
        active=bar_active_values,
        reactive=bar_reactive_values
    ))

    p_bar = figure(
        x_range=bar_labels,
        title="Active and Reactive Power",
        width=500,
        height=450,
        toolbar_location=None,
        tools="",
    )

    # We'll just plot side-by-side bars:
    # Active (blue) vs Reactive (red)
    # Since we don't have fancy transforms here, we'll just shift x's a bit
    x_locations = list(range(len(bar_labels)))

    # One approach is to do small offsets from integer x-locations
    # We'll plot them by creating two separate vbars
    #  => x offset for "active" can be -0.2; for "reactive" +0.2, etc.
    width = 0.4

    # We'll store them in the ColumnDataSource: but we can keep it simple
    # We'll plot with an index
    p_bar.vbar(
        x=[x - 0.15 for x in x_locations],
        top='active',
        width=0.3,
        source=source_bar,
        color="#4a69bd",
        legend_label="Active Power (MW)"
    )

    p_bar.vbar(
        x=[x + 0.15 for x in x_locations],
        top='reactive',
        width=0.3,
        source=source_bar,
        color="#e55039",
        legend_label="Reactive Power (MVAr)"
    )

    p_bar.xaxis.ticker = x_locations
    p_bar.xaxis.major_label_overrides = {i: lbl for i, lbl in enumerate(bar_labels)}
    p_bar.xaxis.axis_label = ""
    p_bar.yaxis.axis_label = "Power (MW / MVAr)"
    p_bar.legend.location = "top_left"

    return p_bar

# ---------------------------
# Streamlit Layout
# ---------------------------
st.set_page_config(layout="wide")  # Use "wide" layout

st.title("Power Flow Simulator for Transmission Lines")

# Row 1: Image + Table side by side
col_image, col_table = st.columns([1,2])

with col_image:
    st.subheader("Power flows on a line PI section")
    # Load image
    img_url = "https://i.postimg.cc/FKStDhY9/Frame-2-1.png"
    try:
        response = requests.get(img_url)
        image = Image.open(BytesIO(response.content))
        st.image(image, width=350)
    except:
        st.write("Image not available.")

with col_table:
    st.subheader("Example cases of different lines")

    # Create a small table for these parameters
    # We'll show them in a simple dataframe
    import pandas as pd

    df_table = pd.DataFrame({
        "Voltage_Level": [case["Voltage_Level"] for case in case_parameters],
        "Type": [case["Type"] for case in case_parameters],
        "Resistance (Ω/km)": [f"{case['R']}" for case in case_parameters],
        "Reactance (Ω/km)": [f"{case['X']}" for case in case_parameters],
        "Capacitance (nF/km)": [f"{case['C']*1e9:.0f}" for case in case_parameters],
        "Length (km)": [case["length"] for case in case_parameters],
    })
    st.dataframe(df_table, use_container_width=True)

# Row 2: Let user pick a predefined case or None
st.markdown("---")
col_case, col_spacer = st.columns([1,4])
with col_case:
    st.subheader("Select a Case")
    case_names = [case["name"] for case in case_parameters]
    selected_case = st.selectbox("Pick a line/cable case:", options=case_names)

    # Find that case in the dictionary
    for case in case_parameters:
        if case["name"] == selected_case:
            default_R = case["R"]
            default_X = case["X"]
            default_C = case["C"]
            default_length = case["length"]
            default_baseV = case["base_voltage"]

# Row 3: Sliders and numeric inputs
st.markdown("---")
col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    Vs = st.slider("Vs (pu)", 0.8, 1.2, 1.0, 0.01)
    Vr = st.slider("Vr (pu)", 0.8, 1.2, 0.95, 0.01)

with col2:
    R = st.number_input("R (Ω/km)", min_value=0.0001, max_value=1.0, value=float(default_R), step=0.0001)
    X = st.number_input("X (Ω/km)", min_value=0.0001, max_value=2.0, value=float(default_X), step=0.0001)

with col3:
    C_nF = st.number_input("C (nF/km)", min_value=0.1, max_value=1000.0, value=float(default_C*1e9), step=1.0)
    C = C_nF * 1e-9

with col4:
    delta_deg = st.slider("Delta (°)", -60, 60, 10, 1)

with col5:
    length = st.number_input("Line Length (km)", min_value=1.0, max_value=1000.0, value=float(default_length), step=1.0)
    base_voltage = st.number_input("Base Voltage (kV)", min_value=1.0, max_value=1000.0, value=float(default_baseV), step=1.0)

# ---------------------------
# Perform Calculations
# ---------------------------
S_s, S_i, S_o, S_loss, S_r, Q_s, Q_r = calculate_power_flow(
    Vs, Vr, R, X, C, delta_deg, base_voltage, length
)

# ---------------------------
# Plots
# ---------------------------
col_plot1, col_plot2 = st.columns(2)

with col_plot1:
    st.subheader("Power Flow Vectors")
    fig_vectors = create_vector_figure(S_s, S_i, S_o, S_r, S_loss)
    st.bokeh_chart(fig_vectors, use_container_width=True)

with col_plot2:
    st.subheader("Active vs Reactive Components")
    fig_bar = create_bar_chart(S_s, S_i, S_o, S_r, S_loss)
    st.bokeh_chart(fig_bar, use_container_width=True)

# ---------------------------
# Display Numeric Results
# ---------------------------
st.markdown("---")
st.subheader("Numeric Results")

col_res1, col_res2, col_res3 = st.columns(3)

with col_res1:
    st.write(f"**Sending End Power (S_s):**")
    st.write(f"{S_s.real:.2f} MW + j {S_s.imag:.2f} MVAr")

    st.write(f"**After Shunt (S_i):**")
    st.write(f"{S_i.real:.2f} MW + j {S_i.imag:.2f} MVAr")

with col_res2:
    st.write(f"**Before Shunt (S_o):**")
    st.write(f"{S_o.real:.2f} MW + j {S_o.imag:.2f} MVAr")

    st.write(f"**Receiving End Power (S_r):**")
    st.write(f"{S_r.real:.2f} MW + j {S_r.imag:.2f} MVAr")

with col_res3:
    st.write(f"**Power Loss (S_loss):**")
    st.write(f"{S_loss.real:.2f} MW + j {S_loss.imag:.2f} MVAr")

    st.write(f"**Q_s (MVAr at Sending):** {Q_s:.2f}")
    st.write(f"**Q_r (MVAr at Receiving):** {Q_r:.2f}")

st.markdown("---")
st.caption("Designed with ❤️ using [Streamlit](https://streamlit.io) + [Bokeh](https://bokeh.org).")
