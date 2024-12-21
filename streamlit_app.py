import streamlit as st
import numpy as np
import pandas as pd
import requests
from io import BytesIO
from PIL import Image

# We will use Bokeh 2.4.3 (pinned in requirements.txt).
from bokeh.plotting import figure
from bokeh.models import Arrow, VeeHead, ColumnDataSource, Legend
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
    Calculates power flow on a line pi-section based on user inputs.

    Returns:
      S_s, S_i, S_o, S_loss, S_r, Q_s, Q_r  (complex except Q_s, Q_r are floats).
    """
    frequency = 50  # Hz
    delta_rad = np.deg2rad(delta_deg)

    # Impedance for the entire line
    Z = (R + 1j * X) * length

    # Actual kV from per-unit
    V_s_actual = Vs * base_voltage
    V_r_actual = Vr * base_voltage

    # Receiving voltage as a complex with angle delta (V_r e^{-j delta})
    V_r_complex = V_r_actual * np.exp(-1j * delta_rad)

    # Line current
    I_line = (V_s_actual - V_r_complex) / Z

    # Power after shunt at sending side (S_i)
    S_i = V_s_actual * np.conjugate(I_line)

    # Reactive power at sending end (capacitance)
    Q_s = 2 * np.pi * frequency * C * length * (V_s_actual ** 2)
    S_s = S_i - 1j * Q_s

    # Power loss
    S_loss = I_line * np.conjugate(I_line) * Z

    # Power before shunt at receiving side
    S_o = S_i - S_loss

    # Reactive power at receiving end
    Q_r = 2 * np.pi * frequency * C * length * (V_r_actual ** 2)
    S_r = S_o + 1j * Q_r

    return S_s, S_i, S_o, S_loss, S_r, Q_s, Q_r

# ---------------------------
# Helper: Create Vector Figure
# ---------------------------
def create_vector_figure(S_s, S_i, S_o, S_r, S_loss):
    """
    Create a Bokeh figure showing all power-flow vectors as arrows.
    We'll store everything as single-element arrays so Bokeh doesn't complain.
    """

    # Single-element arrays for each arrow's start/end
    source = ColumnDataSource(data=dict(
        # S_s: from (0,0) to (S_s.real, S_s.imag)
        S_s_x_start=[0],         S_s_y_start=[0],
        S_s_x_end=[S_s.real],    S_s_y_end=[S_s.imag],

        # S_i: from (0,0) to (S_i.real, S_i.imag)
        S_i_x_start=[0],         S_i_y_start=[0],
        S_i_x_end=[S_i.real],    S_i_y_end=[S_i.imag],

        # S_o: from (0,0) to (S_o.real, S_o.imag)
        S_o_x_start=[0],         S_o_y_start=[0],
        S_o_x_end=[S_o.real],    S_o_y_end=[S_o.imag],

        # S_r: from (0,0) to (S_r.real, S_r.imag)
        S_r_x_start=[0],         S_r_y_start=[0],
        S_r_x_end=[S_r.real],    S_r_y_end=[S_r.imag],

        # Extra arrow from S_s->S_i (like the original code)
        S_si_x_start=[S_s.real], S_si_y_start=[S_s.imag],
        S_si_x_end=[S_i.real],   S_si_y_end=[S_i.imag],

        # Extra arrow from S_o->S_r
        S_or_x_start=[S_o.real], S_or_y_start=[S_o.imag],
        S_or_x_end=[S_r.real],   S_or_y_end=[S_r.imag],

        # S_loss: from S_o to S_o + S_loss
        S_loss_x_start=[S_o.real],
        S_loss_y_start=[S_o.imag],
        S_loss_x_end=[S_o.real + S_loss.real],
        S_loss_y_end=[S_o.imag + S_loss.imag],
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

    # Main arrows from origin
    p.add_layout(Arrow(end=arrow_head,
                       x_start='S_s_x_start', y_start='S_s_y_start',
                       x_end='S_s_x_end', y_end='S_s_y_end',
                       source=source, line_width=3, line_color="red"))
    p.add_layout(Arrow(end=arrow_head,
                       x_start='S_i_x_start', y_start='S_i_y_start',
                       x_end='S_i_x_end', y_end='S_i_y_end',
                       source=source, line_width=3, line_color="green"))
    p.add_layout(Arrow(end=arrow_head,
                       x_start='S_o_x_start', y_start='S_o_y_start',
                       x_end='S_o_x_end', y_end='S_o_y_end',
                       source=source, line_width=3, line_color="blue"))
    p.add_layout(Arrow(end=arrow_head,
                       x_start='S_r_x_start', y_start='S_r_y_start',
                       x_end='S_r_x_end', y_end='S_r_y_end',
                       source=source, line_width=3, line_color="purple"))

    # Extra arrows for partial vectors
    p.add_layout(Arrow(end=arrow_head,
                       x_start='S_si_x_start', y_start='S_si_y_start',
                       x_end='S_si_x_end', y_end='S_si_y_end',
                       source=source, line_width=3, line_color="orange"))
    p.add_layout(Arrow(end=arrow_head,
                       x_start='S_or_x_start', y_start='S_or_y_start',
                       x_end='S_or_x_end', y_end='S_or_y_end',
                       source=source, line_width=3, line_color="cyan"))

    # Arrow for power loss
    p.add_layout(Arrow(end=arrow_head,
                       x_start='S_loss_x_start', y_start='S_loss_y_start',
                       x_end='S_loss_x_end', y_end='S_loss_y_end',
                       source=source, line_width=3, line_color="magenta"))

    # Add hover
    p.add_tools(HoverTool())

    return p

# ---------------------------
# Helper: Create Bar Chart
# ---------------------------
def create_bar_chart(S_s, S_i, S_o, S_r, S_loss):
    """
    Bokeh bar chart for Active (MW) and Reactive (MVAr) components.
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

    # We'll do two vbars, slightly offset
    x_locations = list(range(len(bar_labels)))
    width = 0.3

    # Active bars (blue)
    p_bar.vbar(
        x=[x - 0.15 for x in x_locations],
        top='active',
        width=width,
        source=source_bar,
        color="#4a69bd",
        legend_label="Active Power (MW)"
    )

    # Reactive bars (red)
    p_bar.vbar(
        x=[x + 0.15 for x in x_locations],
        top='reactive',
        width=width,
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
st.set_page_config(layout="wide")

st.title("Power Flow Simulator for Transmission Lines")

# Row 1: Image + Table side-by-side
col_image, col_table = st.columns([1, 2])

with col_image:
    st.subheader("Power flows on a line PI section")

    img_url = "https://i.postimg.cc/FKStDhY9/Frame-2-1.png"
    try:
        resp = requests.get(img_url, timeout=5)
        pi_image = Image.open(BytesIO(resp.content))
        st.image(pi_image, width=350)
    except Exception:
        st.write("Image not available.")

with col_table:
    st.subheader("Example cases of different lines")

    df_table = pd.DataFrame({
        "Voltage_Level": [c["Voltage_Level"] for c in case_parameters],
        "Type": [c["Type"] for c in case_parameters],
        "R (Ω/km)": [c["R"] for c in case_parameters],
        "X (Ω/km)": [c["X"] for c in case_parameters],
        "C (nF/km)": [f"{c['C']*1e9:.0f}" for c in case_parameters],
        "Length (km)": [c["length"] for c in case_parameters],
    })
    st.dataframe(df_table, use_container_width=True)

# Row 2: Case selection
st.markdown("---")
col_case, _ = st.columns([1, 4])
with col_case:
    st.subheader("Select a Case")
    case_names = [case["name"] for case in case_parameters]
    selected_case = st.selectbox("Pick a line/cable case:", options=case_names)

    # Find the matching case
    default_case = next(case for case in case_parameters if case["name"] == selected_case)

# Row 3: Sliders / Inputs
st.markdown("---")
col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    Vs = st.slider("Vs (pu)", 0.8, 1.2, 1.0, 0.01)
    Vr = st.slider("Vr (pu)", 0.8, 1.2, 0.95, 0.01)

with col2:
    R = st.number_input("R (Ω/km)", min_value=0.0001, max_value=2.0,
                        value=float(default_case["R"]), step=0.0001)
    X = st.number_input("X (Ω/km)", min_value=0.0001, max_value=2.0,
                        value=float(default_case["X"]), step=0.0001)

with col3:
    # Convert default C to nF for the display
    default_C_nF = default_case["C"] * 1e9
    C_nF = st.number_input("C (nF/km)", min_value=0.1, max_value=1000.0,
                           value=float(default_C_nF), step=1.0)
    C = C_nF * 1e-9

with col4:
    delta_deg = st.slider("Delta (°)", -60, 60, 10, 1)

with col5:
    length = st.number_input("Line Length (km)", min_value=1.0, max_value=1000.0,
                             value=float(default_case["length"]), step=1.0)
    base_voltage = st.number_input("Base Voltage (kV)", min_value=1.0, max_value=1000.0,
                                   value=float(default_case["base_voltage"]), step=1.0)

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
    st.write("**Sending End Power (S_s):**")
    st.write(f"{S_s.real:.2f} MW + j {S_s.imag:.2f} MVAr")

    st.write("**After Shunt (S_i):**")
    st.write(f"{S_i.real:.2f} MW + j {S_i.imag:.2f} MVAr")

with col_res2:
    st.write("**Before Shunt (S_o):**")
    st.write(f"{S_o.real:.2f} MW + j {S_o.imag:.2f} MVAr")

    st.write("**Receiving End Power (S_r):**")
    st.write(f"{S_r.real:.2f} MW + j {S_r.imag:.2f} MVAr")

with col_res3:
    st.write("**Power Loss (S_loss):**")
    st.write(f"{S_loss.real:.2f} MW + j {S_loss.imag:.2f} MVAr")

    st.write(f"**Q_s (MVAr at Sending):** {Q_s:.2f}")
    st.write(f"**Q_r (MVAr at Receiving):** {Q_r:.2f}")

st.markdown("---")
st.caption("Requires Bokeh 2.4.3. Built with ❤️ using [Streamlit](https://streamlit.io) + Bokeh.")
