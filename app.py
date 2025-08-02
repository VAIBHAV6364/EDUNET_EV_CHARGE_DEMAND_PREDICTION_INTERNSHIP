import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
import plotly.graph_objects as go

# === Set page config ===
st.set_page_config(page_title="EV Forecast", layout="wide")

# === Load Model ===
model = joblib.load('forecasting_ev_model.pkl')

# === Enhanced CSS Styling ===
st.markdown("""
    <style>
        html, body, [class*="css"] {
            font-family: 'Segoe UI', sans-serif;
        }

        .stApp {
            background: linear-gradient(to right, #1f1c2c, #928dab);
            color: #ffffff;
        }

        .main-title {
            font-size: 40px;
            font-weight: bold;
            text-align: center;
            color: #ffffff;
            margin-top: 20px;
        }

        .subtitle {
            text-align: center;
            font-size: 22px;
            font-weight: 600;
            color: #e0e0e0;
            margin-bottom: 30px;
        }

        .section-header {
            font-size: 28px;
            font-weight: 600;
            color: #f1f1f1;
            margin-top: 30px;
        }

        .ev-label {
            font-size: 20px;
            color: #ffffff;
        }

        .success-box {
            background-color: #2e7d32;
            padding: 10px;
            border-radius: 8px;
            font-size: 16px;
            color: white;
        }

        .footer {
            text-align: center;
            color: #aaaaaa;
            font-size: 14px;
            margin-top: 40px;
        }
    </style>
""", unsafe_allow_html=True)

# === Title, Subtitle & Header Buttons ===
st.markdown("""
    <style>
        .main-title {
            font-size: 40px;
            font-weight: bold;
            text-align: center;
            color: #ffffff;
            margin-top: 20px;
        }

        .subtitle {
            text-align: center;
            font-size: 22px;
            font-weight: 600;
            color: #e0e0e0;
            margin-bottom: 10px;
        }

        .header-bar {
            text-align: center;
            margin-top: 10px;
            margin-bottom: 30px;
        }

        .header-bar b {
            font-size: 16px;
            color: #ffffff;
            margin-right: 12px;
        }

        .header-buttons a {
            display: inline-block;
            margin: 5px 10px;
            padding: 8px 18px;
            background-color: #30336b;
            color: white;
            border-radius: 5px;
            text-decoration: none;
            font-weight: 600;
            transition: 0.3s ease;
            font-size: 14px;
        }

        .header-buttons a:hover {
            background-color: #130f40;
        }
    </style>


    <div class="header-bar">
        <b>Designed & Developed by Vaibhav R</b>
        <div class="header-buttons">
            <a href="http://www.linkedin.com/in/vaibhav-r-188b712a2" target="_blank">🔗 LinkedIn</a>
            <a href="https://github.com/VAIBHAV6364" target="_blank">💻 GitHub</a>
            <a href="https://vaibhav6364.github.io/my-first-simple-portfolio-website/index.html" target="_blank">🌐 Portfolio</a>
        </div>
    </div>
            
    <div class="main-title">🔮 EV Adoption Forecast for Washington Counties</div>
    <div class="subtitle">Explore electric vehicle adoption forecasts over the next 3 years.</div>
""", unsafe_allow_html=True)


# === Image Display ===
st.image("ev_representation.jpeg", use_container_width=True)

# === Load Data ===
@st.cache_data
def load_data():
    df = pd.read_csv("preprocessed_ev_data.csv")
    df['Date'] = pd.to_datetime(df['Date'])
    return df

df = load_data()

# === County Selection ===
county_list = sorted(df['County'].dropna().unique().tolist())
st.markdown("<div class='section-header'>🔎 Select a County</div>", unsafe_allow_html=True)
county = st.selectbox("", county_list)

# === Forecast Calculation Logic (unchanged but structured into function) ===
def forecast_county(df, county, model):
    county_df = df[df['County'] == county].sort_values("Date")
    county_code = county_df['county_encoded'].iloc[0]
    historical_ev = list(county_df['Electric Vehicle (EV) Total'].values[-6:])
    cumulative_ev = list(np.cumsum(historical_ev))
    months_since_start = county_df['months_since_start'].max()
    latest_date = county_df['Date'].max()

    future_rows = []
    forecast_horizon = 36

    for i in range(1, forecast_horizon + 1):
        forecast_date = latest_date + pd.DateOffset(months=i)
        months_since_start += 1
        lag1, lag2, lag3 = historical_ev[-1], historical_ev[-2], historical_ev[-3]
        roll_mean = np.mean([lag1, lag2, lag3])
        pct_change_1 = (lag1 - lag2) / lag2 if lag2 != 0 else 0
        pct_change_3 = (lag1 - lag3) / lag3 if lag3 != 0 else 0
        recent_cumulative = cumulative_ev[-6:]
        ev_growth_slope = np.polyfit(range(len(recent_cumulative)), recent_cumulative, 1)[0] if len(recent_cumulative) == 6 else 0

        new_row = {
            'months_since_start': months_since_start,
            'county_encoded': county_code,
            'ev_total_lag1': lag1,
            'ev_total_lag2': lag2,
            'ev_total_lag3': lag3,
            'ev_total_roll_mean_3': roll_mean,
            'ev_total_pct_change_1': pct_change_1,
            'ev_total_pct_change_3': pct_change_3,
            'ev_growth_slope': ev_growth_slope
        }

        pred = model.predict(pd.DataFrame([new_row]))[0]
        future_rows.append({"Date": forecast_date, "Predicted EV Total": round(pred)})

        historical_ev.append(pred)
        if len(historical_ev) > 6:
            historical_ev.pop(0)
        cumulative_ev.append(cumulative_ev[-1] + pred)
        if len(cumulative_ev) > 6:
            cumulative_ev.pop(0)

    return county_df, pd.DataFrame(future_rows)

county_df, forecast_df = forecast_county(df, county, model)

# === Combine & Plot ===
historical_cum = county_df[['Date', 'Electric Vehicle (EV) Total']].copy()
historical_cum['Cumulative EV'] = historical_cum['Electric Vehicle (EV) Total'].cumsum()
forecast_df['Cumulative EV'] = forecast_df['Predicted EV Total'].cumsum() + historical_cum['Cumulative EV'].iloc[-1]

combined = pd.concat([
    historical_cum[['Date', 'Cumulative EV']].assign(Source='Historical'),
    forecast_df[['Date', 'Cumulative EV']].assign(Source='Forecast')
], ignore_index=True)

# === Plot using Plotly ===
fig = go.Figure()
for source, data in combined.groupby("Source"):
    fig.add_trace(go.Scatter(
        x=data['Date'],
        y=data['Cumulative EV'],
        mode='lines+markers',
        name=source
    ))

fig.update_layout(
    title=f"Cumulative EV Forecast for {county} County (3-Year Outlook)",
    title_font=dict(size=20),
    plot_bgcolor='#1f1c2c',
    paper_bgcolor='#1f1c2c',
    font=dict(color='white'),
    xaxis_title="Date",
    yaxis_title="Cumulative EV Count",
    hovermode="x unified"
)

st.plotly_chart(fig, use_container_width=True)

# === Forecast Summary ===
historical_total = historical_cum['Cumulative EV'].iloc[-1]
forecasted_total = forecast_df['Cumulative EV'].iloc[-1]

if historical_total > 0:
    growth = ((forecasted_total - historical_total) / historical_total) * 100
    st.markdown(f"<div class='success-box'>EV adoption in <strong>{county}</strong> is expected to increase by <strong>{growth:.2f}%</strong> over the next 3 years.</div>", unsafe_allow_html=True)
else:
    st.warning("No historical data available to compute percentage change.")


# === Multi-County Comparison Section ===
st.markdown("<div class='section-header'>📍 Compare EV Adoption Trends Across Counties</div>", unsafe_allow_html=True)
multi_counties = st.multiselect("Select up to 3 counties", county_list, max_selections=3)

if multi_counties:
    comparison_data = []

    for cty in multi_counties:
        cty_df, fc_df = forecast_county(df, cty, model)

        hist_cum = cty_df[['Date', 'Electric Vehicle (EV) Total']].copy()
        hist_cum['Cumulative EV'] = hist_cum['Electric Vehicle (EV) Total'].cumsum()
        fc_df['Cumulative EV'] = fc_df['Predicted EV Total'].cumsum() + hist_cum['Cumulative EV'].iloc[-1]

        combined_cty = pd.concat([
            hist_cum[['Date', 'Cumulative EV']].assign(County=cty),
            fc_df[['Date', 'Cumulative EV']].assign(County=cty)
        ], ignore_index=True)

        comparison_data.append(combined_cty)

    all_cty_df = pd.concat(comparison_data, ignore_index=True)

    # === Plot Multi-County ===
    fig_comp = go.Figure()
    for cty, group in all_cty_df.groupby("County"):
        fig_comp.add_trace(go.Scatter(
            x=group['Date'],
            y=group['Cumulative EV'],
            mode='lines+markers',
            name=cty
        ))

    fig_comp.update_layout(
        title="Cumulative EV Trends Across Selected Counties (3-Year Forecast)",
        title_font=dict(size=18),
        plot_bgcolor='#1f1c2c',
        paper_bgcolor='#1f1c2c',
        font=dict(color='white'),
        xaxis_title="Date",
        yaxis_title="Cumulative EV Count",
        hovermode="x unified"
    )

    st.plotly_chart(fig_comp, use_container_width=True)

    # === Display % Growth Summary ===
    summary_lines = []
    for cty in multi_counties:
        data = all_cty_df[all_cty_df['County'] == cty].reset_index(drop=True)
        historical_total = data['Cumulative EV'].iloc[len(data) - 36 - 1]
        forecasted_total = data['Cumulative EV'].iloc[-1]
        if historical_total > 0:
            growth = ((forecasted_total - historical_total) / historical_total) * 100
            summary_lines.append(f"{cty}: {growth:.2f}%")
        else:
            summary_lines.append(f"{cty}: N/A")

    summary_string = " | ".join(summary_lines)
    st.markdown(f"<div class='success-box'>3-Year Growth Forecast — <strong>{summary_string}</strong></div>", unsafe_allow_html=True)


# === Footer ===
st.markdown("<div class='footer'>Completed under Edunet & shell collaborative 4 week internship under <strong>AICTE Internship Cycle 2</strong> by <strong>Vaibhav R</strong> through Skill_4_future</div>", unsafe_allow_html=True)

# === Sticky Footer with Buttons ===
st.markdown("""
    <style>
        .footer-bar {
            background-color: #0e0e0e;
            padding: 20px;
            margin-top: 40px;
            text-align: center;
            border-top: 1px solid #444;
        }
        .footer-bar b {
            font-size: 16px;
            color: #ffffff;
        }
        .footer-buttons a {
            display: inline-block;
            margin: 8px 15px;
            padding: 10px 20px;
            background-color: #30336b;
            color: white;
            border-radius: 6px;
            text-decoration: none;
            font-weight: 600;
            transition: 0.3s ease;
        }
        .footer-buttons a:hover {
            background-color: #130f40;
        }
    </style>

    <div class="footer-bar">
        <b>Designed & Developed by Vaibhav R</b>
    </div>
""", unsafe_allow_html=True)
