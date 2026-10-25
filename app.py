
import streamlit as st
import joblib
import pandas as pd
import plotly.express as px

from pathlib import Path

ROOT = Path(__file__).resolve().parent

st.set_page_config(
    page_title="ChurnIQ | Customer Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

MODEL_PATH = ROOT / "models" / "logistic_regression_pipeline.joblib"
DATA_PATH = ROOT / "data" / "telco_churn_cleaned.csv"


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_data():
    data = pd.read_csv(DATA_PATH)
    data["TotalCharges"] = pd.to_numeric(
        data["TotalCharges"], errors="coerce"
    )
    return data


model = load_model()
df = load_data()

PRIMARY = "#4169F5"
PURPLE = "#7964E8"
GREEN = "#10B981"
RED = "#EF6475"


st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background: #F4F7FC;
    color: #17233C;
}

[data-testid="stHeader"] {
    background: rgba(244,247,252,0.96);
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #10192C, #192945);
}

[data-testid="stSidebar"] * {
    color: #E7EEFF;
}

[data-testid="stSidebar"] [data-testid="stRadio"] label {
    padding: 8px;
    border-radius: 8px;
}

.block-container {
    max-width: 1550px;
    padding-top: 1.7rem;
    padding-bottom: 3rem;
}

h1, h2, h3 {
    color: #17233C;
    letter-spacing: -0.5px;
}

h1 {
    font-weight: 800;
}

[data-testid="stMetric"] {
    background: white;
    padding: 20px;
    border: 1px solid #E6EBF4;
    border-radius: 15px;
    box-shadow: 0 5px 18px rgba(24,44,80,0.04);
}

[data-testid="stMetricLabel"] {
    color: #64748B;
    font-weight: 600;
}

[data-testid="stMetricValue"] {
    color: #17233C;
    font-weight: 800;
}

[data-testid="stVerticalBlockBorderWrapper"] {
    background: white;
    border: 1px solid #E6EBF4;
    border-radius: 15px;
    box-shadow: 0 5px 20px rgba(24,44,80,0.035);
}

.stButton > button,
.stFormSubmitButton > button {
    min-height: 43px;
    border-radius: 9px;
    border: none;
    background: linear-gradient(100deg, #4169F5, #7959E9);
    color: white;
    font-weight: 700;
}

.stButton > button:hover,
.stFormSubmitButton > button:hover {
    color: white;
    border: none;
    box-shadow: 0 5px 16px rgba(65,105,245,0.2);
}

hr {
    border-color: #E6EBF4;
}

.hero {
    background: linear-gradient(115deg, #182C53, #345BD4 60%, #7957E9);
    border-radius: 17px;
    padding: 27px 30px;
    color: white;
    margin: 20px 0 25px 0;
    box-shadow: 0 12px 28px rgba(57,84,180,0.14);
}

.hero h2 {
    color: white;
    font-size: 26px;
    margin: 0 0 8px 0;
}

.hero p {
    color: #E2E9FF;
    margin: 0;
    line-height: 1.7;
}

.eyebrow {
    color: #71809A;
    font-size: 11px;
    font-weight: 800;
    letter-spacing: 1.5px;
    text-transform: uppercase;
}

.small-muted {
    color: #64748B;
    font-size: 13px;
}

.result-card {
    background: linear-gradient(135deg, #182C53, #345BD4);
    padding: 24px;
    border-radius: 15px;
    color: white;
}

.result-card h2 {
    color: white;
    margin-bottom: 7px;
}

.result-card p {
    color: #E0E8FF;
}

[data-testid="stForm"] {
    border: none;
}

</style>
""", unsafe_allow_html=True)


def style_chart(fig):
    fig.update_layout(
        template="plotly_white",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter", color="#64748B", size=12),
        margin=dict(l=10, r=15, t=25, b=10)
    )
    fig.update_xaxes(showgrid=False, zeroline=False)
    fig.update_yaxes(gridcolor="#EDF1F7", zeroline=False)
    return fig


def section_heading(title, subtitle):
    st.markdown(f"### {title}")
    st.markdown(
        f'<p class="small-muted">{subtitle}</p>',
        unsafe_allow_html=True
    )


def segment_summary(data, column):
    summary = (
        data.groupby(column, observed=False)
        .agg(
            Customers=("Churn", "size"),
            Churned=("Churn", lambda values: (values == "Yes").sum())
        )
        .reset_index()
    )

    summary["Churn Rate (%)"] = (
        summary["Churned"] / summary["Customers"] * 100
    ).round(2)

    return summary


st.sidebar.markdown("""
<div style="padding: 10px 4px 24px 4px;">
    <div style="font-size:27px;">▥ <b>ChurnIQ</b></div>
    <div style="font-size:11px;letter-spacing:1.4px;margin-top:6px;">
        CUSTOMER INTELLIGENCE
    </div>
</div>
""", unsafe_allow_html=True)

page = st.sidebar.radio(
    "WORKSPACE",
    ["Overview", "Predict Churn", "Customer Insights"]
)

total_customers = len(df)
churned = int((df["Churn"] == "Yes").sum())
retained = int((df["Churn"] == "No").sum())
churn_rate = churned / total_customers * 100


if page == "Overview":

    st.markdown(
        '<div class="eyebrow">ANALYTICS WORKSPACE</div>',
        unsafe_allow_html=True
    )
    st.title("Customer Overview")
    st.markdown(
        '<p class="small-muted">Understand customer health and explore '
        'historical churn patterns.</p>',
        unsafe_allow_html=True
    )

    st.markdown("""
    <div class="hero">
        <h2>Make every customer insight count.</h2>
        <p>Monitor churn patterns, explore customer segments and use
        predictive analytics to support retention decisions.</p>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Total Customers", f"{total_customers:,}")
    c2.metric("Churned Customers", f"{churned:,}")
    c3.metric("Retained Customers", f"{retained:,}")
    c4.metric("Overall Churn Rate", f"{churn_rate:.2f}%")

    left, right = st.columns(2)

    with left:
        with st.container(border=True):
            section_heading(
                "Customer Status",
                "Distribution in the historical dataset"
            )

            status = pd.DataFrame({
                "Status": ["Retained", "Churned"],
                "Customers": [retained, churned]
            })

            fig = px.pie(
                status,
                names="Status",
                values="Customers",
                hole=0.68,
                color="Status",
                color_discrete_map={
                    "Retained": GREEN,
                    "Churned": RED
                }
            )

            fig.update_traces(
                textposition="outside",
                textinfo="label+percent",
                marker=dict(line=dict(color="white", width=3))
            )

            fig.update_layout(
                showlegend=False,
                annotations=[dict(
                    text=f"<b>{total_customers:,}</b><br>Customers",
                    x=0.5,
                    y=0.5,
                    showarrow=False,
                    font=dict(size=17, color="#17233C")
                )]
            )

            st.plotly_chart(
                style_chart(fig), use_container_width=True
            )

    with right:
        with st.container(border=True):
            section_heading(
                "Churn by Contract",
                "Observed churn rate for each contract type"
            )

            contract_data = segment_summary(df, "Contract")

            fig = px.bar(
                contract_data,
                x="Contract",
                y="Churn Rate (%)",
                color="Contract",
                text=contract_data["Churn Rate (%)"].map(
                    lambda value: f"{value:.1f}%"
                ),
                color_discrete_sequence=[PRIMARY, PURPLE, GREEN]
            )

            fig.update_traces(
                textposition="outside",
                cliponaxis=False,
                showlegend=False
            )
            fig.update_layout(
                xaxis_title="",
                yaxis_title="Churn rate (%)",
                showlegend=False
            )

            st.plotly_chart(
                style_chart(fig), use_container_width=True
            )

    left, right = st.columns(2)

    with left:
        with st.container(border=True):
            section_heading(
                "Churn by Tenure",
                "Observed churn rate across customer tenure groups"
            )

            tenure_data = df.copy()
            tenure_data["Tenure Group"] = pd.cut(
                tenure_data["tenure"],
                bins=[-1, 12, 24, 48, 72],
                labels=[
                    "0–12 months",
                    "13–24 months",
                    "25–48 months",
                    "49–72 months"
                ]
            )

            tenure_summary = segment_summary(
                tenure_data, "Tenure Group"
            )

            fig = px.bar(
                tenure_summary,
                x="Tenure Group",
                y="Churn Rate (%)",
                color="Churn Rate (%)",
                text=tenure_summary["Churn Rate (%)"].map(
                    lambda value: f"{value:.1f}%"
                ),
                color_continuous_scale=["#B6CCFF", PRIMARY]
            )

            fig.update_traces(textposition="outside", cliponaxis=False)
            fig.update_layout(
                coloraxis_showscale=False,
                xaxis_title="Tenure",
                yaxis_title="Churn rate (%)"
            )

            st.plotly_chart(
                style_chart(fig), use_container_width=True
            )

    with right:
        with st.container(border=True):
            section_heading(
                "Churn by Payment Method",
                "Compare churn rates by billing preference"
            )

            payment_summary = segment_summary(df, "PaymentMethod")
            payment_summary = payment_summary.sort_values("Churn Rate (%)")

            fig = px.bar(
                payment_summary,
                x="Churn Rate (%)",
                y="PaymentMethod",
                orientation="h",
                text=payment_summary["Churn Rate (%)"].map(
                    lambda value: f"{value:.1f}%"
                ),
                color="Churn Rate (%)",
                color_continuous_scale=["#B6CCFF", PURPLE]
            )

            fig.update_traces(textposition="outside", cliponaxis=False)
            fig.update_layout(
                coloraxis_showscale=False,
                xaxis_title="Churn rate (%)",
                yaxis_title=""
            )

            st.plotly_chart(
                style_chart(fig), use_container_width=True
            )

    st.info(
        "These charts describe historical associations in the dataset. "
        "They do not prove that a particular service or contract causes churn."
    )


elif page == "Predict Churn":

    st.markdown(
        '<div class="eyebrow">PREDICTIVE ANALYTICS</div>',
        unsafe_allow_html=True
    )
    st.title("Customer Risk Assessment")
    st.markdown(
        '<p class="small-muted">Estimate churn probability using customer '
        'profile, service and billing information.</p>',
        unsafe_allow_html=True
    )

    st.markdown("""
    <div class="hero">
        <h2>Predict risk. Prioritize retention.</h2>
        <p>Complete the customer profile to generate an estimate from
        the trained machine learning pipeline.</p>
    </div>
    """, unsafe_allow_html=True)

    with st.form("customer_prediction_form"):

        st.subheader("Customer Profile")
        a, b, c = st.columns(3)

        with a:
            gender = st.selectbox("Gender", ["Female", "Male"])
            senior = st.selectbox(
                "Senior citizen",
                [0, 1],
                format_func=lambda value: "Yes" if value else "No"
            )
            partner = st.selectbox("Has partner?", ["Yes", "No"])
            dependents = st.selectbox("Has dependents?", ["No", "Yes"])
            tenure = st.slider("Tenure (months)", 0, 72, 12)

        
        with b:
            phone = st.selectbox("Phone service", ["Yes", "No"])

            if phone == "Yes":
                multiple_lines = st.selectbox(
                    "Multiple lines", ["No", "Yes"]
                )
            else:
                multiple_lines = "No phone service"
                st.selectbox(
                    "Multiple lines",
                    ["No phone service"],
                    disabled=True
                )

            internet = st.selectbox(
                "Internet service", ["Fiber optic", "DSL", "No"]
            )

            if internet == "No":
                online_security = "No internet service"
                online_backup = "No internet service"
                device_protection = "No internet service"

                st.selectbox(
                    "Online security",
                    ["No internet service"],
                    disabled=True
                )
                st.selectbox(
                    "Online backup",
                    ["No internet service"],
                    disabled=True
                )
                st.selectbox(
                    "Device protection",
                    ["No internet service"],
                    disabled=True
                )
            else:
                online_security = st.selectbox(
                    "Online security", ["No", "Yes"]
                )
                online_backup = st.selectbox(
                    "Online backup", ["No", "Yes"]
                )
                device_protection = st.selectbox(
                    "Device protection", ["No", "Yes"]
                )

        with c:
            if internet == "No":
                tech_support = "No internet service"
                streaming_tv = "No internet service"
                streaming_movies = "No internet service"

                st.selectbox(
                    "Tech support",
                    ["No internet service"],
                    disabled=True
                )
                st.selectbox(
                    "Streaming TV",
                    ["No internet service"],
                    disabled=True
                )
                st.selectbox(
                    "Streaming movies",
                    ["No internet service"],
                    disabled=True
                )
            else:
                tech_support = st.selectbox(
                    "Tech support", ["No", "Yes"]
                )
                streaming_tv = st.selectbox(
                    "Streaming TV", ["No", "Yes"]
                )
                streaming_movies = st.selectbox(
                    "Streaming movies", ["No", "Yes"]
                )

            contract = st.selectbox(
                "Contract",
                ["Month-to-month", "One year", "Two year"]
            )
            paperless = st.selectbox(
                "Paperless billing", ["Yes", "No"]
            )
            payment = st.selectbox(
                "Payment method",
                [
                    "Electronic check",
                    "Mailed check",
                    "Bank transfer (automatic)",
                    "Credit card (automatic)"
                ]
            )


        st.subheader("Billing Information")
        charge1, charge2 = st.columns(2)

        with charge1:
            monthly_charges = st.number_input(
                "Monthly charges ($)",
                min_value=0.0,
                max_value=200.0,
                value=69.5,
                step=1.0
            )

        with charge2:
            total_charges = st.number_input(
                "Total charges ($)",
                min_value=0.0,
                max_value=100000.0,
                value=69.5,
                step=10.0
            )

        submitted = st.form_submit_button(
            "Calculate Churn Risk",
            use_container_width=True
        )

    if submitted:

        customer = pd.DataFrame([{
            "gender": gender,
            "SeniorCitizen": senior,
            "Partner": partner,
            "Dependents": dependents,
            "tenure": tenure,
            "PhoneService": phone,
            "MultipleLines": multiple_lines,
            "InternetService": internet,
            "OnlineSecurity": online_security,
            "OnlineBackup": online_backup,
            "DeviceProtection": device_protection,
            "TechSupport": tech_support,
            "StreamingTV": streaming_tv,
            "StreamingMovies": streaming_movies,
            "Contract": contract,
            "PaperlessBilling": paperless,
            "PaymentMethod": payment,
            "MonthlyCharges": monthly_charges,
            "TotalCharges": total_charges
        }])

        probability = float(model.predict_proba(customer)[0][1])
        threshold = 0.30
        prediction = int(probability >= threshold)
        percentage = probability * 100

        st.divider()
        st.subheader("Prediction Result")
        st.caption("Classification threshold: 30% | Churn probability ≥ 30% is classified as Churn.")

        result_left, result_right = st.columns([1.2, 1])

        with result_left:
            label = (
                "Churn Risk Detected"
                if prediction == 1
                else "Lower Churn Risk Predicted"
            )

            st.markdown(
                f'<div class="result-card">'
                f'<p>MODEL RESULT</p>'
                f'<h2>{label}</h2>'
                f'<p>Estimated churn probability: '
                f'<b>{percentage:.2f}%</b></p>'
                f'</div>',
                unsafe_allow_html=True
            )
            st.progress(probability)

        with result_right:
            st.metric("Estimated Churn Probability", f"{percentage:.2f}%")
            st.metric(
                "Classification",
                "Churn" if prediction == 1 else "No Churn"
            )

        st.subheader("Suggested Next Step")

        if percentage >= 70:
            st.error(
                "High estimated risk: review the customer's service "
                "experience, billing concerns and contract options."
            )
        elif percentage >= 40:
            st.warning(
                "Moderate estimated risk: consider checking customer "
                "engagement, support history and billing satisfaction."
            )
        else:
            st.success(
                "Lower estimated risk: continue normal engagement and "
                "monitor future customer behavior."
            )

        st.caption(
            "The probability is a model estimate, not a guarantee. "
            "These priority bands are illustrative and have not been "
            "validated as optimal business thresholds."
        )

        result = customer.copy()
        result["Prediction"] = (
            "Churn" if prediction == 1 else "No Churn"
        )
        result["Churn Probability (%)"] = round(percentage, 2)

        st.download_button(
            "Download Prediction Record",
            data=result.to_csv(index=False).encode("utf-8"),
            file_name="customer_churn_prediction.csv",
            mime="text/csv"
        )


elif page == "Customer Insights":

    st.markdown(
        '<div class="eyebrow">SEGMENT ANALYTICS</div>',
        unsafe_allow_html=True
    )
    st.title("Customer Insights")
    st.markdown(
        '<p class="small-muted">Filter customer segments and compare '
        'their observed churn patterns.</p>',
        unsafe_allow_html=True
    )

    filter1, filter2 = st.columns(2)

    with filter1:
        contracts = st.multiselect(
            "Contract type",
            options=sorted(df["Contract"].dropna().unique()),
            default=sorted(df["Contract"].dropna().unique())
        )

    with filter2:
        internet_types = st.multiselect(
            "Internet service",
            options=sorted(df["InternetService"].dropna().unique()),
            default=sorted(df["InternetService"].dropna().unique())
        )

    filtered = df[
        df["Contract"].isin(contracts)
        & df["InternetService"].isin(internet_types)
    ]

    if filtered.empty:
        st.warning("No customers match the selected filters.")

    else:
        f1, f2, f3 = st.columns(3)

        f1.metric("Customers in Segment", f"{len(filtered):,}")
        f2.metric(
            "Churned Customers",
            f"{int((filtered['Churn'] == 'Yes').sum()):,}"
        )
        f3.metric(
            "Segment Churn Rate",
            f"{(filtered['Churn'] == 'Yes').mean() * 100:.2f}%"
        )

        segment_data = (
            filtered.groupby(
                ["Contract", "InternetService"],
                observed=False
            )
            .agg(
                Customers=("Churn", "size"),
                Churned=("Churn", lambda values: (values == "Yes").sum())
            )
            .reset_index()
        )

        segment_data["Churn Rate (%)"] = (
            segment_data["Churned"]
            / segment_data["Customers"] * 100
        ).round(2)

        with st.container(border=True):
            section_heading(
                "Contract and Internet Segments",
                "Compare churn rates across selected customer groups"
            )

            fig = px.bar(
                segment_data,
                x="Contract",
                y="Churn Rate (%)",
                color="InternetService",
                barmode="group",
                hover_data=["Customers", "Churned"],
                color_discrete_sequence=[PRIMARY, PURPLE, GREEN]
            )

            st.plotly_chart(
                style_chart(fig), use_container_width=True
            )

        with st.container(border=True):
            section_heading(
                "Segment Detail",
                "Customer counts and observed churn rate"
            )

            st.dataframe(
                segment_data.sort_values(
                    "Churn Rate (%)", ascending=False
                ),
                use_container_width=True,
                hide_index=True
            )

        st.download_button(
            "Download Segment Report",
            data=segment_data.to_csv(index=False).encode("utf-8"),
            file_name="customer_segment_report.csv",
            mime="text/csv"
        )

