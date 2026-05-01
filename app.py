import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="Churn Intelligence", page_icon="📡", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
html, body, [data-testid="stAppViewContainer"], .stApp {
    background: #020c18 !important;
    font-family: 'Inter', sans-serif !important;
    color: #c9d8e8 !important;
}
.main .block-container { padding: 0 2.5rem 4rem !important; max-width: 1400px !important; }
#MainMenu, footer, header,
[data-testid="stToolbar"],
[data-testid="stDecoration"],
[data-testid="stStatusWidget"],
[data-testid="stSidebarNav"],
.stDeployButton,
[data-testid="collapsedControl"] { display: none !important; }
[data-testid="stSidebar"] { display: none !important; }
.hero-banner {
    background: linear-gradient(135deg, #020c18 0%, #041828 40%, #062840 70%, #041828 100%);
    border-bottom: 1px solid #0a3050;
    padding: 36px 0 32px;
    margin-bottom: 32px;
    position: relative;
    overflow: hidden;
}
.hero-banner::before {
    content: '';
    position: absolute;
    top: -60px; left: -60px;
    width: 300px; height: 300px;
    background: radial-gradient(circle, #00c49a18 0%, transparent 70%);
    pointer-events: none;
}
.hero-banner::after {
    content: '';
    position: absolute;
    bottom: -80px; right: -40px;
    width: 350px; height: 350px;
    background: radial-gradient(circle, #0a9fd418 0%, transparent 70%);
    pointer-events: none;
}
.glass-card {
    background: linear-gradient(135deg, rgba(13,31,53,0.85) 0%, rgba(10,24,40,0.90) 100%);
    backdrop-filter: blur(16px);
    border: 1px solid rgba(14,85,136,0.35);
    border-radius: 16px;
    padding: 22px 20px;
    position: relative;
    overflow: hidden;
}
.glass-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, #00c49a, #0a9fd4);
    border-radius: 16px 16px 0 0;
}
.glass-card .card-icon { font-size: 28px; margin-bottom: 10px; }
.glass-card .card-value { font-size: 32px; font-weight: 800; color: #00c49a; line-height: 1; margin-bottom: 4px; }
.glass-card .card-label { font-size: 11px; color: #3a7a9a; font-weight: 600; text-transform: uppercase; letter-spacing: 1px; }
.glass-card .card-sub { font-size: 12px; color: #2a6a5a; margin-top: 4px; }
.section-title { font-size: 20px; font-weight: 700; color: #e0f0ff; margin: 32px 0 4px; }
.section-sub { font-size: 13px; color: #3a7a9a; margin-bottom: 20px; }
.stTabs [data-baseweb="tab-list"] {
    background: rgba(10,22,40,0.9) !important;
    border: 1px solid #0a3050 !important;
    border-radius: 14px !important;
    padding: 5px !important;
    gap: 4px !important;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 10px !important;
    color: #3a7a9a !important;
    font-weight: 600 !important;
    font-size: 14px !important;
    padding: 10px 24px !important;
    background: transparent !important;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #00b890, #0a9fd4) !important;
    color: #ffffff !important;
    box-shadow: 0 4px 15px rgba(0,196,154,0.3) !important;
}
[data-testid="stSelectbox"] > div > div,
[data-baseweb="select"] > div {
    background: rgba(13,31,53,0.9) !important;
    border: 1px solid #0a3050 !important;
    border-radius: 10px !important;
    color: #c9d8e8 !important;
}
[data-baseweb="popover"], [data-baseweb="menu"] {
    background: #0a1828 !important;
    border: 1px solid #0a3050 !important;
    border-radius: 10px !important;
}
[data-baseweb="option"] { background: #0a1828 !important; color: #c9d8e8 !important; }
[data-baseweb="option"]:hover { background: #0a2a45 !important; }
[data-testid="stSlider"] > div > div > div { background: #0a3050 !important; }
[data-testid="stRadio"] {
    background: rgba(13,31,53,0.8) !important;
    border-radius: 12px !important;
    padding: 12px !important;
    border: 1px solid #0a3050 !important;
}
label, [data-testid="stWidgetLabel"] p {
    color: #3a7a9a !important;
    font-size: 11px !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.8px !important;
}
.stButton > button, [data-testid="stFormSubmitButton"] > button {
    background: linear-gradient(135deg, #00c49a 0%, #0a9fd4 100%) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 14px 28px !important;
    font-weight: 700 !important;
    font-size: 15px !important;
    width: 100% !important;
    box-shadow: 0 6px 24px rgba(0,196,154,0.3) !important;
}
[data-testid="stForm"] {
    background: rgba(10,22,40,0.85) !important;
    border: 1px solid #0a3050 !important;
    border-radius: 18px !important;
    padding: 28px !important;
}
[data-testid="stProgressBar"] > div { background: #0a1828 !important; border-radius: 10px !important; }
[data-testid="stProgressBar"] > div > div {
    background: linear-gradient(90deg, #00c49a, #0a9fd4) !important;
    border-radius: 10px !important;
}
[data-testid="stMetric"] {
    background: rgba(13,31,53,0.85) !important;
    border: 1px solid #0a3050 !important;
    border-radius: 14px !important;
    padding: 16px 20px !important;
}
[data-testid="stMetricLabel"] { color: #3a7a9a !important; font-size: 11px !important; }
[data-testid="stMetricValue"] { color: #00c49a !important; font-size: 28px !important; font-weight: 800 !important; }
[data-testid="stMetricDelta"] { color: #0a9fd4 !important; }
hr { border-color: #0a2a40 !important; margin: 1.5rem 0 !important; }
::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: #020c18; }
::-webkit-scrollbar-thumb { background: #0a3050; border-radius: 5px; }
</style>
""", unsafe_allow_html=True)

GREEN = "#00c49a"
BLUE  = "#0a9fd4"
RED   = "#e05570"
MUTED = "#3a7a9a"
LEGEND = dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#8aacca"))

def base_layout(title="", height=320, **extra):
    d = dict(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter", color="#8aacca"),
        margin=dict(l=20, r=20, t=44, b=20),
        xaxis=dict(gridcolor="#0a2a40", showgrid=True, zeroline=False,
                   tickfont=dict(color="#5a8aaa", size=11)),
        yaxis=dict(gridcolor="#0a2a40", showgrid=True, zeroline=False,
                   tickfont=dict(color="#5a8aaa", size=11)),
        hoverlabel=dict(bgcolor="#0d1f35", bordercolor=BLUE,
                        font=dict(color="#e0f0ff", size=13)),
        height=height,
    )
    if title:
        d["title"] = title
        d["title_font"] = dict(color="#e0f0ff", size=14, family="Inter")
    d.update(extra)
    return d

@st.cache_data
def load_and_train():
    df = pd.read_csv("DATA/WA_Fn-UseC_-Telco-Customer-Churn.csv")
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df.dropna(inplace=True)
    df.drop(columns=["customerID"], inplace=True)
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})
    raw = df.copy()
    le = LabelEncoder()
    for col in df.select_dtypes(include="object").columns:
        df[col] = le.fit_transform(df[col])
    X = df.drop("Churn", axis=1)
    y = df["Churn"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    sc = StandardScaler()
    X_tr = sc.fit_transform(X_train)
    X_te = sc.transform(X_test)
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_tr, y_train)
    lr = LogisticRegression(max_iter=1000)
    lr.fit(X_tr, y_train)
    return raw, df, X, y, X_tr, X_te, y_train, y_test, sc, rf, lr, \
           accuracy_score(y_test, rf.predict(X_te)), accuracy_score(y_test, lr.predict(X_te))

with st.spinner("Initializing intelligence engine..."):
    raw, df, X, y, X_tr, X_te, y_tr, y_te, scaler, rf, lr, rf_acc, lr_acc = load_and_train()

# ── Hero ──────────────────────────────────────────────
st.markdown("""
<div class="hero-banner">
  <div style="position:relative;z-index:2;">
    <div style="font-size:11px;color:#1a6a5a;font-weight:700;letter-spacing:2.5px;text-transform:uppercase;margin-bottom:8px;">
      ● &nbsp;LIVE · ML ANALYTICS PLATFORM
    </div>
    <div style="font-size:38px;font-weight:900;color:#e8f4ff;line-height:1.15;letter-spacing:-0.5px;">
      Churn Intelligence
    </div>
    <div style="font-size:15px;color:#3a7a9a;margin-top:10px;">
      Real-time customer retention analytics powered by machine learning.
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

total      = len(df)
churned    = int(df["Churn"].sum())
retained   = total - churned
churn_rate = churned / total * 100

k1,k2,k3,k4,k5 = st.columns(5)
for col,icon,val,label,sub in zip(
    [k1,k2,k3,k4,k5],
    ["👥","⚠️","✅","🌲","📉"],
    [f"{total:,}",f"{churned:,}",f"{retained:,}",f"{rf_acc*100:.1f}%",f"{lr_acc*100:.1f}%"],
    ["Total Customers","Churned","Retained","RF Accuracy","LR Accuracy"],
    ["Full dataset",f"{churn_rate:.1f}% of base",f"{100-churn_rate:.1f}% of base","Random Forest","Logistic Reg."]
):
    col.markdown(f"""
    <div class="glass-card">
      <div class="card-icon">{icon}</div>
      <div class="card-value">{val}</div>
      <div class="card-label">{label}</div>
      <div class="card-sub">{sub}</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)
tab1, tab2, tab3 = st.tabs(["📊   Overview & Insights","🤖   Model Performance","🔍   Live Prediction"])

# ════ TAB 1 ═══════════════════════════════════════════
with tab1:
    st.markdown('<div class="section-title">Customer Analytics Overview</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Interactive charts — hover anywhere for detailed breakdowns.</div>', unsafe_allow_html=True)

    r1a, r1b = st.columns([1, 1.6])
    with r1a:
        sizes = df["Churn"].value_counts()
        fig = go.Figure(go.Pie(
            labels=["Retained","Churned"], values=sizes.values, hole=0.62,
            marker=dict(colors=[GREEN,RED], line=dict(color="#020c18", width=3)),
            hovertemplate="<b>%{label}</b><br>Count: %{value:,}<br>Share: %{percent}<extra></extra>",
            textfont=dict(color="white", size=13),
        ))
        fig.add_annotation(text=f"<b>{churn_rate:.1f}%</b>", x=0.5, y=0.56,
            font=dict(size=26, color=RED, family="Inter"), showarrow=False)
        fig.add_annotation(text="churn rate", x=0.5, y=0.42,
            font=dict(size=12, color=MUTED, family="Inter"), showarrow=False)
        fig.update_layout(**base_layout("Churn Distribution", height=320,
            showlegend=True, legend=dict(orientation="h", y=-0.05, x=0.5,
            xanchor="center", **LEGEND)))
        st.plotly_chart(fig, use_container_width=True)

    with r1b:
        cc = raw.groupby("Contract")["Churn"].agg(["sum","count"]).reset_index()
        cc.columns = ["Contract","Churned","Total"]
        cc["Retained"] = cc["Total"] - cc["Churned"]
        fig2 = go.Figure()
        fig2.add_trace(go.Bar(name="Retained", x=cc["Contract"], y=cc["Retained"],
            marker_color=GREEN, opacity=0.85,
            hovertemplate="<b>%{x}</b><br>Retained: %{y:,}<extra></extra>"))
        fig2.add_trace(go.Bar(name="Churned", x=cc["Contract"], y=cc["Churned"],
            marker_color=RED, opacity=0.85,
            hovertemplate="<b>%{x}</b><br>Churned: %{y:,}<extra></extra>"))
        fig2.update_layout(**base_layout("Churn by Contract Type", height=320,
            barmode="stack", xaxis_title="Contract Type", yaxis_title="Customers",
            showlegend=True, legend=LEGEND))
        st.plotly_chart(fig2, use_container_width=True)

    r2a, r2b = st.columns(2)
    with r2a:
        samp = raw.sample(min(800,len(raw)), random_state=1)
        fig3 = px.scatter(samp, x="tenure", y="MonthlyCharges",
            color=samp["Churn"].map({1:"Churned",0:"Retained"}),
            color_discrete_map={"Churned":RED,"Retained":GREEN}, opacity=0.65,
            labels={"tenure":"Tenure (months)","MonthlyCharges":"Monthly Charges ($)","color":"Status"})
        fig3.update_traces(marker=dict(size=6),
            hovertemplate="Tenure: %{x} mo<br>Monthly: $%{y:.2f}<extra></extra>")
        fig3.update_layout(**base_layout("Tenure vs Monthly Charges", height=320,
            showlegend=True, legend=LEGEND))
        st.plotly_chart(fig3, use_container_width=True)

    with r2b:
        fig4 = go.Figure()
        fig4.add_trace(go.Box(y=raw[raw["Churn"]==0]["MonthlyCharges"], name="Retained",
            marker_color=GREEN, line_color=GREEN,
            fillcolor="rgba(0,196,154,0.15)", boxmean=True,
            hovertemplate="<b>Retained</b><br>$%{y:.2f}<extra></extra>"))
        fig4.add_trace(go.Box(y=raw[raw["Churn"]==1]["MonthlyCharges"], name="Churned",
            marker_color=RED, line_color=RED,
            fillcolor="rgba(224,85,112,0.15)", boxmean=True,
            hovertemplate="<b>Churned</b><br>$%{y:.2f}<extra></extra>"))
        fig4.update_layout(**base_layout("Monthly Charges Distribution", height=320,
            yaxis_title="Monthly Charges ($)", showlegend=True, legend=LEGEND))
        st.plotly_chart(fig4, use_container_width=True)

    ic = raw.groupby("InternetService")["Churn"].agg(["mean","count"]).reset_index()
    ic.columns = ["Service","Churn Rate","Count"]
    ic["Churn Rate"] = (ic["Churn Rate"]*100).round(1)
    fig5 = px.bar(ic, x="Service", y="Churn Rate",
        color="Churn Rate", color_continuous_scale=["#00c49a","#0a9fd4","#e05570"],
        text=ic["Churn Rate"].apply(lambda x: f"{x}%"))
    fig5.update_traces(hovertemplate="<b>%{x}</b><br>Churn Rate: %{y:.1f}%<extra></extra>",
        textposition="outside", textfont=dict(color="white", size=12))
    fig5.update_layout(**base_layout("Churn Rate by Internet Service", height=300,
        xaxis_title="Internet Service", yaxis_title="Churn Rate (%)",
        coloraxis_showscale=False, yaxis_range=[0,65]))
    st.plotly_chart(fig5, use_container_width=True)

# ════ TAB 2 ═══════════════════════════════════════════
with tab2:
    st.markdown('<div class="section-title">Model Performance</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Random Forest vs Logistic Regression on 20% holdout test data.</div>', unsafe_allow_html=True)

    ma, mb = st.columns(2)
    with ma:
        st.markdown(f"""
        <div class="glass-card" style="border-top:3px solid {GREEN};text-align:center;padding:28px;margin-bottom:16px;">
            <div style="font-size:11px;color:{MUTED};font-weight:700;letter-spacing:1.5px;text-transform:uppercase;">Random Forest</div>
            <div style="font-size:56px;font-weight:900;color:{GREEN};margin:12px 0;line-height:1;">{rf_acc*100:.1f}%</div>
            <div style="font-size:12px;color:{MUTED};">Test Set Accuracy</div>
        </div>""", unsafe_allow_html=True)
        cm = confusion_matrix(y_te, rf.predict(X_te))
        fig_cm = px.imshow(cm, text_auto=True,
            x=["Predicted: Stay","Predicted: Churn"],
            y=["Actual: Stay","Actual: Churn"],
            color_continuous_scale=[[0,"#0d1f35"],[1,GREEN]], aspect="auto")
        fig_cm.update_traces(
            hovertemplate="<b>%{y}</b><br>%{x}<br>Count: <b>%{z}</b><extra></extra>",
            textfont=dict(size=18, color="white"))
        fig_cm.update_layout(**base_layout("Confusion Matrix — Random Forest", height=300,
            coloraxis_showscale=False))
        st.plotly_chart(fig_cm, use_container_width=True)

    with mb:
        st.markdown(f"""
        <div class="glass-card" style="border-top:3px solid {BLUE};text-align:center;padding:28px;margin-bottom:16px;">
            <div style="font-size:11px;color:{MUTED};font-weight:700;letter-spacing:1.5px;text-transform:uppercase;">Logistic Regression</div>
            <div style="font-size:56px;font-weight:900;color:{BLUE};margin:12px 0;line-height:1;">{lr_acc*100:.1f}%</div>
            <div style="font-size:12px;color:{MUTED};">Test Set Accuracy</div>
        </div>""", unsafe_allow_html=True)
        fi = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=True).tail(12)
        fig_fi = go.Figure(go.Bar(
            x=fi.values, y=fi.index, orientation="h",
            marker=dict(color=[GREEN if v > fi.median() else BLUE for v in fi.values]),
            hovertemplate="<b>%{y}</b><br>Importance: %{x:.4f}<extra></extra>"))
        fig_fi.update_layout(**base_layout("Top Feature Importances", height=300,
            xaxis_title="Importance Score"))
        st.plotly_chart(fig_fi, use_container_width=True)

    st.markdown('<div class="section-title" style="margin-top:8px;">Metrics Radar</div>', unsafe_allow_html=True)
    rf_p = rf.predict(X_te)
    lr_p = lr.predict(X_te)
    cats = ["Accuracy","Precision","Recall","F1 Score"]
    rf_v = [rf_acc, precision_score(y_te,rf_p), recall_score(y_te,rf_p), f1_score(y_te,rf_p)]
    lr_v = [lr_acc, precision_score(y_te,lr_p), recall_score(y_te,lr_p), f1_score(y_te,lr_p)]
    fig_r = go.Figure()
    fig_r.add_trace(go.Scatterpolar(r=rf_v+[rf_v[0]], theta=cats+[cats[0]],
        fill="toself", name="Random Forest",
        line=dict(color=GREEN, width=2), fillcolor="rgba(0,196,154,0.15)",
        hovertemplate="<b>Random Forest</b><br>%{theta}: %{r:.3f}<extra></extra>"))
    fig_r.add_trace(go.Scatterpolar(r=lr_v+[lr_v[0]], theta=cats+[cats[0]],
        fill="toself", name="Logistic Regression",
        line=dict(color=BLUE, width=2), fillcolor="rgba(10,159,212,0.15)",
        hovertemplate="<b>Logistic Regression</b><br>%{theta}: %{r:.3f}<extra></extra>"))
    fig_r.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        polar=dict(bgcolor="rgba(13,31,53,0.5)",
            radialaxis=dict(visible=True, range=[0.6,1.0], gridcolor="#0a2a40",
                tickfont=dict(color=MUTED, size=10)),
            angularaxis=dict(gridcolor="#0a2a40", tickfont=dict(color="#8aacca", size=12))),
        legend=LEGEND,
        hoverlabel=dict(bgcolor="#0d1f35", bordercolor=BLUE, font=dict(color="#e0f0ff", size=13)),
        height=380, margin=dict(l=60,r=60,t=30,b=30))
    st.plotly_chart(fig_r, use_container_width=True)

# ════ TAB 3 ═══════════════════════════════════════════
with tab3:
    st.markdown('<div class="section-title">Live Churn Prediction</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Fill in customer details and get an instant ML-powered churn risk score.</div>', unsafe_allow_html=True)

    with st.form("predict"):
        p1,p2,p3 = st.columns(3)
        with p1:
            st.markdown(f"<div style='color:{GREEN};font-size:11px;font-weight:700;letter-spacing:1.5px;margin-bottom:14px;'>👤 CUSTOMER INFO</div>", unsafe_allow_html=True)
            gender       = st.selectbox("Gender", ["Male","Female"])
            senior       = st.selectbox("Senior Citizen", ["No","Yes"])
            partner      = st.selectbox("Has Partner", ["Yes","No"])
            dependents   = st.selectbox("Has Dependents", ["Yes","No"])
            tenure       = st.slider("Tenure (months)", 0, 72, 12)
            phone_service= st.selectbox("Phone Service", ["Yes","No"])
        with p2:
            st.markdown(f"<div style='color:{BLUE};font-size:11px;font-weight:700;letter-spacing:1.5px;margin-bottom:14px;'>🌐 SERVICES</div>", unsafe_allow_html=True)
            multiple_lines   = st.selectbox("Multiple Lines", ["No","Yes","No phone service"])
            internet_service = st.selectbox("Internet Service", ["DSL","Fiber optic","No"])
            online_security  = st.selectbox("Online Security", ["Yes","No","No internet service"])
            online_backup    = st.selectbox("Online Backup", ["Yes","No","No internet service"])
            device_protection= st.selectbox("Device Protection", ["Yes","No","No internet service"])
            tech_support     = st.selectbox("Tech Support", ["Yes","No","No internet service"])
        with p3:
            st.markdown(f"<div style='color:#00b8d4;font-size:11px;font-weight:700;letter-spacing:1.5px;margin-bottom:14px;'>💳 BILLING</div>", unsafe_allow_html=True)
            streaming_tv    = st.selectbox("Streaming TV", ["Yes","No","No internet service"])
            streaming_movies= st.selectbox("Streaming Movies", ["Yes","No","No internet service"])
            contract        = st.selectbox("Contract Type", ["Month-to-month","One year","Two year"])
            paperless       = st.selectbox("Paperless Billing", ["Yes","No"])
            payment         = st.selectbox("Payment Method", ["Electronic check","Mailed check","Bank transfer (automatic)","Credit card (automatic)"])
            monthly_charges = st.slider("Monthly Charges ($)", 18.0, 120.0, 65.0)
            total_charges   = st.slider("Total Charges ($)", 18.0, 9000.0, 500.0)

        st.divider()
        model_choice = st.radio("Prediction Model", ["Random Forest","Logistic Regression"], horizontal=True)
        submitted = st.form_submit_button("🔮  Run Churn Prediction", use_container_width=True)

    if submitted:
        inp = {
            "gender":1 if gender=="Male" else 0,
            "SeniorCitizen":1 if senior=="Yes" else 0,
            "Partner":1 if partner=="Yes" else 0,
            "Dependents":1 if dependents=="Yes" else 0,
            "tenure":tenure,
            "PhoneService":1 if phone_service=="Yes" else 0,
            "MultipleLines":{"No":0,"Yes":1,"No phone service":2}[multiple_lines],
            "InternetService":{"DSL":0,"Fiber optic":1,"No":2}[internet_service],
            "OnlineSecurity":{"No":0,"Yes":1,"No internet service":2}[online_security],
            "OnlineBackup":{"No":0,"Yes":1,"No internet service":2}[online_backup],
            "DeviceProtection":{"No":0,"Yes":1,"No internet service":2}[device_protection],
            "TechSupport":{"No":0,"Yes":1,"No internet service":2}[tech_support],
            "StreamingTV":{"No":0,"Yes":1,"No internet service":2}[streaming_tv],
            "StreamingMovies":{"No":0,"Yes":1,"No internet service":2}[streaming_movies],
            "Contract":{"Month-to-month":0,"One year":1,"Two year":2}[contract],
            "PaperlessBilling":1 if paperless=="Yes" else 0,
            "PaymentMethod":{"Bank transfer (automatic)":0,"Credit card (automatic)":1,
                             "Electronic check":2,"Mailed check":3}[payment],
            "MonthlyCharges":monthly_charges,
            "TotalCharges":total_charges,
        }
        scaled = scaler.transform(pd.DataFrame([inp]))
        model  = rf if model_choice=="Random Forest" else lr
        pred   = model.predict(scaled)[0]
        prob   = model.predict_proba(scaled)[0][1]

        st.divider()
        _, rc, _ = st.columns([1,2,1])
        with rc:
            color = RED   if pred==1 else GREEN
            icon  = "⚠️"  if pred==1 else "✅"
            label = "HIGH CHURN RISK"   if pred==1 else "LOW CHURN RISK"
            desc  = "This customer is likely to cancel." if pred==1 else "This customer is likely to stay."

            st.markdown(f"""
            <div style='background:linear-gradient(135deg,rgba(13,31,53,0.95),rgba(10,22,40,0.98));
            border:1px solid {color}33; border-top:3px solid {color}; border-radius:18px;
            padding:40px 32px; text-align:center; box-shadow:0 8px 40px {color}18;'>
                <div style='font-size:52px;margin-bottom:14px;'>{icon}</div>
                <div style='font-size:11px;color:{color};font-weight:700;letter-spacing:2px;margin-bottom:10px;'>PREDICTION RESULT</div>
                <div style='font-size:28px;font-weight:800;color:{color};'>{label}</div>
                <div style='color:#3a7a9a;font-size:14px;margin:10px 0 24px;'>{desc}</div>
                <div style='font-size:64px;font-weight:900;color:{color};line-height:1;'>{prob*100:.1f}%</div>
                <div style='color:#2a5a7a;font-size:11px;margin-top:6px;letter-spacing:1.5px;'>CHURN PROBABILITY</div>
            </div>""", unsafe_allow_html=True)

            st.markdown("<div style='margin-top:20px;'></div>", unsafe_allow_html=True)

            fig_g = go.Figure(go.Indicator(
                mode="gauge+number", value=prob*100,
                number=dict(suffix="%", font=dict(size=32, color=color, family="Inter")),
                gauge=dict(
                    axis=dict(range=[0,100], tickfont=dict(color=MUTED, size=10)),
                    bar=dict(color=color, thickness=0.25),
                    bgcolor="rgba(13,31,53,0.8)", borderwidth=0,
                    steps=[
                        dict(range=[0,30],   color="rgba(0,196,154,0.15)"),
                        dict(range=[30,60],  color="rgba(10,159,212,0.15)"),
                        dict(range=[60,100], color="rgba(224,85,112,0.15)"),
                    ],
                    threshold=dict(line=dict(color=color, width=3), value=prob*100)
                )
            ))
            fig_g.update_layout(paper_bgcolor="rgba(0,0,0,0)",
                font=dict(family="Inter", color="#8aacca"),
                height=220, margin=dict(l=30,r=30,t=20,b=10),
                hoverlabel=dict(bgcolor="#0d1f35", font=dict(color="#e0f0ff")))
            st.plotly_chart(fig_g, use_container_width=True)

            m1,m2,m3 = st.columns(3)
            m1.metric("Churn Risk",   f"{prob*100:.1f}%")
            m2.metric("Retention",    f"{(1-prob)*100:.1f}%")
            m3.metric("Model",        model_choice.split()[0])
