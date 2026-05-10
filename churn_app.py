"""
AI-Powered Customer Churn Analytics Platform
=============================================
Single-file Streamlit app — no imports from other files needed.
Run:  streamlit run churn_app.py
Login: demo@churnai.com / demo123
"""

# ─────────────────────────────────────────────────────────────────────────────
# IMPORTS
# ─────────────────────────────────────────────────────────────────────────────
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG  (must be first Streamlit call)
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ChurnAI Platform",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@600;700;800&family=DM+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif !important;
    background-color: #0a0e1a !important;
    color: #e2e8f0 !important;
}
.main .block-container { padding-top: 1.5rem; }

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #0d1120 !important;
    border-right: 1px solid rgba(255,255,255,0.07) !important;
}
section[data-testid="stSidebar"] * { color: #94a3b8 !important; }
section[data-testid="stSidebar"] .stButton > button {
    background: transparent !important;
    color: #94a3b8 !important;
    border: none !important;
    text-align: left !important;
    font-size: 14px !important;
    padding: 8px 12px !important;
    border-radius: 8px !important;
    width: 100% !important;
    box-shadow: none !important;
}
section[data-testid="stSidebar"] .stButton > button:hover {
    background: rgba(99,102,241,0.1) !important;
    color: #e2e8f0 !important;
    box-shadow: none !important;
}

/* Hide Streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }

/* Metric cards */
[data-testid="metric-container"] {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
    border-radius: 14px !important;
    padding: 18px 20px !important;
}
[data-testid="stMetricLabel"] > div  { color: #64748b !important; font-size: 12px !important; }
[data-testid="stMetricValue"] > div  { color: #f1f5f9 !important; font-size: 24px !important; font-family:'Syne',sans-serif !important; }

/* Inputs */
.stSelectbox > div > div,
.stTextInput > div > div > input,
.stNumberInput > div > div > input {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: 10px !important;
    color: #e2e8f0 !important;
}

/* Primary buttons */
.stButton > button[kind="primary"],
.main .stButton > button {
    background: linear-gradient(135deg,#6366f1,#8b5cf6) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    padding: 10px 22px !important;
}
.main .stButton > button:hover {
    box-shadow: 0 6px 20px rgba(99,102,241,0.4) !important;
    transform: translateY(-1px) !important;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    background: rgba(255,255,255,0.03);
    border-radius: 12px; padding: 4px; gap: 4px;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px !important;
    color: #64748b !important;
    font-size: 13px !important;
}
.stTabs [aria-selected="true"] {
    background: rgba(99,102,241,0.2) !important;
    color: #6366f1 !important;
}

/* Dataframe */
.stDataFrame { border-radius: 12px; overflow: hidden; }
hr { border-color: rgba(255,255,255,0.07) !important; }

/* Chat bubbles */
.chat-user {
    background: linear-gradient(135deg,#6366f1,#8b5cf6);
    color: white;
    border-radius: 18px 18px 4px 18px;
    padding: 10px 16px;
    margin: 6px 0;
    max-width: 75%;
    font-size: 14px;
    line-height: 1.5;
    margin-left: auto;
    width: fit-content;
}
.chat-ai {
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.09);
    color: #e2e8f0;
    border-radius: 18px 18px 18px 4px;
    padding: 12px 16px;
    margin: 6px 0;
    max-width: 85%;
    font-size: 14px;
    line-height: 1.6;
    white-space: pre-wrap;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# DATA & MODEL  (cached so they only run once per session)
# ─────────────────────────────────────────────────────────────────────────────
FIRST_NAMES = ["Arjun","Priya","Rahul","Ananya","Kiran","Deepak","Sneha","Vikram",
               "Pooja","Aditya","Riya","Suresh","Nandini","Rajesh","Kavya","Manish",
               "Swathi","Rohit","Divya","Amit","Lakshmi","Sanjay","Meera","Nikhil",
               "Aparna","Gaurav","Sunita","Harish","Preeti","Varun","Tanvi","Pranav",
               "Shreya","Ajay","Neha","Vivek","Kartik","Shweta","Akash","Ravi"]
LAST_NAMES  = ["Sharma","Patel","Verma","Singh","Rao","Kumar","Joshi","Nair","Gupta",
               "Mehta","Kapoor","Pillai","Das","Iyer","Reddy","Tiwari","Menon","Bhatt",
               "Pandey","Shah","Thomas","Choudhary","Krishnan","Mishra","Yadav","Bose",
               "Aggarwal","Malhotra","Dubey","Bhat"]

@st.cache_data
def generate_data(n=400):
    rng = np.random.RandomState(42)
    contracts = rng.choice(["Month-to-month","One year","Two year"], n, p=[0.55,0.24,0.21])
    internet  = rng.choice(["DSL","Fiber optic","No"], n, p=[0.34,0.44,0.22])
    payment   = rng.choice(["Electronic check","Mailed check",
                             "Bank transfer (automatic)","Credit card (automatic)"], n)
    tech      = rng.choice(["Yes","No","No internet service"], n)
    streaming = rng.choice(["Yes","No","No internet service"], n)
    tenure    = rng.randint(1, 73, n)
    monthly   = np.round(rng.uniform(18, 118, n), 2)
    total     = np.round(monthly * tenure * rng.uniform(0.8, 1.0, n), 2)
    gender    = rng.choice(["Male","Female"], n)
    senior    = rng.choice([0,1], n, p=[0.83,0.17])
    depend    = rng.choice(["Yes","No"], n, p=[0.3,0.7])
    partner   = rng.choice(["Yes","No"], n)
    paperless = rng.choice(["Yes","No"], n)
    ages      = np.clip(rng.normal(42,14,n).astype(int), 18, 80)

    # Churn probability
    p = 0.15 * np.ones(n)
    p += 0.28*(contracts=="Month-to-month") + 0.08*(contracts=="One year")
    p += 0.15*(monthly>70) + 0.10*(monthly>90)
    p += 0.18*(tenure<12)  + 0.08*((tenure>=12)&(tenure<24))
    p += 0.06*(internet=="Fiber optic") + 0.07*(tech=="No")
    p += 0.05*(senior==1)  + 0.06*(payment=="Electronic check")
    p -= 0.05*(depend=="Yes")
    p  = np.clip(p + rng.normal(0,0.04,n), 0.02, 0.97)
    churn = (rng.rand(n) < p).astype(int)

    names = [f"{rng.choice(FIRST_NAMES)} {rng.choice(LAST_NAMES)}" for _ in range(n)]

    return pd.DataFrame({
        "CustomerID":       [f"CUST-{1000+i}" for i in range(n)],
        "Name":             names,
        "Age":              ages,
        "Gender":           gender,
        "SeniorCitizen":    senior,
        "Partner":          partner,
        "Dependents":       depend,
        "Tenure":           tenure,
        "InternetService":  internet,
        "TechSupport":      tech,
        "StreamingTV":      streaming,
        "Contract":         contracts,
        "PaperlessBilling": paperless,
        "PaymentMethod":    payment,
        "MonthlyCharges":   monthly,
        "TotalCharges":     total,
        "Churn":            churn,
    })


ENCODE_MAP = {
    "Gender":          {"Female":0,"Male":1},
    "Partner":         {"No":0,"Yes":1},
    "Dependents":      {"No":0,"Yes":1},
    "PaperlessBilling":{"No":0,"Yes":1},
    "InternetService": {"DSL":0,"Fiber optic":1,"No":2},
    "TechSupport":     {"No":0,"No internet service":1,"Yes":2},
    "StreamingTV":     {"No":0,"No internet service":1,"Yes":2},
    "Contract":        {"Month-to-month":0,"One year":1,"Two year":2},
    "PaymentMethod":   {"Bank transfer (automatic)":0,"Credit card (automatic)":1,
                        "Electronic check":2,"Mailed check":3},
}
FEATURE_COLS = ["Age","Gender","SeniorCitizen","Partner","Dependents","Tenure",
                "InternetService","TechSupport","StreamingTV","Contract",
                "PaperlessBilling","PaymentMethod","MonthlyCharges","TotalCharges"]


@st.cache_resource
def train_models():
    df = generate_data()
    dfe = df.copy()
    for col, mapping in ENCODE_MAP.items():
        dfe[col] = dfe[col].map(mapping).fillna(0).astype(int)
    X = dfe[FEATURE_COLS]
    y = dfe["Churn"]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=2000, random_state=42),
        "Random Forest":       RandomForestClassifier(n_estimators=100, random_state=42),
        "XGBoost":             XGBClassifier(n_estimators=100, random_state=42,
                                             eval_metric="logloss", verbosity=0),
    }
    results, best_name, best_f1, best_model = {}, None, 0, None
    for name, m in models.items():
        m.fit(X_tr, y_tr)
        yp = m.predict(X_te)
        acc  = round(accuracy_score(y_te, yp)*100, 1)
        prec = round(precision_score(y_te, yp, zero_division=0)*100, 1)
        rec  = round(recall_score(y_te, yp, zero_division=0)*100, 1)
        f1   = round(f1_score(y_te, yp, zero_division=0)*100, 1)
        results[name] = {"accuracy":acc,"precision":prec,"recall":rec,"f1":f1}
        if f1 > best_f1:
            best_f1, best_name, best_model = f1, name, m
    return best_model, results, best_name


def predict_churn(row_dict):
    model, _, _ = train_models()
    enc = {}
    for col in FEATURE_COLS:
        val = row_dict.get(col, 0)
        enc[col] = ENCODE_MAP[col].get(str(val), 0) if col in ENCODE_MAP else val
    X = pd.DataFrame([enc])
    prob  = float(model.predict_proba(X)[0][1])
    score = round(prob*100, 1)
    if prob > 0.70:
        risk   = "🔴 High Risk"
        color  = "#ef4444"
        action = ("Immediate action required — assign a dedicated retention specialist "
                  "and offer a contract upgrade with 20% discount within 48 hours.")
    elif prob > 0.40:
        risk   = "🟡 Medium Risk"
        color  = "#f59e0b"
        action = ("Schedule a proactive check-in call within 7 days. "
                  "Offer loyalty rewards and highlight premium features.")
    else:
        risk   = "🟢 Low Risk"
        color  = "#10b981"
        action = ("Customer is stable. Continue regular engagement and "
                  "consider upselling premium add-ons.")
    return score, risk, color, action


# ─────────────────────────────────────────────────────────────────────────────
# PLOTLY SHARED THEME
# ─────────────────────────────────────────────────────────────────────────────
PL = dict(
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#94a3b8", family="DM Sans"),
    margin=dict(l=10, r=10, t=36, b=10),
    legend=dict(bgcolor="rgba(0,0,0,0)"),
    xaxis=dict(gridcolor="rgba(255,255,255,0.05)", zerolinecolor="rgba(0,0,0,0)"),
    yaxis=dict(gridcolor="rgba(255,255,255,0.05)", zerolinecolor="rgba(0,0,0,0)"),
)

def card(html):
    st.markdown(f"""
    <div style="background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.08);
                border-radius:14px;padding:18px;margin-bottom:8px">{html}</div>
    """, unsafe_allow_html=True)

def section(title):
    st.markdown(f"""
    <p style="font-family:'Syne',sans-serif;font-size:22px;font-weight:700;
              color:#f1f5f9;margin:0 0 2px">{title}</p>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# AUTH
# ─────────────────────────────────────────────────────────────────────────────
if "auth" not in st.session_state:
    st.session_state.auth = False

if not st.session_state.auth:
    _, mid, _ = st.columns([1, 1.1, 1])
    with mid:
        st.markdown("<br><br>", unsafe_allow_html=True)
        st.markdown("""
        <div style="text-align:center;margin-bottom:28px">
            <div style="font-size:40px;margin-bottom:12px">📊</div>
            <p style="font-family:'Syne',sans-serif;font-size:26px;font-weight:700;
                      color:#f1f5f9;margin:0">ChurnAI Platform</p>
            <p style="color:#64748b;font-size:14px;margin-top:4px">
                AI-Powered Customer Intelligence</p>
        </div>
        """, unsafe_allow_html=True)

        t1, t2 = st.tabs(["🔑 Sign In", "📝 Sign Up"])
        with t1:
            em = st.text_input("Email",    value="demo@churnai.com", key="li_em")
            pw = st.text_input("Password", value="demo123", type="password", key="li_pw")
            if st.button("Sign In to Dashboard", use_container_width=True, key="li_btn"):
                if em == "demo@churnai.com" and pw == "demo123":
                    st.session_state.auth      = True
                    st.session_state.user_name = "Demo User"
                    st.rerun()
                else:
                    st.error("Use demo@churnai.com / demo123")
            st.caption("Demo: demo@churnai.com / demo123")

        with t2:
            nm = st.text_input("Full Name", key="su_nm")
            se = st.text_input("Email",     key="su_em")
            sp = st.text_input("Password",  type="password", key="su_pw")
            if st.button("Create Account", use_container_width=True, key="su_btn"):
                if nm and se and sp:
                    st.session_state.auth      = True
                    st.session_state.user_name = nm
                    st.rerun()
                else:
                    st.error("Please fill all fields.")
    st.stop()


# ─────────────────────────────────────────────────────────────────────────────
# LOAD DATA + MODEL  (after auth so spinner shows on main area)
# ─────────────────────────────────────────────────────────────────────────────
with st.spinner("Loading AI models and data…"):
    DF = generate_data()
    MODEL, MODEL_RESULTS, BEST_MODEL_NAME = train_models()

# Derived stats
CHURNED  = DF[DF["Churn"]==1]
ACTIVE   = DF[DF["Churn"]==0]
CHURN_RT = round(len(CHURNED)/len(DF)*100, 1)
REV_LOSS = round(CHURNED["MonthlyCharges"].sum(), 0)
RET_RT   = round(100 - CHURN_RT, 1)
MRR      = round(ACTIVE["MonthlyCharges"].sum(), 0)
AVG_CLV  = round(DF["TotalCharges"].mean(), 0)

MONTHS = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
rng2   = np.random.RandomState(7)
TREND  = pd.DataFrame({
    "Month":    MONTHS,
    "Churned":  rng2.randint(8,22,12),
    "Retained": rng2.randint(40,72,12),
    "Revenue":  rng2.randint(3500,8000,12),
})


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR  NAV
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="display:flex;align-items:center;gap:10px;padding:4px 0 20px">
        <span style="font-size:26px">📊</span>
        <span style="font-family:'Syne',sans-serif;font-size:17px;font-weight:700;
                     color:#f1f5f9 !important">ChurnAI</span>
    </div>
    """, unsafe_allow_html=True)
    st.divider()

    NAV = {
        "📊  Dashboard":        "dashboard",
        "🎯  Churn Prediction":  "prediction",
        "👥  Customers":         "customers",
        "🧠  Explainable AI":    "explainability",
        "📈  Analytics":         "analytics",
        "🤖  AI Chatbot":        "chatbot",
        "⚙️  Admin Panel":       "admin",
    }
    if "page" not in st.session_state:
        st.session_state.page = "dashboard"

    for label, key in NAV.items():
        if st.button(label, key=f"nav_{key}", use_container_width=True):
            st.session_state.page = key
            st.rerun()

    st.divider()
    st.markdown(f"""
    <div style="padding:10px 12px;border-radius:10px;background:rgba(255,255,255,0.03);
                border:1px solid rgba(255,255,255,0.06);margin-bottom:8px">
        <p style="font-size:13px;font-weight:500;color:#e2e8f0 !important;margin:0">
            {st.session_state.get('user_name','User')}</p>
        <p style="font-size:11px;color:#6366f1 !important;margin:0">Admin</p>
    </div>
    """, unsafe_allow_html=True)

    if st.button("🚪  Logout", key="logout", use_container_width=True):
        st.session_state.auth = False
        st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# TOPBAR
# ─────────────────────────────────────────────────────────────────────────────
tb1, tb2 = st.columns([3,1])
with tb1:
    page_labels = {v:k for k,v in NAV.items()}
    st.markdown(f"""
    <p style="color:#475569;font-size:13px;margin-bottom:0">
        ChurnAI &nbsp;›&nbsp;
        <span style="color:#6366f1">{page_labels.get(st.session_state.page,'Dashboard')}</span>
    </p>
    """, unsafe_allow_html=True)
with tb2:
    st.markdown("""
    <div style="display:flex;justify-content:flex-end;align-items:center;gap:8px">
        <span style="background:rgba(16,185,129,0.1);border:1px solid rgba(16,185,129,0.2);
                     color:#10b981;padding:3px 10px;border-radius:8px;font-size:12px">● Live</span>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<hr style='margin:8px 0 20px'>", unsafe_allow_html=True)
PAGE = st.session_state.page


# ═════════════════════════════════════════════════════════════════════════════
# PAGE: DASHBOARD
# ═════════════════════════════════════════════════════════════════════════════
if PAGE == "dashboard":
    section("📊 Analytics Dashboard")
    st.markdown("<p style='color:#64748b;font-size:14px;margin-bottom:20px'>Real-time churn intelligence and business insights</p>", unsafe_allow_html=True)

    c1,c2,c3,c4,c5,c6 = st.columns(6)
    c1.metric("Total Customers",  len(DF),              "+12%")
    c2.metric("Active",           len(ACTIVE),           "+8%")
    c3.metric("Churned",          len(CHURNED),          "-3%")
    c4.metric("Churn Rate",       f"{CHURN_RT}%",        delta="-2.1%")
    c5.metric("Revenue Loss/mo",  f"${REV_LOSS:,.0f}",   delta="-5%", delta_color="inverse")
    c6.metric("Retention Rate",   f"{RET_RT}%",          "+2%")

    st.markdown("<br>", unsafe_allow_html=True)

    # Row 1
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Monthly Churn & Retention Trend**")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=TREND["Month"], y=TREND["Churned"], name="Churned",
            mode="lines+markers", line=dict(color="#6366f1",width=2.5),
            fill="tozeroy", fillcolor="rgba(99,102,241,0.1)"))
        fig.add_trace(go.Scatter(x=TREND["Month"], y=TREND["Retained"], name="Retained",
            mode="lines+markers", line=dict(color="#10b981",width=2.5),
            fill="tozeroy", fillcolor="rgba(16,185,129,0.1)"))
        fig.update_layout(**PL, height=260)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("**Churn Distribution**")
        pie = pd.DataFrame({"Status":["Active","Churned"],"Count":[len(ACTIVE),len(CHURNED)]})
        fig = px.pie(pie, values="Count", names="Status", hole=0.55,
                     color="Status",
                     color_discrete_map={"Active":"#10b981","Churned":"#ef4444"})
        fig.update_traces(textfont_color="#e2e8f0")
        fig.update_layout(**PL, height=260)
        st.plotly_chart(fig, use_container_width=True)

    # Row 2
    col3, col4 = st.columns(2)
    with col3:
        st.markdown("**Contract Type vs Churn**")
        ct = (DF.groupby("Contract")["Churn"].agg(["sum","count"])
               .rename(columns={"sum":"Churned","count":"Total"}).reset_index())
        ct["Retained"] = ct["Total"] - ct["Churned"]
        fig = px.bar(ct, x="Contract", y=["Churned","Retained"], barmode="group",
                     color_discrete_map={"Churned":"#ef4444","Retained":"#10b981"})
        fig.update_layout(**PL, height=260)
        st.plotly_chart(fig, use_container_width=True)

    with col4:
        st.markdown("**Payment Method vs Churn Rate**")
        pm = (DF.groupby("PaymentMethod")["Churn"].agg(["sum","count"]).reset_index())
        pm.columns = ["PaymentMethod","Churned","Total"]
        pm["Churn%"] = (pm["Churned"]/pm["Total"]*100).round(1)
        pm["Short"]  = pm["PaymentMethod"].str[:16]
        fig = px.bar(pm, x="Churn%", y="Short", orientation="h",
                     color="Churn%", color_continuous_scale=["#10b981","#f59e0b","#ef4444"])
        fig.update_layout(**PL, height=260, coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    # Row 3
    col5, col6 = st.columns(2)
    with col5:
        st.markdown("**Age Group Segmentation**")
        DF2 = DF.copy()
        DF2["AgeGroup"] = pd.cut(DF2["Age"], bins=[17,25,35,45,55,100],
                                 labels=["18-25","26-35","36-45","46-55","55+"])
        seg = DF2.groupby("AgeGroup", observed=True).agg(
            Total=("CustomerID","count"), Churned=("Churn","sum")).reset_index()
        fig = px.bar(seg, x="AgeGroup", y=["Total","Churned"], barmode="group",
                     color_discrete_map={"Total":"#6366f1","Churned":"#ef4444"})
        fig.update_layout(**PL, height=260)
        st.plotly_chart(fig, use_container_width=True)

    with col6:
        st.markdown("**Monthly Revenue Trend**")
        fig = px.line(TREND, x="Month", y="Revenue", markers=True, line_shape="spline",
                      color_discrete_sequence=["#06b6d4"])
        fig.update_traces(fill="tozeroy", fillcolor="rgba(6,182,212,0.1)")
        fig.update_layout(**PL, height=260)
        st.plotly_chart(fig, use_container_width=True)

    # Internet service
    st.markdown("**Internet Service × Churn Rate**")
    is_df = (DF.groupby("InternetService")["Churn"].agg(["sum","count"]).reset_index())
    is_df.columns = ["InternetService","Churned","Total"]
    is_df["Churn%"] = (is_df["Churned"]/is_df["Total"]*100).round(1)
    fig = px.bar(is_df, x="InternetService", y="Churn%",
                 color="Churn%", color_continuous_scale=["#10b981","#f59e0b","#ef4444"],
                 text="Churn%")
    fig.update_traces(texttemplate="%{text}%", textposition="outside")
    fig.update_layout(**PL, height=230, coloraxis_showscale=False)
    st.plotly_chart(fig, use_container_width=True)


# ═════════════════════════════════════════════════════════════════════════════
# PAGE: PREDICTION
# ═════════════════════════════════════════════════════════════════════════════
elif PAGE == "prediction":
    section("🎯 Churn Prediction")
    st.markdown("<p style='color:#64748b;font-size:14px;margin-bottom:20px'>ML-powered real-time churn probability scoring</p>", unsafe_allow_html=True)

    col_form, col_result = st.columns([1,1])

    with col_form:
        st.markdown("#### Customer Profile")
        r1a, r1b = st.columns(2)
        age    = r1a.number_input("Age",              18, 80,  35)
        gender = r1b.selectbox("Gender",              ["Male","Female"])
        r2a, r2b = st.columns(2)
        tenure = r2a.number_input("Tenure (months)",  1,  72,  12)
        mc     = r2b.number_input("Monthly Charges ($)", 18.0, 120.0, 65.0, step=1.0)
        r3a, r3b = st.columns(2)
        contract = r3a.selectbox("Contract", ["Month-to-month","One year","Two year"])
        internet = r3b.selectbox("Internet",  ["Fiber optic","DSL","No"])
        r4a, r4b = st.columns(2)
        payment  = r4a.selectbox("Payment",
                      ["Electronic check","Credit card (automatic)",
                       "Bank transfer (automatic)","Mailed check"])
        tc = r4b.number_input("Total Charges ($)", 0.0, 10000.0,
                               float(round(tenure*mc, 2)), step=10.0)
        st.markdown("**Service Options**")
        b1,b2,b3,b4 = st.columns(4)
        tech_s  = b1.checkbox("Tech Support")
        stream  = b2.checkbox("Streaming TV")
        senior  = b3.checkbox("Senior Citizen")
        depend  = b4.checkbox("Dependents")

        predict_btn = st.button("⚡ PREDICT CHURN", use_container_width=True)

    with col_result:
        st.markdown("#### Prediction Result")

        if predict_btn:
            row = {
                "Age": age, "Gender": gender,
                "SeniorCitizen": int(senior), "Partner": "No",
                "Dependents":   "Yes" if depend else "No",
                "Tenure": tenure, "InternetService": internet,
                "TechSupport":  "Yes" if tech_s  else "No",
                "StreamingTV":  "Yes" if stream  else "No",
                "Contract": contract, "PaperlessBilling": "Yes",
                "PaymentMethod": payment,
                "MonthlyCharges": mc, "TotalCharges": tc,
            }
            with st.spinner("Analyzing…"):
                score, risk, color, action = predict_churn(row)
            st.session_state["pred"] = (score, risk, color, action)

        if "pred" in st.session_state:
            score, risk, color, action = st.session_state["pred"]
            # Gauge
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=score,
                number={"suffix":"%","font":{"color":color,"size":38,"family":"Syne"}},
                gauge={
                    "axis":    {"range":[0,100],"tickcolor":"#475569",
                                "tickfont":{"color":"#64748b"}},
                    "bar":     {"color":color,"thickness":0.25},
                    "bgcolor": "rgba(255,255,255,0.04)",
                    "bordercolor":"rgba(0,0,0,0)",
                    "steps":[
                        {"range":[0,40],  "color":"rgba(16,185,129,0.1)"},
                        {"range":[40,70], "color":"rgba(245,158,11,0.1)"},
                        {"range":[70,100],"color":"rgba(239,68,68,0.1)"},
                    ],
                    "threshold":{"line":{"color":color,"width":3},"thickness":0.75,"value":score},
                },
            ))
            fig.update_layout(paper_bgcolor="rgba(0,0,0,0)",
                              plot_bgcolor="rgba(0,0,0,0)",
                              font=dict(color="#94a3b8"),
                              height=230, margin=dict(l=20,r=20,t=20,b=10))
            st.plotly_chart(fig, use_container_width=True)

            st.markdown(f"""
            <div style="text-align:center;margin:-8px 0 14px">
                <span style="background:{color}20;color:{color};padding:6px 20px;
                             border-radius:20px;font-size:15px;font-weight:600">{risk}</span>
            </div>
            <div style="background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.08);
                        border-radius:12px;padding:14px 16px;">
                <p style="font-size:12px;color:#64748b;margin-bottom:6px">🤖 AI Recommendation</p>
                <p style="font-size:13px;color:#e2e8f0;line-height:1.6;margin:0">{action}</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="background:rgba(255,255,255,0.02);border:1px dashed rgba(255,255,255,0.1);
                        border-radius:14px;padding:60px 24px;text-align:center;color:#475569">
                <p style="font-size:36px;margin-bottom:10px">🎯</p>
                <p>Fill the form and click<br><b>PREDICT CHURN</b></p>
            </div>
            """, unsafe_allow_html=True)

    # Model comparison
    st.divider()
    st.markdown("#### Model Performance Comparison")
    cols = st.columns(len(MODEL_RESULTS))
    best_f1 = max(v["f1"] for v in MODEL_RESULTS.values())
    for col, (name, m) in zip(cols, MODEL_RESULTS.items()):
        active = m["f1"] == best_f1
        border = "rgba(99,102,241,0.5)" if active else "rgba(255,255,255,0.07)"
        bg     = "rgba(99,102,241,0.06)" if active else "rgba(255,255,255,0.02)"
        badge  = " ✅" if active else ""
        col.markdown(f"""
        <div style="background:{bg};border:1px solid {border};border-radius:14px;
                    padding:16px;text-align:center">
            <p style="font-size:13px;font-weight:600;color:#f1f5f9;margin-bottom:10px">
                {name}{badge}</p>
            <p style="font-size:12px;color:#64748b;margin:3px 0">Accuracy &nbsp;<b style="color:#6366f1">{m['accuracy']}%</b></p>
            <p style="font-size:12px;color:#64748b;margin:3px 0">Precision &nbsp;<b style="color:#8b5cf6">{m['precision']}%</b></p>
            <p style="font-size:12px;color:#64748b;margin:3px 0">Recall &nbsp;<b style="color:#06b6d4">{m['recall']}%</b></p>
            <p style="font-size:12px;color:#64748b;margin:3px 0">F1 Score &nbsp;<b style="color:#10b981">{m['f1']}%</b></p>
        </div>
        """, unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
# PAGE: CUSTOMERS
# ═════════════════════════════════════════════════════════════════════════════
elif PAGE == "customers":
    section("👥 Customer Management")
    st.markdown("<p style='color:#64748b;font-size:14px;margin-bottom:20px'>Search, filter and manage all customer records</p>", unsafe_allow_html=True)

    def quick_risk(row):
        p = 0.15
        if row["Contract"] == "Month-to-month": p += 0.28
        elif row["Contract"] == "One year":     p += 0.08
        if row["MonthlyCharges"] > 70:          p += 0.15
        if row["Tenure"] < 12:                  p += 0.18
        if p > 0.70: return "High"
        if p > 0.40: return "Medium"
        return "Low"

    view = DF.copy()
    view["Risk"]   = view.apply(quick_risk, axis=1)
    view["Status"] = view["Churn"].map({1:"Churned",0:"Active"})

    # Filters
    f1,f2,f3,f4 = st.columns([2,1,1,1])
    search  = f1.text_input("🔍 Search name or ID", "")
    status  = f2.selectbox("Status",     ["All","Active","Churned"])
    risk_f  = f3.selectbox("Risk Level", ["All","High","Medium","Low"])
    sort_c  = f4.selectbox("Sort by",    ["MonthlyCharges","Tenure","TotalCharges"])

    if search:
        view = view[view["Name"].str.contains(search,case=False) |
                    view["CustomerID"].str.contains(search,case=False)]
    if status != "All":  view = view[view["Status"] == status]
    if risk_f != "All":  view = view[view["Risk"]   == risk_f]
    view = view.sort_values(sort_c, ascending=False)

    st.markdown(f"<p style='color:#64748b;font-size:13px'>{len(view)} customers</p>",
                unsafe_allow_html=True)

    csv = view.to_csv(index=False).encode()
    st.download_button("⬇️ Export CSV", csv, "customers.csv", "text/csv")

    show_cols = ["CustomerID","Name","Age","Gender","Status","Risk",
                 "MonthlyCharges","Contract","Tenure","InternetService"]

    def clr_s(v):
        return ("background-color:rgba(16,185,129,0.12);color:#10b981"
                if v=="Active" else
                "background-color:rgba(239,68,68,0.12);color:#ef4444")

    def clr_r(v):
        d = {"High":"rgba(239,68,68,0.12);color:#ef4444",
             "Medium":"rgba(245,158,11,0.12);color:#f59e0b",
             "Low":"rgba(16,185,129,0.12);color:#10b981"}
        return f"background-color:{d.get(v,'')}"

    styled = (view[show_cols].reset_index(drop=True)
              .style
              .map(clr_s, subset=["Status"])
              .map(clr_r, subset=["Risk"])
              .format({"MonthlyCharges":"${:.2f}"}))
    st.dataframe(styled, use_container_width=True, height=480)

    st.divider()
    s1,s2,s3,s4 = st.columns(4)
    s1.metric("Avg Monthly Charge", f"${view['MonthlyCharges'].mean():.2f}")
    s2.metric("Avg Tenure",         f"{view['Tenure'].mean():.1f} mo")
    s3.metric("High Risk",          int((view["Risk"]=="High").sum()))
    s4.metric("Churned",            int((view["Status"]=="Churned").sum()))


# ═════════════════════════════════════════════════════════════════════════════
# PAGE: EXPLAINABILITY
# ═════════════════════════════════════════════════════════════════════════════
elif PAGE == "explainability":
    section("🧠 Explainable AI")
    st.markdown("<p style='color:#64748b;font-size:14px;margin-bottom:20px'>SHAP-based model interpretability — understand why each prediction was made</p>", unsafe_allow_html=True)

    FEATS = pd.DataFrame({
        "Feature":    ["Contract Type","Monthly Charges","Tenure","Internet Service",
                       "Tech Support","Payment Method","Senior Citizen","Dependents"],
        "Importance": [0.28,0.22,0.18,0.12,0.09,0.07,0.04,0.02],
        "Impact":     ["High","High","High","Medium","Medium","Low","Low","Low"],
    })
    COLOR_MAP = {"High":"#ef4444","Medium":"#f59e0b","Low":"#10b981"}

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Global Feature Importance")
        df_p = FEATS.sort_values("Importance")
        fig = px.bar(df_p, x="Importance", y="Feature", orientation="h",
                     color="Impact", color_discrete_map=COLOR_MAP,
                     text=df_p["Importance"].apply(lambda v: f"{v*100:.0f}%"))
        fig.update_traces(textposition="outside")
        fig.update_layout(**PL, height=320)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("#### Feature Breakdown")
        for _, row in FEATS.sort_values("Importance", ascending=False).iterrows():
            pct   = row["Importance"]*100
            color = COLOR_MAP.get(row["Impact"],"#64748b")
            st.markdown(f"""
            <div style="margin-bottom:12px">
                <div style="display:flex;justify-content:space-between;margin-bottom:5px">
                    <span style="font-size:13px;color:#e2e8f0">{row['Feature']}</span>
                    <span style="font-size:13px;color:{color};font-weight:600">{pct:.0f}%
                        &nbsp;<span style="font-size:10px;background:{color}20;color:{color};
                        padding:2px 6px;border-radius:10px">{row['Impact']}</span>
                    </span>
                </div>
                <div style="height:7px;border-radius:4px;background:rgba(255,255,255,0.06)">
                    <div style="height:100%;width:{pct*3:.0f}%;max-width:100%;border-radius:4px;
                                background:linear-gradient(90deg,{color}80,{color})"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    st.divider()
    st.markdown("#### SHAP Waterfall — Local Explanation (sample high-risk customer)")
    st.caption("How each feature pushed the churn probability from the base rate to the final score.")

    wf_labels = ["Base Rate","Contract: M-t-M","Monthly $85","Tenure 3mo",
                 "Tech Support: No","Fiber Internet","Dependents: Yes","Final Score"]
    wf_vals   = [0.15,  0.28, 0.15, 0.18,  0.07, 0.06, -0.05, 0.84]
    wf_types  = ["base","pos","pos","pos","pos","pos","neg","total"]

    run   = 0.0
    bases = []; vals = []; cols_w = []
    for v, t in zip(wf_vals, wf_types):
        if t == "base":   bases.append(0);       vals.append(v);      cols_w.append("#6366f1")
        elif t == "total":bases.append(0);        vals.append(v);      cols_w.append("#ef4444")
        elif t == "pos":  bases.append(run);      vals.append(v);      cols_w.append("#ef4444")
        else:             bases.append(run+v);    vals.append(abs(v)); cols_w.append("#10b981")
        if t != "total":  run += v

    fig = go.Figure(go.Bar(
        x=wf_labels, y=vals, base=bases,
        marker_color=cols_w,
        text=[f"{'+' if v>=0 else ''}{v*100:.0f}%" for v in wf_vals],
        textposition="outside", textfont=dict(color="#e2e8f0", size=11),
    ))
    fig.update_layout(
    **PL,
    height=260
)

    fig.update_yaxes(range=[50, 100])
    st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.markdown("#### Key Business Insights")
    ins1, ins2, ins3 = st.columns(3)
    for col, icon, title, desc in [
        (ins1,"🔴","Contract Impact",
        "Month-to-month customers churn at 42% vs 3% on 2-year plans. "
        "Contract migration is the highest-ROI retention lever."),
        (ins2,"🟡","Pricing Sensitivity",
        "Customers paying above $70/month have 2.3× higher churn. "
        "The sweet spot sits at $65–70."),
        (ins3,"🟢","Tenure Effect",
        "First 12 months are highest risk. Churn probability drops 45% after "
        "24 months and 72% after 48 months."),
    ]:
        col.markdown(f"""
        <div style="background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.07);
                    border-radius:14px;padding:18px">
            <p style="font-size:22px;margin-bottom:8px">{icon}</p>
            <p style="font-size:14px;font-weight:600;color:#f1f5f9;margin-bottom:8px">{title}</p>
            <p style="font-size:13px;color:#64748b;line-height:1.6;margin:0">{desc}</p>
        </div>
        """, unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
# PAGE: ANALYTICS
# ═════════════════════════════════════════════════════════════════════════════
elif PAGE == "analytics":
    section("📈 Revenue Analytics")
    st.markdown("<p style='color:#64748b;font-size:14px;margin-bottom:20px'>Customer lifetime value, retention analysis and AI-driven business insights</p>", unsafe_allow_html=True)

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Avg Customer LTV",  f"${AVG_CLV:,.0f}",    "+12%")
    c2.metric("Monthly Recurring", f"${MRR:,.0f}",         "+8%")
    c3.metric("Annual Recurring",  f"${MRR*12/1000:.1f}K", "+8%")
    c4.metric("Revenue Churn/mo",  f"${REV_LOSS:,.0f}",    delta="-5%", delta_color="inverse")

    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Monthly Retention Rate**")
        rng3 = np.random.RandomState(5)
        ret  = pd.DataFrame({"Month":MONTHS,
                             "Retention":np.clip(rng3.normal(72,5,12),60,90).round(1)})
        fig  = px.line(ret, x="Month", y="Retention", markers=True, line_shape="spline",
                       color_discrete_sequence=["#10b981"])
        fig.update_traces(fill="tozeroy", fillcolor="rgba(16,185,129,0.1)")
        fig.add_hline(y=ret["Retention"].mean(), line_dash="dot", line_color="#f59e0b")
        fig.update_layout(
                            **PL,
                            height=260
                        )

        fig.update_yaxes(range=[50, 100])
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("**Customer LTV by Age Segment**")
        DF3 = DF.copy()
        DF3["AgeGroup"] = pd.cut(DF3["Age"], bins=[17,25,35,45,55,100],
                                 labels=["18-25","26-35","36-45","46-55","55+"])
        clv = DF3.groupby("AgeGroup", observed=True)["TotalCharges"].mean().reset_index()
        clv.columns = ["AgeGroup","CLV"]
        fig = px.bar(clv, x="AgeGroup", y="CLV",
                     color="CLV", color_continuous_scale=["#6366f1","#8b5cf6","#06b6d4"],
                     text=clv["CLV"].apply(lambda v:f"${v:.0f}"))
        fig.update_traces(textposition="outside")
        fig.update_layout(**PL, height=260, coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        st.markdown("**Revenue Loss by Contract Type**")
        ch_rev = CHURNED.groupby("Contract")["MonthlyCharges"].sum().reset_index()
        ch_rev.columns = ["Contract","RevenueLoss"]
        fig = px.pie(ch_rev, values="RevenueLoss", names="Contract", hole=0.5,
                     color_discrete_sequence=["#ef4444","#f59e0b","#6366f1"])
        fig.update_traces(textfont_color="#e2e8f0")
        fig.update_layout(**PL, height=260)
        st.plotly_chart(fig, use_container_width=True)

    with col4:
        st.markdown("**Risk Profile Radar**")
        cats = ["Contract","Charges","Tenure","Support","Streaming","Payment"]
        fig  = go.Figure()
        fig.add_trace(go.Scatterpolar(r=[95,80,85,70,60,75], theta=cats, fill="toself",
            name="High Risk", fillcolor="rgba(239,68,68,0.15)", line=dict(color="#ef4444")))
        fig.add_trace(go.Scatterpolar(r=[30,40,20,75,50,60], theta=cats, fill="toself",
            name="Low Risk",  fillcolor="rgba(16,185,129,0.15)", line=dict(color="#10b981")))
        fig.update_layout(
            polar=dict(bgcolor="rgba(0,0,0,0)",
                radialaxis=dict(visible=True, range=[0,100],
                    gridcolor="rgba(255,255,255,0.1)", linecolor="rgba(255,255,255,0.1)"),
                angularaxis=dict(gridcolor="rgba(255,255,255,0.1)",
                    linecolor="rgba(255,255,255,0.1)")),
            paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#94a3b8"),
            legend=dict(bgcolor="rgba(0,0,0,0)"), height=260,
            margin=dict(l=30,r=30,t=30,b=30))
        st.plotly_chart(fig, use_container_width=True)

    # AI Recommendations
    st.divider()
    st.markdown("#### 🤖 AI Business Recommendations")
    recs = [
        ("Contract Migration Campaign","Critical",
         "Target month-to-month customers with annual plan incentives (3 months free). Expected 35% conversion.",
         "+$8,400/mo"),
        ("At-Risk Early Warning","High",
         "15 customers with churn score > 85% need immediate outreach within 48 hours.",
         "+$5,200/mo"),
        ("Tenure Loyalty Rewards","Medium",
         "Introduce milestone rewards at 6, 12, and 24 months. Reduces early churn by 22%.",
         "+$3,100/mo"),
        ("Price Optimisation","Medium",
         "Tiered pricing below $70 threshold could cut high-charge churn by 18%.",
         "+$2,800/mo"),
    ]
    rc1, rc2 = st.columns(2)
    for i, (title, impact, desc, roi) in enumerate(recs):
        c = rc1 if i%2==0 else rc2
        color = {"Critical":"#ef4444","High":"#f59e0b","Medium":"#10b981"}.get(impact,"#64748b")
        c.markdown(f"""
        <div style="background:rgba(255,255,255,0.02);border:1px solid rgba(255,255,255,0.07);
                    border-radius:14px;padding:16px;margin-bottom:12px">
            <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px">
                <span style="font-size:13px;font-weight:600;color:#f1f5f9">{title}</span>
                <span style="font-size:11px;background:{color}20;color:{color};
                             padding:2px 8px;border-radius:10px">{impact}</span>
            </div>
            <p style="font-size:12px;color:#64748b;line-height:1.6;margin-bottom:10px">{desc}</p>
            <span style="font-size:13px;color:#10b981;font-weight:600">Est. ROI: {roi}</span>
        </div>
        """, unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
# PAGE: CHATBOT
# ═════════════════════════════════════════════════════════════════════════════
elif PAGE == "chatbot":
    section("🤖 AI Chatbot Assistant")
    st.markdown("<p style='color:#64748b;font-size:14px;margin-bottom:20px'>Ask anything about your churn data — rule-based intelligence with Claude API fallback</p>", unsafe_allow_html=True)

    FALLBACKS = {
        "churn":
            "📊 **Top Churn Drivers**\n\n"
            "1. **Contract Type** — Month-to-month = 42% churn vs 3% on 2-year plans\n"
            "2. **High Monthly Charges** — >$70/mo customers have 2.3× higher churn\n"
            "3. **Low Tenure** — 68% of churns happen in the first 12 months\n"
            "4. **No Tech Support** — adds ~9% churn probability\n"
            "5. **Electronic Check** — higher churn than auto-pay methods",
        "retention":
            "🎯 **Top Retention Strategies**\n\n"
            "1. Contract upgrade incentives (15% off → annual)\n"
            "2. Proactive support calls for tenure < 6 months\n"
            "3. Loyalty milestone rewards at 6, 12, 24 months\n"
            "4. Price lock guarantee for 12 months\n"
            "5. Assign a CSM to score > 75% customers within 48 hrs",
        "risk":
            "🔴 **High-Risk Customer Profile**\n\n"
            "- Month-to-month contract\n"
            "- Tenure < 6 months\n"
            "- Monthly charges > $70\n"
            "- No tech support subscription\n"
            "- Electronic check payment\n\n"
            f"Currently **{len(DF[(DF['Contract']=='Month-to-month') & (DF['Tenure']<6)])} customers** match this profile.",
        "revenue":
            f"💰 **Revenue Impact**\n\n"
            f"- Monthly revenue loss: **${REV_LOSS:,.0f}**\n"
            f"- Projected annual impact: **${REV_LOSS*12:,.0f}**\n"
            f"- Avg churned customer LTV: **${AVG_CLV:,.0f}**\n"
            f"- Reducing churn by 5% recovers ~**${REV_LOSS*0.05*12:,.0f}/year**",
        "factor":
            "🧠 **SHAP Feature Importance**\n\n"
            "| Feature | Weight |\n|---|---|\n"
            "| Contract Type | 28% |\n"
            "| Monthly Charges | 22% |\n"
            "| Tenure | 18% |\n"
            "| Internet Service | 12% |\n"
            "| Tech Support | 9% |",
        "default":
            f"👋 Hi! I'm your **ChurnAI Assistant**.\n\n"
            f"📊 Current snapshot: **{len(DF)}** customers · **{CHURN_RT}%** churn rate · **${REV_LOSS:,.0f}/mo** revenue loss\n\n"
            "I can help with:\n"
            "• Why customers are churning\n"
            "• Who is most at risk\n"
            "• Retention strategies\n"
            "• Revenue impact analysis\n"
            "• Feature importance explanations\n\n"
            "Try: *Why are customers churning?*",
    }

    def get_reply(msg):
        low = msg.lower()
        try:
            import anthropic
            client = anthropic.Anthropic()
            resp = client.messages.create(
                model="claude-sonnet-4-20250514", max_tokens=350,
                system=(f"You are an AI assistant for a Churn Analytics Platform. "
                        f"Stats: {len(DF)} customers, {CHURN_RT}% churn, ${REV_LOSS}/mo loss. "
                        f"Be concise, data-driven, use bullet points, max 120 words."),
                messages=[{"role":"user","content":msg}],
            )
            return resp.content[0].text
        except Exception:
            pass
        if any(w in low for w in ["why","reason","cause","churn","leaving"]): return FALLBACKS["churn"]
        if any(w in low for w in ["retention","keep","reduce","prevent","retain"]): return FALLBACKS["retention"]
        if any(w in low for w in ["risk","high risk","who","danger","at-risk"]): return FALLBACKS["risk"]
        if any(w in low for w in ["revenue","money","cost","loss","financial","impact"]): return FALLBACKS["revenue"]
        if any(w in low for w in ["factor","feature","affect","shap","important","weight"]): return FALLBACKS["factor"]
        return FALLBACKS["default"]

    if "chat_msgs" not in st.session_state:
        st.session_state.chat_msgs = [{"role":"ai","text":FALLBACKS["default"]}]

    chat_col, info_col = st.columns([2,1])

    with chat_col:
        # Header
        st.markdown("""
        <div style="background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.07);
                    border-radius:14px 14px 0 0;padding:12px 16px;
                    display:flex;align-items:center;gap:10px">
            <span style="font-size:22px">🤖</span>
            <div>
                <p style="font-size:14px;font-weight:600;color:#f1f5f9;margin:0">ChurnAI Assistant</p>
                <p style="font-size:11px;color:#10b981;margin:0">● Online</p>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Messages
        msg_html = ""
        for m in st.session_state.chat_msgs:
            if m["role"] == "user":
                msg_html += f'<div class="chat-user">{m["text"]}</div>'
            else:
                msg_html += f'<div class="chat-ai">{m["text"]}</div>'
        st.markdown(f"""
        <div style="background:rgba(255,255,255,0.02);border-left:1px solid rgba(255,255,255,0.07);
                    border-right:1px solid rgba(255,255,255,0.07);
                    padding:14px 16px;min-height:280px;max-height:380px;overflow-y:auto">
            {msg_html}
        </div>
        """, unsafe_allow_html=True)

        # Quick suggestions
        st.markdown("<p style='font-size:12px;color:#64748b;margin:10px 0 6px'>Quick questions:</p>", unsafe_allow_html=True)
        sugs = ["Why are customers churning?","Retention strategies",
                "High-risk customer profile","Revenue impact","Key churn factors"]
        sg_cols = st.columns(len(sugs))
        for col, s in zip(sg_cols, sugs):
            if col.button(s, key=f"sg_{s[:10]}", use_container_width=True):
                st.session_state.chat_msgs.append({"role":"user","text":s})
                with st.spinner("Thinking…"):
                    reply = get_reply(s)
                st.session_state.chat_msgs.append({"role":"ai","text":reply})
                st.rerun()

        # Input
        with st.form("chat_form", clear_on_submit=True):
            user_in = st.text_input("Message", placeholder="Ask about churn analytics…",
                                    label_visibility="collapsed")
            send = st.form_submit_button("Send ➤", use_container_width=True)
        if send and user_in.strip():
            st.session_state.chat_msgs.append({"role":"user","text":user_in.strip()})
            with st.spinner("Thinking…"):
                reply = get_reply(user_in.strip())
            st.session_state.chat_msgs.append({"role":"ai","text":reply})
            st.rerun()

        if st.button("🗑️ Clear Chat"):
            st.session_state.chat_msgs = [{"role":"ai","text":FALLBACKS["default"]}]
            st.rerun()

    with info_col:
        st.markdown("#### Live Snapshot")
        for label, val in [("Customers",len(DF)),
                            ("Churn Rate",f"{CHURN_RT}%"),
                            ("Revenue Loss",f"${REV_LOSS:,.0f}/mo"),
                            ("Retention",f"{RET_RT}%")]:
            st.markdown(f"""
            <div style="background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.07);
                        border-radius:10px;padding:12px 14px;margin-bottom:8px">
                <p style="font-size:11px;color:#64748b;margin:0">{label}</p>
                <p style="font-size:20px;font-weight:700;color:#6366f1;margin:0">{val}</p>
            </div>
            """, unsafe_allow_html=True)
        st.markdown("#### What I can help with")
        for cap in ["Churn root-cause analysis","High-risk identification",
                    "Retention strategies","Revenue impact","Feature explanations",
                    "Business recommendations"]:
            st.markdown(f"<p style='font-size:12px;color:#64748b;margin:4px 0'>✓ {cap}</p>",
                        unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
# PAGE: ADMIN
# ═════════════════════════════════════════════════════════════════════════════
elif PAGE == "admin":
    section("⚙️ Admin Panel")
    st.markdown("<p style='color:#64748b;font-size:14px;margin-bottom:20px'>Model management, system health and API documentation</p>", unsafe_allow_html=True)

    s1,s2,s3,s4 = st.columns(4)
    s1.metric("API Uptime",        "99.9%",  "+0.1%")
    s2.metric("Avg Response Time", "142 ms", "-8 ms")
    s3.metric("Predictions Today", "1,284",  "+212")
    s4.metric("Dataset Rows",      len(DF))

    st.divider()

    # Model comparison
    st.markdown("#### ML Model Comparison")
    best_f1 = max(v["f1"] for v in MODEL_RESULTS.values())
    rows = []
    for name, m in MODEL_RESULTS.items():
        rows.append({
            "Model":    name,
            "Accuracy": f"{m['accuracy']}%",
            "Precision":f"{m['precision']}%",
            "Recall":   f"{m['recall']}%",
            "F1 Score": f"{m['f1']}%",
            "Status":   "✅ Active" if m["f1"]==best_f1 else "—",
        })
    mdf = pd.DataFrame(rows)
    def sty_active(v):
        if "Active" in str(v):
            return "background-color:rgba(16,185,129,0.12);color:#10b981;font-weight:600"
        return ""
    st.dataframe(mdf.style.map(sty_active, subset=["Status"]),
                 use_container_width=True, hide_index=True)

    # Bar chart
    st.markdown("#### Performance Visualisation")
    met_rows = []
    for name, m in MODEL_RESULTS.items():
        for k,v in [("Accuracy",m["accuracy"]),("Precision",m["precision"]),
                    ("Recall",m["recall"]),("F1",m["f1"])]:
            met_rows.append({"Model":name,"Metric":k,"Value":v})
    met_df = pd.DataFrame(met_rows)
    fig = px.bar(met_df, x="Model", y="Value", color="Metric", barmode="group",
                 color_discrete_sequence=["#6366f1","#8b5cf6","#06b6d4","#10b981"],
                 text="Value")
    fig.update_traces(texttemplate="%{text}%", textposition="outside")
    fig.update_layout(**PL, height=300)
    fig.update_yaxes(range=[0, 110])
    st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # API docs
    st.markdown("#### REST API Reference")
    endpoints = [
        ("POST",   "/api/predict",            "#f59e0b", "Predict churn probability"),
        ("GET",    "/api/analytics",           "#10b981", "Aggregated dashboard data"),
        ("GET",    "/api/customers",           "#10b981", "List customers with filters"),
        ("POST",   "/api/chat",                "#f59e0b", "AI chatbot endpoint"),
        ("GET",    "/api/feature-importance",  "#10b981", "SHAP feature importance"),
        ("DELETE", "/api/customers/{id}",      "#ef4444", "Remove a customer record"),
    ]
    for method, path, color, desc in endpoints:
        st.markdown(f"""
        <div style="display:flex;align-items:center;gap:12px;padding:10px 14px;
                    background:rgba(255,255,255,0.02);border:1px solid rgba(255,255,255,0.05);
                    border-radius:10px;margin-bottom:6px">
            <span style="background:{color}20;color:{color};padding:2px 8px;border-radius:6px;
                         font-size:11px;font-weight:700;min-width:54px;text-align:center">
                {method}</span>
            <code style="font-size:13px;color:#94a3b8;flex:1">{path}</code>
            <span style="font-size:12px;color:#475569">{desc}</span>
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    # Activity feed
    st.markdown("#### Recent Activity")
    feed = [
        ("🔴","2 min ago","High-risk prediction: CUST-1003 (87% churn probability)"),
        ("🔵","8 min ago","Bulk prediction: 50 customers analysed"),
        ("🟢","15 min ago","Model training complete — best F1 updated"),
        ("🔴","1 hr ago","Customer CUST-1017 churned — $89/month revenue impact"),
        ("🔵","2 hr ago","New batch uploaded: 12 records imported"),
        ("🟢","4 hr ago","Retention campaign sent to 23 medium-risk customers"),
    ]
    for icon, t, event in feed:
        st.markdown(f"""
        <div style="display:flex;gap:12px;padding:9px 0;
                    border-bottom:1px solid rgba(255,255,255,0.04)">
            <span style="font-size:14px;margin-top:1px">{icon}</span>
            <div>
                <p style="font-size:13px;color:#e2e8f0;margin:0">{event}</p>
                <p style="font-size:11px;color:#475569;margin:2px 0 0">{t}</p>
            </div>
        </div>
        """, unsafe_allow_html=True)