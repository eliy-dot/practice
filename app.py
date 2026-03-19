import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
import statsmodels.api as sm

st.set_page_config(page_title="Air Quality Policy Dashboard", layout="wide")

st.title("🌫️ Air Quality Policy Impact Dashboard")

# ---------------------------
# 1. 데이터 로드 (실무 스타일)
# ---------------------------
@st.cache_data
def load_data():
    data = {
        "year": [2025]*10 + [2026]*10,
        "month": list(range(1, 11)) * 2,
        "pm25": [42,39,45,38,35,33,30,28,31,36,
                 30,28,32,27,25,23,22,21,24,26],
        "respiratory": [310,295,320,280,260,250,240,235,245,270,
                        240,225,250,220,205,195,190,185,200,215]
    }
    df = pd.DataFrame(data)
    df["date"] = pd.to_datetime(df["year"].astype(str) + "-" + df["month"].astype(str))
    df = df.sort_values("date")
    return df

df = load_data()

# ---------------------------
# 2. KPI 계산
# ---------------------------
def calculate_kpi(df):
    result = {}

    for col in ["pm25", "respiratory"]:
        pre = df[df.year == 2025][col].mean()
        post = df[df.year == 2026][col].mean()
        result[col] = {
            "pre": pre,
            "post": post,
            "reduction": (pre - post) / pre * 100
        }

    corr = df["pm25"].corr(df["respiratory"])

    t_pm, p_pm = stats.ttest_ind(
        df[df.year == 2025]["pm25"],
        df[df.year == 2026]["pm25"]
    )

    return result, corr, p_pm

kpi, corr, p_pm = calculate_kpi(df)

# ---------------------------
# 3. KPI UI
# ---------------------------
st.subheader("📊 KPI Overview")

col1, col2, col3 = st.columns(3)

col1.metric("PM2.5 감소율", f"{kpi['pm25']['reduction']:.1f}%")
col2.metric("질환 감소율", f"{kpi['respiratory']['reduction']:.1f}%")
col3.metric("상관계수", f"{corr:.2f}")

st.caption(f"PM2.5 p-value: {p_pm:.5f}")

# ---------------------------
# 4. 인터랙티브 시계열
# ---------------------------
st.subheader("📈 Time Series Analysis")

fig = px.line(
    df,
    x="date",
    y=["pm25", "respiratory"],
    color="year",
    markers=True,
    title="PM2.5 vs Respiratory Cases",
)

fig.update_layout(hovermode="x unified")

st.plotly_chart(fig, use_container_width=True)

# ---------------------------
# 5. 상관관계 (클릭/hover 가능)
# ---------------------------
st.subheader("🔗 Correlation Analysis")

fig2 = px.scatter(
    df,
    x="pm25",
    y="respiratory",
    color="year",
    trendline="ols",  # 회귀선 자동 추가
    title="PM2.5 vs Respiratory Cases"
)

st.plotly_chart(fig2, use_container_width=True)

# ---------------------------
# 6. Lag 분석 (핵심🔥)
# ---------------------------
st.subheader("⏱️ Lag Analysis (시차 효과)")

lag_range = st.slider("Lag (개월)", 0, 3, 1)

df["pm25_lag"] = df["pm25"].shift(lag_range)

lag_corr = df[["pm25_lag", "respiratory"]].corr().iloc[0,1]

st.write(f"📌 Lag {lag_range}개월 상관계수: {lag_corr:.2f}")

fig3 = px.scatter(
    df,
    x="pm25_lag",
    y="respiratory",
    title=f"Lag {lag_range} Correlation"
)

st.plotly_chart(fig3, use_container_width=True)

# ---------------------------
# 7. 회귀 분석 (실무 포인트🔥)
# ---------------------------
st.subheader("📉 Regression Analysis")

X = df["pm25"]
X = sm.add_constant(X)
y = df["respiratory"]

model = sm.OLS(y, X).fit()

st.text(model.summary())

# ---------------------------
# 8. 정책 효과 시뮬레이션
# ---------------------------
st.subheader("🧪 Policy Simulation")

pm_input = st.slider("가정 PM2.5 수준", 20, 50, 30)

pred = model.predict([1, pm_input])[0]

st.write(f"예상 호흡기 질환 발생률: {pred:.1f}")

# ---------------------------
# 9. 해석
# ---------------------------
st.subheader("🧠 Insight")

st.markdown(f"""
- 정책 이후 PM2.5는 **{kpi['pm25']['reduction']:.1f}% 감소**
- 질환 발생률은 **{kpi['respiratory']['reduction']:.1f}% 감소**
- 상관계수 **{corr:.2f} → 매우 강한 관계**
- Lag 분석을 통해 시차 효과 확인 가능
- 회귀 모델 기반 정책 효과 시뮬레이션 가능

👉 **환경 정책 → 건강 개선으로 이어지는 데이터 기반 근거 확보**
""")
