import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from scipy import stats
import statsmodels.api as sm

st.set_page_config(page_title="대기질 정책 분석 대시보드", layout="wide")

# ---------------------------
# 1. 데이터 로드
# ---------------------------
@st.cache_data
def load_data():
    data = {
        "연도": [2025]*10 + [2026]*10,
        "월": list(range(1, 11)) * 2,
        "PM25": [42,39,45,38,35,33,30,28,31,36,
                 30,28,32,27,25,23,22,21,24,26],
        "호흡기질환": [310,295,320,280,260,250,240,235,245,270,
                       240,225,250,220,205,195,190,185,200,215]
    }
    df = pd.DataFrame(data)
    df["날짜"] = pd.to_datetime(df["연도"].astype(str) + "-" + df["월"].astype(str))
    return df.sort_values("날짜")

df = load_data()

# ---------------------------
# 2. 사이드바
# ---------------------------
st.sidebar.title("⚙️ 분석 설정")

selected_year = st.sidebar.multiselect(
    "연도 선택",
    options=df["연도"].unique(),
    default=df["연도"].unique()
)

lag = st.sidebar.slider("시차(Lag, 개월)", 0, 3, 1)

show_regression = st.sidebar.checkbox("회귀 분석 보기", True)

filtered_df = df[df["연도"].isin(selected_year)].copy()

# lag 변수 생성
filtered_df["PM25_LAG"] = filtered_df["PM25"].shift(lag)

# ---------------------------
# 3. KPI 계산
# ---------------------------
def calculate_kpi(df):
    result = {}

    for col in ["PM25", "호흡기질환"]:
        pre = df[df["연도"] == 2025][col].mean()
        post = df[df["연도"] == 2026][col].mean()
        result[col] = {
            "감소율": (pre - post) / pre * 100
        }

    corr = df["PM25"].corr(df["호흡기질환"])
    return result, corr

kpi, corr = calculate_kpi(filtered_df)

# ---------------------------
# 4. 제목
# ---------------------------
st.title("🌫️ 서울시 대기질 정책 효과 분석 대시보드")

# ---------------------------
# 5. KPI
# ---------------------------
st.subheader("📊 핵심 지표")

col1, col2, col3 = st.columns(3)

col1.metric("PM2.5 감소율", f"{kpi['PM25']['감소율']:.1f}%")
col2.metric("호흡기 질환 감소율", f"{kpi['호흡기질환']['감소율']:.1f}%")
col3.metric("상관계수", f"{corr:.2f}")

# ---------------------------
# 6. 시계열 (인터랙티브)
# ---------------------------
st.subheader("📈 시계열 분석")

fig = px.line(
    filtered_df,
    x="날짜",
    y=["PM25", "호흡기질환"],
    color="연도",
    markers=True
)

fig.update_layout(hovermode="x unified")

st.plotly_chart(fig, use_container_width=True)

# ---------------------------
# 7. 상관관계
# ---------------------------
st.subheader("🔗 상관관계 분석")

fig2 = px.scatter(
    filtered_df,
    x="PM25",
    y="호흡기질환",
    color="연도",
    trendline="ols"
)

st.plotly_chart(fig2, use_container_width=True)

# ---------------------------
# 8. Lag 분석
# ---------------------------
st.subheader("⏱️ 시차(Lag) 분석")

lag_corr = filtered_df[["PM25_LAG", "호흡기질환"]].corr().iloc[0,1]

st.write(f"👉 Lag {lag}개월 상관계수: {lag_corr:.2f}")

fig3 = px.scatter(
    filtered_df,
    x="PM25_LAG",
    y="호흡기질환",
    title=f"Lag {lag} 분석"
)

st.plotly_chart(fig3, use_container_width=True)

# ---------------------------
# 9. 회귀 분석
# ---------------------------
if show_regression:
    st.subheader("📉 회귀 분석")

    reg_df = filtered_df.dropna()

    X = sm.add_constant(reg_df["PM25"])
    y = reg_df["호흡기질환"]

    model = sm.OLS(y, X).fit()

    st.text(model.summary())

# ---------------------------
# 10. 정책 시뮬레이션
# ---------------------------
st.subheader("🧪 정책 시뮬레이션")

pm_input = st.slider("가정 PM2.5 수준", 20, 50, 30)

reg_df = filtered_df.dropna()
X = sm.add_constant(reg_df["PM25"])
y = reg_df["호흡기질환"]
model = sm.OLS(y, X).fit()

pred = model.predict([1, pm_input])[0]

st.write(f"👉 예상 호흡기 질환 발생률: {pred:.1f}")

# ---------------------------
# 11. 인사이트
# ---------------------------
st.subheader("🧠 인사이트")

st.markdown(f"""
- PM2.5 감소 → 질환 감소 확인
- 상관계수 **{corr:.2f}** → 매우 강한 관계
- 시차 분석으로 정책 효과의 시간 지연 확인 가능
- 회귀 기반으로 정책 효과 예측 가능

👉 **환경 정책이 실제 건강 개선으로 이어짐**
""")
