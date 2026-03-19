import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

st.set_page_config(page_title="서울시 대기질 정책 분석", layout="wide")

st.title("🌫️ 서울시 대기질 개선 정책 효과 분석 대시보드")

# ---------------------------
# 1. 데이터 생성
# ---------------------------
data = {
    "year": [2025]*10 + [2026]*10,
    "month": list(range(1, 11)) * 2,
    "pm25": [42,39,45,38,35,33,30,28,31,36,
             30,28,32,27,25,23,22,21,24,26],
    "respiratory": [310,295,320,280,260,250,240,235,245,270,
                    240,225,250,220,205,195,190,185,200,215]
}

df = pd.DataFrame(data)

# ---------------------------
# 2. KPI 계산
# ---------------------------
avg_2025_pm = df[df.year == 2025]["pm25"].mean()
avg_2026_pm = df[df.year == 2026]["pm25"].mean()

avg_2025_resp = df[df.year == 2025]["respiratory"].mean()
avg_2026_resp = df[df.year == 2026]["respiratory"].mean()

pm_reduction = (avg_2025_pm - avg_2026_pm) / avg_2025_pm * 100
resp_reduction = (avg_2025_resp - avg_2026_resp) / avg_2025_resp * 100

corr = df["pm25"].corr(df["respiratory"])

t_pm, p_pm = stats.ttest_ind(
    df[df.year == 2025]["pm25"],
    df[df.year == 2026]["pm25"]
)

t_resp, p_resp = stats.ttest_ind(
    df[df.year == 2025]["respiratory"],
    df[df.year == 2026]["respiratory"]
)

# ---------------------------
# 3. KPI 출력
# ---------------------------
st.subheader("📊 주요 성과 지표")

col1, col2, col3 = st.columns(3)

col1.metric("PM2.5 감소율", f"{pm_reduction:.1f}%")
col2.metric("호흡기 질환 감소율", f"{resp_reduction:.1f}%")
col3.metric("상관계수 (r)", f"{corr:.2f}")

st.write(f"PM2.5 p-value: {p_pm:.4f}")
st.write(f"질환 p-value: {p_resp:.4f}")

# ---------------------------
# 4. 시계열 시각화
# ---------------------------
st.subheader("📈 월별 추이 비교")

fig, ax = plt.subplots()

for year, color in zip([2025, 2026], ["orange", "blue"]):
    temp = df[df.year == year]
    ax.plot(temp["month"], temp["pm25"], marker='o', label=f"{year} PM2.5", linestyle='-')
    ax.plot(temp["month"], temp["respiratory"], marker='x', label=f"{year} 질환", linestyle='--')

ax.set_xlabel("Month")
ax.set_ylabel("Value")
ax.legend()

st.pyplot(fig)

# ---------------------------
# 5. 상관관계 시각화
# ---------------------------
st.subheader("🔗 PM2.5 vs 호흡기 질환 상관관계")

fig2, ax2 = plt.subplots()

ax2.scatter(df["pm25"], df["respiratory"])

# 회귀선
m, b = np.polyfit(df["pm25"], df["respiratory"], 1)
ax2.plot(df["pm25"], m*df["pm25"] + b)

ax2.set_xlabel("PM2.5")
ax2.set_ylabel("Respiratory Cases")

st.pyplot(fig2)

# ---------------------------
# 6. 해석 요약
# ---------------------------
st.subheader("🧠 해석")

st.markdown(f"""
- PM2.5는 약 **{pm_reduction:.1f}% 감소**
- 호흡기 질환은 약 **{resp_reduction:.1f}% 감소**
- 상관계수 **{corr:.2f}** → 매우 강한 양의 관계
- p-value < 0.001 수준 → 통계적으로 유의미

👉 **환경 정책이 실제 건강 개선으로 이어졌음을 시사**
""")
