import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
import statsmodels.api as sm
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────
# 페이지 설정
# ─────────────────────────────────────────
st.set_page_config(
    page_title="서울시 대기질 정책 분석",
    page_icon="🌫️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────
# 커스텀 CSS (고급 스타일링)
# ─────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@300;400;500;700;900&family=Space+Mono:wght@400;700&display=swap');

/* 전체 배경 */
.stApp {
    background: linear-gradient(135deg, #0a0e1a 0%, #0d1b2a 40%, #0a1628 100%);
    font-family: 'Noto Sans KR', sans-serif;
}

/* 사이드바 */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1b2a 0%, #0a1220 100%);
    border-right: 1px solid rgba(0, 212, 255, 0.15);
}
section[data-testid="stSidebar"] .stMarkdown p {
    color: #8fa8c0;
}

/* 히어로 헤더 */
.hero-header {
    background: linear-gradient(135deg, rgba(0,212,255,0.08) 0%, rgba(0,128,255,0.05) 50%, rgba(100,0,255,0.08) 100%);
    border: 1px solid rgba(0, 212, 255, 0.2);
    border-radius: 20px;
    padding: 40px 50px;
    margin-bottom: 32px;
    position: relative;
    overflow: hidden;
}
.hero-header::before {
    content: '';
    position: absolute;
    top: -50%;
    left: -50%;
    width: 200%;
    height: 200%;
    background: radial-gradient(circle at 30% 50%, rgba(0,212,255,0.05) 0%, transparent 60%);
    pointer-events: none;
}
.hero-title {
    font-size: 2.4rem;
    font-weight: 900;
    background: linear-gradient(90deg, #00d4ff, #0080ff, #a855f7);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0 0 8px 0;
    letter-spacing: -0.5px;
}
.hero-subtitle {
    color: #8fa8c0;
    font-size: 0.95rem;
    font-weight: 300;
    letter-spacing: 2px;
    text-transform: uppercase;
}
.hero-badge {
    display: inline-block;
    background: rgba(0, 212, 255, 0.1);
    border: 1px solid rgba(0, 212, 255, 0.3);
    color: #00d4ff;
    padding: 4px 14px;
    border-radius: 20px;
    font-size: 0.75rem;
    font-weight: 700;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    margin-bottom: 16px;
}

/* KPI 카드 */
.kpi-card {
    background: linear-gradient(135deg, rgba(255,255,255,0.04) 0%, rgba(255,255,255,0.02) 100%);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 16px;
    padding: 24px 28px;
    position: relative;
    overflow: hidden;
    transition: all 0.3s ease;
}
.kpi-card::after {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 2px;
    border-radius: 16px 16px 0 0;
}
.kpi-card.blue::after { background: linear-gradient(90deg, #00d4ff, #0080ff); }
.kpi-card.green::after { background: linear-gradient(90deg, #00ff88, #00cc66); }
.kpi-card.purple::after { background: linear-gradient(90deg, #a855f7, #7c3aed); }
.kpi-card.orange::after { background: linear-gradient(90deg, #f97316, #ea580c); }

.kpi-label {
    color: #6b8aaa;
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 2px;
    text-transform: uppercase;
    margin-bottom: 10px;
}
.kpi-value {
    font-size: 2.6rem;
    font-weight: 900;
    line-height: 1;
    margin-bottom: 6px;
    font-family: 'Space Mono', monospace;
}
.kpi-value.blue { color: #00d4ff; }
.kpi-value.green { color: #00ff88; }
.kpi-value.purple { color: #c084fc; }
.kpi-value.orange { color: #fb923c; }

.kpi-delta {
    font-size: 0.78rem;
    color: #4ade80;
    font-weight: 600;
}
.kpi-desc {
    font-size: 0.72rem;
    color: #4a6680;
    margin-top: 6px;
}

/* 섹션 헤더 */
.section-header {
    display: flex;
    align-items: center;
    gap: 12px;
    margin: 36px 0 20px 0;
}
.section-icon {
    width: 36px;
    height: 36px;
    border-radius: 10px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.1rem;
}
.section-title {
    font-size: 1.15rem;
    font-weight: 700;
    color: #e2eaf5;
    letter-spacing: -0.3px;
}
.section-line {
    flex: 1;
    height: 1px;
    background: linear-gradient(90deg, rgba(0,212,255,0.3), transparent);
    margin-left: 8px;
}

/* 인사이트 카드 */
.insight-card {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.07);
    border-left: 3px solid #00d4ff;
    border-radius: 0 12px 12px 0;
    padding: 16px 20px;
    margin-bottom: 12px;
}
.insight-card.warning { border-left-color: #f97316; }
.insight-card.success { border-left-color: #00ff88; }
.insight-card.info { border-left-color: #a855f7; }

.insight-title {
    font-size: 0.8rem;
    font-weight: 700;
    color: #00d4ff;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 6px;
}
.insight-card.warning .insight-title { color: #f97316; }
.insight-card.success .insight-title { color: #00ff88; }
.insight-card.info .insight-title { color: #c084fc; }
.insight-text {
    font-size: 0.88rem;
    color: #9ab0c6;
    line-height: 1.6;
}

/* 탭 스타일 */
.stTabs [data-baseweb="tab-list"] {
    background: rgba(255,255,255,0.03);
    border-radius: 12px;
    padding: 4px;
    border: 1px solid rgba(255,255,255,0.08);
    gap: 4px;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px;
    color: #6b8aaa !important;
    font-weight: 600;
    font-size: 0.85rem;
    padding: 8px 18px;
    transition: all 0.2s;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, rgba(0,212,255,0.15), rgba(0,128,255,0.1)) !important;
    color: #00d4ff !important;
    border: 1px solid rgba(0,212,255,0.3) !important;
}

/* 슬라이더 */
.stSlider [data-baseweb="slider"] {
    padding: 0;
}

/* 메트릭 기본값 숨김 후 커스텀 사용 */
[data-testid="stMetric"] {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 12px;
    padding: 16px;
}
[data-testid="stMetricLabel"] { color: #6b8aaa !important; font-size: 0.75rem !important; }
[data-testid="stMetricValue"] { color: #e2eaf5 !important; }
[data-testid="stMetricDelta"] { font-size: 0.8rem !important; }

/* 데이터프레임 */
.stDataFrame { border-radius: 12px; overflow: hidden; }

/* 구분선 */
hr { border-color: rgba(255,255,255,0.07) !important; }

/* 사이드바 텍스트 */
.sidebar-section {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 12px;
    padding: 16px;
    margin-bottom: 16px;
}
.sidebar-section-title {
    font-size: 0.7rem;
    font-weight: 700;
    color: #4a6680;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    margin-bottom: 12px;
}

/* 경고창 */
.stAlert { border-radius: 12px !important; }

/* 버튼 */
.stButton button {
    background: linear-gradient(135deg, rgba(0,212,255,0.15), rgba(0,128,255,0.1));
    border: 1px solid rgba(0,212,255,0.3);
    color: #00d4ff;
    border-radius: 10px;
    font-weight: 700;
    letter-spacing: 0.5px;
    transition: all 0.2s;
}
.stButton button:hover {
    background: linear-gradient(135deg, rgba(0,212,255,0.25), rgba(0,128,255,0.2));
    border-color: rgba(0,212,255,0.6);
    transform: translateY(-1px);
}

/* 체크박스, 셀렉트박스 */
.stMultiSelect [data-baseweb="select"] {
    background: rgba(255,255,255,0.04) !important;
    border-color: rgba(255,255,255,0.12) !important;
    border-radius: 10px !important;
}

/* 정책 제언 테이블 */
.policy-table {
    width: 100%;
    border-collapse: separate;
    border-spacing: 0 8px;
}
.policy-table th {
    color: #4a6680;
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    padding: 8px 16px;
    text-align: left;
}
.policy-table td {
    background: rgba(255,255,255,0.03);
    border-top: 1px solid rgba(255,255,255,0.06);
    border-bottom: 1px solid rgba(255,255,255,0.06);
    color: #9ab0c6;
    padding: 14px 16px;
    font-size: 0.85rem;
}
.policy-table td:first-child {
    border-left: 1px solid rgba(255,255,255,0.06);
    border-radius: 10px 0 0 10px;
}
.policy-table td:last-child {
    border-right: 1px solid rgba(255,255,255,0.06);
    border-radius: 0 10px 10px 0;
}
.stage-badge {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 20px;
    font-size: 0.72rem;
    font-weight: 700;
}
.stage-1 { background: rgba(0,212,255,0.15); color: #00d4ff; border: 1px solid rgba(0,212,255,0.3); }
.stage-2 { background: rgba(168,85,247,0.15); color: #c084fc; border: 1px solid rgba(168,85,247,0.3); }
.stage-3 { background: rgba(0,255,136,0.15); color: #00ff88; border: 1px solid rgba(0,255,136,0.3); }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────
# Plotly 공통 레이아웃 테마
# ─────────────────────────────────────────
CHART_THEME = dict(
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    font=dict(family='Noto Sans KR, sans-serif', color='#8fa8c0', size=12),
    xaxis=dict(
        gridcolor='rgba(255,255,255,0.05)',
        linecolor='rgba(255,255,255,0.1)',
        tickcolor='rgba(255,255,255,0.1)',
        zerolinecolor='rgba(255,255,255,0.05)',
    ),
    yaxis=dict(
        gridcolor='rgba(255,255,255,0.05)',
        linecolor='rgba(255,255,255,0.1)',
        tickcolor='rgba(255,255,255,0.1)',
        zerolinecolor='rgba(255,255,255,0.05)',
    ),
    legend=dict(
        bgcolor='rgba(255,255,255,0.04)',
        bordercolor='rgba(255,255,255,0.1)',
        borderwidth=1,
        font=dict(color='#8fa8c0', size=11)
    ),
    margin=dict(t=40, b=40, l=10, r=10),
)

COLOR_2025 = '#f97316'
COLOR_2026 = '#00d4ff'
COLOR_ACCENT = '#a855f7'

# ─────────────────────────────────────────
# 데이터 로드
# ─────────────────────────────────────────
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
    df["월명"] = df["월"].map({1:"1월",2:"2월",3:"3월",4:"4월",5:"5월",
                               6:"6월",7:"7월",8:"8월",9:"9월",10:"10월"})
    return df.sort_values("날짜").reset_index(drop=True)

df = load_data()

# ─────────────────────────────────────────
# 사이드바
# ─────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 20px 0 24px 0;'>
        <div style='font-size:2.4rem; margin-bottom:8px;'>🌫️</div>
        <div style='font-size:0.7rem; font-weight:700; letter-spacing:3px; color:#4a6680; text-transform:uppercase;'>Seoul Air Quality</div>
        <div style='font-size:1rem; font-weight:700; color:#e2eaf5; margin-top:4px;'>정책 분석 대시보드</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sidebar-section-title">📅 분석 기간</div>', unsafe_allow_html=True)
    selected_year = st.multiselect(
        "연도 선택",
        options=sorted(df["연도"].unique()),
        default=sorted(df["연도"].unique()),
        label_visibility="collapsed"
    )

    st.markdown("---")
    st.markdown('<div class="sidebar-section-title">⚙️ 분석 옵션</div>', unsafe_allow_html=True)

    lag = st.slider("시차(Lag) 개월", 0, 3, 1,
                    help="미세먼지 농도와 질환 발생 사이의 시간 지연 설정")

    show_regression = st.checkbox("회귀 분석 결과 표시", value=True)
    show_ci = st.checkbox("신뢰구간(95%) 표시", value=True)
    show_anomaly = st.checkbox("이상값 하이라이트", value=True)

    st.markdown("---")
    st.markdown('<div class="sidebar-section-title">🎨 차트 설정</div>', unsafe_allow_html=True)
    chart_height = st.slider("차트 높이", 300, 600, 420, step=20)

    st.markdown("---")
    st.markdown("""
    <div style='text-align:center; padding-top:8px;'>
        <div style='font-size:0.68rem; color:#2a3a4a; letter-spacing:1px;'>서울특별시 정책본부</div>
        <div style='font-size:0.68rem; color:#2a3a4a;'>환경보건팀 · 2026.03</div>
    </div>
    """, unsafe_allow_html=True)

if not selected_year:
    st.warning("연도를 하나 이상 선택하세요.")
    st.stop()

# ─────────────────────────────────────────
# 데이터 필터링 및 파생 변수
# ─────────────────────────────────────────
filtered_df = df[df["연도"].isin(selected_year)].copy()
filtered_df["PM25_LAG"] = filtered_df["PM25"].shift(lag)

df_2025 = df[df["연도"] == 2025]
df_2026 = df[df["연도"] == 2026]

# ─────────────────────────────────────────
# KPI 계산
# ─────────────────────────────────────────
pm25_2025 = df_2025["PM25"].mean()
pm25_2026 = df_2026["PM25"].mean() if 2026 in selected_year else None
pm25_reduce = (pm25_2025 - pm25_2026) / pm25_2025 * 100 if pm25_2026 else None

health_2025 = df_2025["호흡기질환"].mean()
health_2026 = df_2026["호흡기질환"].mean() if 2026 in selected_year else None
health_reduce = (health_2025 - health_2026) / health_2025 * 100 if health_2026 else None

corr = filtered_df["PM25"].corr(filtered_df["호흡기질환"])

t_stat, p_val = stats.ttest_ind(df_2025["PM25"], df_2026["PM25"]) if (len(selected_year) == 2) else (None, None)

# ─────────────────────────────────────────
# 히어로 헤더
# ─────────────────────────────────────────
st.markdown(f"""
<div class="hero-header">
    <div class="hero-badge">Policy Analytics · Real-time Dashboard</div>
    <div class="hero-title">서울시 대기질 정책 효과 분석</div>
    <div class="hero-subtitle">PM2.5 저감 → 호흡기 질환 발생률 변화 정량 분석 · 2025–2026</div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────
# KPI 카드 (커스텀 HTML)
# ─────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)

with col1:
    val = f"{pm25_reduce:.1f}%" if pm25_reduce else "N/A"
    st.markdown(f"""
    <div class="kpi-card blue">
        <div class="kpi-label">PM2.5 평균 감소율</div>
        <div class="kpi-value blue">{val}</div>
        <div class="kpi-delta">↓ {pm25_2025:.1f} → {pm25_2026:.1f} μg/m³</div>
        <div class="kpi-desc">정책 시행 전후 연평균 비교</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    val2 = f"{health_reduce:.1f}%" if health_reduce else "N/A"
    st.markdown(f"""
    <div class="kpi-card green">
        <div class="kpi-label">호흡기 질환 감소율</div>
        <div class="kpi-value green">{val2}</div>
        <div class="kpi-delta">↓ {health_2025:.0f} → {health_2026:.0f} 명/10만</div>
        <div class="kpi-desc">월평균 발생률 기준</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="kpi-card purple">
        <div class="kpi-label">상관계수 (Pearson r)</div>
        <div class="kpi-value purple">{corr:.3f}</div>
        <div class="kpi-delta">{'매우 강한 양의 상관' if corr > 0.9 else '강한 양의 상관'}</div>
        <div class="kpi-desc">R² = {corr**2:.3f}</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    pval_str = f"{p_val:.4f}" if p_val is not None else "—"
    sig_str = "통계적 유의 (p<0.001)" if (p_val is not None and p_val < 0.001) else ("유의" if p_val is not None and p_val < 0.05 else "—")
    st.markdown(f"""
    <div class="kpi-card orange">
        <div class="kpi-label">T-Test p-value</div>
        <div class="kpi-value orange">{pval_str}</div>
        <div class="kpi-delta">✓ {sig_str}</div>
        <div class="kpi-desc">독립표본 t-검정 (PM2.5)</div>
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────
# 탭 레이아웃
# ─────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈 시계열 분석",
    "🔗 상관 & 회귀",
    "⏱️ 시차(Lag) 분석",
    "🧪 정책 시뮬레이션",
    "📋 종합 보고"
])

# ══════════════════════════════════════════
# TAB 1: 시계열 분석
# ══════════════════════════════════════════
with tab1:
    st.markdown("""
    <div class="section-header">
        <div class="section-title">월별 PM2.5 농도 추이 비교</div>
        <div class="section-line"></div>
    </div>
    """, unsafe_allow_html=True)

    # 이중 축 시계열
    fig_ts = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        row_heights=[0.55, 0.45],
        vertical_spacing=0.06,
        subplot_titles=["PM2.5 농도 (μg/m³)", "호흡기 질환 발생률 (명/10만명)"]
    )

    colors = {2025: COLOR_2025, 2026: COLOR_2026}
    for yr in sorted(selected_year):
        sub = filtered_df[filtered_df["연도"] == yr]

        # PM25 라인
        fig_ts.add_trace(go.Scatter(
            x=sub["날짜"], y=sub["PM25"],
            name=f"{yr}년 PM2.5",
            line=dict(color=colors[yr], width=2.5),
            mode='lines+markers',
            marker=dict(size=7, symbol='circle',
                        line=dict(color='rgba(0,0,0,0.5)', width=1.5)),
            hovertemplate=f"<b>{yr}년</b><br>날짜: %{{x|%Y년 %m월}}<br>PM2.5: %{{y}} μg/m³<extra></extra>"
        ), row=1, col=1)

        # CI 영역 (show_ci)
        if show_ci:
            std = sub["PM25"].std()
            fig_ts.add_trace(go.Scatter(
                x=pd.concat([sub["날짜"], sub["날짜"].iloc[::-1]]),
                y=pd.concat([sub["PM25"]+std*0.5, (sub["PM25"]-std*0.5).iloc[::-1]]),
                fill='toself',
                fillcolor=f"rgba({int(colors[yr][1:3],16)},{int(colors[yr][3:5],16)},{int(colors[yr][5:7],16)},0.08)",
                line=dict(color='rgba(255,255,255,0)'),
                showlegend=False, hoverinfo='skip'
            ), row=1, col=1)

        # 호흡기질환 라인
        fig_ts.add_trace(go.Scatter(
            x=sub["날짜"], y=sub["호흡기질환"],
            name=f"{yr}년 질환발생률",
            line=dict(color=colors[yr], width=2, dash='dot'),
            mode='lines+markers',
            marker=dict(size=6, symbol='diamond'),
            hovertemplate=f"<b>{yr}년</b><br>날짜: %{{x|%Y년 %m월}}<br>발생률: %{{y}} 명<extra></extra>"
        ), row=2, col=1)

    # WHO 기준선
    fig_ts.add_hline(y=15, row=1, col=1,
                     line=dict(color='rgba(255,100,100,0.5)', width=1.5, dash='dash'),
                     annotation_text="WHO 기준 15μg/m³",
                     annotation_font=dict(color='rgba(255,100,100,0.8)', size=10))

    # 이상값 하이라이트
    if show_anomaly:
        max_row = filtered_df.loc[filtered_df["PM25"].idxmax()]
        fig_ts.add_annotation(
            x=max_row["날짜"], y=max_row["PM25"],
            text=f"⚠ 최고 {max_row['PM25']}μg/m³",
            showarrow=True, arrowhead=2, arrowcolor='#f97316',
            font=dict(color='#f97316', size=10),
            bgcolor='rgba(249,115,22,0.1)',
            bordercolor='rgba(249,115,22,0.3)',
            ax=30, ay=-30, row=1, col=1
        )

    fig_ts.update_layout(
        height=chart_height + 100,
        hovermode="x unified",
        **CHART_THEME
    )
    fig_ts.update_xaxes(
        gridcolor='rgba(255,255,255,0.05)',
        linecolor='rgba(255,255,255,0.1)',
    )
    fig_ts.update_yaxes(
        gridcolor='rgba(255,255,255,0.05)',
        linecolor='rgba(255,255,255,0.1)',
    )
    st.plotly_chart(fig_ts, use_container_width=True)

    # 월별 비교 히트맵
    if len(selected_year) == 2:
        st.markdown("""
        <div class="section-header">
            <div class="section-title">월별 연도간 차이 히트맵</div>
            <div class="section-line"></div>
        </div>
        """, unsafe_allow_html=True)

        pivot_pm = df.pivot_table(index="연도", columns="월명", values="PM25",
                                   aggfunc="mean")
        pivot_pm = pivot_pm[["1월","2월","3월","4월","5월","6월","7월","8월","9월","10월"]]

        fig_heat = go.Figure(go.Heatmap(
            z=pivot_pm.values,
            x=pivot_pm.columns.tolist(),
            y=[str(y) for y in pivot_pm.index.tolist()],
            colorscale=[[0, '#00ff88'], [0.4, '#0080ff'], [1, '#ff4444']],
            text=pivot_pm.values,
            texttemplate="%{text:.0f}",
            hovertemplate="연도: %{y}<br>월: %{x}<br>PM2.5: %{z:.1f} μg/m³<extra></extra>",
            showscale=True,
            colorbar=dict(
                title=dict(text="μg/m³", font=dict(color='#8fa8c0')),
                tickfont=dict(color='#8fa8c0'),
                outlinecolor='rgba(255,255,255,0.1)',
            )
        ))
        fig_heat.update_layout(height=200, **CHART_THEME, margin=dict(t=20,b=20,l=10,r=10))
        st.plotly_chart(fig_heat, use_container_width=True)

# ══════════════════════════════════════════
# TAB 2: 상관 & 회귀
# ══════════════════════════════════════════
with tab2:
    col_left, col_right = st.columns([3, 2])

    with col_left:
        st.markdown("""
        <div class="section-header">
            <div class="section-title">PM2.5 vs 호흡기 질환 산점도</div>
            <div class="section-line"></div>
        </div>
        """, unsafe_allow_html=True)

        fig_scatter = go.Figure()

        for yr in sorted(selected_year):
            sub = filtered_df[filtered_df["연도"] == yr]
            c = colors[yr]

            # OLS 회귀선 추가
            if len(sub) > 2:
                x_range = np.linspace(sub["PM25"].min()-2, sub["PM25"].max()+2, 100)
                slope, intercept, r, p, se = stats.linregress(sub["PM25"], sub["호흡기질환"])
                y_pred = slope * x_range + intercept

                fig_scatter.add_trace(go.Scatter(
                    x=x_range, y=y_pred,
                    mode='lines',
                    name=f"{yr}년 회귀선",
                    line=dict(color=c, width=2, dash='dash'),
                    hoverinfo='skip'
                ))

                # CI 밴드
                if show_ci:
                    n = len(sub)
                    x_mean = sub["PM25"].mean()
                    se_line = se * np.sqrt(1/n + (x_range - x_mean)**2 / ((n-1)*sub["PM25"].var()))
                    t_crit = stats.t.ppf(0.975, df=n-2)
                    ci = t_crit * se_line
                    fig_scatter.add_trace(go.Scatter(
                        x=np.concatenate([x_range, x_range[::-1]]),
                        y=np.concatenate([y_pred+ci, (y_pred-ci)[::-1]]),
                        fill='toself',
                        fillcolor=f"rgba({int(c[1:3],16)},{int(c[3:5],16)},{int(c[5:7],16)},0.08)",
                        line=dict(color='rgba(0,0,0,0)'),
                        name=f"{yr}년 95% CI",
                        hoverinfo='skip'
                    ))

            # 포인트
            fig_scatter.add_trace(go.Scatter(
                x=sub["PM25"], y=sub["호흡기질환"],
                mode='markers+text',
                name=f"{yr}년",
                marker=dict(
                    size=12, color=c,
                    opacity=0.85,
                    line=dict(color='rgba(0,0,0,0.4)', width=2),
                    symbol='circle'
                ),
                text=sub["월명"],
                textposition="top center",
                textfont=dict(size=8, color='rgba(255,255,255,0.4)'),
                hovertemplate=f"<b>{yr}년 %{{text}}</b><br>PM2.5: %{{x}} μg/m³<br>발생률: %{{y}} 명<extra></extra>"
            ))

        fig_scatter.update_layout(
            height=chart_height,
            xaxis_title="PM2.5 농도 (μg/m³)",
            yaxis_title="호흡기 질환 발생률 (명/10만명)",
            **CHART_THEME
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

    with col_right:
        st.markdown("""
        <div class="section-header">
            <div class="section-title">회귀 분석 결과</div>
            <div class="section-line"></div>
        </div>
        """, unsafe_allow_html=True)

        reg_df = filtered_df.dropna(subset=["PM25", "호흡기질환"])
        X = sm.add_constant(reg_df["PM25"])
        model = sm.OLS(reg_df["호흡기질환"], X).fit()

        slope = model.params["PM25"]
        intercept = model.params["const"]
        r2 = model.rsquared
        p_pm25 = model.pvalues["PM25"]
        f_stat = model.fvalue
        f_p = model.f_pvalue

        st.markdown(f"""
        <div class="insight-card success">
            <div class="insight-title">📐 모델 요약</div>
            <div class="insight-text">
                <b>방정식:</b> 질환 = {slope:.2f} × PM2.5 + {intercept:.1f}<br>
                <b>R²:</b> {r2:.4f} &nbsp;|&nbsp; <b>Adj. R²:</b> {model.rsquared_adj:.4f}<br>
                <b>F-통계:</b> {f_stat:.2f} (p={f_p:.4f})<br>
                <b>AIC:</b> {model.aic:.1f} &nbsp;|&nbsp; <b>BIC:</b> {model.bic:.1f}
            </div>
        </div>
        <div class="insight-card {'success' if p_pm25 < 0.001 else 'warning'}">
            <div class="insight-title">📊 계수 유의성</div>
            <div class="insight-text">
                <b>PM2.5 계수:</b> {slope:.4f}<br>
                <b>p-value:</b> {p_pm25:.6f}<br>
                <b>95% CI:</b> [{model.conf_int().loc['PM25', 0]:.3f}, {model.conf_int().loc['PM25', 1]:.3f}]<br>
                <b>판정:</b> {'✅ 통계적으로 유의 (p<0.001)' if p_pm25 < 0.001 else '⚠ p<0.05 수준'}
            </div>
        </div>
        <div class="insight-card info">
            <div class="insight-title">💡 해석</div>
            <div class="insight-text">
                PM2.5가 1μg/m³ 증가할 때마다 호흡기 질환 발생률이 평균 <b>{slope:.1f}명/10만명</b> 증가.<br>
                모델이 전체 분산의 <b>{r2*100:.1f}%</b>를 설명.
            </div>
        </div>
        """, unsafe_allow_html=True)

        if show_regression:
            with st.expander("📄 전체 회귀 분석 결과 (OLS Summary)"):
                st.text(model.summary())

    # 잔차 분포
    st.markdown("""
    <div class="section-header">
        <div class="section-title">잔차 분석 (Residual Analysis)</div>
        <div class="section-line"></div>
    </div>
    """, unsafe_allow_html=True)

    residuals = model.resid
    fitted = model.fittedvalues

    fig_resid = make_subplots(
        rows=1, cols=3,
        subplot_titles=["잔차 vs 적합값", "잔차 분포 히스토그램", "Q-Q Plot"]
    )

    # 잔차 vs 적합값
    fig_resid.add_trace(go.Scatter(
        x=fitted, y=residuals,
        mode='markers',
        marker=dict(color=COLOR_2026, size=8, opacity=0.7,
                    line=dict(color='rgba(0,0,0,0.3)', width=1)),
        name='잔차',
        hovertemplate="적합값: %{x:.1f}<br>잔차: %{y:.1f}<extra></extra>"
    ), row=1, col=1)
    fig_resid.add_hline(y=0, row=1, col=1,
                        line=dict(color='rgba(255,100,100,0.5)', width=1.5, dash='dash'))

    # 히스토그램
    fig_resid.add_trace(go.Histogram(
        x=residuals, nbinsx=10,
        marker_color=COLOR_ACCENT,
        opacity=0.7,
        name='빈도',
        hovertemplate="범위: %{x}<br>빈도: %{y}<extra></extra>"
    ), row=1, col=2)

    # Q-Q Plot
    qq_theoretical = stats.norm.ppf(np.linspace(0.01, 0.99, len(residuals)))
    qq_sample = np.sort(residuals)
    fig_resid.add_trace(go.Scatter(
        x=qq_theoretical, y=qq_sample,
        mode='markers',
        marker=dict(color=COLOR_2025, size=8, opacity=0.7),
        name='Q-Q',
    ), row=1, col=3)
    fig_resid.add_trace(go.Scatter(
        x=[-3, 3],
        y=[qq_sample.min(), qq_sample.max()],
        mode='lines',
        line=dict(color='rgba(255,255,255,0.3)', dash='dash', width=1),
        showlegend=False
    ), row=1, col=3)

    fig_resid.update_layout(height=320, showlegend=False, **CHART_THEME)
    fig_resid.update_xaxes(gridcolor='rgba(255,255,255,0.05)', linecolor='rgba(255,255,255,0.1)')
    fig_resid.update_yaxes(gridcolor='rgba(255,255,255,0.05)', linecolor='rgba(255,255,255,0.1)')
    st.plotly_chart(fig_resid, use_container_width=True)

# ══════════════════════════════════════════
# TAB 3: 시차(Lag) 분석
# ══════════════════════════════════════════
with tab3:
    st.markdown("""
    <div class="section-header">
        <div class="section-title">다중 Lag 상관계수 비교</div>
        <div class="section-line"></div>
    </div>
    """, unsafe_allow_html=True)

    # 모든 Lag에 대해 상관계수 계산
    lag_results = []
    for l in range(0, 4):
        tmp = filtered_df.copy()
        tmp["PM25_LAG"] = tmp["PM25"].shift(l)
        c_val = tmp[["PM25_LAG", "호흡기질환"]].dropna().corr().iloc[0, 1]
        lag_results.append({"Lag": l, "상관계수": c_val})
    lag_df = pd.DataFrame(lag_results)

    col_la, col_lb = st.columns([2, 3])

    with col_la:
        fig_lag_bar = go.Figure(go.Bar(
            x=[f"Lag {r['Lag']}개월" for _, r in lag_df.iterrows()],
            y=lag_df["상관계수"],
            marker=dict(
                color=lag_df["상관계수"],
                colorscale=[[0, '#a855f7'], [0.5, '#0080ff'], [1, '#00d4ff']],
                line=dict(color='rgba(255,255,255,0.15)', width=1),
            ),
            text=[f"{v:.3f}" for v in lag_df["상관계수"]],
            textposition='outside',
            textfont=dict(color='#e2eaf5', size=12, family='Space Mono'),
            hovertemplate="<b>%{x}</b><br>상관계수: %{y:.4f}<extra></extra>"
        ))
        # 선택된 Lag 강조
        fig_lag_bar.add_vline(
            x=lag,
            line=dict(color='rgba(0,255,136,0.5)', width=2, dash='dash'),
            annotation_text=f"현재 선택: Lag {lag}",
            annotation_font=dict(color='#00ff88', size=10)
        )
        fig_lag_bar.update_layout(
            height=300,
            yaxis_range=[lag_df["상관계수"].min()-0.05, 1.05],
            **CHART_THEME
        )
        st.plotly_chart(fig_lag_bar, use_container_width=True)

    with col_lb:
        st.markdown("""
        <div class="section-header">
            <div class="section-title">시차 적용 산점도</div>
            <div class="section-line"></div>
        </div>
        """, unsafe_allow_html=True)

        lag_plot_df = filtered_df.dropna(subset=["PM25_LAG", "호흡기질환"])
        lag_corr = lag_plot_df["PM25_LAG"].corr(lag_plot_df["호흡기질환"])

        fig_lag_sc = go.Figure()

        # 회귀선
        if len(lag_plot_df) > 2:
            x_r = np.linspace(lag_plot_df["PM25_LAG"].min()-1, lag_plot_df["PM25_LAG"].max()+1, 100)
            sl, ic, _, _, _ = stats.linregress(lag_plot_df["PM25_LAG"], lag_plot_df["호흡기질환"])
            fig_lag_sc.add_trace(go.Scatter(
                x=x_r, y=sl*x_r+ic, mode='lines',
                line=dict(color=COLOR_ACCENT, width=2, dash='dash'),
                name='회귀선', hoverinfo='skip'
            ))

        fig_lag_sc.add_trace(go.Scatter(
            x=lag_plot_df["PM25_LAG"], y=lag_plot_df["호흡기질환"],
            mode='markers',
            marker=dict(size=11, color=COLOR_ACCENT, opacity=0.8,
                        line=dict(color='rgba(0,0,0,0.3)', width=1.5)),
            name=f"Lag {lag}개월",
            hovertemplate=f"PM2.5(Lag {lag}): %{{x:.1f}} μg/m³<br>발생률: %{{y}} 명<extra></extra>"
        ))

        fig_lag_sc.update_layout(
            height=300,
            title=dict(text=f"Lag {lag}개월 | r = {lag_corr:.3f}", font=dict(color='#e2eaf5', size=13)),
            xaxis_title=f"PM2.5 (t-{lag}개월 전)",
            yaxis_title="호흡기 질환 발생률",
            **CHART_THEME
        )
        st.plotly_chart(fig_lag_sc, use_container_width=True)

    # 크로스 상관 함수 분석
    st.markdown("""
    <div class="section-header">
        <div class="section-title">교차 상관 분석 요약표</div>
        <div class="section-line"></div>
    </div>
    """, unsafe_allow_html=True)

    st.dataframe(
        lag_df.style
            .background_gradient(subset=["상관계수"], cmap="Blues")
            .format({"상관계수": "{:.4f}"})
            .set_properties(**{
                'font-family': 'Space Mono, monospace',
                'font-size': '13px'
            }),
        use_container_width=True, height=200
    )

# ══════════════════════════════════════════
# TAB 4: 정책 시뮬레이션
# ══════════════════════════════════════════
with tab4:
    st.markdown("""
    <div class="section-header">
        <div class="section-title">대화형 정책 시뮬레이션</div>
        <div class="section-line"></div>
    </div>
    """, unsafe_allow_html=True)

    # 회귀 모델
    reg_base = filtered_df.dropna(subset=["PM25", "호흡기질환"])
    X_base = sm.add_constant(reg_base["PM25"])
    model_base = sm.OLS(reg_base["호흡기질환"], X_base).fit()

    col_sim1, col_sim2 = st.columns([1, 2])

    with col_sim1:
        st.markdown("#### ⚙️ 시나리오 설정")
        pm_input = st.slider("목표 PM2.5 수준 (μg/m³)", 15, 50, 28, step=1)
        target_year = st.selectbox("시나리오 연도", ["2027 (예측)", "2028 (예측)", "2029 (예측)"])
        policy_intensity = st.select_slider(
            "정책 강도",
            options=["최소", "보통", "강력", "초강력"],
            value="강력"
        )

        intensity_map = {"최소": 1.0, "보통": 0.95, "강력": 0.9, "초강력": 0.82}
        intensity_factor = intensity_map[policy_intensity]
        adjusted_pm = pm_input * intensity_factor

        pred = model_base.predict([1, adjusted_pm])[0]
        pred_ci = model_base.get_prediction([1, adjusted_pm]).conf_int(alpha=0.05)
        pred_lo, pred_hi = pred_ci[0][0], pred_ci[0][1]

        baseline_health = health_2025
        reduction = (baseline_health - pred) / baseline_health * 100

        st.markdown(f"""
        <br>
        <div class="insight-card success">
            <div class="insight-title">📊 시뮬레이션 결과</div>
            <div class="insight-text">
                조정 PM2.5: <b>{adjusted_pm:.1f} μg/m³</b><br>
                예측 발생률: <b>{pred:.1f} 명/10만</b><br>
                95% CI: [{pred_lo:.1f}, {pred_hi:.1f}]<br>
                기준 대비 감소: <b>{reduction:.1f}%</b>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col_sim2:
        # 시나리오 비교 시각화
        pm_range = np.linspace(15, 50, 200)
        pred_range = model_base.predict(sm.add_constant(pd.Series(pm_range)))

        pred_all = model_base.get_prediction(sm.add_constant(pd.Series(pm_range)))
        ci_all = pred_all.conf_int(alpha=0.05)

        fig_sim = go.Figure()

        # CI 영역
        if show_ci:
            fig_sim.add_trace(go.Scatter(
                x=np.concatenate([pm_range, pm_range[::-1]]),
                y=np.concatenate([ci_all[:, 1], ci_all[:, 0][::-1]]),
                fill='toself',
                fillcolor='rgba(0,128,255,0.07)',
                line=dict(color='rgba(0,0,0,0)'),
                name='95% 예측 CI', hoverinfo='skip'
            ))

        # 예측선
        fig_sim.add_trace(go.Scatter(
            x=pm_range, y=pred_range,
            mode='lines',
            line=dict(color=COLOR_2026, width=2.5),
            name='예측 발생률',
            hovertemplate="PM2.5: %{x:.1f} μg/m³<br>예측 발생률: %{y:.1f} 명<extra></extra>"
        ))

        # 실제 데이터 포인트
        for yr in sorted(selected_year):
            sub = filtered_df[filtered_df["연도"] == yr]
            fig_sim.add_trace(go.Scatter(
                x=sub["PM25"], y=sub["호흡기질환"],
                mode='markers',
                name=f"{yr}년 실제",
                marker=dict(size=9, color=colors[yr], opacity=0.7,
                            line=dict(color='rgba(0,0,0,0.3)', width=1))
            ))

        # 선택 지점 강조
        fig_sim.add_trace(go.Scatter(
            x=[adjusted_pm], y=[pred],
            mode='markers',
            marker=dict(size=18, color='#00ff88', symbol='star',
                        line=dict(color='rgba(0,0,0,0.5)', width=2)),
            name=f"선택: {adjusted_pm:.1f}μg/m³"
        ))

        # 수직/수평 참조선
        fig_sim.add_vline(x=adjusted_pm, line=dict(color='rgba(0,255,136,0.4)', dash='dash', width=1.5))
        fig_sim.add_hline(y=pred, line=dict(color='rgba(0,255,136,0.4)', dash='dash', width=1.5))
        fig_sim.add_vline(x=15, line=dict(color='rgba(255,100,100,0.4)', dash='dot', width=1),
                          annotation_text="WHO 기준", annotation_font=dict(color='rgba(255,100,100,0.7)', size=9))

        fig_sim.update_layout(
            height=chart_height,
            xaxis_title="PM2.5 농도 (μg/m³)",
            yaxis_title="예측 호흡기 질환 발생률 (명/10만명)",
            **CHART_THEME
        )
        st.plotly_chart(fig_sim, use_container_width=True)

    # 다중 시나리오 비교
    st.markdown("""
    <div class="section-header">
        <div class="section-title">시나리오별 정책 효과 비교</div>
        <div class="section-line"></div>
    </div>
    """, unsafe_allow_html=True)

    scenarios = {
        "현상 유지 (35μg/m³)": 35,
        "소극적 감축 (30μg/m³)": 30,
        "적극적 감축 (25μg/m³)": 25,
        "WHO 기준 달성 (15μg/m³)": 15,
    }
    sc_preds = {k: model_base.predict([1, v])[0] for k, v in scenarios.items()}
    sc_df = pd.DataFrame({
        "시나리오": list(sc_preds.keys()),
        "PM2.5": list(scenarios.values()),
        "예측 발생률": [round(v, 1) for v in sc_preds.values()],
        "감소율(%)": [round((baseline_health - v) / baseline_health * 100, 1) for v in sc_preds.values()]
    })

    fig_sc = go.Figure(go.Bar(
        x=sc_df["시나리오"],
        y=sc_df["예측 발생률"],
        marker=dict(
            color=sc_df["예측 발생률"],
            colorscale=[[0, '#00ff88'], [0.5, '#0080ff'], [1, '#f97316']],
            line=dict(color='rgba(255,255,255,0.1)', width=1)
        ),
        text=[f"{v}명<br>({r}% 감소)" for v, r in zip(sc_df["예측 발생률"], sc_df["감소율(%)"])],
        textposition='outside',
        textfont=dict(color='#e2eaf5', size=11),
        hovertemplate="<b>%{x}</b><br>예측 발생률: %{y:.1f}명<extra></extra>"
    ))
    fig_sc.update_layout(
        height=340,
        yaxis_range=[0, sc_df["예측 발생률"].max() * 1.25],
        **CHART_THEME
    )
    st.plotly_chart(fig_sc, use_container_width=True)

# ══════════════════════════════════════════
# TAB 5: 종합 보고
# ══════════════════════════════════════════
with tab5:
    col_r1, col_r2 = st.columns([3, 2])

    with col_r1:
        st.markdown("""
        <div class="section-header">
            <div class="section-title">종합 분석 결과 요약</div>
            <div class="section-line"></div>
        </div>
        """, unsafe_allow_html=True)

        pm_r = f"{pm25_reduce:.1f}%" if pm25_reduce else "—"
        h_r = f"{health_reduce:.1f}%" if health_reduce else "—"

        st.markdown(f"""
        <div class="insight-card success">
            <div class="insight-title">✅ 핵심 성과</div>
            <div class="insight-text">
                • PM2.5 연평균 농도 <b>{pm_r} 감소</b> (35.7 → 25.8 μg/m³)<br>
                • 호흡기 질환 발생률 <b>{h_r} 감소</b> (270.5 → 212.5 명/10만)<br>
                • 상관계수 <b>r = {corr:.3f}</b> — 극히 강한 양의 상관관계<br>
                • 독립표본 t-검정 <b>p < 0.001</b> — 통계적으로 매우 유의
            </div>
        </div>
        <div class="insight-card">
            <div class="insight-title">🌡️ 계절별 분석</div>
            <div class="insight-text">
                • 겨울-봄철(1~4월) 감소 폭 가장 큼: 3월 45 → 32 μg/m³<br>
                • 여름철 기저 농도도 개선: 8월 28 → 21 μg/m³<br>
                • 계절관리제 및 구조적 배출원 감축의 복합 효과
            </div>
        </div>
        <div class="insight-card warning">
            <div class="insight-title">⚠️ 분석의 한계</div>
            <div class="insight-text">
                • 2개년 단기 데이터 — 최소 5년 이상 추적 권고<br>
                • 기상(풍속, 강수) 및 국외 유입 변수 미통제<br>
                • 서울 대도시 특수성 — 소도시 일반화 제한
            </div>
        </div>
        <div class="insight-card info">
            <div class="insight-title">🔮 향후 정책 방향</div>
            <div class="insight-text">
                • 수도권 광역 단위 확대 적용 검토<br>
                • 실시간 AI 연계 '스마트 헬스케어 경보 시스템' 구축<br>
                • 취약계층(노인·아동) 밀집지역 '청정 공기 세이프존' 확대<br>
                • 시민 참여형 '환경보건 참여예산제' 도입
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col_r2:
        st.markdown("""
        <div class="section-header">
            <div class="section-title">정책 추진 단계</div>
            <div class="section-line"></div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <table class="policy-table">
            <tr>
                <th>단계</th><th>추진 과제</th><th>기대 효과</th>
            </tr>
            <tr>
                <td><span class="stage-badge stage-1">1단계 · 단기</span></td>
                <td style="color:#e2eaf5; font-weight:600;">저감 정책 공고화</td>
                <td>기후동행카드 확대<br>노후 보일러 100% 교체</td>
            </tr>
            <tr>
                <td><span class="stage-badge stage-2">2단계 · 중기</span></td>
                <td style="color:#e2eaf5; font-weight:600;">데이터 통합 관리</td>
                <td>실시간 AI 연계<br>고위험군 경보 시스템</td>
            </tr>
            <tr>
                <td><span class="stage-badge stage-3">3단계 · 장기</span></td>
                <td style="color:#e2eaf5; font-weight:600;">그린 인프라 완성</td>
                <td>도심 바람길 숲 조성<br>벽면 녹화 의무화</td>
            </tr>
        </table>
        <br>
        """, unsafe_allow_html=True)

        # 레이더 차트 (정책 성과 지표)
        categories = ['PM2.5 감소', '질환 감소', '통계 유의성', '시민 만족도', '데이터 신뢰도']
        values = [
            min(100, (pm25_reduce or 0) * 3.6),
            min(100, (health_reduce or 0) * 4.7),
            95 if (p_val is not None and p_val < 0.001) else 70,
            80,
            75
        ]
        values_closed = values + [values[0]]
        cats_closed = categories + [categories[0]]

        fig_radar = go.Figure(go.Scatterpolar(
            r=values_closed,
            theta=cats_closed,
            fill='toself',
            fillcolor='rgba(0,212,255,0.1)',
            line=dict(color='#00d4ff', width=2),
            marker=dict(size=6, color='#00d4ff'),
            name='정책 성과'
        ))
        fig_radar.update_layout(
            polar=dict(
                bgcolor='rgba(0,0,0,0)',
                radialaxis=dict(
                    visible=True, range=[0, 100],
                    tickfont=dict(color='#4a6680', size=9),
                    gridcolor='rgba(255,255,255,0.08)',
                    linecolor='rgba(255,255,255,0.08)',
                ),
                angularaxis=dict(
                    tickfont=dict(color='#8fa8c0', size=10),
                    gridcolor='rgba(255,255,255,0.08)',
                    linecolor='rgba(255,255,255,0.1)',
                )
            ),
            height=320,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            margin=dict(t=20, b=20, l=30, r=30),
            showlegend=False,
            font=dict(color='#8fa8c0')
        )
        st.plotly_chart(fig_radar, use_container_width=True)

    # 원본 데이터 테이블
    st.markdown("""
    <div class="section-header">
        <div class="section-title">분석 원데이터</div>
        <div class="section-line"></div>
    </div>
    """, unsafe_allow_html=True)

    display_df = filtered_df[["연도","월","PM25","호흡기질환"]].copy()
    display_df.columns = ["연도", "월", "PM2.5 (μg/m³)", "호흡기 질환 (명/10만)"]

    st.dataframe(
        display_df.style
            .background_gradient(subset=["PM2.5 (μg/m³)"], cmap="YlOrRd")
            .background_gradient(subset=["호흡기 질환 (명/10만)"], cmap="PuBu")
            .format({"PM2.5 (μg/m³)": "{:.0f}", "호흡기 질환 (명/10만)": "{:.0f}"}),
        use_container_width=True,
        height=380
    )

# ─────────────────────────────────────────
# 푸터
# ─────────────────────────────────────────
st.markdown("""
<div style='text-align:center; padding: 40px 0 20px 0; border-top: 1px solid rgba(255,255,255,0.06); margin-top:40px;'>
    <div style='font-size:0.75rem; color:#2a3a4a; letter-spacing:2px; text-transform:uppercase;'>
        서울특별시 정책본부 환경보건팀 · 2026.03 · Confidential
    </div>
    <div style='font-size:0.68rem; color:#1e2d3a; margin-top:6px; letter-spacing:1px;'>
        Built with Streamlit · Plotly · SciPy · Statsmodels
    </div>
</div>
""", unsafe_allow_html=True)
