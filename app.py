"""
Health Symptom Analyzer — 증상 기반 질병 예측 헬스케어 대시보드
v1.0 · 2026.03
개발 스택: Python · Streamlit · scikit-learn · Plotly · pandas
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import streamlit.components.v1 as components
import warnings
warnings.filterwarnings("ignore")

# ────────────────────────────────────────────────────────────────────────────────
# 페이지 설정
# ────────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Health Symptom Analyzer",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ────────────────────────────────────────────────────────────────────────────────
# 글로벌 CSS
# ────────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* 전체 배경 */
.stApp { background-color: #f8fafc; }

/* 면책 조항 배너 */
.disclaimer-banner {
    background: linear-gradient(135deg, #fff3cd, #ffeaa7);
    border: 1px solid #f59e0b;
    border-left: 4px solid #f59e0b;
    border-radius: 8px;
    padding: 10px 16px;
    margin-bottom: 16px;
    font-size: 13px;
    color: #92400e;
    font-weight: 500;
}

/* 긴급도 카드 */
.urgency-card {
    border-radius: 12px;
    padding: 16px;
    margin-bottom: 12px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.08);
}
.urgency-red   { background: linear-gradient(135deg, #fff5f5, #fed7d7); border-left: 4px solid #e53e3e; }
.urgency-orange{ background: linear-gradient(135deg, #fffaf0, #feebc8); border-left: 4px solid #dd6b20; }
.urgency-green { background: linear-gradient(135deg, #f0fff4, #c6f6d5); border-left: 4px solid #38a169; }

/* 배지 */
.badge-red    { background:#e53e3e; color:white; padding:3px 10px; border-radius:20px; font-size:12px; font-weight:600; }
.badge-orange { background:#dd6b20; color:white; padding:3px 10px; border-radius:20px; font-size:12px; font-weight:600; }
.badge-green  { background:#38a169; color:white; padding:3px 10px; border-radius:20px; font-size:12px; font-weight:600; }

/* 치료 정보 카드 */
.treatment-card {
    background: white;
    border-radius: 10px;
    padding: 16px;
    box-shadow: 0 2px 6px rgba(0,0,0,0.06);
    height: 100%;
}
.treatment-card h4 { margin-top: 0; color: #2d3748; font-size: 15px; }
.treatment-card ul { padding-left: 18px; color: #4a5568; font-size: 14px; }

/* 메트릭 카드 */
.metric-box {
    background: white;
    border-radius: 10px;
    padding: 14px 20px;
    text-align: center;
    box-shadow: 0 2px 6px rgba(0,0,0,0.06);
}
.metric-box .value { font-size: 28px; font-weight: 700; color: #2b6cb0; }
.metric-box .label { font-size: 12px; color: #718096; margin-top: 4px; }

/* 사이드바 */
[data-testid="stSidebar"] { background-color: #edf2f7; }
[data-testid="stSidebar"] .stCheckbox label { font-size: 13px; }

/* 탭 */
.stTabs [data-baseweb="tab"] { font-weight: 600; font-size: 14px; }
</style>
""", unsafe_allow_html=True)

# ────────────────────────────────────────────────────────────────────────────────
# 한국어 증상 매핑 (132개)
# ────────────────────────────────────────────────────────────────────────────────
SYMPTOM_KR = {
    "itching": "가려움증", "skin_rash": "피부 발진", "nodal_skin_eruptions": "결절성 피부 발진",
    "continuous_sneezing": "지속적 재채기", "shivering": "오한", "chills": "냉기/몸살",
    "joint_pain": "관절 통증", "stomach_pain": "복통", "acidity": "위산 과다",
    "ulcers_on_tongue": "혀 궤양", "muscle_wasting": "근육 소실", "vomiting": "구토",
    "burning_micturition": "배뇨 시 작열감", "spotting_urination": "혈뇨", "fatigue": "피로감",
    "weight_gain": "체중 증가", "anxiety": "불안감", "cold_hands_and_feets": "손발 냉증",
    "mood_swings": "기분 변동", "weight_loss": "체중 감소", "restlessness": "안절부절",
    "lethargy": "무기력증", "patches_in_throat": "인후 반점", "irregular_sugar_level": "혈당 불규칙",
    "cough": "기침", "high_fever": "고열", "sunken_eyes": "눈꺼풀 함몰",
    "breathlessness": "호흡 곤란", "sweating": "발한", "dehydration": "탈수",
    "indigestion": "소화불량", "headache": "두통", "yellowish_skin": "황달(피부)",
    "dark_urine": "소변 색 진함", "nausea": "메스꺼움", "loss_of_appetite": "식욕 부진",
    "pain_behind_the_eyes": "눈 통증", "back_pain": "허리 통증", "constipation": "변비",
    "abdominal_pain": "복부 통증", "diarrhoea": "설사", "mild_fever": "미열",
    "yellow_urine": "소변 황색", "yellowing_of_eyes": "황달(눈)", "acute_liver_failure": "급성 간부전",
    "fluid_overload": "체액 과부하", "swelling_of_stomach": "복부 팽창", "swelled_lymph_nodes": "림프절 종창",
    "malaise": "권태감", "blurred_and_distorted_vision": "시력 흐림", "phlegm": "가래",
    "throat_irritation": "인후 자극", "redness_of_eyes": "충혈", "sinus_pressure": "부비동 압박",
    "runny_nose": "콧물", "congestion": "코막힘", "chest_pain": "흉통",
    "weakness_in_limbs": "사지 쇠약", "fast_heart_rate": "빠른 심박수", "pain_during_bowel_movements": "배변 시 통증",
    "pain_in_anal_region": "항문 통증", "bloody_stool": "혈변", "irritation_in_anus": "항문 자극",
    "neck_pain": "목 통증", "dizziness": "어지러움", "cramps": "경련",
    "bruising": "멍", "obesity": "비만", "swollen_legs": "다리 부종",
    "swollen_blood_vessels": "혈관 팽창", "puffy_face_and_eyes": "얼굴/눈 부종", "enlarged_thyroid": "갑상선 비대",
    "brittle_nails": "손발톱 부서짐", "swollen_extremeties": "말단 부종", "excessive_hunger": "과도한 식욕",
    "extra_marital_contacts": "위험 접촉", "drying_and_tingling_lips": "입술 건조/따끔", "slurred_speech": "발음 불명확",
    "knee_pain": "무릎 통증", "hip_joint_pain": "고관절 통증", "muscle_weakness": "근육 쇠약",
    "stiff_neck": "목 뻣뻣함", "swelling_joints": "관절 부종", "movement_stiffness": "운동 강직",
    "spinning_movements": "회전성 어지러움", "loss_of_balance": "균형 감각 상실", "unsteadiness": "불안정 보행",
    "weakness_of_one_body_side": "편측 쇠약", "loss_of_smell": "후각 상실", "bladder_discomfort": "방광 불편",
    "foul_smell_of_urine": "소변 악취", "continuous_feel_of_urine": "잦은 요의", "passage_of_gases": "가스 방출",
    "internal_itching": "내부 가려움", "toxic_look_typhos": "독성 외모", "depression": "우울증",
    "irritability": "과민성", "muscle_pain": "근육 통증", "altered_sensorium": "의식 변화",
    "red_spots_over_body": "전신 붉은 반점", "belly_pain": "배 통증", "abnormal_menstruation": "월경 이상",
    "dischromic_patches": "색소 이상 반점", "watering_from_eyes": "눈물 흘림", "increased_appetite": "식욕 증가",
    "polyuria": "다뇨", "family_history": "가족력", "mucoid_sputum": "점액성 가래",
    "rusty_sputum": "녹슨색 가래", "lack_of_concentration": "집중력 저하", "visual_disturbances": "시각 장애",
    "receiving_blood_transfusion": "수혈 이력", "receiving_unsterile_injections": "비위생적 주사 이력",
    "coma": "혼수 상태", "stomach_bleeding": "위 출혈", "distention_of_abdomen": "복부 팽만",
    "history_of_alcohol_consumption": "음주 이력", "fluid_overload_1": "체액 과부하(2)", "blood_in_sputum": "가래 혈액",
    "prominent_veins_on_calf": "종아리 정맥 돌출", "palpitations": "심계항진", "painful_walking": "보행 시 통증",
    "pus_filled_pimples": "고름 여드름", "blackheads": "블랙헤드", "scurring": "딱지",
    "skin_peeling": "피부 벗겨짐", "silver_like_dusting": "은빛 각질", "small_dents_in_nails": "손발톱 함몰",
    "inflammatory_nails": "염증성 손발톱", "blister": "물집", "red_sore_around_nose": "코 주변 홍반",
    "yellow_crust_ooze": "황색 분비물",
}

# ────────────────────────────────────────────────────────────────────────────────
# 증상 카테고리 분류
# ────────────────────────────────────────────────────────────────────────────────
SYMPTOM_CATEGORIES = {
    "전신·발열 증상": [
        "fatigue", "high_fever", "mild_fever", "shivering", "chills", "sweating",
        "dehydration", "malaise", "weight_loss", "weight_gain", "lethargy",
        "restlessness", "anxiety", "mood_swings", "depression", "irritability",
        "altered_sensorium", "coma", "toxic_look_typhos",
    ],
    "통증·근골격 증상": [
        "joint_pain", "muscle_pain", "muscle_weakness", "muscle_wasting",
        "back_pain", "neck_pain", "knee_pain", "hip_joint_pain",
        "chest_pain", "headache", "pain_behind_the_eyes", "cramps",
        "stiff_neck", "swelling_joints", "movement_stiffness", "painful_walking",
        "weakness_in_limbs", "weakness_of_one_body_side",
    ],
    "소화기 증상": [
        "stomach_pain", "abdominal_pain", "belly_pain", "nausea", "vomiting",
        "diarrhoea", "constipation", "indigestion", "acidity", "loss_of_appetite",
        "increased_appetite", "excessive_hunger", "passage_of_gases",
        "pain_during_bowel_movements", "bloody_stool", "stomach_bleeding",
        "distention_of_abdomen", "swelling_of_stomach", "fluid_overload",
        "ulcers_on_tongue",
    ],
    "호흡기 증상": [
        "cough", "breathlessness", "phlegm", "throat_irritation",
        "patches_in_throat", "continuous_sneezing", "runny_nose", "congestion",
        "sinus_pressure", "blood_in_sputum", "mucoid_sputum", "rusty_sputum",
        "loss_of_smell",
    ],
    "피부·외형 증상": [
        "itching", "skin_rash", "nodal_skin_eruptions", "yellowish_skin",
        "red_spots_over_body", "dischromic_patches", "pus_filled_pimples",
        "blackheads", "skin_peeling", "silver_like_dusting", "blister",
        "red_sore_around_nose", "yellow_crust_ooze", "bruising",
        "brittle_nails", "small_dents_in_nails", "inflammatory_nails", "scurring",
    ],
    "기타 증상": [
        "dizziness", "spinning_movements", "loss_of_balance", "unsteadiness",
        "blurred_and_distorted_vision", "visual_disturbances", "redness_of_eyes",
        "watering_from_eyes", "yellowish_skin", "yellowing_of_eyes",
        "dark_urine", "yellow_urine", "burning_micturition", "spotting_urination",
        "bladder_discomfort", "foul_smell_of_urine", "continuous_feel_of_urine",
        "swelled_lymph_nodes", "enlarged_thyroid", "swollen_legs",
        "swollen_blood_vessels", "puffy_face_and_eyes", "obesity",
        "fast_heart_rate", "palpitations", "irregular_sugar_level", "polyuria",
        "abnormal_menstruation", "slurred_speech", "lack_of_concentration",
        "cold_hands_and_feets", "drying_and_tingling_lips", "family_history",
        "history_of_alcohol_consumption", "receiving_blood_transfusion",
        "receiving_unsterile_injections", "internal_itching", "prominent_veins_on_calf",
        "pain_in_anal_region", "irritation_in_anus", "extra_marital_contacts",
        "sunken_eyes",
    ],
}

# ────────────────────────────────────────────────────────────────────────────────
# 전체 증상 목록 (카테고리 합산 → 중복 제거)
# ────────────────────────────────────────────────────────────────────────────────
ALL_SYMPTOMS = []
seen = set()
for syms in SYMPTOM_CATEGORIES.values():
    for s in syms:
        if s not in seen:
            ALL_SYMPTOMS.append(s)
            seen.add(s)

# 나머지 SYMPTOM_KR 키 중 미포함된 것 추가
for s in SYMPTOM_KR:
    if s not in seen:
        ALL_SYMPTOMS.append(s)
        seen.add(s)

# ────────────────────────────────────────────────────────────────────────────────
# 증상 심각도 가중치 (1~7)
# ────────────────────────────────────────────────────────────────────────────────
SEVERITY_WEIGHT = {
    "itching": 1, "skin_rash": 3, "nodal_skin_eruptions": 4, "continuous_sneezing": 4,
    "shivering": 5, "chills": 3, "joint_pain": 3, "stomach_pain": 4, "acidity": 3,
    "ulcers_on_tongue": 4, "muscle_wasting": 3, "vomiting": 5, "burning_micturition": 6,
    "spotting_urination": 5, "fatigue": 4, "weight_gain": 3, "anxiety": 4,
    "cold_hands_and_feets": 5, "mood_swings": 3, "weight_loss": 3, "restlessness": 5,
    "lethargy": 2, "patches_in_throat": 6, "irregular_sugar_level": 5, "cough": 4,
    "high_fever": 7, "sunken_eyes": 4, "breathlessness": 7, "sweating": 3,
    "dehydration": 4, "indigestion": 5, "headache": 3, "yellowish_skin": 3,
    "dark_urine": 4, "nausea": 5, "loss_of_appetite": 4, "pain_behind_the_eyes": 4,
    "back_pain": 3, "constipation": 4, "abdominal_pain": 4, "diarrhoea": 4,
    "mild_fever": 5, "yellow_urine": 4, "yellowing_of_eyes": 5, "acute_liver_failure": 7,
    "fluid_overload": 6, "swelling_of_stomach": 5, "swelled_lymph_nodes": 6,
    "malaise": 3, "blurred_and_distorted_vision": 3, "phlegm": 5, "throat_irritation": 4,
    "redness_of_eyes": 4, "sinus_pressure": 4, "runny_nose": 5, "congestion": 4,
    "chest_pain": 7, "weakness_in_limbs": 6, "fast_heart_rate": 5,
    "pain_during_bowel_movements": 6, "pain_in_anal_region": 6, "bloody_stool": 6,
    "irritation_in_anus": 6, "neck_pain": 5, "dizziness": 4, "cramps": 4,
    "bruising": 4, "obesity": 4, "swollen_legs": 5, "swollen_blood_vessels": 5,
    "puffy_face_and_eyes": 5, "enlarged_thyroid": 6, "brittle_nails": 3,
    "swollen_extremeties": 5, "excessive_hunger": 4, "extra_marital_contacts": 4,
    "drying_and_tingling_lips": 4, "slurred_speech": 6, "knee_pain": 4,
    "hip_joint_pain": 4, "muscle_weakness": 3, "stiff_neck": 6, "swelling_joints": 6,
    "movement_stiffness": 5, "spinning_movements": 6, "loss_of_balance": 5,
    "unsteadiness": 5, "weakness_of_one_body_side": 7, "loss_of_smell": 3,
    "bladder_discomfort": 4, "foul_smell_of_urine": 5, "continuous_feel_of_urine": 6,
    "passage_of_gases": 5, "internal_itching": 4, "toxic_look_typhos": 5,
    "depression": 3, "irritability": 3, "muscle_pain": 2, "altered_sensorium": 6,
    "red_spots_over_body": 5, "belly_pain": 4, "abnormal_menstruation": 6,
    "dischromic_patches": 5, "watering_from_eyes": 4, "increased_appetite": 5,
    "polyuria": 4, "family_history": 5, "mucoid_sputum": 4, "rusty_sputum": 4,
    "lack_of_concentration": 3, "visual_disturbances": 5,
    "receiving_blood_transfusion": 5, "receiving_unsterile_injections": 5,
    "coma": 7, "stomach_bleeding": 6, "distention_of_abdomen": 4,
    "history_of_alcohol_consumption": 5, "fluid_overload_1": 6, "blood_in_sputum": 6,
    "prominent_veins_on_calf": 5, "palpitations": 4, "painful_walking": 3,
    "pus_filled_pimples": 2, "blackheads": 1, "scurring": 1, "skin_peeling": 3,
    "silver_like_dusting": 2, "small_dents_in_nails": 1, "inflammatory_nails": 2,
    "blister": 4, "red_sore_around_nose": 3, "yellow_crust_ooze": 4,
}

# ────────────────────────────────────────────────────────────────────────────────
# 질병 데이터 (41종 질병 × 증상 매핑)
# ────────────────────────────────────────────────────────────────────────────────
DISEASE_KR = {
    "Fungal infection": "진균 감염", "Allergy": "알레르기", "GERD": "위식도역류질환",
    "Chronic cholestasis": "만성 담즙 정체", "Drug Reaction": "약물 반응",
    "Peptic ulcer disease": "소화성 궤양", "AIDS": "에이즈", "Diabetes": "당뇨병",
    "Gastroenteritis": "장염", "Bronchial Asthma": "기관지 천식",
    "Hypertension": "고혈압", "Migraine": "편두통",
    "Cervical spondylosis": "경추 척추증", "Paralysis (brain hemorrhage)": "뇌출혈/마비",
    "Jaundice": "황달", "Malaria": "말라리아", "Chicken pox": "수두",
    "Dengue": "뎅기열", "Typhoid": "장티푸스", "Hepatitis A": "A형 간염",
    "Hepatitis B": "B형 간염", "Hepatitis C": "C형 간염",
    "Hepatitis D": "D형 간염", "Hepatitis E": "E형 간염",
    "Alcoholic hepatitis": "알코올성 간염", "Tuberculosis": "결핵",
    "Common Cold": "일반 감기", "Pneumonia": "폐렴",
    "Dimorphic hemmorhoids(piles)": "치질", "Heart attack": "심장마비",
    "Varicose veins": "정맥류", "Hypothyroidism": "갑상선 기능 저하증",
    "Hyperthyroidism": "갑상선 기능 항진증", "Hypoglycemia": "저혈당증",
    "Osteoarthritis": "골관절염", "Arthritis": "관절염",
    "(vertigo) Paroxysmal Positional Vertigo": "양성발작성 체위성 현기증",
    "Acne": "여드름", "Urinary tract infection": "요로 감염",
    "Psoriasis": "건선", "Impetigo": "농가진",
}

DISEASE_SYMPTOMS = {
    "Fungal infection": ["itching", "skin_rash", "nodal_skin_eruptions", "dischromic_patches"],
    "Allergy": ["continuous_sneezing", "shivering", "chills", "watering_from_eyes", "redness_of_eyes"],
    "GERD": ["stomach_pain", "acidity", "vomiting", "indigestion", "chest_pain", "nausea"],
    "Chronic cholestasis": ["itching", "vomiting", "yellowish_skin", "nausea", "loss_of_appetite", "abdominal_pain", "yellowing_of_eyes"],
    "Drug Reaction": ["itching", "skin_rash", "stomach_pain", "vomiting", "burning_micturition"],
    "Peptic ulcer disease": ["vomiting", "indigestion", "loss_of_appetite", "abdominal_pain", "passage_of_gases", "internal_itching"],
    "AIDS": ["muscle_wasting", "patches_in_throat", "high_fever", "extra_marital_contacts", "fatigue", "weight_loss", "receiving_blood_transfusion", "receiving_unsterile_injections"],
    "Diabetes": ["fatigue", "weight_loss", "restlessness", "lethargy", "irregular_sugar_level", "polyuria", "excessive_hunger", "increased_appetite"],
    "Gastroenteritis": ["vomiting", "sunken_eyes", "dehydration", "diarrhoea"],
    "Bronchial Asthma": ["fatigue", "cough", "high_fever", "breathlessness", "family_history", "mucoid_sputum"],
    "Hypertension": ["headache", "chest_pain", "dizziness", "loss_of_balance", "lack_of_concentration"],
    "Migraine": ["acidity", "indigestion", "headache", "blurred_and_distorted_vision", "excessive_hunger", "stiff_neck", "depression", "irritability", "visual_disturbances"],
    "Cervical spondylosis": ["back_pain", "weakness_in_limbs", "neck_pain", "dizziness", "loss_of_balance"],
    "Paralysis (brain hemorrhage)": ["vomiting", "headache", "weakness_of_one_body_side", "altered_sensorium"],
    "Jaundice": ["itching", "vomiting", "fatigue", "weight_loss", "high_fever", "yellowish_skin", "dark_urine", "abdominal_pain"],
    "Malaria": ["chills", "vomiting", "high_fever", "sweating", "headache", "nausea", "diarrhoea", "muscle_pain"],
    "Chicken pox": ["itching", "skin_rash", "fatigue", "lethargy", "high_fever", "headache", "loss_of_appetite", "mild_fever", "swelled_lymph_nodes", "malaise", "red_spots_over_body", "blister"],
    "Dengue": ["skin_rash", "chills", "joint_pain", "vomiting", "fatigue", "high_fever", "headache", "nausea", "loss_of_appetite", "pain_behind_the_eyes", "back_pain", "malaise", "muscle_pain", "red_spots_over_body"],
    "Typhoid": ["chills", "vomiting", "fatigue", "high_fever", "headache", "nausea", "constipation", "abdominal_pain", "diarrhoea", "toxic_look_typhos", "belly_pain", "malaise"],
    "Hepatitis A": ["joint_pain", "vomiting", "yellowish_skin", "dark_urine", "nausea", "loss_of_appetite", "abdominal_pain", "diarrhoea", "mild_fever", "yellowing_of_eyes", "muscle_pain"],
    "Hepatitis B": ["itching", "fatigue", "lethargy", "yellowish_skin", "dark_urine", "loss_of_appetite", "abdominal_pain", "fluid_overload", "malaise", "yellowing_of_eyes", "receiving_blood_transfusion", "receiving_unsterile_injections"],
    "Hepatitis C": ["fatigue", "yellowish_skin", "nausea", "loss_of_appetite", "yellowing_of_eyes", "receiving_blood_transfusion", "receiving_unsterile_injections", "family_history"],
    "Hepatitis D": ["joint_pain", "vomiting", "fatigue", "yellowish_skin", "dark_urine", "nausea", "loss_of_appetite", "abdominal_pain", "yellowing_of_eyes", "receiving_blood_transfusion"],
    "Hepatitis E": ["joint_pain", "vomiting", "fatigue", "high_fever", "yellowish_skin", "dark_urine", "nausea", "loss_of_appetite", "abdominal_pain", "acute_liver_failure", "coma", "stomach_bleeding"],
    "Alcoholic hepatitis": ["vomiting", "yellowish_skin", "abdominal_pain", "swelling_of_stomach", "history_of_alcohol_consumption", "fluid_overload", "renal_failure"],
    "Tuberculosis": ["chills", "vomiting", "fatigue", "weight_loss", "cough", "high_fever", "breathlessness", "sweating", "loss_of_appetite", "mild_fever", "yellowing_of_eyes", "phlegm", "blood_in_sputum", "malaise"],
    "Common Cold": ["continuous_sneezing", "chills", "fatigue", "cough", "headache", "swelled_lymph_nodes", "malaise", "phlegm", "runny_nose", "congestion", "chest_pain", "loss_of_smell", "muscle_pain"],
    "Pneumonia": ["chills", "fatigue", "cough", "high_fever", "breathlessness", "sweating", "malaise", "phlegm", "rusty_sputum", "fast_heart_rate"],
    "Dimorphic hemmorhoids(piles)": ["constipation", "pain_during_bowel_movements", "pain_in_anal_region", "bloody_stool", "irritation_in_anus"],
    "Heart attack": ["vomiting", "breathlessness", "sweating", "chest_pain", "fast_heart_rate"],
    "Varicose veins": ["fatigue", "cramps", "bruising", "obesity", "swollen_legs", "swollen_blood_vessels", "prominent_veins_on_calf"],
    "Hypothyroidism": ["fatigue", "weight_gain", "cold_hands_and_feets", "mood_swings", "lethargy", "depression", "enlarged_thyroid", "brittle_nails", "swollen_extremeties", "puffy_face_and_eyes", "abnormal_menstruation"],
    "Hyperthyroidism": ["fatigue", "mood_swings", "weight_loss", "restlessness", "sweating", "diarrhoea", "fast_heart_rate", "enlarged_thyroid", "excessive_hunger", "irritability", "abnormal_menstruation"],
    "Hypoglycemia": ["vomiting", "fatigue", "anxiety", "sweating", "headache", "nausea", "blurred_and_distorted_vision", "excessive_hunger", "drying_and_tingling_lips", "slurred_speech", "irritability", "palpitations"],
    "Osteoarthritis": ["joint_pain", "neck_pain", "knee_pain", "hip_joint_pain", "swelling_joints", "painful_walking"],
    "Arthritis": ["muscle_weakness", "stiff_neck", "swelling_joints", "movement_stiffness", "loss_of_balance"],
    "(vertigo) Paroxysmal Positional Vertigo": ["vomiting", "headache", "nausea", "spinning_movements", "loss_of_balance", "unsteadiness"],
    "Acne": ["skin_rash", "pus_filled_pimples", "blackheads", "scurring"],
    "Urinary tract infection": ["burning_micturition", "bladder_discomfort", "foul_smell_of_urine", "continuous_feel_of_urine"],
    "Psoriasis": ["skin_rash", "joint_pain", "skin_peeling", "silver_like_dusting", "small_dents_in_nails", "inflammatory_nails"],
    "Impetigo": ["skin_rash", "itching", "high_fever", "red_sore_around_nose", "blister", "yellow_crust_ooze"],
}

# ────────────────────────────────────────────────────────────────────────────────
# 질병-신체 부위 매핑
# ────────────────────────────────────────────────────────────────────────────────
DISEASE_BODY_PARTS = {
    "Fungal infection": ["skin"],
    "Allergy": ["nose", "eyes", "skin"],
    "GERD": ["esophagus", "stomach"],
    "Chronic cholestasis": ["liver", "gallbladder"],
    "Drug Reaction": ["skin", "stomach"],
    "Peptic ulcer disease": ["stomach"],
    "AIDS": ["lymph", "immune"],
    "Diabetes": ["pancreas", "kidneys"],
    "Gastroenteritis": ["stomach", "intestines"],
    "Bronchial Asthma": ["lungs"],
    "Hypertension": ["heart", "brain"],
    "Migraine": ["brain"],
    "Cervical spondylosis": ["spine", "joints"],
    "Paralysis (brain hemorrhage)": ["brain"],
    "Jaundice": ["liver"],
    "Malaria": ["liver", "blood"],
    "Chicken pox": ["skin", "lymph"],
    "Dengue": ["blood", "lymph"],
    "Typhoid": ["intestines", "lymph"],
    "Hepatitis A": ["liver"],
    "Hepatitis B": ["liver"],
    "Hepatitis C": ["liver"],
    "Hepatitis D": ["liver"],
    "Hepatitis E": ["liver"],
    "Alcoholic hepatitis": ["liver"],
    "Tuberculosis": ["lungs"],
    "Common Cold": ["nose", "throat", "lungs"],
    "Pneumonia": ["lungs"],
    "Dimorphic hemmorhoids(piles)": ["intestines"],
    "Heart attack": ["heart"],
    "Varicose veins": ["legs", "blood"],
    "Hypothyroidism": ["thyroid"],
    "Hyperthyroidism": ["thyroid"],
    "Hypoglycemia": ["pancreas", "brain"],
    "Osteoarthritis": ["joints", "legs"],
    "Arthritis": ["joints"],
    "(vertigo) Paroxysmal Positional Vertigo": ["brain", "ears"],
    "Acne": ["skin"],
    "Urinary tract infection": ["bladder", "kidneys"],
    "Psoriasis": ["skin", "joints"],
    "Impetigo": ["skin"],
}

# ────────────────────────────────────────────────────────────────────────────────
# 치료 정보 DB
# ────────────────────────────────────────────────────────────────────────────────
TREATMENT_DB = {
    "Fungal infection": {
        "drugs": [{"name": "플루코나졸", "type": "항진균제", "note": "처방 필요"},
                  {"name": "클로트리마졸 크림", "type": "외용 항진균제", "note": "일반의약품"},
                  {"name": "테르비나핀", "type": "항진균제", "note": "처방 필요"}],
        "treatments": ["감염 부위 건조 유지", "통기성 좋은 옷 착용", "공용 수건 사용 금지", "면 소재 속옷 착용"],
        "folk_remedies": ["티트리 오일 희석 도포", "사과식초 희석 세척", "코코넛 오일 외용"],
        "urgency": "경과 관찰", "urgency_color": "green"
    },
    "Allergy": {
        "drugs": [{"name": "세티리진", "type": "항히스타민제", "note": "일반의약품"},
                  {"name": "로라타딘", "type": "항히스타민제", "note": "일반의약품"},
                  {"name": "몬테루카스트", "type": "류코트리엔 억제제", "note": "처방 필요"}],
        "treatments": ["알레르겐 회피", "HEPA 공기청정기 사용", "침구 자주 세탁", "알레르기 검사 시행"],
        "folk_remedies": ["국산 꿀 소량 섭취", "쐐기풀 차 음용", "식염수 비강 세척"],
        "urgency": "경과 관찰", "urgency_color": "green"
    },
    "GERD": {
        "drugs": [{"name": "오메프라졸", "type": "PPI", "note": "처방 필요"},
                  {"name": "란소프라졸", "type": "PPI", "note": "처방 필요"},
                  {"name": "라니티딘", "type": "H2차단제", "note": "일반의약품"}],
        "treatments": ["취침 2시간 전 식사 금지", "침대 머리 15cm 올리기", "소식 다식", "금연·금주"],
        "folk_remedies": ["알로에베라 주스(무당)", "생강차 식전 섭취", "사과식초 희석 음용"],
        "urgency": "빠른 진료", "urgency_color": "orange"
    },
    "Diabetes": {
        "drugs": [{"name": "메트포르민", "type": "혈당강하제", "note": "처방 필요"},
                  {"name": "인슐린", "type": "호르몬 요법", "note": "처방 필요"},
                  {"name": "글리피지드", "type": "설포닐우레아", "note": "처방 필요"}],
        "treatments": ["혈당 자가 모니터링", "저탄수화물 식단", "규칙적 유산소 운동", "발 건강 관리"],
        "folk_remedies": ["여주 주스", "계피 가루 섭취", "차전자피 보충"],
        "urgency": "빠른 진료", "urgency_color": "orange"
    },
    "Hypertension": {
        "drugs": [{"name": "암로디핀", "type": "칼슘채널차단제", "note": "처방 필요"},
                  {"name": "리시노프릴", "type": "ACE억제제", "note": "처방 필요"},
                  {"name": "로살탄", "type": "ARB", "note": "처방 필요"}],
        "treatments": ["저염식 (하루 5g 미만)", "규칙적 유산소 운동", "체중 감량", "금연·절주", "스트레스 관리"],
        "folk_remedies": ["마늘 섭취", "비트 주스", "히비스커스 차"],
        "urgency": "빠른 진료", "urgency_color": "orange"
    },
    "Heart attack": {
        "drugs": [{"name": "아스피린", "type": "항혈소판제", "note": "즉시 복용"},
                  {"name": "니트로글리세린", "type": "혈관확장제", "note": "처방 필요"},
                  {"name": "헤파린", "type": "항응고제", "note": "병원 투여"}],
        "treatments": ["즉시 119 신고", "안정 취하기", "꽉 조이는 옷 느슨하게", "의식 있으면 아스피린 복용"],
        "folk_remedies": ["※ 민간요법 적용 금지 — 즉시 응급실 방문"],
        "urgency": "즉시 병원", "urgency_color": "red"
    },
    "Tuberculosis": {
        "drugs": [{"name": "이소니아지드", "type": "항결핵제", "note": "처방 필요"},
                  {"name": "리팜피신", "type": "항결핵제", "note": "처방 필요"},
                  {"name": "피라진아미드", "type": "항결핵제", "note": "처방 필요"}],
        "treatments": ["6개월 이상 규칙적 복약", "마스크 착용 (KF94)", "환기 철저", "접촉자 검사"],
        "folk_remedies": ["생강·강황 차 보조", "마늘 섭취 보조", "충분한 영양 섭취"],
        "urgency": "즉시 병원", "urgency_color": "red"
    },
    "Pneumonia": {
        "drugs": [{"name": "아목시실린", "type": "항생제", "note": "처방 필요"},
                  {"name": "아지스로마이신", "type": "항생제", "note": "처방 필요"},
                  {"name": "이부프로펜", "type": "해열진통제", "note": "일반의약품"}],
        "treatments": ["충분한 휴식", "수분 충분히 섭취", "증기 흡입", "반좌위 자세"],
        "folk_remedies": ["생강차", "꿀물", "양파 찜질(민간)"],
        "urgency": "즉시 병원", "urgency_color": "red"
    },
    "Malaria": {
        "drugs": [{"name": "클로로퀸", "type": "항말라리아제", "note": "처방 필요"},
                  {"name": "아르테미시닌", "type": "항말라리아제", "note": "처방 필요"}],
        "treatments": ["모기 기피제 사용", "모기장 취침", "방충 의류 착용"],
        "folk_remedies": ["님(Neem) 잎 차", "파파야 잎 추출물(보조)"],
        "urgency": "즉시 병원", "urgency_color": "red"
    },
    "Common Cold": {
        "drugs": [{"name": "타이레놀", "type": "해열진통제", "note": "일반의약품"},
                  {"name": "슈다페드린", "type": "코막힘 완화", "note": "일반의약품"},
                  {"name": "덱스트로메토르판", "type": "기침 억제", "note": "일반의약품"}],
        "treatments": ["충분한 수분 섭취", "충분한 휴식", "소금물 가글", "스팀 흡입"],
        "folk_remedies": ["생강·꿀 레몬차", "닭고기 수프", "유칼립투스 오일 흡입"],
        "urgency": "경과 관찰", "urgency_color": "green"
    },
    "Migraine": {
        "drugs": [{"name": "수마트립탄", "type": "트립탄제", "note": "처방 필요"},
                  {"name": "이부프로펜", "type": "NSAIDs", "note": "일반의약품"},
                  {"name": "토피라메이트", "type": "예방약", "note": "처방 필요"}],
        "treatments": ["어두운 조용한 방 휴식", "냉찜질·온찜질", "규칙적 수면", "트리거 음식 회피"],
        "folk_remedies": ["페버퓨 허브", "생강차", "라벤더 오일 흡입"],
        "urgency": "빠른 진료", "urgency_color": "orange"
    },
    "Urinary tract infection": {
        "drugs": [{"name": "트리메토프림", "type": "항생제", "note": "처방 필요"},
                  {"name": "니트로푸란토인", "type": "항생제", "note": "처방 필요"},
                  {"name": "페나조피리딘", "type": "진통제", "note": "처방 필요"}],
        "treatments": ["수분 섭취 증가 (하루 2L)", "배뇨 후 앞에서 뒤로 닦기", "면 속옷 착용"],
        "folk_remedies": ["크랜베리 주스", "블루베리 섭취", "프로바이오틱스"],
        "urgency": "빠른 진료", "urgency_color": "orange"
    },
    "Acne": {
        "drugs": [{"name": "벤조일퍼옥사이드", "type": "외용 항균제", "note": "일반의약품"},
                  {"name": "레티노이드 크림", "type": "비타민A 유도체", "note": "처방 필요"},
                  {"name": "독시사이클린", "type": "항생제", "note": "처방 필요"}],
        "treatments": ["하루 2회 순한 세안", "오일 프리 보습제 사용", "자외선차단제 필수", "짜지 않기"],
        "folk_remedies": ["티트리 오일 희석 도포", "알로에베라 젤", "꿀 마스크"],
        "urgency": "경과 관찰", "urgency_color": "green"
    },
    "Psoriasis": {
        "drugs": [{"name": "코르티코스테로이드 크림", "type": "스테로이드", "note": "처방 필요"},
                  {"name": "칼시포트리올", "type": "비타민D 유도체", "note": "처방 필요"},
                  {"name": "메토트렉세이트", "type": "면역억제제", "note": "처방 필요"}],
        "treatments": ["보습제 자주 도포", "적당한 햇빛 노출", "스트레스 관리", "목욕 후 바로 보습"],
        "folk_remedies": ["오트밀 목욕", "알로에베라 도포", "어유(오메가3) 복용"],
        "urgency": "빠른 진료", "urgency_color": "orange"
    },
    "Hypothyroidism": {
        "drugs": [{"name": "레보티록신", "type": "갑상선 호르몬", "note": "처방 필요"}],
        "treatments": ["매일 같은 시간 복약", "갑상선 기능 정기 검사", "요오드 적정 섭취", "운동 병행"],
        "folk_remedies": ["셀레늄 풍부 음식 섭취", "아슈와간다 보충제(보조)", "글루텐 제한(일부 효과)"],
        "urgency": "빠른 진료", "urgency_color": "orange"
    },
    "Dengue": {
        "drugs": [{"name": "아세트아미노펜", "type": "해열진통제", "note": "일반의약품"},
                  {"name": "수액 보충", "type": "지지 요법", "note": "병원 투여"}],
        "treatments": ["충분한 수분 보충", "안정 취하기", "모기 기피 방역"],
        "folk_remedies": ["파파야 잎 주스(보조)", "구아바 잎 차"],
        "urgency": "즉시 병원", "urgency_color": "red"
    },
    "Hepatitis A": {
        "drugs": [{"name": "지지 요법", "type": "수액·영양", "note": "병원 투여"}],
        "treatments": ["충분한 휴식", "저지방·고탄수 식이", "금주 필수", "A형 간염 예방접종"],
        "folk_remedies": ["민들레 차(간 보호 보조)", "밀크씨슬(보조)"],
        "urgency": "빠른 진료", "urgency_color": "orange"
    },
    "Jaundice": {
        "drugs": [{"name": "원인 질환 치료", "type": "원인별 상이", "note": "처방 필요"}],
        "treatments": ["충분한 수분 섭취", "지방 제한 식이", "금주", "정기 간 기능 검사"],
        "folk_remedies": ["레몬 주스", "토마토 주스", "사탕수수 주스(보조)"],
        "urgency": "즉시 병원", "urgency_color": "red"
    },
    "Impetigo": {
        "drugs": [{"name": "무피로신 연고", "type": "항생제 외용", "note": "처방 필요"},
                  {"name": "세팔렉신", "type": "경구 항생제", "note": "처방 필요"}],
        "treatments": ["감염 부위 청결 유지", "손 자주 씻기", "수건 개별 사용", "딱지 제거 금지"],
        "folk_remedies": ["생꿀 도포(항균 보조)", "마늘 즙 희석 도포"],
        "urgency": "빠른 진료", "urgency_color": "orange"
    },
    "Typhoid": {
        "drugs": [{"name": "시프로플록사신", "type": "항생제", "note": "처방 필요"},
                  {"name": "아지스로마이신", "type": "항생제", "note": "처방 필요"}],
        "treatments": ["깨끗한 물 음용", "끓인 음식 섭취", "충분한 휴식", "격리"],
        "folk_remedies": ["바나나(전해질)", "꿀물 섭취", "생강 레몬차"],
        "urgency": "즉시 병원", "urgency_color": "red"
    },
}

# 기본 치료 정보 (DB에 없는 질병용 폴백)
DEFAULT_TREATMENT = {
    "drugs": [{"name": "해당 질환 전문의 처방 필요", "type": "전문 처방", "note": "병원 방문 권장"}],
    "treatments": ["전문의 진료 후 치료 계획 수립", "충분한 휴식", "수분 섭취"],
    "folk_remedies": ["전문의 상담 후 보조 요법 결정"],
    "urgency": "빠른 진료", "urgency_color": "orange"
}

# ────────────────────────────────────────────────────────────────────────────────
# 데이터 생성 및 모델 학습
# ────────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def train_models():
    """앙상블 ML 모델 학습 (GaussianNB + RandomForest) — @cache_resource로 1회만 실행"""
    np.random.seed(42)
    diseases = list(DISEASE_SYMPTOMS.keys())
    symptom_list = ALL_SYMPTOMS

    X, y = [], []
    for disease, symptoms in DISEASE_SYMPTOMS.items():
        for _ in range(60):  # 질병당 60샘플 (노이즈 증강)
            vec = np.zeros(len(symptom_list))
            for s in symptoms:
                if s in symptom_list:
                    idx = symptom_list.index(s)
                    weight = SEVERITY_WEIGHT.get(s, 3) / 7.0
                    # 노이즈: 70~100% 확률로 해당 증상 포함
                    if np.random.random() > 0.25:
                        vec[idx] = weight
            # 노이즈 증상 0~2개 추가
            n_noise = np.random.randint(0, 3)
            noise_idxs = np.random.choice(len(symptom_list), n_noise, replace=False)
            for ni in noise_idxs:
                if vec[ni] == 0:
                    vec[ni] = np.random.uniform(0.05, 0.2)
            X.append(vec)
            y.append(disease)

    X = np.array(X)
    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    nb = GaussianNB()
    nb.fit(X, y_enc)

    rf = RandomForestClassifier(n_estimators=200, max_depth=20, random_state=42, n_jobs=-1)
    rf.fit(X, y_enc)

    return nb, rf, le, symptom_list, diseases

# ────────────────────────────────────────────────────────────────────────────────
# 예측 함수
# ────────────────────────────────────────────────────────────────────────────────
def predict_diseases(selected_symptoms, model_choice, nb, rf, le, symptom_list, top_n=8):
    vec = np.zeros((1, len(symptom_list)))
    for s in selected_symptoms:
        if s in symptom_list:
            idx = symptom_list.index(s)
            weight = SEVERITY_WEIGHT.get(s, 3) / 7.0
            vec[0, idx] = weight

    nb_proba = nb.predict_proba(vec)[0]
    rf_proba = rf.predict_proba(vec)[0]

    if model_choice == "앙상블":
        proba = (nb_proba + rf_proba) / 2
    elif model_choice == "Naive Bayes":
        proba = nb_proba
    else:
        proba = rf_proba

    classes = le.classes_
    result = sorted(zip(classes, proba), key=lambda x: x[1], reverse=True)[:top_n]

    rows = []
    for disease, prob in result:
        rows.append({
            "disease": disease,
            "disease_kr": DISEASE_KR.get(disease, disease),
            "probability": prob,
            "prob_pct": round(prob * 100, 1),
        })
    return pd.DataFrame(rows)

# ────────────────────────────────────────────────────────────────────────────────
# 신체 부위 연관도 계산
# ────────────────────────────────────────────────────────────────────────────────
def calc_body_intensity(result_df):
    part_scores = {}
    part_diseases = {}
    for _, row in result_df.iterrows():
        parts = DISEASE_BODY_PARTS.get(row["disease"], [])
        for part in parts:
            part_scores[part] = part_scores.get(part, 0) + row["probability"]
            if part not in part_diseases:
                part_diseases[part] = []
            part_diseases[part].append({
                "name": row["disease"], "kr": row["disease_kr"], "prob": row["prob_pct"]
            })
    # 정규화
    if part_scores:
        max_v = max(part_scores.values())
        part_scores = {k: v / max_v for k, v in part_scores.items()}
    return part_scores, part_diseases

# ────────────────────────────────────────────────────────────────────────────────
# Plotly 차트
# ────────────────────────────────────────────────────────────────────────────────
def make_chart(result_df):
    colors = []
    for _, row in result_df.iterrows():
        disease = row["disease"]
        treatment = TREATMENT_DB.get(disease, DEFAULT_TREATMENT)
        uc = treatment.get("urgency_color", "green")
        if uc == "red":
            colors.append("#e53e3e")
        elif uc == "orange":
            colors.append("#dd6b20")
        else:
            colors.append("#38a169")

    fig = go.Figure(go.Bar(
        x=result_df["prob_pct"],
        y=[f"{r['disease_kr']}<br><span style='font-size:10px;color:#718096'>{r['disease']}</span>"
           for _, r in result_df.iterrows()],
        orientation="h",
        marker_color=colors,
        text=[f"{p}%" for p in result_df["prob_pct"]],
        textposition="outside",
        hovertemplate="<b>%{y}</b><br>확률: %{x:.1f}%<extra></extra>",
    ))
    fig.update_layout(
        height=max(320, len(result_df) * 48),
        margin=dict(l=0, r=60, t=20, b=20),
        xaxis=dict(title="확률 (%)", range=[0, min(100, result_df["prob_pct"].max() * 1.3)]),
        yaxis=dict(autorange="reversed", tickfont=dict(size=12)),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Malgun Gothic, sans-serif"),
    )
    return fig

# ────────────────────────────────────────────────────────────────────────────────
# 긴급도 카드
# ────────────────────────────────────────────────────────────────────────────────
def render_urgency_cards(result_df):
    for i, (_, row) in enumerate(result_df.head(3).iterrows()):
        treatment = TREATMENT_DB.get(row["disease"], DEFAULT_TREATMENT)
        uc = treatment.get("urgency_color", "green")
        urgency = treatment.get("urgency", "경과 관찰")
        badge_cls = f"badge-{uc}"
        card_cls = f"urgency-{uc}"
        rank_emoji = ["🥇", "🥈", "🥉"][i]
        st.markdown(f"""
        <div class="urgency-card {card_cls}">
            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:6px;">
                <span style="font-weight:700; font-size:15px;">{rank_emoji} {row['disease_kr']}</span>
                <span class="{badge_cls}">{urgency}</span>
            </div>
            <div style="color:#4a5568; font-size:13px;">{row['disease']} · <b>{row['prob_pct']}%</b></div>
        </div>
        """, unsafe_allow_html=True)

# ────────────────────────────────────────────────────────────────────────────────
# 인체 해부도 SVG — 귀엽고 둥근 디자인 + 한국어 라벨 + 장기 상세 정보
# ────────────────────────────────────────────────────────────────────────────────

# 장기별 기능 및 병리 설명
ORGAN_INFO = {
    "brain": {
        "emoji": "🧠", "name": "뇌",
        "function": "인체의 최고 사령부. 사고·감각·운동·기억·언어를 총괄하며 자율신경계를 통해 전신 기능을 조절합니다.",
        "pathology": "뇌졸중(뇌경색·뇌출혈), 편두통, 알츠하이머, 뇌전증, 뇌수막염 등이 발생할 수 있습니다.",
    },
    "eyes": {
        "emoji": "👁️", "name": "눈",
        "function": "빛을 수용해 시각 정보를 뇌로 전달하는 감각기관. 각막·수정체·망막이 협력해 초점을 맺습니다.",
        "pathology": "결막염, 녹내장, 백내장, 황반변성, 당뇨망막병증 등에 취약합니다.",
    },
    "ears": {
        "emoji": "👂", "name": "귀",
        "function": "소리를 감지하고 평형 감각을 유지합니다. 외이·중이·내이의 3단계 구조로 이루어져 있습니다.",
        "pathology": "중이염, 이명, 메니에르병(어지럼증), 이석증(체위성 현기증) 등이 흔합니다.",
    },
    "nose": {
        "emoji": "👃", "name": "코/부비동",
        "function": "공기를 여과·가습·가온하고 후각을 담당합니다. 부비동은 공기 통로와 연결된 빈 공간입니다.",
        "pathology": "알레르기 비염, 부비동염(축농증), 비용종, 후각 장애 등이 발생합니다.",
    },
    "throat": {
        "emoji": "🗣️", "name": "인후/편도",
        "function": "음식물과 공기의 교차로이며 발성에 관여합니다. 편도는 면역 방어 1차 관문 역할을 합니다.",
        "pathology": "편도염, 인두염, 후두염, 연하곤란, 성대결절 등이 발생할 수 있습니다.",
    },
    "esophagus": {
        "emoji": "🔴", "name": "식도",
        "function": "입에서 위까지 음식물을 운반하는 근육성 관(길이 약 25cm). 연동운동으로 내용물을 이동시킵니다.",
        "pathology": "위식도역류질환(GERD), 식도염, 식도협착, 바렛식도, 식도암 등이 발생합니다.",
    },
    "thyroid": {
        "emoji": "🦋", "name": "갑상선",
        "function": "나비 모양의 내분비샘. 갑상선 호르몬(T3·T4)을 분비해 대사율·체온·심박수를 조절합니다.",
        "pathology": "갑상선 기능 저하증(피로·체중증가), 갑상선 기능 항진증(빠른 심박·체중감소), 갑상선암이 주요 질환입니다.",
    },
    "lungs": {
        "emoji": "🫁", "name": "폐",
        "function": "산소를 흡수하고 이산화탄소를 배출하는 호흡기관. 약 3억 개의 폐포로 구성됩니다.",
        "pathology": "천식, 폐렴, 결핵, 만성폐쇄성폐질환(COPD), 폐암, 기관지염이 주요 질환입니다.",
    },
    "heart": {
        "emoji": "❤️", "name": "심장",
        "function": "하루 약 10만 번 박동하며 전신에 혈액을 공급하는 펌프. 4개의 방으로 이루어져 있습니다.",
        "pathology": "심근경색(심장마비), 협심증, 심부전, 부정맥, 심장판막질환이 대표적입니다.",
    },
    "liver": {
        "emoji": "🟤", "name": "간",
        "function": "500가지 이상의 기능을 담당하는 최대 내장기관. 해독·담즙생성·단백질 합성·혈당 조절을 수행합니다.",
        "pathology": "간염(A/B/C형), 지방간, 간경변증, 간암, 알코올성 간질환이 흔합니다.",
    },
    "gallbladder": {
        "emoji": "🟡", "name": "담낭",
        "function": "간에서 생성된 담즙을 저장·농축했다가 지방 소화 시 십이지장으로 분비합니다.",
        "pathology": "담석증, 담낭염, 담낭 폴립, 담낭암이 주요 질환입니다.",
    },
    "stomach": {
        "emoji": "🫙", "name": "위",
        "function": "강력한 산(pH 1~3)과 효소로 음식물을 분해합니다. 음식물을 4~6시간 저장·혼합합니다.",
        "pathology": "위염, 소화성 궤양, 위식도역류, 헬리코박터 감염, 위암이 발생할 수 있습니다.",
    },
    "spleen": {
        "emoji": "🟣", "name": "비장",
        "function": "오래된 적혈구를 파괴하고 면역세포를 생산·저장합니다. 혈액 저장소 역할도 합니다.",
        "pathology": "비장 비대(비종), 말라리아·뎅기열에 의한 파열, 혈액 질환에서 이차적으로 영향받습니다.",
    },
    "pancreas": {
        "emoji": "🟠", "name": "췌장",
        "function": "인슐린·글루카곤을 분비해 혈당을 조절(내분비)하고, 소화효소를 십이지장으로 분비(외분비)합니다.",
        "pathology": "당뇨병(1·2형), 급성/만성 췌장염, 췌장암이 대표적 질환입니다.",
    },
    "kidneys": {
        "emoji": "🫘", "name": "신장",
        "function": "하루 약 180L의 혈액을 여과해 소변을 생성합니다. 혈압·전해질·산-염기 균형을 조절합니다.",
        "pathology": "신우신염(요로감염 합병), 신장결석, 만성신장병, 신부전이 주요 질환입니다.",
    },
    "intestines": {
        "emoji": "🌀", "name": "소장·대장",
        "function": "소장(6~7m)에서 영양소를 흡수하고, 대장(1.5m)에서 수분을 흡수·변을 형성합니다.",
        "pathology": "장염, 크론병, 과민성대장증후군(IBS), 대장암, 치질, 장 폐색이 발생합니다.",
    },
    "bladder": {
        "emoji": "💧", "name": "방광",
        "function": "신장에서 생성된 소변을 저장(400~600mL)했다가 배뇨 시 요도를 통해 배출합니다.",
        "pathology": "방광염(요로감염), 과민성 방광, 방광결석, 방광암이 주요 질환입니다.",
    },
    "joints": {
        "emoji": "🦴", "name": "관절",
        "function": "두 뼈를 연결해 움직임을 가능하게 합니다. 연골·활액막·인대·힘줄로 구성됩니다.",
        "pathology": "골관절염(퇴행성), 류마티스 관절염, 통풍, 강직성 척추염이 흔한 관절 질환입니다.",
    },
    "blood": {
        "emoji": "🩸", "name": "혈액/혈관",
        "function": "적혈구(산소운반)·백혈구(면역)·혈소판(지혈)으로 구성됩니다. 혈관은 전신에 약 96,000km 뻗어 있습니다.",
        "pathology": "빈혈, 고혈압, 동맥경화, 정맥류, 혈전증, 백혈병이 발생할 수 있습니다.",
    },
    "lymph": {
        "emoji": "🔵", "name": "림프계",
        "function": "조직액을 수집해 혈액으로 돌려보내고, 림프절에서 병원체를 걸러내는 면역 네트워크입니다.",
        "pathology": "림프절 종창(감염·암), 림프부종, 림프종(혹킨·비호킨)이 주요 질환입니다.",
    },
    "skin": {
        "emoji": "🧴", "name": "피부",
        "function": "인체 최대 기관(체표면적 약 1.7㎡). 외부 자극 차단·체온 조절·감각수용·비타민D 합성을 담당합니다.",
        "pathology": "아토피·건선·여드름·두드러기·대상포진·피부암이 대표적입니다.",
    },
    "immune": {
        "emoji": "🛡️", "name": "면역계",
        "function": "병원체·이물질로부터 신체를 방어합니다. 선천면역(신속·비특이적)과 후천면역(느린·특이적)으로 나뉩니다.",
        "pathology": "자가면역질환(루푸스·류마티스), 면역결핍(HIV/AIDS), 알레르기, 과민반응이 발생합니다.",
    },
}

def render_body_svg(part_intensity, part_diseases):
    def intensity_color(score):
        if score == 0:    return "#dbeafe"   # 연파랑(비연관)
        elif score < 0.25: return "#bbf7d0"  # 연초록
        elif score < 0.5:  return "#fde68a"  # 노랑
        elif score < 0.75: return "#fdba74"  # 주황
        else:              return "#fca5a5"  # 연빨강

    def stroke_color(score):
        if score == 0:    return "#93c5fd"
        elif score < 0.25: return "#4ade80"
        elif score < 0.5:  return "#fbbf24"
        elif score < 0.75: return "#f97316"
        else:              return "#ef4444"

    import json

    # 장기 데이터 (연관 질병 + 기능·병리)
    part_full_data = {}
    for part, info in ORGAN_INFO.items():
        diseases_str = ""
        if part in part_diseases:
            diseases_str = " / ".join([f"{d['kr']} {d['prob']}%" for d in part_diseases[part][:4]])
        part_full_data[part] = {
            "emoji":     info["emoji"],
            "name":      info["name"],
            "function":  info["function"],
            "pathology": info["pathology"],
            "diseases":  diseases_str,
            "score":     round(part_intensity.get(part, 0) * 100),
        }

    part_data_str = json.dumps(part_full_data, ensure_ascii=False)

    def c(p):   return intensity_color(part_intensity.get(p, 0))
    def sc(p):  return stroke_color(part_intensity.get(p, 0))

    svg_html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%);
    font-family: 'Malgun Gothic', 'Apple SD Gothic Neo', sans-serif;
    min-height: 100vh;
  }}

  /* ── 범례 ── */
  .legend {{
    display: flex; gap: 10px; justify-content: center;
    padding: 8px 0 4px; flex-wrap: wrap;
  }}
  .legend-item {{
    display: flex; align-items: center; gap: 5px;
    font-size: 11px; color: #475569; font-weight: 500;
  }}
  .legend-dot {{
    width: 13px; height: 13px; border-radius: 50%;
    border: 2px solid rgba(0,0,0,0.12);
  }}

  /* ── SVG 장기 공통 ── */
  .organ {{
    cursor: pointer;
    transition: transform 0.18s cubic-bezier(.34,1.56,.64,1),
                filter 0.18s ease,
                opacity 0.18s;
    transform-origin: center;
    transform-box: fill-box;
  }}
  .organ:hover {{
    transform: scale(1.12);
    filter: drop-shadow(0 4px 10px rgba(59,130,246,0.45));
  }}
  .organ:active {{ transform: scale(0.96); }}

  /* ── 한국어 라벨 ── */
  .organ-label {{
    pointer-events: none;
    user-select: none;
    font-size: 8.5px;
    font-weight: 700;
    fill: #1e3a5f;
    letter-spacing: -0.3px;
    paint-order: stroke;
    stroke: rgba(255,255,255,0.85);
    stroke-width: 3px;
  }}

  /* ── 정보 패널 ── */
  #info-panel {{
    position: fixed;
    bottom: 14px; left: 50%;
    transform: translateX(-50%) translateY(30px);
    background: white;
    border-radius: 18px;
    padding: 0;
    box-shadow: 0 8px 32px rgba(0,0,0,0.18);
    width: 92%; max-width: 420px;
    display: none;
    overflow: hidden;
    animation: slideUp 0.28s cubic-bezier(.34,1.56,.64,1) forwards;
  }}
  @keyframes slideUp {{
    from {{ transform: translateX(-50%) translateY(30px); opacity: 0; }}
    to   {{ transform: translateX(-50%) translateY(0);   opacity: 1; }}
  }}
  #panel-header {{
    padding: 12px 16px 10px;
    display: flex; align-items: center; gap: 10px;
  }}
  #panel-emoji {{ font-size: 26px; line-height: 1; }}
  #panel-title {{
    font-size: 16px; font-weight: 800; color: #0f172a; line-height: 1.2;
  }}
  #panel-score {{
    margin-left: auto;
    font-size: 11px; font-weight: 700;
    padding: 3px 9px; border-radius: 20px;
    background: #eff6ff; color: #1d4ed8;
  }}
  .panel-section {{
    padding: 8px 16px;
    border-top: 1px solid #f1f5f9;
    font-size: 12px; line-height: 1.55; color: #334155;
  }}
  .panel-section-title {{
    font-size: 10px; font-weight: 700; color: #64748b;
    text-transform: uppercase; letter-spacing: 0.6px;
    margin-bottom: 3px;
  }}
  #panel-diseases {{
    padding: 8px 16px 12px;
    border-top: 1px solid #f1f5f9;
  }}
  .disease-chips {{
    display: flex; flex-wrap: wrap; gap: 5px; margin-top: 4px;
  }}
  .disease-chip {{
    background: #fef2f2; color: #b91c1c;
    font-size: 11px; font-weight: 600;
    padding: 2px 9px; border-radius: 20px;
    border: 1px solid #fecaca;
  }}
  .no-disease {{
    font-size: 12px; color: #94a3b8; font-style: italic;
  }}
  #close-btn {{
    position: absolute; top: 10px; right: 12px;
    background: #f1f5f9; border: none; border-radius: 50%;
    width: 22px; height: 22px; cursor: pointer;
    font-size: 13px; color: #64748b; line-height: 22px; text-align: center;
  }}
  #close-btn:hover {{ background: #e2e8f0; }}
</style>
</head>
<body>

<!-- 범례 -->
<div class="legend">
  <div class="legend-item"><div class="legend-dot" style="background:#dbeafe;border-color:#93c5fd"></div>비연관</div>
  <div class="legend-item"><div class="legend-dot" style="background:#bbf7d0;border-color:#4ade80"></div>낮음</div>
  <div class="legend-item"><div class="legend-dot" style="background:#fde68a;border-color:#fbbf24"></div>중간</div>
  <div class="legend-item"><div class="legend-dot" style="background:#fdba74;border-color:#f97316"></div>높음</div>
  <div class="legend-item"><div class="legend-dot" style="background:#fca5a5;border-color:#ef4444"></div>매우 높음</div>
</div>

<!-- 메인 SVG -->
<svg viewBox="0 0 380 680" xmlns="http://www.w3.org/2000/svg"
     width="100%" style="max-height:640px; display:block;">
  <defs>
    <!-- 몸통 그라디언트 -->
    <linearGradient id="bodyGrad" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0%" stop-color="#e0f2fe"/>
      <stop offset="100%" stop-color="#bae6fd"/>
    </linearGradient>
    <!-- 장기 기본 그라디언트 (밝기 오버레이) -->
    <radialGradient id="organShine" cx="35%" cy="30%" r="60%">
      <stop offset="0%" stop-color="white" stop-opacity="0.35"/>
      <stop offset="100%" stop-color="white" stop-opacity="0"/>
    </radialGradient>
    <!-- 그림자 필터 -->
    <filter id="softShadow" x="-20%" y="-20%" width="140%" height="140%">
      <feDropShadow dx="0" dy="2" stdDeviation="3" flood-color="#94a3b8" flood-opacity="0.25"/>
    </filter>
    <filter id="bodyShadow" x="-5%" y="-2%" width="110%" height="108%">
      <feDropShadow dx="0" dy="3" stdDeviation="5" flood-color="#7dd3fc" flood-opacity="0.3"/>
    </filter>
  </defs>

  <!-- ── 제목 ── -->
  <text x="190" y="22" text-anchor="middle" font-size="15" font-weight="800"
        fill="#0f172a" letter-spacing="-0.5">🧬 인체 해부도</text>
  <text x="190" y="37" text-anchor="middle" font-size="10" fill="#64748b">
    장기를 클릭하면 기능·병리 정보를 확인할 수 있어요
  </text>

  <!-- ══════════════ 몸통 실루엣 ══════════════ -->
  <!-- 머리 -->
  <ellipse cx="190" cy="82" rx="52" ry="60"
           fill="url(#bodyGrad)" stroke="#93c5fd" stroke-width="2" filter="url(#bodyShadow)"/>
  <!-- 목 -->
  <rect x="172" y="138" width="36" height="36" rx="14"
        fill="url(#bodyGrad)" stroke="#93c5fd" stroke-width="2"/>
  <!-- 몸통 -->
  <path d="M112,172 Q94,196 96,274 Q98,336 112,372 Q136,394 190,396 Q244,394 268,372 Q282,336 284,274 Q286,196 268,172 Z"
        fill="url(#bodyGrad)" stroke="#93c5fd" stroke-width="2" filter="url(#bodyShadow)"/>
  <!-- 왼팔 -->
  <path d="M112,178 Q78,196 68,254 Q62,288 70,314 Q78,332 90,332 Q104,332 110,314 Q118,288 118,254 Z"
        fill="url(#bodyGrad)" stroke="#93c5fd" stroke-width="2"/>
  <!-- 오른팔 -->
  <path d="M268,178 Q302,196 312,254 Q318,288 310,314 Q302,332 290,332 Q276,332 270,314 Q262,288 262,254 Z"
        fill="url(#bodyGrad)" stroke="#93c5fd" stroke-width="2"/>
  <!-- 왼다리 -->
  <path d="M148,392 Q138,420 134,480 Q128,532 130,566 Q132,582 148,584 Q164,584 168,566 Q172,532 170,480 Q168,432 164,394 Z"
        fill="url(#bodyGrad)" stroke="#93c5fd" stroke-width="2"/>
  <!-- 오른다리 -->
  <path d="M232,392 Q242,420 246,480 Q252,532 250,566 Q248,582 232,584 Q216,584 212,566 Q208,532 210,480 Q212,432 216,394 Z"
        fill="url(#bodyGrad)" stroke="#93c5fd" stroke-width="2"/>

  <!-- ══════════════ 피부 오버레이 (클릭 불가) ══════════════ -->
  <rect x="96" y="172" width="188" height="224" rx="28"
        fill="{c('skin')}" opacity="0.18" style="pointer-events:none;"/>

  <!-- ══════════════ 장기들 ══════════════ -->

  <!-- 눈 (좌·우) -->
  <g class="organ" data-part="eyes" filter="url(#softShadow)">
    <ellipse cx="170" cy="72" rx="10" ry="7" fill="{c('eyes')}" stroke="{sc('eyes')}" stroke-width="2"/>
    <ellipse cx="210" cy="72" rx="10" ry="7" fill="{c('eyes')}" stroke="{sc('eyes')}" stroke-width="2"/>
    <ellipse cx="170" cy="72" rx="10" ry="7" fill="url(#organShine)"/>
    <ellipse cx="210" cy="72" rx="10" ry="7" fill="url(#organShine)"/>
    <circle cx="170" cy="73" r="3.5" fill="{sc('eyes')}" opacity="0.55"/>
    <circle cx="210" cy="73" r="3.5" fill="{sc('eyes')}" opacity="0.55"/>
  </g>
  <text class="organ-label" x="190" y="67" text-anchor="middle">눈</text>

  <!-- 귀 (좌·우) -->
  <g class="organ" data-part="ears" filter="url(#softShadow)">
    <ellipse cx="140" cy="88" rx="8" ry="13" fill="{c('ears')}" stroke="{sc('ears')}" stroke-width="2"/>
    <ellipse cx="240" cy="88" rx="8" ry="13" fill="{c('ears')}" stroke="{sc('ears')}" stroke-width="2"/>
    <ellipse cx="140" cy="88" rx="8" ry="13" fill="url(#organShine)"/>
    <ellipse cx="240" cy="88" rx="8" ry="13" fill="url(#organShine)"/>
  </g>
  <text class="organ-label" x="140" y="103" text-anchor="middle">귀</text>
  <text class="organ-label" x="240" y="103" text-anchor="middle">귀</text>

  <!-- 코 -->
  <g class="organ" data-part="nose" filter="url(#softShadow)">
    <ellipse cx="190" cy="96" rx="8" ry="7" fill="{c('nose')}" stroke="{sc('nose')}" stroke-width="2"/>
    <ellipse cx="190" cy="96" rx="8" ry="7" fill="url(#organShine)"/>
  </g>
  <text class="organ-label" x="190" y="109" text-anchor="middle">코</text>

  <!-- 뇌 -->
  <g class="organ" data-part="brain" filter="url(#softShadow)">
    <ellipse cx="190" cy="66" rx="42" ry="30" fill="{c('brain')}" stroke="{sc('brain')}" stroke-width="2.5"/>
    <ellipse cx="190" cy="66" rx="42" ry="30" fill="url(#organShine)"/>
    <!-- 뇌 주름 장식 -->
    <path d="M162,58 Q172,52 182,58 Q192,52 202,58 Q212,52 218,60"
          fill="none" stroke="{sc('brain')}" stroke-width="1.5" opacity="0.5" stroke-linecap="round"/>
    <path d="M160,68 Q170,62 180,68 Q190,62 200,68 Q210,62 220,68"
          fill="none" stroke="{sc('brain')}" stroke-width="1.5" opacity="0.5" stroke-linecap="round"/>
  </g>
  <text class="organ-label" x="190" y="68" text-anchor="middle">뇌</text>

  <!-- 인후/편도 -->
  <g class="organ" data-part="throat" filter="url(#softShadow)">
    <rect x="176" y="141" width="28" height="22" rx="11"
          fill="{c('throat')}" stroke="{sc('throat')}" stroke-width="2"/>
    <rect x="176" y="141" width="28" height="22" rx="11" fill="url(#organShine)"/>
  </g>
  <text class="organ-label" x="190" y="157" text-anchor="middle">인후</text>

  <!-- 갑상선 -->
  <g class="organ" data-part="thyroid" filter="url(#softShadow)">
    <ellipse cx="190" cy="172" rx="22" ry="12" fill="{c('thyroid')}" stroke="{sc('thyroid')}" stroke-width="2"/>
    <ellipse cx="190" cy="172" rx="22" ry="12" fill="url(#organShine)"/>
  </g>
  <text class="organ-label" x="190" y="175" text-anchor="middle">갑상선</text>

  <!-- 식도 (갑상선 뒤 배경, 클릭 불가) -->
  <rect x="184" y="155" width="12" height="34" rx="6"
        fill="{c('esophagus')}" stroke="{sc('esophagus')}" stroke-width="1.5"
        opacity="0.7" style="pointer-events:none;"/>

  <!-- 림프 (좌·우 목 아래) -->
  <g class="organ" data-part="lymph" filter="url(#softShadow)">
    <circle cx="120" cy="198" r="10" fill="{c('lymph')}" stroke="{sc('lymph')}" stroke-width="2"/>
    <circle cx="260" cy="198" r="10" fill="{c('lymph')}" stroke="{sc('lymph')}" stroke-width="2"/>
    <circle cx="120" cy="198" r="10" fill="url(#organShine)"/>
    <circle cx="260" cy="198" r="10" fill="url(#organShine)"/>
  </g>
  <text class="organ-label" x="120" y="213" text-anchor="middle">림프</text>
  <text class="organ-label" x="260" y="213" text-anchor="middle">림프</text>

  <!-- 혈관 (팔, 클릭 가능) -->
  <g class="organ" data-part="blood" filter="url(#softShadow)">
    <rect x="72" y="204" width="16" height="88" rx="8"
          fill="{c('blood')}" stroke="{sc('blood')}" stroke-width="2" opacity="0.85"/>
    <rect x="292" y="204" width="16" height="88" rx="8"
          fill="{c('blood')}" stroke="{sc('blood')}" stroke-width="2" opacity="0.85"/>
  </g>
  <text class="organ-label" x="80" y="252" text-anchor="middle">혈관</text>

  <!-- 폐 (좌·우) — 둥글둥글한 형태 -->
  <g class="organ" data-part="lungs" filter="url(#softShadow)">
    <path d="M118,192 Q104,204 102,234 Q100,262 114,278 Q128,290 146,282 Q156,272 156,248 Q156,216 148,196 Z"
          fill="{c('lungs')}" stroke="{sc('lungs')}" stroke-width="2.5"/>
    <path d="M262,192 Q276,204 278,234 Q280,262 266,278 Q252,290 234,282 Q224,272 224,248 Q224,216 232,196 Z"
          fill="{c('lungs')}" stroke="{sc('lungs')}" stroke-width="2.5"/>
    <path d="M118,192 Q104,204 102,234 Q100,262 114,278 Q128,290 146,282 Q156,272 156,248 Q156,216 148,196 Z"
          fill="url(#organShine)"/>
    <path d="M262,192 Q276,204 278,234 Q280,262 266,278 Q252,290 234,282 Q224,272 224,248 Q224,216 232,196 Z"
          fill="url(#organShine)"/>
    <!-- 폐 세엽 장식선 -->
    <path d="M116,214 Q124,210 132,216" fill="none" stroke="{sc('lungs')}" stroke-width="1.5" opacity="0.5" stroke-linecap="round"/>
    <path d="M264,214 Q256,210 248,216" fill="none" stroke="{sc('lungs')}" stroke-width="1.5" opacity="0.5" stroke-linecap="round"/>
  </g>
  <text class="organ-label" x="122" y="242" text-anchor="middle">좌폐</text>
  <text class="organ-label" x="258" y="242" text-anchor="middle">우폐</text>

  <!-- 심장 — 귀여운 하트형 -->
  <g class="organ" data-part="heart" filter="url(#softShadow)">
    <path d="M185,196 Q174,188 168,196 Q162,204 174,216 Q182,224 190,230 Q198,224 206,216 Q218,204 212,196 Q206,188 195,196 Q193,199 190,202 Q187,199 185,196 Z"
          fill="{c('heart')}" stroke="{sc('heart')}" stroke-width="2.5"/>
    <path d="M185,196 Q174,188 168,196 Q162,204 174,216 Q182,224 190,230 Q198,224 206,216 Q218,204 212,196 Q206,188 195,196 Q193,199 190,202 Q187,199 185,196 Z"
          fill="url(#organShine)"/>
  </g>
  <text class="organ-label" x="190" y="218" text-anchor="middle">심장</text>

  <!-- 간 — 오른쪽 둥근 덩어리 -->
  <g class="organ" data-part="liver" filter="url(#softShadow)">
    <path d="M216,236 Q242,230 250,248 Q256,266 244,280 Q228,292 208,288 Q194,282 192,266 Q196,244 216,236 Z"
          fill="{c('liver')}" stroke="{sc('liver')}" stroke-width="2.5"/>
    <path d="M216,236 Q242,230 250,248 Q256,266 244,280 Q228,292 208,288 Q194,282 192,266 Q196,244 216,236 Z"
          fill="url(#organShine)"/>
  </g>
  <text class="organ-label" x="224" y="264" text-anchor="middle">간</text>

  <!-- 담낭 -->
  <g class="organ" data-part="gallbladder" filter="url(#softShadow)">
    <ellipse cx="238" cy="292" rx="12" ry="16" fill="{c('gallbladder')}" stroke="{sc('gallbladder')}" stroke-width="2"/>
    <ellipse cx="238" cy="292" rx="12" ry="16" fill="url(#organShine)"/>
  </g>
  <text class="organ-label" x="238" y="314" text-anchor="middle">담낭</text>

  <!-- 비장 — 왼쪽 -->
  <g class="organ" data-part="spleen" filter="url(#softShadow)">
    <ellipse cx="144" cy="270" rx="18" ry="24" fill="{c('spleen')}" stroke="{sc('spleen')}" stroke-width="2"/>
    <ellipse cx="144" cy="270" rx="18" ry="24" fill="url(#organShine)"/>
  </g>
  <text class="organ-label" x="144" y="274" text-anchor="middle">비장</text>

  <!-- 위 -->
  <g class="organ" data-part="stomach" filter="url(#softShadow)">
    <path d="M166,254 Q155,264 154,282 Q154,300 166,310 Q180,318 196,310 Q204,298 202,280 Q200,260 188,252 Z"
          fill="{c('stomach')}" stroke="{sc('stomach')}" stroke-width="2.5"/>
    <path d="M166,254 Q155,264 154,282 Q154,300 166,310 Q180,318 196,310 Q204,298 202,280 Q200,260 188,252 Z"
          fill="url(#organShine)"/>
    <!-- 위 주름 -->
    <path d="M162,272 Q174,268 186,274" fill="none" stroke="{sc('stomach')}" stroke-width="1.5" opacity="0.5" stroke-linecap="round"/>
  </g>
  <text class="organ-label" x="178" y="284" text-anchor="middle">위</text>

  <!-- 췌장 — 가로 가지 모양 -->
  <g class="organ" data-part="pancreas" filter="url(#softShadow)">
    <rect x="154" y="314" width="62" height="18" rx="9"
          fill="{c('pancreas')}" stroke="{sc('pancreas')}" stroke-width="2"/>
    <rect x="154" y="314" width="62" height="18" rx="9" fill="url(#organShine)"/>
  </g>
  <text class="organ-label" x="185" y="326" text-anchor="middle">췌장</text>

  <!-- 신장 (좌·우) -->
  <g class="organ" data-part="kidneys" filter="url(#softShadow)">
    <ellipse cx="142" cy="308" rx="15" ry="22" fill="{c('kidneys')}" stroke="{sc('kidneys')}" stroke-width="2"/>
    <ellipse cx="238" cy="308" rx="15" ry="22" fill="{c('kidneys')}" stroke="{sc('kidneys')}" stroke-width="2"/>
    <ellipse cx="142" cy="308" rx="15" ry="22" fill="url(#organShine)"/>
    <ellipse cx="238" cy="308" rx="15" ry="22" fill="url(#organShine)"/>
  </g>
  <text class="organ-label" x="142" y="312" text-anchor="middle">신장</text>
  <text class="organ-label" x="238" y="312" text-anchor="middle">신장</text>

  <!-- 소장·대장 — 둥글게 감긴 모양 -->
  <g class="organ" data-part="intestines" filter="url(#softShadow)">
    <path d="M140,334 Q122,344 124,366 Q126,386 148,394 Q172,400 196,394 Q218,386 220,366 Q222,344 204,334 Q186,326 190,350 Q190,368 170,370 Q150,368 152,350 Q152,336 140,334 Z"
          fill="{c('intestines')}" stroke="{sc('intestines')}" stroke-width="2.5"/>
    <path d="M140,334 Q122,344 124,366 Q126,386 148,394 Q172,400 196,394 Q218,386 220,366 Q222,344 204,334 Q186,326 190,350 Q190,368 170,370 Q150,368 152,350 Q152,336 140,334 Z"
          fill="url(#organShine)"/>
  </g>
  <text class="organ-label" x="190" y="365" text-anchor="middle">소장·대장</text>

  <!-- 방광 -->
  <g class="organ" data-part="bladder" filter="url(#softShadow)">
    <ellipse cx="190" cy="398" rx="24" ry="18" fill="{c('bladder')}" stroke="{sc('bladder')}" stroke-width="2"/>
    <ellipse cx="190" cy="398" rx="24" ry="18" fill="url(#organShine)"/>
  </g>
  <text class="organ-label" x="190" y="401" text-anchor="middle">방광</text>

  <!-- 무릎 관절 -->
  <g class="organ" data-part="joints" filter="url(#softShadow)">
    <circle cx="148" cy="508" r="18" fill="{c('joints')}" stroke="{sc('joints')}" stroke-width="2.5"/>
    <circle cx="232" cy="508" r="18" fill="{c('joints')}" stroke="{sc('joints')}" stroke-width="2.5"/>
    <circle cx="148" cy="508" r="18" fill="url(#organShine)"/>
    <circle cx="232" cy="508" r="18" fill="url(#organShine)"/>
    <!-- 무릎 십자선 장식 -->
    <line x1="140" y1="508" x2="156" y2="508" stroke="{sc('joints')}" stroke-width="1.5" opacity="0.5"/>
    <line x1="148" y1="500" x2="148" y2="516" stroke="{sc('joints')}" stroke-width="1.5" opacity="0.5"/>
    <line x1="224" y1="508" x2="240" y2="508" stroke="{sc('joints')}" stroke-width="1.5" opacity="0.5"/>
    <line x1="232" y1="500" x2="232" y2="516" stroke="{sc('joints')}" stroke-width="1.5" opacity="0.5"/>
  </g>
  <text class="organ-label" x="148" y="530" text-anchor="middle">무릎관절</text>
  <text class="organ-label" x="232" y="530" text-anchor="middle">무릎관절</text>

  <!-- 다리 하부 (정강이) -->
  <g class="organ" data-part="legs" filter="url(#softShadow)">
    <rect x="133" y="527" width="30" height="52" rx="15"
          fill="{c('legs')}" stroke="{sc('legs')}" stroke-width="2" opacity="0.85"/>
    <rect x="217" y="527" width="30" height="52" rx="15"
          fill="{c('legs')}" stroke="{sc('legs')}" stroke-width="2" opacity="0.85"/>
  </g>

  <!-- 면역계 배지 (우측 상단) -->
  <g class="organ" data-part="immune" filter="url(#softShadow)">
    <rect x="298" y="155" width="54" height="28" rx="14"
          fill="{c('immune')}" stroke="{sc('immune')}" stroke-width="2"/>
    <rect x="298" y="155" width="54" height="28" rx="14" fill="url(#organShine)"/>
  </g>
  <text class="organ-label" x="325" y="173" text-anchor="middle">면역계</text>

  <!-- 피부 배지 (좌측 상단) -->
  <g class="organ" data-part="skin" filter="url(#softShadow)">
    <rect x="28" y="155" width="50" height="28" rx="14"
          fill="{c('skin')}" stroke="{sc('skin')}" stroke-width="2"/>
    <rect x="28" y="155" width="50" height="28" rx="14" fill="url(#organShine)"/>
  </g>
  <text class="organ-label" x="53" y="173" text-anchor="middle">피부</text>

</svg>

<!-- ── 정보 패널 ── -->
<div id="info-panel">
  <button id="close-btn" onclick="closePanel()">✕</button>
  <div id="panel-header">
    <span id="panel-emoji">🫀</span>
    <span id="panel-title">장기명</span>
    <span id="panel-score">연관도 0%</span>
  </div>
  <div class="panel-section">
    <div class="panel-section-title">⚙️ 주요 기능</div>
    <div id="panel-function">—</div>
  </div>
  <div class="panel-section">
    <div class="panel-section-title">🔬 주요 병리</div>
    <div id="panel-pathology">—</div>
  </div>
  <div id="panel-diseases">
    <div class="panel-section-title">🩺 예측 연관 질병</div>
    <div class="disease-chips" id="panel-chips"></div>
  </div>
</div>

<script>
const PART_DATA = {part_data_str};
let hideTimer = null;

function closePanel() {{
  const p = document.getElementById('info-panel');
  p.style.display = 'none';
  if (hideTimer) clearTimeout(hideTimer);
}}

document.querySelectorAll('.organ').forEach(el => {{
  el.addEventListener('click', function(e) {{
    e.stopPropagation();
    // data-part는 <g> 또는 자식에 있을 수 있음
    const part = this.getAttribute('data-part')
                 || this.closest('[data-part]')?.getAttribute('data-part');
    if (!part || !PART_DATA[part]) return;

    const d = PART_DATA[part];
    const panel = document.getElementById('info-panel');

    document.getElementById('panel-emoji').textContent   = d.emoji;
    document.getElementById('panel-title').textContent   = d.name;
    document.getElementById('panel-score').textContent   = '연관도 ' + d.score + '%';
    document.getElementById('panel-function').textContent  = d.function;
    document.getElementById('panel-pathology').textContent = d.pathology;

    const chipsEl = document.getElementById('panel-chips');
    chipsEl.innerHTML = '';
    if (d.diseases) {{
      d.diseases.split(' / ').forEach(item => {{
        const chip = document.createElement('span');
        chip.className = 'disease-chip';
        chip.textContent = item;
        chipsEl.appendChild(chip);
      }});
    }} else {{
      chipsEl.innerHTML = '<span class="no-disease">선택 증상과 직접 연관된 질병 없음</span>';
    }}

    panel.style.animation = 'none';
    panel.offsetHeight; // reflow
    panel.style.animation = '';
    panel.style.display = 'block';

    if (hideTimer) clearTimeout(hideTimer);
    hideTimer = setTimeout(closePanel, 8000);
  }});
}});

// 패널 바깥 클릭 시 닫기
document.addEventListener('click', function(e) {{
  const panel = document.getElementById('info-panel');
  if (!e.target.closest('.organ') && !e.target.closest('#info-panel')) {{
    closePanel();
  }}
}});
</script>
</body>
</html>"""
    return svg_html

# ────────────────────────────────────────────────────────────────────────────────
# 치료 정보 탭 렌더링
# ────────────────────────────────────────────────────────────────────────────────
def render_treatment(result_df):
    top_diseases = result_df.head(3)
    tabs = st.tabs([f"{'🥇🥈🥉'[i]} {row['disease_kr']}" for i, (_, row) in enumerate(top_diseases.iterrows())])

    for i, (tab, (_, row)) in enumerate(zip(tabs, top_diseases.iterrows())):
        with tab:
            treatment = TREATMENT_DB.get(row["disease"], DEFAULT_TREATMENT)
            uc = treatment.get("urgency_color", "green")
            urgency = treatment.get("urgency", "경과 관찰")
            badge_cls = f"badge-{uc}"

            st.markdown(f"""
            <div style="margin-bottom:12px;">
              <span style="font-size:16px; font-weight:700;">{row['disease_kr']}</span>
              &nbsp;&nbsp;<span class="{badge_cls}">{urgency}</span>
              &nbsp;<span style="color:#718096; font-size:13px;">확률: {row['prob_pct']}%</span>
            </div>
            """, unsafe_allow_html=True)

            col1, col2, col3 = st.columns(3)
            with col1:
                drugs_html = "".join([
                    f"<li><b>{d['name']}</b> <span style='color:#718096;font-size:12px;'>({d['type']})</span><br>"
                    f"<span style='color:#9ca3af;font-size:11px;'>{d['note']}</span></li>"
                    for d in treatment["drugs"]
                ])
                st.markdown(f"""
                <div class="treatment-card">
                  <h4>💊 추천 약품</h4>
                  <ul style="list-style:none; padding:0;">{drugs_html}</ul>
                </div>
                """, unsafe_allow_html=True)
            with col2:
                treats_html = "".join([f"<li style='margin-bottom:6px;'>{t}</li>" for t in treatment["treatments"]])
                st.markdown(f"""
                <div class="treatment-card">
                  <h4>🏥 치료·관리법</h4>
                  <ul>{treats_html}</ul>
                </div>
                """, unsafe_allow_html=True)
            with col3:
                folk_html = "".join([f"<li style='margin-bottom:6px;'>{f}</li>" for f in treatment["folk_remedies"]])
                st.markdown(f"""
                <div class="treatment-card">
                  <h4>🌿 민간요법</h4>
                  <ul>{folk_html}</ul>
                  <p style="font-size:11px;color:#9ca3af;margin-top:8px;">※ 민간요법은 보조적 참고용이며 의학적 효능을 보증하지 않습니다.</p>
                </div>
                """, unsafe_allow_html=True)

# ────────────────────────────────────────────────────────────────────────────────
# 메인 앱
# ────────────────────────────────────────────────────────────────────────────────
def main():
    # ── 면책 조항 (고정 배너) ──────────────────────────────────────────────────
    st.markdown("""
    <div class="disclaimer-banner">
      ⚠️ <b>본 서비스는 건강 정보 참고용이며 전문의 진단을 대체할 수 없습니다.</b>
      &nbsp;의료기기 아님 · 진단 대체 불가 · 참고용 건강 정보 도구
    </div>
    """, unsafe_allow_html=True)

    # ── 헤더 ──────────────────────────────────────────────────────────────────
    col_h1, col_h2 = st.columns([3, 1])
    with col_h1:
        st.markdown("## 🏥 Health Symptom Analyzer")
        st.caption("증상 기반 질병 예측 헬스케어 대시보드 · 41종 질병 · 132개 증상 · NB+RF 앙상블 모델")
    with col_h2:
        st.markdown("""
        <div style="display:flex; gap:8px; justify-content:flex-end; padding-top:10px; flex-wrap:wrap;">
          <span style="background:#e0f2fe; color:#0369a1; padding:4px 10px; border-radius:20px; font-size:12px; font-weight:600;">📊 41종 질병</span>
          <span style="background:#f0fdf4; color:#166534; padding:4px 10px; border-radius:20px; font-size:12px; font-weight:600;">🔬 NB+RF 앙상블</span>
          <span style="background:#fef9c3; color:#854d0e; padding:4px 10px; border-radius:20px; font-size:12px; font-weight:600;">0원 서버비</span>
        </div>
        """, unsafe_allow_html=True)
    st.markdown("---")

    # ── 모델 로딩 ─────────────────────────────────────────────────────────────
    with st.spinner("🔬 ML 모델 초기화 중..."):
        nb, rf, le, symptom_list, diseases = train_models()

    # ── 사이드바 ──────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("### ⚙️ 설정")
        model_choice = st.radio("ML 모델 선택", ["앙상블", "Naive Bayes", "Random Forest"],
                                 help="앙상블: NB+RF 확률 평균 (권장)")
        top_n = st.slider("결과 표시 개수", min_value=3, max_value=15, value=8,
                           help="상위 N개 예측 질병 표시")
        st.markdown("---")
        st.markdown("### 🩺 증상 선택")
        st.caption("해당하는 증상을 모두 선택하세요 (최소 1개)")

        # 카테고리 간 중복 증상 제거: 먼저 등장한 카테고리에만 표시
        already_rendered: set = set()
        selected_symptoms = []
        first_cat = True
        for cat_idx, (cat_name, cat_symptoms) in enumerate(SYMPTOM_CATEGORIES.items()):
            # 이 카테고리에서 처음 등장하는 증상만, SYMPTOM_KR에 존재하는 것만
            valid = [
                s for s in cat_symptoms
                if s in SYMPTOM_KR and s not in already_rendered
            ]
            already_rendered.update(valid)
            if not valid:
                continue
            with st.expander(f"{'▼' if first_cat else '▶'} {cat_name} ({len(valid)}개)", expanded=first_cat):
                for sym in valid:
                    kr_label = SYMPTOM_KR.get(sym, sym)
                    # key = 카테고리 인덱스 + 증상명 → 전역 고유 보장
                    if st.checkbox(kr_label, key=f"cb_{cat_idx}_{sym}"):
                        selected_symptoms.append(sym)
            first_cat = False

        st.markdown("---")
        st.caption(f"✅ 선택된 증상: **{len(selected_symptoms)}개**")

        if len(selected_symptoms) == 0:
            st.warning("최소 1개 이상의 증상을 선택하세요.")

        predict_btn = st.button("🔍 질병 예측 실행", type="primary",
                                 disabled=(len(selected_symptoms) == 0),
                                 use_container_width=True)

    # ── 메인 콘텐츠 ───────────────────────────────────────────────────────────
    if len(selected_symptoms) == 0:
        # 초기 화면
        st.markdown("""
        <div style="text-align:center; padding: 60px 20px;">
          <div style="font-size: 64px; margin-bottom: 16px;">🩺</div>
          <h3 style="color:#2d3748; margin-bottom:8px;">증상을 선택하고 질병을 예측하세요</h3>
          <p style="color:#718096;">좌측 사이드바에서 현재 느끼는 증상을 선택한 후<br>
          <b>질병 예측 실행</b> 버튼을 클릭하면 AI가 분석합니다.</p>
        </div>
        """, unsafe_allow_html=True)

        # 요약 메트릭
        m1, m2, m3, m4 = st.columns(4)
        for col, val, lbl in zip(
            [m1, m2, m3, m4],
            ["41종", "132개", "NB+RF", "0원"],
            ["지원 질병", "분석 증상", "앙상블 모델", "초기 서버 비용"],
        ):
            col.markdown(f"""
            <div class="metric-box">
              <div class="value">{val}</div>
              <div class="label">{lbl}</div>
            </div>
            """, unsafe_allow_html=True)
        return

    # ── 예측 실행 ─────────────────────────────────────────────────────────────
    result_df = predict_diseases(selected_symptoms, model_choice, nb, rf, le, symptom_list, top_n)
    part_intensity, part_diseases = calc_body_intensity(result_df)

    # 선택 증상 표시
    kr_symptoms = [SYMPTOM_KR.get(s, s) for s in selected_symptoms]
    st.markdown(
        "**선택된 증상:** " + " · ".join([f"`{s}`" for s in kr_symptoms])
    )

    # ── 3탭 레이아웃 ──────────────────────────────────────────────────────────
    tab1, tab2, tab3 = st.tabs(["📊 예측 결과", "🫀 인체 해부도", "💊 치료·약품 정보"])

    # ─ Tab 1: 예측 결과 ───────────────────────────────────────────────────────
    with tab1:
        col_chart, col_cards = st.columns([3, 2])

        with col_chart:
            st.markdown("#### 📊 질병 예측 확률")
            fig = make_chart(result_df)
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

            with st.expander("📋 전체 결과 테이블"):
                display_df = result_df[["disease_kr", "disease", "prob_pct"]].copy()
                display_df.columns = ["질병명 (한국어)", "Disease (EN)", "확률 (%)"]
                st.dataframe(display_df, use_container_width=True, hide_index=True)

        with col_cards:
            st.markdown("#### 🏷️ 상위 3개 질병 분석")
            render_urgency_cards(result_df)
            st.markdown("""
            <div style="font-size:11px; color:#9ca3af; margin-top:8px;">
              🔴 즉시 병원 &nbsp; 🟠 빠른 진료 &nbsp; 🟢 경과 관찰
            </div>
            """, unsafe_allow_html=True)

    # ─ Tab 2: 인체 해부도 ─────────────────────────────────────────────────────
    with tab2:
        col_svg, col_info = st.columns([2, 1])

        with col_svg:
            st.markdown("#### 🫀 인체 해부도 — 연관 부위 시각화")
            st.caption("장기를 클릭하면 연관 질병 정보를 확인할 수 있습니다")
            svg_html = render_body_svg(part_intensity, part_diseases)
            components.html(svg_html, height=620, scrolling=False)

        with col_info:
            st.markdown("#### 📍 부위별 연관도")
            sorted_parts = sorted(part_intensity.items(), key=lambda x: x[1], reverse=True)
            part_kr = {
                "brain": "🧠 뇌", "heart": "❤️ 심장", "lungs": "🫁 폐",
                "liver": "🟤 간", "stomach": "🟡 위", "intestines": "🔵 장",
                "kidneys": "🟣 신장", "bladder": "🔷 방광", "skin": "🟠 피부",
                "joints": "🦴 관절", "thyroid": "🟢 갑상선", "nose": "👃 코",
                "throat": "🔴 인후", "eyes": "👁️ 눈", "ears": "👂 귀",
                "pancreas": "🔶 췌장", "esophagus": "🟤 식도", "gallbladder": "🟡 담낭",
                "spine": "⬜ 척추", "legs": "🦵 다리", "blood": "🩸 혈액",
                "lymph": "🔵 림프", "immune": "🛡️ 면역", "spleen": "🟣 비장",
            }
            for part, score in sorted_parts[:8]:
                label = part_kr.get(part, part)
                bar_color = "#ef4444" if score > 0.75 else "#f97316" if score > 0.5 else "#eab308" if score > 0.25 else "#22c55e"
                st.markdown(f"""
                <div style="margin-bottom:8px;">
                  <div style="display:flex; justify-content:space-between; font-size:13px;">
                    <span>{label}</span><span style="color:#718096;">{score*100:.0f}%</span>
                  </div>
                  <div style="background:#e2e8f0; border-radius:4px; height:6px; margin-top:3px;">
                    <div style="background:{bar_color}; width:{score*100:.0f}%; height:6px; border-radius:4px;"></div>
                  </div>
                </div>
                """, unsafe_allow_html=True)

    # ─ Tab 3: 치료·약품 정보 ──────────────────────────────────────────────────
    with tab3:
        st.markdown("#### 💊 치료·약품·민간요법 정보")
        st.caption("상위 3개 예측 질병의 치료 정보를 제공합니다")
        render_treatment(result_df)
        st.markdown("""
        <br>
        <div style="background:#f1f5f9; border-radius:8px; padding:12px 16px; font-size:12px; color:#64748b;">
          ⚠️ <b>주의사항:</b> 약품 정보는 참고용이며 반드시 전문의·약사의 처방 및 지도 하에 복용하세요.
          민간요법은 의학적 효능이 검증되지 않은 경우가 있습니다.
          본 서비스는 의료기기가 아니며 전문의 진단을 대체할 수 없습니다.
        </div>
        """, unsafe_allow_html=True)

    # ── 푸터 ──────────────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("""
    <div style="text-align:center; color:#94a3b8; font-size:12px; padding:8px 0;">
      Health Symptom Analyzer · v1.0 · 2026.03 &nbsp;|&nbsp;
      데이터: Kaggle(kaushil268) · itachi9604 GitHub &nbsp;|&nbsp;
      의료기기 아님 · 진단 대체 불가 · 참고용 건강 정보 도구
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
