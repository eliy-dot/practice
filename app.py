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
# 인체 해부도 SVG
# ────────────────────────────────────────────────────────────────────────────────
def render_body_svg(part_intensity, part_diseases):
    def intensity_color(score):
        if score == 0:
            return "#e2e8f0"
        elif score < 0.25:
            return "#fef9c3"
        elif score < 0.5:
            return "#fed7aa"
        elif score < 0.75:
            return "#fca5a5"
        else:
            return "#ef4444"

    part_data_json = {}
    for part, diseases in part_diseases.items():
        diseases_str = ", ".join([f"{d['kr']}({d['prob']}%)" for d in diseases[:3]])
        part_data_json[part] = diseases_str

    colors = {
        "brain": intensity_color(part_intensity.get("brain", 0)),
        "eyes": intensity_color(part_intensity.get("eyes", 0)),
        "ears": intensity_color(part_intensity.get("ears", 0)),
        "nose": intensity_color(part_intensity.get("nose", 0)),
        "throat": intensity_color(part_intensity.get("throat", 0)),
        "esophagus": intensity_color(part_intensity.get("esophagus", 0)),
        "thyroid": intensity_color(part_intensity.get("thyroid", 0)),
        "lungs": intensity_color(part_intensity.get("lungs", 0)),
        "heart": intensity_color(part_intensity.get("heart", 0)),
        "liver": intensity_color(part_intensity.get("liver", 0)),
        "gallbladder": intensity_color(part_intensity.get("gallbladder", 0)),
        "stomach": intensity_color(part_intensity.get("stomach", 0)),
        "spleen": intensity_color(part_intensity.get("spleen", 0)),
        "pancreas": intensity_color(part_intensity.get("pancreas", 0)),
        "kidneys": intensity_color(part_intensity.get("kidneys", 0)),
        "intestines": intensity_color(part_intensity.get("intestines", 0)),
        "bladder": intensity_color(part_intensity.get("bladder", 0)),
        "spine": intensity_color(part_intensity.get("spine", 0)),
        "joints": intensity_color(part_intensity.get("joints", 0)),
        "legs": intensity_color(part_intensity.get("legs", 0)),
        "blood": intensity_color(part_intensity.get("blood", 0)),
        "skin": intensity_color(part_intensity.get("skin", 0)),
        "lymph": intensity_color(part_intensity.get("lymph", 0)),
        "immune": intensity_color(part_intensity.get("immune", 0)),
    }

    import json
    part_data_str = json.dumps(part_data_json, ensure_ascii=False)

    svg_html = f"""
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
  body {{ margin: 0; background: transparent; font-family: 'Malgun Gothic', sans-serif; }}
  .organ {{ cursor: pointer; transition: opacity 0.2s; stroke: #94a3b8; stroke-width: 1.5; }}
  .organ:hover {{ opacity: 0.75; stroke: #3b82f6; stroke-width: 2.5; }}
  #info-panel {{
    position: fixed; bottom: 10px; left: 50%; transform: translateX(-50%);
    background: white; border-radius: 10px; padding: 12px 18px;
    box-shadow: 0 4px 16px rgba(0,0,0,0.15); font-size: 13px;
    max-width: 400px; width: 90%; display: none; border-left: 4px solid #3b82f6;
  }}
  .legend {{ display: flex; gap: 12px; justify-content: center; margin-bottom: 8px; font-size: 12px; }}
  .legend-item {{ display: flex; align-items: center; gap: 4px; }}
  .legend-dot {{ width: 12px; height: 12px; border-radius: 3px; }}
</style>
</head>
<body>
<div class="legend">
  <div class="legend-item"><div class="legend-dot" style="background:#e2e8f0"></div> 비연관</div>
  <div class="legend-item"><div class="legend-dot" style="background:#fef9c3"></div> 낮음</div>
  <div class="legend-item"><div class="legend-dot" style="background:#fed7aa"></div> 중간</div>
  <div class="legend-item"><div class="legend-dot" style="background:#fca5a5"></div> 높음</div>
  <div class="legend-item"><div class="legend-dot" style="background:#ef4444"></div> 매우 높음</div>
</div>
<svg viewBox="0 0 340 600" xmlns="http://www.w3.org/2000/svg" width="100%" style="max-height:560px">
  <!-- 배경 몸통 실루엣 -->
  <ellipse cx="170" cy="60" rx="42" ry="52" fill="#f1f5f9" stroke="#cbd5e1" stroke-width="1"/>
  <!-- 목 -->
  <rect x="155" y="108" width="30" height="30" rx="6" fill="#f1f5f9" stroke="#cbd5e1" stroke-width="1"/>
  <!-- 몸통 -->
  <path d="M100,138 Q85,160 88,230 Q90,290 100,340 Q120,360 170,362 Q220,360 240,340 Q250,290 252,230 Q255,160 240,138 Z" fill="#f1f5f9" stroke="#cbd5e1" stroke-width="1"/>
  <!-- 팔 -->
  <path d="M100,145 Q70,160 62,210 Q58,240 65,265 Q72,285 80,285 Q90,285 94,265 Q98,240 98,210 Z" fill="#f1f5f9" stroke="#cbd5e1" stroke-width="1"/>
  <path d="M240,145 Q270,160 278,210 Q282,240 275,265 Q268,285 260,285 Q250,285 246,265 Q242,240 242,210 Z" fill="#f1f5f9" stroke="#cbd5e1" stroke-width="1"/>
  <!-- 다리 -->
  <path d="M130,358 Q122,380 118,430 Q114,480 116,530 Q118,548 130,550 Q142,550 146,530 Q150,480 150,430 Q150,390 148,360 Z" fill="#f1f5f9" stroke="#cbd5e1" stroke-width="1"/>
  <path d="M210,358 Q218,380 222,430 Q226,480 224,530 Q222,548 210,550 Q198,550 194,530 Q190,480 190,430 Q190,390 192,360 Z" fill="#f1f5f9" stroke="#cbd5e1" stroke-width="1"/>

  <!-- 뇌 -->
  <ellipse class="organ" id="brain" cx="170" cy="52" rx="32" ry="38" fill="{colors['brain']}" data-part="brain" data-kr="뇌"/>

  <!-- 눈 (좌우) -->
  <ellipse class="organ" id="eyes" cx="152" cy="44" rx="7" ry="5" fill="{colors['eyes']}" data-part="eyes" data-kr="눈"/>
  <ellipse class="organ" id="eyes2" cx="188" cy="44" rx="7" ry="5" fill="{colors['eyes']}" data-part="eyes" data-kr="눈"/>

  <!-- 귀 -->
  <ellipse class="organ" id="ears" cx="130" cy="58" rx="6" ry="9" fill="{colors['ears']}" data-part="ears" data-kr="귀"/>
  <ellipse class="organ" id="ears2" cx="210" cy="58" rx="6" ry="9" fill="{colors['ears']}" data-part="ears" data-kr="귀"/>

  <!-- 코 -->
  <ellipse class="organ" id="nose" cx="170" cy="68" rx="6" ry="5" fill="{colors['nose']}" data-part="nose" data-kr="코"/>

  <!-- 목/인후 -->
  <rect class="organ" id="throat" x="158" y="110" width="24" height="18" rx="5" fill="{colors['throat']}" data-part="throat" data-kr="인후"/>

  <!-- 갑상선 -->
  <ellipse class="organ" id="thyroid" cx="170" cy="136" rx="18" ry="10" fill="{colors['thyroid']}" data-part="thyroid" data-kr="갑상선"/>

  <!-- 식도 -->
  <rect class="organ" id="esophagus" x="164" y="128" width="12" height="28" rx="4" fill="{colors['esophagus']}" data-part="esophagus" data-kr="식도"/>

  <!-- 폐 (좌우) -->
  <path class="organ" id="lungs" d="M108,155 Q96,170 96,200 Q96,230 108,248 Q124,258 140,250 Q148,238 148,210 Q148,175 140,158 Z" fill="{colors['lungs']}" data-part="lungs" data-kr="폐"/>
  <path class="organ" id="lungs2" d="M232,155 Q244,170 244,200 Q244,230 232,248 Q216,258 200,250 Q192,238 192,210 Q192,175 200,158 Z" fill="{colors['lungs']}" data-part="lungs" data-kr="폐"/>

  <!-- 심장 -->
  <path class="organ" id="heart" d="M156,158 Q148,152 144,160 Q140,168 150,178 Q160,186 170,192 Q180,186 190,178 Q200,168 196,160 Q192,152 184,158 Q180,162 170,170 Q160,162 156,158 Z" fill="{colors['heart']}" data-part="heart" data-kr="심장"/>

  <!-- 간 -->
  <path class="organ" id="liver" d="M196,200 Q220,196 228,212 Q232,228 222,240 Q208,250 190,248 Q178,244 176,230 Q180,210 196,200 Z" fill="{colors['liver']}" data-part="liver" data-kr="간"/>

  <!-- 담낭 -->
  <ellipse class="organ" id="gallbladder" cx="208" cy="252" rx="10" ry="14" fill="{colors['gallbladder']}" data-part="gallbladder" data-kr="담낭"/>

  <!-- 위 -->
  <path class="organ" id="stomach" d="M148,220 Q140,228 138,248 Q138,264 148,272 Q162,278 176,272 Q182,260 180,244 Q178,226 168,218 Z" fill="{colors['stomach']}" data-part="stomach" data-kr="위"/>

  <!-- 비장 -->
  <ellipse class="organ" id="spleen" cx="118" cy="234" rx="14" ry="18" fill="{colors['spleen'] if 'spleen' in colors else '#e2e8f0'}" data-part="spleen" data-kr="비장"/>

  <!-- 췌장 -->
  <rect class="organ" id="pancreas" x="138" y="272" width="52" height="14" rx="7" fill="{colors['pancreas']}" data-part="pancreas" data-kr="췌장"/>

  <!-- 신장 (좌우) -->
  <ellipse class="organ" id="kidneys" cx="118" cy="268" rx="12" ry="18" fill="{colors['kidneys']}" data-part="kidneys" data-kr="신장"/>
  <ellipse class="organ" id="kidneys2" cx="222" cy="268" rx="12" ry="18" fill="{colors['kidneys']}" data-part="kidneys" data-kr="신장"/>

  <!-- 장 -->
  <path class="organ" id="intestines" d="M120,288 Q106,298 108,320 Q110,340 126,350 Q148,358 170,358 Q192,358 214,350 Q230,340 232,320 Q234,298 220,288 Q200,282 170,284 Q140,282 120,288 Z" fill="{colors['intestines']}" data-part="intestines" data-kr="장"/>

  <!-- 방광 -->
  <ellipse class="organ" id="bladder" cx="170" cy="352" rx="20" ry="14" fill="{colors['bladder']}" data-part="bladder" data-kr="방광"/>

  <!-- 척추 -->
  <rect class="organ" id="spine" x="163" y="148" width="14" height="200" rx="5" fill="{colors['spine']}" opacity="0.4" data-part="spine" data-kr="척추"/>

  <!-- 관절 표시 (무릎 등) -->
  <circle class="organ" id="joints" cx="130" cy="470" r="14" fill="{colors['joints']}" data-part="joints" data-kr="관절"/>
  <circle class="organ" id="joints2" cx="210" cy="470" r="14" fill="{colors['joints']}" data-part="joints" data-kr="관절"/>

  <!-- 다리/혈관 -->
  <rect class="organ" id="legs" x="115" y="485" width="30" height="60" rx="6" fill="{colors['legs']}" data-part="legs" data-kr="다리"/>
  <rect class="organ" id="legs2" x="195" y="485" width="30" height="60" rx="6" fill="{colors['legs']}" data-part="legs" data-kr="다리"/>

  <!-- 피부 오버레이 (전신 반투명) -->
  <rect class="organ" id="skin" x="88" y="138" width="164" height="224" rx="20" fill="{colors['skin']}" opacity="0.15" data-part="skin" data-kr="피부"/>

  <!-- 림프 (측면 점) -->
  <circle class="organ" id="lymph" cx="104" cy="170" r="8" fill="{colors['lymph']}" data-part="lymph" data-kr="림프"/>
  <circle class="organ" id="lymph2" cx="236" cy="170" r="8" fill="{colors['lymph']}" data-part="lymph" data-kr="림프"/>

  <!-- 혈관/혈액 (팔 부위) -->
  <rect class="organ" id="blood" x="64" y="170" width="14" height="80" rx="5" fill="{colors['blood']}" opacity="0.5" data-part="blood" data-kr="혈액/혈관"/>
  <rect class="organ" id="blood2" x="262" y="170" width="14" height="80" rx="5" fill="{colors['blood']}" opacity="0.5" data-part="blood" data-kr="혈액/혈관"/>

  <!-- 레이블 -->
  <text x="170" y="20" text-anchor="middle" font-size="13" font-weight="700" fill="#1e293b">인체 해부도</text>
  <text x="170" y="33" text-anchor="middle" font-size="10" fill="#64748b">장기를 클릭하면 연관 질병을 확인할 수 있습니다</text>
</svg>

<div id="info-panel">
  <div id="info-title" style="font-weight:700; color:#1e293b; margin-bottom:4px;"></div>
  <div id="info-diseases" style="color:#4a5568;"></div>
</div>

<script>
const partData = {part_data_str};

document.querySelectorAll('.organ').forEach(el => {{
  el.addEventListener('click', function() {{
    const part = this.getAttribute('data-part');
    const kr = this.getAttribute('data-kr');
    const panel = document.getElementById('info-panel');
    const titleEl = document.getElementById('info-title');
    const disEl = document.getElementById('info-diseases');

    titleEl.textContent = '🫀 ' + kr + ' — 연관 질병';
    if (partData[part]) {{
      disEl.textContent = partData[part];
    }} else {{
      disEl.textContent = '선택한 증상과 직접 연관된 질병이 없습니다.';
    }}
    panel.style.display = 'block';
    setTimeout(() => {{ panel.style.display = 'none'; }}, 5000);
  }});
}});
</script>
</body>
</html>
"""
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

        selected_symptoms = []
        first_cat = True
        for cat_name, cat_symptoms in SYMPTOM_CATEGORIES.items():
            # 중복 없는 증상만
            valid = [s for s in cat_symptoms if s in SYMPTOM_KR]
            with st.expander(f"{'▼' if first_cat else '▶'} {cat_name} ({len(valid)}개)", expanded=first_cat):
                for sym in valid:
                    kr_label = SYMPTOM_KR.get(sym, sym)
                    if st.checkbox(f"{kr_label}", key=f"cb_{sym}"):
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
