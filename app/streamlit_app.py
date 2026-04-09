import sys
from pathlib import Path
import streamlit as st

SEX_MAP = {
    "Masculino": 1,
    "Feminino": 2
}

MAR_MAP = {
    "Casado": 1,
    "Viúvo": 2,
    "Divorciado": 3,
    "Separado": 4,
    "Nunca casou": 5
}

SCHL_MAP = {
    "Sem escolaridade": 1,
    "Fundamental incompleto": 8,
    "Fundamental completo": 12,
    "Ensino médio incompleto": 14,
    "Ensino médio completo": 16,
    "Alguma faculdade": 18,
    "Bacharelado": 21,
    "Mestrado": 22,
    "Doutorado": 24
}

COW_MAP = {
    "Empregado empresa privada": 1,
    "Empregado governo": 2,
    "Autônomo": 3,
    "Trabalhador familiar": 4,
    "Desempregado": 9
}

RACE_MAP = {
    "Branco": 1,
    "Negro": 2,
    "Indígena americano": 3,
    "Alasca nativo": 4,
    "Indiano/asiático": 5,
    "Havaiano/pacífico": 6,
    "Outro": 7,
    "Múltiplas raças": 8,
    "Não informado": 9
}

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR / "src"))

from predict import predict_income

st.set_page_config(
    page_title="Preditor de Renda",
    page_icon="💼",
    layout="centered"
)

st.title("💼 Preditor de Renda")
st.caption("Aplicação de Machine Learning baseada no ACS 2021 (Folktables)")

st.markdown("Preencha os dados abaixo para estimar se o indivíduo pertence à classe de renda mais alta.")

with st.form("income_form"):

    agep = st.slider("Idade", 18, 90, 40)

    sex_label = st.selectbox("Sexo", list(SEX_MAP.keys()))
    sex = SEX_MAP[sex_label]

    mar_label = st.selectbox("Estado civil", list(MAR_MAP.keys()))
    mar = MAR_MAP[mar_label]

    schl_label = st.selectbox("Escolaridade", list(SCHL_MAP.keys()))
    schl = SCHL_MAP[schl_label]

    cow_label = st.selectbox("Tipo de trabalho", list(COW_MAP.keys()))
    cow = COW_MAP[cow_label]

    race_label = st.selectbox("Raça", list(RACE_MAP.keys()))
    rac1p = RACE_MAP[race_label]

    wkhp = st.slider("Horas trabalhadas por semana", 1, 80, 40)

    occp = st.number_input("Código da ocupação", 0, 9999, 2200)

    pobp = st.number_input("Local de nascimento (código)", 1, 100, 6)

    submitted = st.form_submit_button("Prever renda")

if submitted:
    input_data = {
        "AGEP": agep,
        "COW": cow,
        "SCHL": schl,
        "MAR": mar,
        "OCCP": occp,
        "POBP": pobp,
        "WKHP": wkhp,
        "SEX": sex,
        "RAC1P": rac1p,
    }

    prediction, probability = predict_income(input_data)

    st.divider()

    if prediction == 1:
        st.success("Previsão: pertence à classe de renda mais alta.")
    else:
        st.warning("Previsão: não pertence à classe de renda mais alta.")

    if probability is not None:
        st.metric("Probabilidade estimada da classe alta", f"{probability:.2%}")

with st.expander("Exemplo de entrada"):
    st.json({
        "AGEP": 40,
        "COW": 1,
        "SCHL": 16,
        "MAR": 1,
        "OCCP": 2200,
        "POBP": 6,
        "WKHP": 40,
        "SEX": 1,
        "RAC1P": 1
    })