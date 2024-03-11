import pandas as pd
import numpy as np
import scorecardpy as sc
from xgboost import XGBClassifier
import joblib
import json
import streamlit as st

features = ['person_income', 'person_home_ownership', 'loan_intent', 'loan_grade', 'loan_amnt',
       'loan_int_rate', 'loan_percent_income','cb_person_default_on_file', 'cb_person_cred_hist_length']

#----- load model ----
loaded_model = joblib.load('xgboost_model.pkl')

#----- Bins ----
with open('woe_bins.json', 'r') as f:
    loaded_bins = json.load(f)

for key, value in loaded_bins.items():
    if isinstance(value, dict):
        loaded_bins[key] = pd.DataFrame(value)

#---- streamlit -----
#layout
c30, c31, c32 = st.columns([4,1,3])

with c30:
    st.title('''💳 Prediksi Approval Nasabah''')
    st.header("")

with st.expander("ℹ️ - Tentang Aplikasi",expanded=True):
    st.write(
        '''
        - *Prediksi Approval* mudah digunakan dan aplikasi ini dibuat menggunakan streamlit
        - *Prediksi Approval* menggunakan XGBoost untuk menentukan apakah permohonan pinjaman akan di Approve atau Reject
        '''
    )

    st.markdown("")

st.markdown('## 💸 Enter Profile details below')

with st.form(key='Form'):
    ce,c1,ce,c2,c3 = st.columns([0.07,1,0.07,5,0.007])

    person_income = st.text_input(
        "Person Income",
        value="0",
        help="Please enter your Income"
    )

    person_income = int(person_income)

    loan_amount = st.text_input(
        "Loan Mount",
        value="0",
        help="Please enter your Loan"
    )

    loan_amount = int(loan_amount)

    loan_int_rate = st.text_input(
        "Loan Interest Rate",
        value="0.0",
        help="Please enter your Loan Interest Rate"
    )

    loan_int_rate = float(loan_int_rate)

    cb_person_cred_hist_length = st.slider(
        "Credit History Length",
        min_value=0,
        max_value=30,
        value=2,
        help="Please select your Credit Hitory Lenght"
    )

    cb_person_cred_hist_length=int(cb_person_cred_hist_length)
    
    home_ownership = st.selectbox(
        "Home Ownership Status",['OWN','MORTGAGE','RENT','OTHER'],index=0
    )

    loan_intent = st.selectbox(
        "Loan Intent type",['PERSONAL', 'EDUCATION', 'MEDICAL', 'VENTURE', 'HOMEIMPROVEMENT','DEBTCONSOLIDATION'],index=0
    )

    loan_grade = st.selectbox(
        "Loan Grade",['A', 'B', 'C', 'D', 'E', 'F', 'G'],index=0
    )

    cb_person_default_on_file = st.selectbox(
        "person Default on File",['Y', 'N'],index=0
    )

    features = [[person_income,loan_amount,loan_int_rate,home_ownership,loan_intent,loan_grade,cb_person_default_on_file]]

    submit_button = st.form_submit_button(label='Submit')

if submit_button:
    df = pd.DataFrame(features,columns=['person_income','loan_amnt','loan_int_rate','person_home_ownership','loan_intent','loan_grade','cb_person_default_on_file'])
    try:
        df["loan_percent_income"] = round((df['loan_amnt']) / df['person_income'], 2)
    except:
        df["loan_percent_income"] = 0


    df_main  = sc.woebin_ply(df, loaded_bins)
    df_main = df_main[['loan_amnt_woe', 'loan_int_rate_woe', 'person_income_woe', 'loan_percent_income_woe', 'person_home_ownership_woe', 'loan_grade_woe', 'loan_intent_woe', 'cb_person_default_on_file_woe']]
    # st.dataframe(df_main)
    
    pred = loaded_model.predict_proba(df_main)[:,1]
    pred = np.where(pred >= 0.109, "Default", "Approve")
    st.write(f'🤖 believes that the nasabah is  {"".join(pred)}')
