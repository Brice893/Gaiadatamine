from tkinter import W
import streamlit as st
from streamlit_shap import st_shap
import shap
import pandas as pd
from pandas import MultiIndex, Int16Dtype
import numpy as np
import matplotlib
#import seaborn as sns
import requests
import json
import pickle
#import os
from starlette.responses import Response
#from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import io
import plotly.express as px
import plotly.figure_factory as f
import plotly.graph_objs as go
import streamlit.components.v1 as components
#import xgboost as xgb
#from sklearn.ensemble import RandomForestClassifier
from PIL import Image
import warnings
warnings.filterwarnings("ignore", category=UserWarning)


# Command to execute script locally: streamlit run app.py
# Command to run Docker image: docker run -d -p 8501:8501 <streamlit-app-name>:latest

st.sidebar.title("Prêt à dépenser")
#st.write ('---debug chargement image ')
########################################################
# Loading images to the website
########################################################
image = Image.open("storage/credit.png")
st.sidebar.image(image)

# charger le modèle
pickle_lgb = open("storage/final_model_lgb.pkl", "rb")
model = pickle.load(pickle_lgb)

def main_page():

    st.sidebar.markdown("# Calcul du risque")
    st.title('Calcul du risque de remboursement de prêt')

    st.subheader("Prédictions de scoring client et positionnement dans l'ensemble des clients")

    if 'client' not in st.session_state:
        st.session_state.client = 0
    else:
        id_input = st.session_state.client

    entities = requests.post('http://127.0.0.1:8000/liste_id',timeout=8500)

    liste_id = entities.json()

    id_input = st.selectbox('Choisissez le client que vous souhaitez visualiser', liste_id)
    st.session_state.client = id_input

    #
    #-- récup infos
    entities = requests.post('http://127.0.0.1:8000/liste_df',timeout=8500)
    liste_df = entities.json()

    df = pd.DataFrame(liste_df, columns =[
    'TARGET',
    'SK_ID_CURR',
     'AGE',
     'DAYS_EMPLOYED_PERCENT',
       'DAYS_LAST_PHONE_CHANGE',
       'REG_CITY_NOT_WORK_CITY',
       'FLAG_EMP_PHONE',
       'DEF_60_CNT_SOCIAL_CIRCLE',
       'DEF_30_CNT_SOCIAL_CIRCLE',
       'AMT_REQ_CREDIT_BUREAU_YEAR',
       'CNT_CHILDREN',
       'EXT_SOURCE_3',
       'EXT_SOURCE_2',
       'FLOORSMAX_AVG',
       'FLOORSMAX_MEDI',
       'FLOORSMAX_MODE',
       'AMT_GOODS_PRICE',
       'REGION_POPULATION_RELATIVE',
       'AMT_CREDIT',
       'AMT_ANNUITY',
       'AMT_REQ_CREDIT_BUREAU_MON',
       'CODE_GENDER_F'
       ])

    client_infos = df[df['SK_ID_CURR'] == id_input]


    # pour variables locales SHAP plus loin
    X_client_infos = df[df['SK_ID_CURR'] == id_input]

    TARGET = client_infos['TARGET'].values

    client_infos.drop(['SK_ID_CURR', 'TARGET'], axis=1, inplace= True)

    features = client_infos.columns

    client = json.dumps({"num_client": id_input})
    header = {'Content-Type': 'application/json'}
    response = requests.request("POST","http://127.0.0.1:8000/client_infos",headers=header,data=client)

    prediction = response.text

    if "1" in prediction:
        if TARGET == 1:
            st.error('Crédit Refusé (TP)')
        else:
            st.success('Crédit Accepté (FP)')
    else:
        if TARGET == 1:
            st.error('Crédit Refusé (FN)')
        else:
            st.success('Crédit Accepté (TN)')


    # Variables globales du modèle
    st.header('Variables globales du modèle :')
    image_feature = Image.open("storage/barplot_icare.png")
    st.image(image_feature)

    focus_var = st.selectbox('Choisissez la variable de focus',
                                     ['AMT_REQ_CREDIT_BUREAU_YEAR',
                                     'EXT_SOURCE_2',
                                     'EXT_SOURCE_3'])

    st.header("Positionnement du client")

    if focus_var == 'AMT_REQ_CREDIT_BUREAU_YEAR':
        client = json.dumps({"num_client": id_input})
        header = {'Content-Type': 'application/json'}
        response = requests.request("POST","http://127.0.0.1:8000/get_bar_plot_1",headers=header,data=client)
        bar_plot = Image.open(io.BytesIO(response.content)).convert("RGB")
        st.image(bar_plot, use_column_width=True)

    if focus_var == 'EXT_SOURCE_2':
        client = json.dumps({"num_client": id_input})
        header = {'Content-Type': 'application/json'}
        response = requests.request("POST","http://127.0.0.1:8000/get_bar_plot_2",headers=header,data=client)
        bar_plot = Image.open(io.BytesIO(response.content)).convert("RGB")
        st.image(bar_plot, use_column_width=True)

    if focus_var == 'EXT_SOURCE_3':
        client = json.dumps({"num_client": id_input})
        header = {'Content-Type': 'application/json'}
        response = requests.request("POST","http://127.0.0.1:8000/get_bar_plot_3",headers=header,data=client)
        bar_plot = Image.open(io.BytesIO(response.content)).convert("RGB")
        st.image(bar_plot, use_column_width=True)

    # SHAP
    # Variables locales
    st.header('Variables locales du modèle :')
    # compute SHAP values

    X_client_infos.drop(['SK_ID_CURR', 'TARGET'], axis=1, inplace= True)

    # standardiser les variables
    scaler = MinMaxScaler(feature_range = (0, 1))
    scaler.fit(X_client_infos)
    scaled_var = scaler.transform(X_client_infos)

    df = pd.DataFrame(scaled_var, index=X_client_infos.index, columns=X_client_infos.columns)

    # Objet permettant de calculer les shap values
    data_for_prediction_array = df.values.reshape(1, -1)
    #data_for_prediction_array = X.values.reshape(1, -1)

    model.predict_proba(data_for_prediction_array)
    explainer = shap.TreeExplainer(model)

    # Calculate Shap values

    shap_values = explainer.shap_values(df)
    #st_shap(shap.summary_plot(shap_values, features=df, feature_names=df.columns))
    st_shap(shap.summary_plot(shap_values[0], df))

    st_shap(shap.force_plot(explainer.expected_value[1], shap_values[1], df))

    # informations du client
    st.header("Informations du client")
    st.write(X_client_infos)

    st.header("Transparence des informations du client ",id_input)

    #print (int(X_client_infos['AMT_REQ_CREDIT_BUREAU_YEAR']))
    AMT_REQ_CREDIT_BUREAU_YEAR = st.slider("AMT_REQ_CREDIT_BUREAU_YEAR", min_value=0, max_value=4, value= int(X_client_infos['AMT_REQ_CREDIT_BUREAU_YEAR']))
    X_client_infos['AMT_REQ_CREDIT_BUREAU_YEAR'] = AMT_REQ_CREDIT_BUREAU_YEAR

    #AMT_REQ_CREDIT_BUREAU_MON = st.slider("AMT_REQ_CREDIT_BUREAU_MON", min_value=0, max_value=4, value= int(X_client_infos['AMT_REQ_CREDIT_BUREAU_MON']))
    #X_client_infos['AMT_REQ_CREDIT_BUREAU_MON'] = AMT_REQ_CREDIT_BUREAU_MON

    #AMT_GOODS_PRICE = st.slider("AMT_GOODS_PRICE", min_value=0.0, max_value=800000.0, value = float(X_client_infos['AMT_GOODS_PRICE']))
    #X_client_infos['AMT_GOODS_PRICE'] = AMT_GOODS_PRICE

    CNT_CHILDREN = st.slider("CNT_CHILDREN",  min_value=0, max_value=4,  value= int(X_client_infos['CNT_CHILDREN']))
    X_client_infos['CNT_CHILDREN'] = CNT_CHILDREN

    EXT_SOURCE_2 = st.slider("EXT_SOURCE_2", min_value=0.0, max_value=1.0,value= float(X_client_infos['EXT_SOURCE_2']))
    X_client_infos['EXT_SOURCE_2'] = EXT_SOURCE_2

    EXT_SOURCE_3 = st.slider("EXT_SOURCE_3", min_value=0.0, max_value=1.0,value= float(X_client_infos['EXT_SOURCE_3']))
    X_client_infos['EXT_SOURCE_3'] = EXT_SOURCE_3

    DEF_30_CNT_SOCIAL_CIRCLE = st.slider("DEF_30_CNT_SOCIAL_CIRCLE", min_value=0.0, max_value=4.0, value= float(X_client_infos['DEF_30_CNT_SOCIAL_CIRCLE']))
    X_client_infos['DEF_30_CNT_SOCIAL_CIRCLE'] = DEF_30_CNT_SOCIAL_CIRCLE

    #AMT_CREDIT = st.slider("AMT_CREDIT", 0, 1600000, 10000)
    #X_client_infos['AMT_CREDIT'] = AMT_CREDIT

    scaler = MinMaxScaler(feature_range = (0, 1))
    scaler.fit(X_client_infos)
    scaled_var = scaler.transform(X_client_infos)

    df_pred = pd.DataFrame(scaled_var, index=X_client_infos.index, columns=X_client_infos.columns)

    predict = model.predict(df_pred)
    st.write("Nouvelle prédiction :")
    st.write(predict)

    predict_probability = model.predict_proba(X_client_infos)

    st.write("Probabilité d'appartenance aux classes : ", predict_probability)

    if predict_probability [0][0] > predict_probability [0][1]:
       st.success('Votre crédit serait accordé')
       st.subheader('Le client {} aurait une probabilité de remboursement de {}%'.format
                              (id_input ,round(predict_probability[0][0]*100 , 2)))
    else:
       st.error('Votre crédit serait refusé')
       st.subheader('Le client {} aurait une probabilité de défaut de paiement de {}%'.format
                                  (id_input ,round(predict_probability[0][1]*100 , 2)))


my_dict = {
    "Calcul du risque": main_page }


keys = list(my_dict.keys())

selected_page = st.sidebar.selectbox("Select a page", keys)
my_dict[selected_page]()
