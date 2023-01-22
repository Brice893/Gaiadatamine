import json
import pickle
import httpx
from fastapi import Body, FastAPI, Response
from sklearn.preprocessing import MinMaxScaler
from fastapi.logger import logger
from pydantic import BaseModel
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import io

# Modeling
import lightgbm

#import os
# Command to execute script locally: uvicorn main:app --host 0.0.0.0 --port 8000 --reload
# Command to run Docker image: docker run -d -p 8000:8000 <fastapi-app-name>:latest

app = FastAPI()

class Client(BaseModel):
    num_client: int

class BankCredit(BaseModel):
     AGE: int
     DAYS_EMPLOYED_PERCENT: float
     DAYS_LAST_PHONE_CHANGE: float
     REG_CITY_NOT_WORK_CITY: int
     FLAG_EMP_PHONE: int
     DEF_60_CNT_SOCIAL_CIRCLE: float
     DEF_30_CNT_SOCIAL_CIRCLE: float
     AMT_REQ_CREDIT_BUREAU_YEAR: int
     CNT_CHILDREN: int
     EXT_SOURCE_3: float
     EXT_SOURCE_2: float
     FLOORSMAX_AVG: int
     FLOORSMAX_MEDI: int
     FLOORSMAX_MODE: int
     AMT_GOODS_PRICE: float
     REGION_POPULATION_RELATIVE: int
     AMT_CREDIT: float
     AMT_ANNUITY: float
     AMT_REQ_CREDIT_BUREAU_MON: int
     CODE_GENDER_F : int


# Chargement des fichiers
def chargement_data(path):
        if path == "df1":
            pickle_fi = open("storage/df1.pkl", "rb")

        dataframe = pickle.load(pickle_fi)
        liste_id = dataframe['SK_ID_CURR'].tolist()
        liste_df = dataframe.values.tolist()
        return dataframe, liste_id, liste_df

# charger le modèle
pickle_lgb = open("storage/final_model_lgb.pkl", "rb")
model = pickle.load(pickle_lgb)

# datas encodées non standardisées pour dashboard examples
path = "df1"
dataframe, liste_id, liste_df  = chargement_data(path)

@app.post('/liste_id')
async def get_liste_id():
    return liste_id

@app.post('/liste_df')
async def get_liste_id():
    return liste_df

@app.post('/client_infos')
async def get_client_infos(data: Client):
    #print (data.num_client)
    data_df  = dataframe[dataframe['SK_ID_CURR'] == data.num_client]
    data_df.drop(['SK_ID_CURR', 'TARGET'], axis=1, inplace= True)
    #print (data_df)
    scaler = MinMaxScaler(feature_range = (0, 1))
    scaler.fit(data_df)
    scaled_var = scaler.transform(data_df)
    df_var = pd.DataFrame(scaled_var, index=data_df.index, columns=data_df.columns)

    # = model.predict(data_df)
    prediction = model.predict(df_var)

    if prediction == 1:
        return {'prediction': str(prediction)}
        headers = {
        "Content-Type": "appplication/json",
        "Link": '<http://context:ngsi-context.jsonld>; rel="http://www.w3c.org/ns/json-ld#context"; type="lapplication/ld+json"'
        }
    else:
        return {"prediction": 0}
        headers = {
        "Content-Type": "appplication/json",
        "Link": '<http://context:ngsi-context.jsonld>; rel="http://www.w3c.org/ns/json-ld#context"; type="lapplication/ld+json"'
        }

@app.post('/get_bar_plot_1')
def get_bar_1(data: Client):
    """
    Fonction qui prend les infos du client sous forme json et retourne
    le graph en barre

    Parameters
    ----------
    data : Client

    Returns
    -------
    Response
        bar plot for AMT_REQ_CREDIT_BUREAU_YEAR
    """
    barplot_df = pd.read_csv("storage/barplot_df.csv")
    barplot_df = barplot_df[barplot_df['indicator'] == 'AMT_REQ_CREDIT_BUREAU_YEAR']
    data_df  = dataframe[dataframe['SK_ID_CURR'] == data.num_client]
    data_df.drop(['SK_ID_CURR', 'TARGET'], axis=1, inplace= True)
    #data = data.dict()
    #data_df = pd.DataFrame.from_dict([data])
    df_user = pd.DataFrame(columns=['TARGET', 'value', 'indicator'])
    df_user = df_user.append({'TARGET': 'user value',
                              'indicator': 'AMT_REQ_CREDIT_BUREAU_YEAR',
                              'value': float(data_df['AMT_REQ_CREDIT_BUREAU_YEAR'])},
                             ignore_index=True)
    barplot_df = barplot_df.append(df_user)
    bar_plot = plt.figure()
    sns.barplot(data=barplot_df, x='indicator', y='value', hue='TARGET')
    bytes_io = io.BytesIO()
    bar_plot.savefig(bytes_io, format="PNG")
    plt.close()
    return Response(bytes_io.getvalue(), media_type="image/png")

@app.post('/get_bar_plot_2')
def get_bar_1(data: Client):
    """
    Fonction qui prend les infos du client sous forme json et retourne
    le graph en barre

    Parameters
    ----------
    data : Client

    Returns
    -------
    Response
        bar plot for  EXT_SOURCE_2
    """
    barplot_df = pd.read_csv("storage/barplot_df.csv")
    barplot_df = barplot_df[barplot_df['indicator'] == 'EXT_SOURCE_2']
    data_df  = dataframe[dataframe['SK_ID_CURR'] == data.num_client]
    data_df.drop(['SK_ID_CURR', 'TARGET'], axis=1, inplace= True)
    #data = data.dict()
    #data_df = pd.DataFrame.from_dict([data])
    df_user = pd.DataFrame(columns=['TARGET', 'value', 'indicator'])
    df_user = df_user.append({'TARGET': 'user value',
                              'indicator': 'EXT_SOURCE_2',
                              'value': float(data_df['EXT_SOURCE_2'])},
                             ignore_index=True)
    barplot_df = barplot_df.append(df_user)
    bar_plot = plt.figure()
    sns.barplot(data=barplot_df, x='indicator', y='value', hue='TARGET')
    bytes_io = io.BytesIO()
    bar_plot.savefig(bytes_io, format="PNG")
    plt.close()
    return Response(bytes_io.getvalue(), media_type="image/png")

@app.post('/get_bar_plot_3')
def get_bar_1(data: Client):
    """
    Fonction qui prend les infos du client sous forme json et retourne
    le graph en barre

    Parameters
    ----------
    data : Client

    Returns
    -------
    Response
        bar plot for EXT_SOURCE_3
    """
    barplot_df = pd.read_csv("storage/barplot_df.csv")
    barplot_df = barplot_df[barplot_df['indicator'] == 'EXT_SOURCE_3']
    data_df  = dataframe[dataframe['SK_ID_CURR'] == data.num_client]
    data_df.drop(['SK_ID_CURR', 'TARGET'], axis=1, inplace= True)
    #data = data.dict()
    #data_df = pd.DataFrame.from_dict([data])
    df_user = pd.DataFrame(columns=['TARGET', 'value', 'indicator'])
    df_user = df_user.append({'TARGET': 'user value',
                              'indicator': 'EXT_SOURCE_3',
                              'value': float(data_df['EXT_SOURCE_3'])},
                             ignore_index=True)
    barplot_df = barplot_df.append(df_user)
    bar_plot = plt.figure()
    sns.barplot(data=barplot_df, x='indicator', y='value', hue='TARGET')
    bytes_io = io.BytesIO()
    bar_plot.savefig(bytes_io, format="PNG")
    plt.close()
    return Response(bytes_io.getvalue(), media_type="image/png")

@app.post('/liste_df_1')
async def get_liste_id_1():
  return liste_df_1

# homepage route
@app.get("/")
def read_root():
  return {'message': 'This is the homepage of the API '}
