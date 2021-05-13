import pandas as pd
from tes import tes
from sarima import sarima
from ml import ml
from ml2 import ml2


tes20=tes(2020)
sarima20=sarima(2020)
ml20=ml(2020)
tes19=tes(2019)
sarima19=sarima(2019)
ml19=ml(2019)
ml2_20 =ml2(2020)
ml2_19 =ml2(2019)

cols=['gbm','xgb','catb','GBM_smape','XGB_smape','CATB_smape']
ml20[cols]=ml2_20[cols].values
ml19[cols]=ml2_19[cols].values

ml20["tes"]=tes20["Tes"]
ml20["TES_smape"]=tes20["smape"]
ml20["sarima"]=sarima20["Sarima"]
ml20["SARIMA_smape"]=sarima20["smape"]

ml19["tes"]=tes19["Tes"]
ml19["TES_smape"]=tes19["smape"]
ml19["sarima"]=sarima19["Sarima"]
ml19["SARIMA_smape"]=sarima19["smape"]

#ml20.to_excel("predictions_20.xlsx",index=False)
#ml19.to_excel("predictions_19.xlsx",index=False)
ml20=pd.read_excel("predictions_20.xlsx")
ml19=pd.read_excel("predictions_19.xlsx")

ml20.groupby(by="Product").agg({"LGBM_smape": ["mean"]}).sort_values(by= ("LGBM_smape","mean"), ascending=True)
ml20.groupby(by="Product").agg({"RF_smape": ["mean"]}).sort_values(by= ("RF_smape","mean"), ascending=True)
ml20.groupby(by="Product").agg({"MLP_smape": ["mean"]}).sort_values(by= ("MLP_smape","mean"), ascending=True)
ml20.groupby(by="Product").agg({"TES_smape": ["mean"]}).sort_values(by= ("TES_smape","mean"), ascending=True)
ml20.groupby(by="Product").agg({"SARIMA_smape": ["mean"]}).sort_values(by= ("SARIMA_smape","mean"), ascending=True)
ml20.groupby(by="Product").agg({"GBM_smape": ["mean"]}).sort_values(by= ("GBM_smape","mean"), ascending=True)
ml20.groupby(by="Product").agg({"XGB_smape": ["mean"]}).sort_values(by= ("XGB_smape","mean"), ascending=True)
ml20.groupby(by="Product").agg({"CATB_smape": ["mean"]}).sort_values(by= ("CATB_smape","mean"), ascending=True)



ml19.groupby(by="Product").agg({"LGBM_smape": ["mean"]}).sort_values(by= ("LGBM_smape","mean"), ascending=True)
ml19.groupby(by="Product").agg({"RF_smape": ["mean"]}).sort_values(by= ("RF_smape","mean"), ascending=True)
ml19.groupby(by="Product").agg({"MLP_smape": ["mean"]}).sort_values(by= ("MLP_smape","mean"), ascending=True)
ml19.groupby(by="Product").agg({"TES_smape": ["mean"]}).sort_values(by= ("TES_smape","mean"), ascending=True)
ml19.groupby(by="Product").agg({"SARIMA_smape": ["mean"]}).sort_values(by= ("SARIMA_smape","mean"), ascending=True)
ml19.groupby(by="Product").agg({"GBM_smape": ["mean"]}).sort_values(by= ("GBM_smape","mean"), ascending=True)
ml19.groupby(by="Product").agg({"XGB_smape": ["mean"]}).sort_values(by= ("XGB_smape","mean"), ascending=True)
ml19.groupby(by="Product").agg({"CATB_smape": ["mean"]}).sort_values(by= ("CATB_smape","mean"), ascending=True)
