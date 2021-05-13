##################################################
# SARIMA(p, d, q): (Seasonal Autoregressive Integrated Moving-Average)
##################################################

def sarima(year):
    import os
    import itertools
    import warnings
    import numpy as np
    import pandas as pd
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    warnings.filterwarnings('ignore')
    pd.set_option('display.max_rows', None)
    pd.set_option('display.width', 500)

    tahminler=pd.DataFrame()
    products=os.listdir("C:\\Users\Ringolog\Desktop\Yüksek Lisans\BitirmeProjesi\script\data")
    for product in products:
        path=f"C:\\Users\Ringolog\Desktop\Yüksek Lisans\BitirmeProjesi\script\data\\{product}"
        excels=os.listdir(path)
        u1=pd.read_excel(os.path.join(path,excels[0]))
        u2=pd.read_excel(os.path.join(path,excels[1]))
        u3=pd.read_excel(os.path.join(path,excels[2]))
        u4=pd.read_excel(os.path.join(path,excels[3]))
        u5=pd.read_excel(os.path.join(path,excels[4]))
        u6=pd.read_excel(os.path.join(path,excels[5]))

        data= u6.merge(u5).merge(u4).merge(u3).merge(u2).merge(u1)

        Importers = data["Importers"]
        data = data.T
        data.columns = Importers
        data = data[data.index != 'Importers']
        data.reset_index(inplace=True)
        columns = ["index", "United Kingdom", "Canada", "France", "Netherlands",
                     "Germany", "Finland", "Sweden", "Singapore", "Denmark", "Austria"]
        countries=["United Kingdom", "Canada", "France", "Netherlands",
                     "Germany", "Finland", "Sweden", "Singapore", "Denmark", "Austria"]
        data = data[columns]
        data[countries]=data[countries] +1
        df = data.copy()

        df['index'] = pd.to_datetime(df['index'])
        for col in df.columns:
            if col != "index":
                df[col] = df[col].astype(float)

        df = df[df["index"] >= "2012"]
        df=df[df["index"]< str(int(year+1))]

        train = df[df["index"] < str(year)]
        test = df[df["index"] >= str(year)]

        df.index = df["index"]
        df.drop("index", axis=1, inplace=True)
        notnull_columns = [col for col in df.columns if df[col].isnull().sum() == 0]
        null_columns = [col for col in df.columns if df[col].isnull().sum() != 0]
        df.drop(null_columns, axis=1, inplace=True)



        def fit_model_sarima(train, val, pdq, seasonal_pdq):
            sarima_model = SARIMAX(train, order=pdq, seasonal_order=seasonal_pdq).fit(disp=0)
            y_pred_val = sarima_model.get_forecast(steps=12)
            y_pred = y_pred_val.predicted_mean
            return mean_absolute_error(val, y_pred)

        def sarima_optimizer_mae(train, val, pdq, seasonal_pdq):
            best_mae, best_order, best_seasonal_order = float("inf"), float("inf"), None
            for param in pdq:
                print(f"Optimizing SARIMA parameters...for {product}")
                for param_seasonal in seasonal_pdq:
                    try:
                        mae = fit_model_sarima(train, val, param, param_seasonal)
                        if mae < best_mae:
                            best_mae, best_order, best_seasonal_order = mae, param, param_seasonal
                    except:
                        continue
            print('SARIMA{}x{}12 - MAE:{}'.format(best_order, best_seasonal_order, best_mae))
            return best_order, best_seasonal_order

        p = d = q = range(0, 2)
        pdq = list(itertools.product(p, d, q))
        seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]

        mae_best_orders = pd.DataFrame()
        mae_best_seasonal_orders = pd.DataFrame()
        for col in countries:
            try:
                best_order, best_seasonal_order = sarima_optimizer_mae(train[col].values, test[col].values, pdq,
                                                                       seasonal_pdq)
                mae_best_orders[col] = best_order
                mae_best_seasonal_orders[col] = best_seasonal_order
            except:
                continue

        mae_best_orders.to_pickle(f"C:\\Users\Ringolog\Desktop\Yüksek Lisans\BitirmeProjesi\script\\sarima_combs\\{year}\\{product}_bo.pkl")
        mae_best_seasonal_orders.to_pickle(f"C:\\Users\Ringolog\Desktop\Yüksek Lisans\BitirmeProjesi\script\\sarima_combs\\{year}\\{product}_bso.pkl")

        mae_best_orders = pd.read_pickle(f"C:\\Users\Ringolog\Desktop\Yüksek Lisans\BitirmeProjesi\script\\sarima_combs\\{year}\\{product}_bo.pkl")
        mae_best_seasonal_orders = pd.read_pickle(f"C:\\Users\Ringolog\Desktop\Yüksek Lisans\BitirmeProjesi\script\\sarima_combs\\{year}\\{product}_bso.pkl")

        mo_tahmin20 = pd.DataFrame()
        for col in mae_best_orders.columns:
            model = SARIMAX(train[col],
                            order=(mae_best_orders[col][0], mae_best_orders[col][1], mae_best_orders[col][2]),
                            seasonal_order=(mae_best_seasonal_orders[col][0], mae_best_seasonal_orders[col][1],
                                            mae_best_seasonal_orders[col][2], 12))
            sarima_model = model.fit(disp=0)
            pred = sarima_model.get_forecast(steps=12)
            y_pred = pred.predicted_mean
            mo_tahmin20[col] = y_pred

        reals = test[mo_tahmin20.columns].melt()["value"]
        mo_tahmin20 = mo_tahmin20.melt(ignore_index=False)
        mo_tahmin20["real"] = reals.values
        mo_tahmin20["product"] = f"{product}"
        mo_tahmin20.columns = ["Country", "Sarima", "Real", "Product"]
        tahminler=pd.concat([tahminler, mo_tahmin20],ignore_index=True)

    def smape(preds, target):
        import numpy as np
        n = len(preds)
        masked_arr = ~((preds == 0) & (target == 0))
        preds, target = preds[masked_arr], target[masked_arr]
        num = np.abs(preds - target)
        denom = np.abs(preds) + np.abs(target)
        smape_val = (200 * np.sum(num / denom)) / n
        return smape_val

    tahminler["smape"] = [smape([tahminler["Real"][i]], [tahminler["Sarima"][i]]) for i in range(0, tahminler.shape[0])]
    return tahminler
"""
sarima=sarima()

sarima.head()
def smape(preds, target):
    n = len(preds)
    masked_arr = ~((preds == 0) & (target == 0))
    preds, target = preds[masked_arr], target[masked_arr]
    num = np.abs(preds-target)
    denom = np.abs(preds)+np.abs(target)
    smape_val = (200*np.sum(num/denom))/n
    return smape_val

sarima["smape"]=[smape([sarima["Real"][i]],[sarima["Sarima"][i]]) for i in range(0,sarima.shape[0])]
sarima.groupby(by="Product").agg({"smape": ["mean"]}).sort_values(by= ("smape","mean"), ascending=True)
sarima.groupby(by=["Country","Product"]).agg({"smape": ["mean"]}).sort_values(by= ["Country",("smape","mean")], ascending=True)

sarima.groupby(by="Product")["Country"].nunique()

import pandas as pd

elma = pd.read_pickle(f"C:\\Users\Ringolog\Desktop\Yüksek Lisans\BitirmeProjesi\script\\sarima_combs\\elma_bso.pkl")
muz.value_counts().sum()

armut = pd.read_pickle(f"C:\\Users\Ringolog\Desktop\Yüksek Lisans\BitirmeProjesi\script\\sarima_combs\\armut_bso.pkl")

armut.value_counts().sum()

armut=pd.read_pickle(f"C:\\Users\Ringolog\Desktop\Yüksek Lisans\BitirmeProjesi\script\\sarima_combs\\armut_bo.pkl")

"""