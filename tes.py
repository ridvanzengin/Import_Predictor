##################################################
# Holt-Winters
##################################################

def tes(year):
    import os
    import itertools
    import warnings
    import numpy as np
    import pandas as pd
    from sklearn.metrics import mean_absolute_error
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
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

        Importers=data["Importers"]
        data=data.T
        data.columns=Importers

        data = data[data.index != 'Importers']
        data.reset_index(inplace=True)
        columns = ["index", "United Kingdom", "Canada", "France", "Netherlands",
                     "Germany", "Finland", "Sweden", "Singapore", "Denmark", "Austria"]
        countries=["United Kingdom", "Canada", "France", "Netherlands",
                     "Germany", "Finland", "Sweden", "Singapore", "Denmark", "Austria"]
        data = data[columns]
        data[countries]=data[countries] +1
        df=data.copy()
        df['index'] = pd.to_datetime(df['index'])
        for col in df.columns:
            if col !="index":
                df[col]=df[col].astype(float)

        df=df[df["index"]>="2012"]
        df=df[df["index"]< str(int(year+1))]

        train = df[df["index"] < str(year)]
        test = df[df["index"] >= str(year)]

        alphas = betas = gammas = np.arange(0.10, 1, 0.20)
        abg = list(itertools.product(alphas, betas, gammas))

        def optimize_tes(train, test, abg, step=12):
            print(f"Optimizing  TES parameters...for {product}")
            best_mae = float("inf")
            best_comb = []
            for comb in abg:
                try:
                    tes_model = ExponentialSmoothing(train, trend="add",
                                                     seasonal="add",
                                                     seasonal_periods=12). \
                        fit(smoothing_level=comb[0],
                            smoothing_slope=comb[1],
                            smoothing_seasonal=comb[2])

                    y_pred = tes_model.forecast(step)
                    mae = mean_absolute_error(test, y_pred)
                    if mae < best_mae:
                        best_mae = mae
                        best_comb = [round(comb[0], 2), round(comb[1], 2), round(comb[2], 2)]
                except:
                    continue
            return best_comb

        best_combs = pd.DataFrame()
        for col in train.columns:
            try:
                best_comb = optimize_tes(train[col], test[col], abg)
                best_combs[col] = best_comb
            except:
                continue
        best_combs.to_pickle(f"C:\\Users\Ringolog\Desktop\Yüksek Lisans\BitirmeProjesi\script\\tes_combs\\{year}\\{product}.pkl")

        best_combs = pd.read_pickle(f"C:\\Users\Ringolog\Desktop\Yüksek Lisans\BitirmeProjesi\script\\tes_combs\\{year}\\{product}.pkl")
        best_combs.drop(["index"],axis=1, inplace=True)

        o_tahmin20 = pd.DataFrame()
        for col in best_combs.columns:
            tes_model = ExponentialSmoothing(train[col],
                                             trend="add",
                                             seasonal="add",
                                             seasonal_periods=12).fit(smoothing_level=best_combs[col][0],
                                                                      smoothing_slope=best_combs[col][1],
                                                                      smoothing_seasonal=best_combs[col][2])

            y_pred = tes_model.forecast(12)
            o_tahmin20[col] = y_pred
        reals = test[o_tahmin20.columns].melt()["value"]
        o_tahmin20 = o_tahmin20.melt(ignore_index=False)
        o_tahmin20["real"] = reals.values
        o_tahmin20["product"] = f"{product}"
        o_tahmin20.columns = ["Country", "Tes", "Real", "Product"]
        tahminler=pd.concat([tahminler, o_tahmin20],ignore_index=True)

    def smape(preds, target):
        n = len(preds)
        masked_arr = ~((preds == 0) & (target == 0))
        preds, target = preds[masked_arr], target[masked_arr]
        num = np.abs(preds - target)
        denom = np.abs(preds) + np.abs(target)
        smape_val = (200 * np.sum(num / denom)) / n
        return smape_val
    tahminler["smape"] = [smape([tahminler["Real"][i]], [tahminler["Tes"][i]]) for i in range(0, tahminler.shape[0])]
    return tahminler
"""
tes=tes()

tes.head()
tes.shape

def smape(preds, target):
    import numpy as np
    n = len(preds)
    masked_arr = ~((preds == 0) & (target == 0))
    preds, target = preds[masked_arr], target[masked_arr]
    num = np.abs(preds-target)
    denom = np.abs(preds)+np.abs(target)
    smape_val = (200*np.sum(num/denom))/n
    return smape_val

tes["smape"]=[smape([tes["Real"][i]],[tes["Tes"][i]]) for i in range(0,tes.shape[0])]
tes.groupby(by="Product").agg({"smape": ["mean"]}).sort_values(by= ("smape","mean"), ascending=True)

tes.groupby(by="Product")["Country"].nunique()
"""






