
##################################################
# Machine Learning Approach
##################################################


def ml(year):
    import os
    import pandas as pd
    import numpy as np
    pd.set_option('display.max_columns', 8)
    pd.set_option('display.width', 500)
    combined=pd.DataFrame()
    products = os.listdir("C:\\Users\Ringolog\Desktop\Yüksek Lisans\BitirmeProjesi\script\data")
    for product in products:
        path = f"C:\\Users\Ringolog\Desktop\Yüksek Lisans\BitirmeProjesi\script\data\\{product}"
        excels = os.listdir(path)
        u1 = pd.read_excel(os.path.join(path, excels[0]))
        u2 = pd.read_excel(os.path.join(path, excels[1]))
        u3 = pd.read_excel(os.path.join(path, excels[2]))
        u4 = pd.read_excel(os.path.join(path, excels[3]))
        u5 = pd.read_excel(os.path.join(path, excels[4]))
        u6 = pd.read_excel(os.path.join(path, excels[5]))

        data = u6.merge(u5).merge(u4).merge(u3).merge(u2).merge(u1)
        data.head()

        ##################################################
        # Data Prep
        ##################################################
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

        notnull_columns = [col for col in df.columns if df[col].isnull().sum() == 0]
        null_columns = [col for col in df.columns if df[col].isnull().sum() != 0]
        df.drop(null_columns, axis=1, inplace=True)

        df = df.melt("index")
        df.columns = ["Time", "Country", "Import"]

        df['Time'] = pd.to_datetime(df['Time'])
        df['Import'] = df['Import'].astype(float)
        df['Product'] = f"{product}"
        combined = pd.concat([combined, df], ignore_index=True)

    df=combined

    df['month'] = df.Time.dt.month
    df['year'] = df.Time.dt.year
    testyear = df[df["year"] == year]

    def random_noise(dataframe):
        return np.random.normal(scale=round(df["Import"].mean()/20), size=(len(dataframe),))

    def lag_features(dataframe, lags):
        dataframe = dataframe.copy()
        for lag in lags:
            dataframe['Import_lag_' + str(lag)] = dataframe.groupby(["Country"])['Import'].transform(
                lambda x: x.shift(lag)) #+ random_noise(dataframe)
        return dataframe

    df = lag_features(df, [ 1,3,6,12,24])


    #####################################################
    # Rolling Mean Features
    #####################################################

    def roll_mean_features(dataframe, windows):
        dataframe = dataframe.copy()
        for window in windows:
            dataframe['Import_roll_mean_' + str(window)] = dataframe.groupby(["Country"])['Import']. \
                                                              transform(
                lambda x: x.shift(1).rolling(window=window, win_type="triang").mean()) #+ random_noise(dataframe)
        return dataframe

    df = roll_mean_features(df, [2,3,4,5,6,9,12])

    #####################################################
    # Exponentially Weighted Mean Features
    #####################################################

    def ewm_features(dataframe, alphas, lags):
        dataframe = dataframe.copy()
        for alpha in alphas:
            for lag in lags:
                dataframe['Import_ewm_alpha_' + str(alpha).replace(".", "") + "_lag_" + str(lag)] = \
                    dataframe.groupby(["Country"])['Import']. \
                        transform(lambda x: x.shift(lag).ewm(alpha=alpha).mean())
        return dataframe

    alphas = [0.9, 0.7, 0.5, 0.2, 0.1]
    lags = [1,2,3,4,5,6,12,24]

    df = ewm_features(df, alphas, lags)

    #####################################################
    # One-Hot Encoding
    #####################################################

    df = pd.get_dummies(df, columns=['Country','Product'])

    #####################################################
    # Converting Import to log(1+Import)
    #####################################################

    df['Import'] = np.log1p(df["Import"].values)

    #####################################################
    # Train Validasyon Split
    #####################################################

    import re
    df = df.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))

    train = df.loc[(df["year"]!= year), :]
    val = df.loc[(df["year"] == year), :]

    cols = [col for col in train.columns if col not in ['Time', "Import"]]

    Y_train = train['Import']
    X_train = train[cols]

    Y_val = val['Import']
    X_val = val[cols]

    #####################################################
    # Light GBM
    #####################################################
    print("lgbm optimization started")
    from sklearn.model_selection import GridSearchCV
    from lightgbm import LGBMRegressor

    lgbm_grid = {
        'colsample_bytree': [0.2, 0.4, 0.6],
        'learning_rate': [ 0.01, 0.1],
        'n_estimators': [1000, 2500,4000],
        'max_depth': [5, 10, 20],
        'num_leaves': [20, 40,60]}

    lgbm = LGBMRegressor()
    lgbm_cv_model = GridSearchCV(lgbm, lgbm_grid, cv=5, n_jobs=-1, verbose=2)
    lgbm_cv_model.fit(X_train, Y_train)

    lgbm_params = lgbm_cv_model.best_params_
    lgbm_params = pd.DataFrame([lgbm_params])
    lgbm_params.to_pickle(f"C:\\Users\Ringolog\Desktop\Yüksek Lisans\BitirmeProjesi\script\\ml_combs\\{year}\\lgbm.pkl")
    lgbm_params = pd.read_pickle(f"C:\\Users\Ringolog\Desktop\Yüksek Lisans\BitirmeProjesi\script\\ml_combs\\{year}\\lgbm.pkl")

    lgbm_tuned = LGBMRegressor(learning_rate=lgbm_params["learning_rate"][0],
                               max_depth=lgbm_params["max_depth"][0],
                               n_estimators=lgbm_params["n_estimators"][0],
                               colsample_bytree=lgbm_params["colsample_bytree"][0],
                               num_leaves=lgbm_params["num_leaves"][0])
    lgbm_tuned = lgbm_tuned.fit(X_train, Y_train)
    y_pred = lgbm_tuned.predict(X_val)

    testyear["lgbm"] = np.expm1(y_pred)

    #####################################################
    # Random Forest
    #####################################################
    print("rf optimization started")
    from sklearn.ensemble import RandomForestRegressor

    Y_train = Y_train.fillna(0)
    X_train = X_train.fillna(0)
    X_val = X_val.fillna(0)
    Y_val = Y_val.fillna(0)

    rf_params = {'max_depth': [10, 20, 30],
                 'max_features': [20, 40, 60],
                 'n_estimators': [100, 200, 500]}

    rf_model = RandomForestRegressor(random_state=4)
    rf_cv_model = GridSearchCV(rf_model,
                               rf_params,
                               cv=5,
                               n_jobs=-1,
                               verbose=2)
    rf_cv_model.fit(X_train, Y_train)

    rf_params = rf_cv_model.best_params_
    rf_params = pd.DataFrame([rf_params])
    rf_params.to_pickle(f"C:\\Users\Ringolog\Desktop\Yüksek Lisans\BitirmeProjesi\script\\ml_combs\\{year}\\rf.pkl")
    rf_params = pd.read_pickle(f"C:\\Users\Ringolog\Desktop\Yüksek Lisans\BitirmeProjesi\script\\ml_combs\\{year}\\rf.pkl")

    rf_tuned = RandomForestRegressor(max_depth=rf_params["max_depth"][0],
                                     max_features=rf_params["max_features"][0],
                                     n_estimators=rf_params["n_estimators"][0])
    rf_tuned.fit(X_train, Y_train)

    y_pred = rf_tuned.predict(X_val)
    testyear["rf"] = np.expm1(y_pred)

    #####################################################
    # YSA
    #####################################################
    print("mlp optimization started")
    from sklearn.neural_network import MLPRegressor
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train= scaler.transform(X_train)
    X_val= scaler.transform(X_val)
    mlp_model = MLPRegressor()
    mlp_params = {'alpha': [0.1, 0.01, 0.005],
                  'hidden_layer_sizes': [(450, 250), (500, 150), (150, 250, 100), (250, 400, 100)],
                  'activation': ['logistic'],
                  'solver': ['lbfgs']}

    mlp_cv_model = GridSearchCV(mlp_model, mlp_params, n_jobs=-1, verbose=2, cv=5)
    mlp_cv_model.fit(X_train, Y_train)

    mlp_params = mlp_cv_model.best_params_
    mlp_params = pd.DataFrame([mlp_params])
    mlp_params.to_pickle(f"C:\\Users\Ringolog\Desktop\Yüksek Lisans\BitirmeProjesi\script\\ml_combs\\{year}\\mlp.pkl")
    mlp_params = pd.read_pickle(f"C:\\Users\Ringolog\Desktop\Yüksek Lisans\BitirmeProjesi\script\\ml_combs\\{year}\\mlp.pkl")

    mlp_tuned = MLPRegressor(alpha=mlp_params["alpha"][0],
                             hidden_layer_sizes=mlp_params["hidden_layer_sizes"][0],
                             activation=mlp_params["activation"][0],
                             solver=mlp_params["solver"][0])
    mlp_tuned.fit(X_train, Y_train)

    y_pred = mlp_tuned.predict(X_val)
    testyear["mlp"] = np.expm1(y_pred)

    def smape(preds, target):
        import numpy as np
        n = len(preds)
        masked_arr = ~((preds == 0) & (target == 0))
        preds, target = preds[masked_arr], target[masked_arr]
        num = np.abs(preds - target)
        denom = np.abs(preds) + np.abs(target)
        smape_val = (200 * np.sum(num / denom)) / n
        return smape_val

    testyear["MLP_smape"] = [smape([testyear["Import"].values[i]], [testyear["mlp"].values[i]]) for i in
                           range(0, testyear.shape[0])]
    testyear["LGBM_smape"] = [smape([testyear["Import"].values[i]], [testyear["lgbm"].values[i]]) for i in
                            range(0, testyear.shape[0])]
    testyear["RF_smape"] = [smape([testyear["Import"].values[i]], [testyear["rf"].values[i]]) for i in
                          range(0, testyear.shape[0])]

    return testyear
"""
testyear=ml()

def smape(preds, target):
    n = len(preds)
    masked_arr = ~((preds == 0) & (target == 0))
    preds, target = preds[masked_arr], target[masked_arr]
    num = np.abs(preds-target)
    denom = np.abs(preds)+np.abs(target)
    smape_val = (200*np.sum(num/denom))/n
    return smape_val

testyear.head()

testyear["MLP_smape"]=[smape([testyear["Import"].values[i]],[testyear["mlp"].values[i]]) for i in range(0,testyear.shape[0])]
testyear.groupby(by=["Country","Product"]).agg({"smape": ["mean"]}).sort_values(by= ["Country",("smape","mean")], ascending=True)
"""

import pandas as pd
mlp_params = pd.read_pickle(f"C:\\Users\Ringolog\Desktop\Yüksek Lisans\BitirmeProjesi\script\\ml_combs\\mlp.pkl")
