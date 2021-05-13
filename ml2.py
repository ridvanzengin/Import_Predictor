
##################################################
# Machine Learning Approach
##################################################
import pandas as pd
import numpy as np

def ml2(year):
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
    # Gradient Boosting Machines
    #####################################################
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.model_selection import GridSearchCV

    Y_train = Y_train.fillna(0)
    X_train = X_train.fillna(0)
    X_val = X_val.fillna(0)
    Y_val = Y_val.fillna(0)

    gbm_params = {
        'learning_rate': [0.001, 0.01, 0.1],
        'max_depth': [5, 8, 12],
        'n_estimators': [500, 1000, 2000]
    }
    gbm_model = GradientBoostingRegressor()
    gbm_cv_model = GridSearchCV(gbm_model, gbm_params, cv=5, n_jobs=-1, verbose=2)
    gbm_cv_model.fit(X_train, Y_train)

    gbm_params = gbm_cv_model.best_params_
    gbm_params = pd.DataFrame([gbm_params])
    gbm_params.to_pickle(f"C:\\Users\Ringolog\Desktop\Yüksek Lisans\BitirmeProjesi\script\\ml_combs\\{year}\\gbm.pkl")
    gbm_params = pd.read_pickle(
        f"C:\\Users\Ringolog\Desktop\Yüksek Lisans\BitirmeProjesi\script\\ml_combs\\{year}\\gbm.pkl")

    gbm_tuned = GradientBoostingRegressor(learning_rate=gbm_params["learning_rate"][0],
                                          max_depth=gbm_params["max_depth"][0],
                                          n_estimators=gbm_params["n_estimators"][0])
    gbm_tuned = gbm_tuned.fit(X_train, Y_train)
    y_pred = gbm_tuned.predict(X_val)
    testyear["gbm"] = np.expm1(y_pred)

    #####################################################
    # XGBoost
    #####################################################

    from xgboost import XGBRegressor

    xgb_grid = {
        'colsample_bytree': [0.3, 0.5, 0.7],
        'n_estimators': [500, 1000, 2000],
        'max_depth': [5, 8, 12],
        'learning_rate': [0.1, 0.01]
    }
    xgb = XGBRegressor()

    xgb_cv = GridSearchCV(xgb,
                          param_grid=xgb_grid,
                          cv=5,
                          n_jobs=-1,
                          verbose=2)
    xgb_cv.fit(X_train, Y_train)

    xgb_params = xgb_cv.best_params_
    xgb_params = pd.DataFrame([xgb_params])
    xgb_params.to_pickle(f"C:\\Users\Ringolog\Desktop\Yüksek Lisans\BitirmeProjesi\script\\ml_combs\\{year}\\xgb.pkl")
    xgb_params = pd.read_pickle(
        f"C:\\Users\Ringolog\Desktop\Yüksek Lisans\BitirmeProjesi\script\\ml_combs\\{year}\\xgb.pkl")

    xgb_tuned = XGBRegressor(colsample_bytree=xgb_params["colsample_bytree"][0],
                             max_depth=xgb_params["max_depth"][0],
                             n_estimators=xgb_params["n_estimators"][0],
                             learning_rate=xgb_params["learning_rate"][0])

    xgb_tuned = xgb_tuned.fit(X_train, Y_train)
    y_pred = xgb_tuned.predict(X_val)
    testyear["xgb"] = np.expm1(y_pred)

    #####################################################
    # CatBoost
    #####################################################
    from catboost import CatBoostRegressor

    catb_grid = {
        'iterations': [500, 1000, 2000],
        'learning_rate': [0.01, 0.1],
        'depth': [5, 8, 12]}

    catb = CatBoostRegressor()
    catb_cv_model = GridSearchCV(catb, catb_grid, cv=5, n_jobs=-1, verbose=2)
    catb_cv_model.fit(X_train, Y_train)

    catb_params = catb_cv_model.best_params_
    catb_params = pd.DataFrame([catb_params])
    catb_params.to_pickle(f"C:\\Users\Ringolog\Desktop\Yüksek Lisans\BitirmeProjesi\script\\ml_combs\\{year}\\catb.pkl")
    catb_params = pd.read_pickle(
        f"C:\\Users\Ringolog\Desktop\Yüksek Lisans\BitirmeProjesi\script\\ml_combs\\{year}\\catb.pkl")

    catb_tuned = CatBoostRegressor(iterations=catb_params["iterations"][0],
                                   depth=catb_params["depth"][0],
                                   learning_rate=catb_params["learning_rate"][0])

    catb_tuned = catb_tuned.fit(X_train, Y_train)
    y_pred = catb_tuned.predict(X_val)
    testyear["catb"] = np.expm1(y_pred)

    def smape(preds, target):
        import numpy as np
        n = len(preds)
        masked_arr = ~((preds == 0) & (target == 0))
        preds, target = preds[masked_arr], target[masked_arr]
        num = np.abs(preds - target)
        denom = np.abs(preds) + np.abs(target)
        smape_val = (200 * np.sum(num / denom)) / n
        return smape_val

    testyear["GBM_smape"] = [smape([testyear["Import"].values[i]], [testyear["gbm"].values[i]]) for i in
                             range(0, testyear.shape[0])]
    testyear["XGB_smape"] = [smape([testyear["Import"].values[i]], [testyear["xgb"].values[i]]) for i in
                              range(0, testyear.shape[0])]
    testyear["CATB_smape"] = [smape([testyear["Import"].values[i]], [testyear["catb"].values[i]]) for i in
                            range(0, testyear.shape[0])]

    return testyear















