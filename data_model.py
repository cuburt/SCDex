import numpy as np
import pandas as pd
from sklearn import preprocessing, gaussian_process
from sklearn.model_selection import train_test_split
from sklearn.gaussian_process.kernels import WhiteKernel, ExpSineSquared, ConstantKernel, RBF
from datetime import datetime
import math

def _filter(df,contractor_name, project_name):

    df = df.loc[df['Implementor'] == contractor_name]
    df = df.loc[df['Name of Project'] == project_name]
    project_list = upsample(df)

    return forecast(project_list)


def dataframe(contractor_name=None, project_name=None):
    df = pd.read_csv('dataset/full_slippage_dataset.csv', encoding='utf-8', engine='python')

    # Convert to numeric immediately. Strip commas if they exist.
    if df['Contract Amount'].dtype == object:
        df['Contract Amount'] = df['Contract Amount'].astype(str).str.replace(',', '', regex=True)

    df['Contract Amount'] = pd.to_numeric(df['Contract Amount'], errors='coerce')
    df['No. of Days'] = pd.to_numeric(df['No. of Days'], errors='coerce')

    df.set_index('index', inplace=True)
    df.index = pd.to_datetime(df.index)
    if project_name and contractor_name:
        return _filter(df, contractor_name, project_name)
    return df.reset_index()


def upsample(df):
    proj_list = [proj for proj in df['Name of Project'].values]
    project_list = []
    for project in set(proj_list):
        rec_df = df.loc[df['Name of Project'] == project].copy()
        order = (rec_df.shape[0]) - 1
        if order >= 5: order = 5

        # 1. Resample to get the daily index and populate missing days with NaNs
        itpd_df = rec_df.resample("D").first()

        # 2. Isolate numeric columns so interpolation doesn't crash on text strings
        numeric_cols = itpd_df.select_dtypes(include=[np.number]).columns
        itpd_df[numeric_cols] = itpd_df[numeric_cols].interpolate(method='spline', order=order)

        # 3. Fill the string columns (and any leftover NaNs) using pad
        itpd_df = itpd_df.ffill()

        # 4. Final safety casting
        itpd_df['Contract Amount'] = itpd_df['Contract Amount'].astype(float)
        itpd_df['No. of Days'] = itpd_df['No. of Days'].astype(int)

        project_list.append(itpd_df)
    return project_list

def define_model():
    k0 = WhiteKernel(noise_level=0.3**2, noise_level_bounds=(0.1**2, 0.5**2))
    k1 = ConstantKernel(constant_value=2)* \
         ExpSineSquared(length_scale=1.0, periodicity=40, periodicity_bounds=(35,45))
    k2 = ConstantKernel(constant_value=10, constant_value_bounds=(1e-2, 1e3))* \
         RBF(length_scale=100.0, length_scale_bounds=(1, 1e4))
    kernel_1 = k0 + k1 + k2
    linear_model = gaussian_process.GaussianProcessRegressor(kernel=kernel_1, n_restarts_optimizer=10, normalize_y=True, alpha=0.0)
    return linear_model

def score_model(X, y_test, y_pred):
    ssr = sum((y_test-y_pred)**2)
    mse = 1/len(y_pred)*ssr
    rmse = np.sqrt(mse)
    sst = ((y_test-np.mean(y_test))**2)
    r2 = 1 - (ssr/sst)
    adj_r2 = 1-(1-r2)*(len(y_test)-1)/(len(y_test)-(len(X.columns))-1)
    return [np.mean(adj_r2),rmse]


def train_model(X_lately, X, y_lately, y, max_iter, min_perc):
    for r in range(max_iter):
        X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True, test_size=0.2, random_state=r)
        model = define_model().fit(X_train, y_train)
        # model_score = model.score(X_test, y_test)
        # if model_score >= min_perc/100:
        #     y_test_pred = model.predict(X_test)
        #     y_pred = model.predict(X_lately)
        y_test_pred = model.predict(X_test)
        val_model_score = score_model(X_test, y_test, y_test_pred)
        if val_model_score[0] >= (min_perc / 100):
            try:
                mainmodel = define_model().fit(X, y)

                # --- NEW: Ask the GPR model to return the standard deviation (sigma) ---
                y_pred, sigma = mainmodel.predict(X_lately, return_std=True)

                model_score = score_model(X_lately, y_lately, y_pred)

                # --- NEW: Append `sigma` to the very end of your return list ---
                return [X_train, X_test, y_train, y_test, y_pred, model_score[0], model_score[1], sigma]
            except Exception as e:
                print(str(e))
                break

class CustomWalkForward:
    def __init__(self, test_size, gap):
        self.test_size = test_size
        self.train_size = 1 - test_size
        self.gap = gap

    def split(self,df):
        X = df
        n = len(X)
        folds = int(math.ceil(n/(n*self.test_size)))
        q = int(n/folds)
        res = n%folds
        for k in range(1,folds+1):
            train_range = int((q*k)*self.train_size)
            if k == folds: train_range = int((q*k)*self.train_size+res)
            train_set = X.head(train_range-self.gap)
            test_set = X[train_range:train_range+int((q*k)*self.test_size)]
            yield np.array(train_set.index),np.array(test_set.index)


def forecast(project_list):
    max_iter = 500
    min_perc = 80
    tscv = CustomWalkForward(test_size=0.2, gap=0)
    itpdf = pd.concat(project_list)
    itpdf = itpdf[['Slippage', '% WT Plan']]
    ave_rmse = []
    ave_score = []

    for train_index, test_index in tscv.split(itpdf):
        cv_df = pd.concat(project_list)
        cv_df = cv_df[['Slippage', '% WT Plan']]

        # --- NEW: Keep a pristine copy of the data for the final chart ---
        final_df = cv_df.copy()

        forecast_col = 'Slippage'
        X = cv_df.copy()
        scaler = preprocessing.MinMaxScaler()
        scaled_X = scaler.fit_transform(X)
        data = {'Slippage': scaled_X[:, 0], '% WT Plan': scaled_X[:, 1]}
        X = pd.DataFrame(data=data, index=X.index)
        X_train_split, X_lately = X.loc[train_index], X.loc[test_index]

        # Prepare training targets (this step safely drops NA in background)
        cv_df['train'] = cv_df[forecast_col].head(len(train_index) + len(test_index)).shift(-(len(test_index)))
        train_df = cv_df.copy().dropna()

        cv_df['test'] = cv_df[forecast_col].head(len(test_index))
        test_df = cv_df.dropna(subset=['test'])

        y = np.array(train_df['train'])
        y_lately = np.array(test_df['test'])
        y_lately = y_lately[np.logical_not(np.isnan(y_lately))]

        # Train and get predictions + standard deviation bounds
        full_set = train_model(X_lately, X_train_split, y_lately, y, max_iter, min_perc)

        y_pred = full_set[4]
        ave_score.append(full_set[5])
        ave_rmse.append(full_set[6])
        sigma = full_set[7]

        # --- NEW: Append the Forecast to the UNTOUCHED final_df ---
        final_df['Forecast'] = np.nan
        final_df['Upper Bound'] = np.nan
        final_df['Lower Bound'] = np.nan

        last_date = final_df.iloc[-1].name
        last_unix = last_date.timestamp()
        one_day = 86400
        next_unix = last_unix + one_day

        for pred, sig in zip(y_pred, sigma):
            next_date = datetime.fromtimestamp(next_unix)
            next_unix += one_day

            final_df.loc[next_date] = np.nan  # Add new row
            final_df.loc[next_date, 'Forecast'] = pred
            final_df.loc[next_date, 'Upper Bound'] = pred + (1.96 * sig)
            final_df.loc[next_date, 'Lower Bound'] = pred - (1.96 * sig)

        # Connect the forecast line to the historical line smoothly
        final_df.loc[last_date, 'Forecast'] = final_df.loc[last_date, 'Slippage']

    print('SCORE: ', np.mean(ave_score), '\nRMSE: ', np.mean(ave_rmse))
    return [final_df, final_df.reset_index(), np.mean(ave_score), np.mean(ave_rmse)]

