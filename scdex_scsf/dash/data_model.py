import numpy as np
import pandas as pd
from sklearn import preprocessing, gaussian_process
from sklearn.model_selection import train_test_split
from sklearn.gaussian_process.kernels import WhiteKernel, ExpSineSquared, ConstantKernel, RBF
from datetime import datetime
import math

def filter(df,contractor_name, project_name, date_range):

    df = df.loc[df['Implementor'] == contractor_name]
    df = df.loc[df['Name of Project'] == project_name]
    project_list = upsample(df)

    return forecast(project_list)

def dataframe(contractor_name=None, project_name=None, date_range=None):
    df = pd.read_csv('scdex_scsf/dataset/full_slippage_dataset.csv', encoding='utf-8', engine='python')
    df['Contract Amount'] = df['Contract Amount'].astype(str)
    df['No. of Days'] = df['No. of Days'].astype(str)
    df.set_index('index', inplace=True)
    df.index = pd.to_datetime(df.index)
    if project_name and contractor_name:
        return filter(df, contractor_name, project_name, date_range)
    return df.reset_index()

def upsample(df):
    proj_list = [proj for proj in df['Name of Project'].values]
    project_list = []
    for project in set(proj_list):
        rec_df = df.loc[df['Name of Project'] == project]
        order = (rec_df.shape[0]) - 1
        if order >= 5: order = 5
        itpd_df = rec_df.resample("D").interpolate(method='spline', order=order)
        itpd_df = itpd_df.fillna(method='pad')
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

    return [adj_r2,rmse]

def train_model(X_lately, X, y, max_iter, min_perc, ave_score, test_set):
    for r in range(max_iter):
        X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True, test_size=0.2, random_state=r)
        model = define_model().fit(X_train, y_train)
        model_score = model.score(X_test, y_test)
        if model_score >= min_perc/100:
            y_test_pred = model.predict(X_test)
            y_pred = model.predict(X_lately)

            return [X_train, X_test, y_train, y_test, y_pred, model_score,score_model(X, y_test, y_test_pred)[1]]

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
        # print('TRAIN: ', train_index, 'TEST: ', test_index)
        cv_df = pd.concat(project_list)
        cv_df = cv_df[['Slippage', '% WT Plan']]
        forecast_col = 'Slippage'
        X = cv_df
        scaler = preprocessing.StandardScaler()
        scaled_X = scaler.fit_transform(X)
        data = {'Slippage': scaled_X[:, 0], '% WT Plan': scaled_X[:, 1]}
        X = pd.DataFrame(data=data, index=X.index)
        X, X_lately = X.loc[train_index], X.loc[test_index]

        cv_df['label'] = cv_df[forecast_col].head(len(train_index) + len(test_index)).shift(-(len(test_index)))

        cv_df.dropna(inplace=True)
        y = np.array(cv_df['label'])

        full_set = train_model(X_lately, X, y, max_iter, min_perc, ave_score,test_set=True)
        y_pred = full_set[4]
        ave_score.append(full_set[5])
        ave_rmse.append(full_set[6])
        cv_df['Forecast'] = np.nan

        last_date = cv_df.iloc[-1].name
        last_unix = last_date.timestamp()
        one_day = 86400
        next_unix = last_unix + one_day

        for i in y_pred:
            next_date = datetime.fromtimestamp(next_unix)
            next_unix += one_day
            cv_df.loc[next_date] = [np.nan for _ in range(len(cv_df.columns) - 1)] + [i]

        cv_df['label'] = cv_df['label'].shift(len(test_index))
        cv_df['Forecast'] = np.nan

        for i in y_pred:
            next_date = datetime.fromtimestamp(next_unix)
            next_unix += one_day
            cv_df.loc[next_date] = [np.nan for _ in range(len(cv_df.columns) - 1)] + [i]

    for lead in cv_df['label'].values:
        for slippage in cv_df['Slippage'].values:
            if lead == slippage:
                cv_df['label'].loc[cv_df['label']==lead] = np.nan
    print('SCORE: ', ave_score, '\nRMSE: ', ave_rmse)
    return [cv_df, cv_df.reset_index(), np.mean(ave_score), np.mean(ave_rmse)]

