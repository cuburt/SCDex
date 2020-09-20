import flask
import sys
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Output, Input
import plotly
import plotly.graph_objs as go


import numpy as np
import pandas as pd
from sklearn import preprocessing, gaussian_process
from sklearn.model_selection import train_test_split
from sklearn.gaussian_process.kernels import WhiteKernel, ExpSineSquared, ConstantKernel, RBF
from datetime import datetime
import math

def start(contractor_name, project_name):
    df = pd.read_csv('D:/devs/flask/scsf/dataset/full_slippage_dataset.csv',encoding='utf-8',engine='python')
    df['Contract Amount'] = df['Contract Amount'].astype(str)
    df['No. of Days'] = df['No. of Days'].astype(str)
    df = df.loc[df['Implementor']==contractor_name]
    df = df.loc[df['Name of Project']==project_name]
    df.set_index('index',inplace=True)
    df.index = pd.to_datetime(df.index)
    project_list = upsample(df)

    return forecast(project_list)

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

def train_model(X_lately, X, y, max_iter, min_perc, cross_val_score, test_set):
    for r in range(max_iter):
        X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True, test_size=0.2, random_state=r)
        model = define_model().fit(X_train, y_train)
        if model.score(X_test, y_test) >= min_perc/100:
            y_test_pred = model.predict(X_test)
            y_pred = model.predict(X_lately)
            #print('ADJ. R2: ',score_model(X, y_test, y_test_pred)[0], '\nRMSE: ',score_model(X, y_test, y_test_pred)[1])
            cross_val_score.append(score_model(X, y_test, y_test_pred)[0])
            return [X_train, X_test, y_train, y_test, y_pred, score_model(X, y_test, y_test_pred)[1]]
        else: continue

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
    cross_val_score = []
    tscv = CustomWalkForward(test_size=0.2, gap=0)
    itpdf = pd.concat(project_list)
    itpdf = itpdf[['Slippage', '% WT Plan']]
    ave_rmse = []

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

        full_set = train_model(X_lately, X, y, max_iter, min_perc, cross_val_score, test_set=True)
        y_pred = full_set[4]
        ave_rmse.append(full_set[5])
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

    return cv_df

contractor_name, project_name = 'FREDEN CONSTRUCTION','Const. of 2-Storey Barangay Hall, Brgy. Kablon'

server = flask.Flask(__name__)

graph = dash.Dash(__name__,server=server, url_base_pathname='/')

# graph.layout = html.Div(
#     html.Div([
#         html.H4('Slippage Dashboard'),
#         html.Div(id='live-update-text'),
#         dcc.Graph(id='live-graph'),
#         dcc.Interval(
#             id='graph-update',
#             interval=2*1000,
#             n_intervals = 0
#         )
#     ])
#
# )
#
# @graph.callback(Output('live-graph','figure'),[Input('graph-update','n_intervals')])
# def update_data(input_data):
#
#     df = pd.read_csv('D:/devs/flask/scsf/dataset/full_slippage_dataset.csv', encoding='utf-8', engine='python')
#     df['Contract Amount'] = df['Contract Amount'].astype(str)
#     df['No. of Days'] = df['No. of Days'].astype(str)
#     df = df.loc[df['Implementor'] == contractor_name]
#     df = df.loc[df['Name of Project'] == project_name]
#     df.set_index('index', inplace=True)
#     df.index = pd.to_datetime(df.index)
#     project_list = upsample(df)
#
#     new_df = next(forecast(project_list))
#     # if not new_df['label'].empty: Y = new_df['label']
#     # if not new_df['Forecast'].empty: Y = new_df['Forecast']
#
#     data = plotly.graph_objs.Scatter(
#         x=new_df.index,
#         y = new_df['Slippage'],
#         name='Scatter',
#         mode='lines+markers'
#     )
#     return {'data':[data],
#             # 'layout': go.Layout(xaxis=dict(range=[min(X),max(X)]),
#             #                                    yaxis=dict(range=[min(Y),max(Y)]),
#             #                                    )
#             }

df = start(contractor_name, project_name)
graph.layout = html.Div(children=[html.H1('Dashboard'),
                                  dcc.Input(id='input', value='Enter Project Name here',type='text'),
                                  dcc.Graph(id='example',
                                            figure={
                                                'data':[
                                                    {'x':df.index,'y':df['Slippage'],'type':'bar','name':'Slippage'},
                                                    {'x':df.index,'y':df['label'],'type':'bar','name':'Lead'},
                                                    {'x':df.index,'y':df['Forecast'],'type':'bar','name':'Forecast'}
                                                ],
                                                'layout':{
                                                    'title':project_name
                                                }
                                            })])

# @server.route('/')
# def homepage():
#     return flask.redirect('/graph')

@server.route('/about')
def about():
    return flask.render_template('about.html')
#
# application = DispatcherMiddleware(server, {
#     '/dash':graph.server
# })

if __name__ == '__main__':
    server.run(debug=True)