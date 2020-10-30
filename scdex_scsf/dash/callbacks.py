import dash
from dash.dependencies import Input, Output, State
import plotly.graph_objects as go
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
from .data_model import dataframe
import time

def create_contractor_dropdown():
    dropdown = dcc.Dropdown(
        id='contractor-dropdown',
        options=[{'label': contractor, 'value': contractor} for contractor in
                 set(dataFrame()['Implementor'])]
    )
    return dropdown

def dataFrame(contractor=None, project=None):
    data_frame = dataframe(contractor, project)
    return data_frame

def init_callbacks(app):
    @app.callback(
        [Output(f"collapse-{i}", "is_open") for i in range(1, 3)],
        [Input(f"group-{i}-toggle", "n_clicks") for i in range(1, 3)],
        [State(f"collapse-{i}", "is_open") for i in range(1, 3)],
    )
    def toggle_accordion(n1, n2, is_open1, is_open2):
        ctx = dash.callback_context

        if not ctx.triggered:
            return False, False
        else:
            button_id = ctx.triggered[0]["prop_id"].split(".")[0]

        if button_id == "group-1-toggle" and n1:
            return not is_open1, False
        elif button_id == "group-2-toggle" and n2:
            return False, not is_open2
        return False, False

    @app.callback(Output('project-dropdown','options'),
                  [Input('contractor-dropdown','value')])
    def update_project_dropdown(contractor):
        return [{'label':project, 'value':project} for project in set(dataFrame()['Name of Project'].loc[dataFrame()['Implementor']==contractor].values)]

    @app.callback([Output('notif-space', 'children'),
                   Output('forecast-graph', 'figure'),
                   Output('distribution-graph', 'figure'),
                   Output('regression-graph', 'figure'),
                   Output('score_lbl', 'children'),
                   Output('rmse_lbl', 'children'),
                   Output('collapse-1', 'children'),
                   Output('collapse-2', 'children')],
                  [Input('contractor-dropdown', 'value'),
                   Input('project-dropdown', 'value')])
    def update_forecast_graph(contractor, project):
        time.sleep(1)
        if not contractor and not project:
            contractor, project = 'FREDEN CONSTRUCTION', 'Const. of 2-Storey Barangay Hall, Brgy. Kablon'

        df = dataFrame(contractor, project)

        accordion_table2 = dbc.CardBody(children=[
            dbc.Table.from_dataframe((dataFrame()), striped=True, bordered=True, hover=True, className="md-10")
        ])

        try:
            figure = go.Figure(
                data=[{'x': df[0].index, 'y': df[0]['Slippage'],
                       'mode': 'lines+markers', 'name': 'Slippage'},
                      {'x': df[0].index, 'y': df[0]['train'],
                       'mode': 'lines+markers', 'name': 'Lead'},
                      {'x': df[0].index, 'y': df[0]['Forecast'],
                       'mode': 'lines+markers', 'name': 'Forecast'}],
                layout=go.Layout({"title": project,
                                  "xaxis": {
                                      'rangeslider': {'visible': True},
                                      'rangeselector': {'visible': True,
                                                        'buttons': [{'step': 'all'}, {'step': 'year'},
                                                                    {'step': 'month'},
                                                                    {'step': 'day'}]}},
                                  "showlegend": False,
                                  "height": 500}
                                 )
            )

            dist_figure = {'layout': {'title': 'Distribution'}}

            reg_figure = {'layout': {'title': 'Regression'}}

            accordion_table1 = dbc.CardBody(children=[
                dbc.Table.from_dataframe((df[0]), striped=True, bordered=True, hover=True, className="md-10")
            ])

            score = dbc.Alert('AdjR2: %.4f'% df[2], color="info")
            rmse = dbc.Alert('RMSE: %.4f'% df[3], color="info")

            return  None, figure, dist_figure, reg_figure, score, rmse, accordion_table1, accordion_table2

        except Exception as e:
            children = dbc.Col([dbc.Alert("ERROR: " + str(e), color="danger", style={'margin-bottom':'8px'})], md=12, style={'margin': '0px', 'padding': '0px 4px'})
            return  children, None, None, None, None, None, None, accordion_table2

