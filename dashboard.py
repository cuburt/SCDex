import pandas as pd
import pickle

import dash
from dash import html
import dash_bootstrap_components as dbc
from content import navbar, body, header, footer
from callbacks import init_callbacks



def init_dashboard(server=None):
    if server:
        app = dash.Dash(
            server=server,
            routes_pathname_prefix='/',
            external_stylesheets=[dbc.themes.SIMPLEX]
        )

        app.layout = html.Div(id='dash-container')



        app.layout = html.Div([
            navbar(),
            header(),
            body(),
            footer(),
        ])
    else:
        app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
        app.title = "SCDex"
        app.layout = html.Div(children=[
            navbar(),
            header(),
            body(),
            footer()
        ])
    init_callbacks(app)

    return app.server if server else app



