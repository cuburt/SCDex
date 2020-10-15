import pandas as pd
import pickle

import dash
import dash_html_components as html
import dash_core_components as dcc
import dash_bootstrap_components as dbc
from .content import navbar, body, header, footer
from .callbacks import init_callbacks



def init_dashboard(server):
    dash_app = dash.Dash(
        server=server,
        routes_pathname_prefix='/',
        external_stylesheets=[dbc.themes.SIMPLEX]
    )

    dash_app.layout = html.Div(id='dash-container')



    dash_app.layout = html.Div([
        navbar(),
        header(),
        body(),
        footer(),
    ])

    init_callbacks(dash_app)

    return dash_app.server



