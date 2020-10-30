import dash_bootstrap_components as dbc
import dash_html_components as html
import dash_core_components as dcc
from .callbacks import create_contractor_dropdown
import dash_table
import plotly.graph_objects as go


def navbar():
    navbar = dbc.NavbarSimple(children=[
        dbc.NavItem(dbc.NavLink('Upload CSV', href='#')),
        dbc.NavItem(dbc.NavLink('API Docs', href='#')),

    ],
        brand='SCDex',
        brand_href='/',
        sticky='top',
    )

    return navbar

def header():
    header = dbc.Container([
        dbc.Card(
            dbc.CardBody([
                dbc.Row([
                    dbc.Col([
                        html.P('Forecasting contractors\' slippage for infrastructure projects in South Cotabato')
                    ])
                ]),
                html.Hr(),
                dbc.Row([
                    dbc.Col([
                        create_contractor_dropdown()
                    ], md=6),
                    dbc.Col([
                        dcc.Dropdown(
                            id='project-dropdown'
                        )
                    ], md=6)
                ]),
            ]), style={'margin-top':'8px', 'margin-bottom':'8px'}
        )
    ], style={'padding':'0px 8px'})
    return header

def body():
    body = dbc.Container([
        dbc.Row(id='notif-space'),
        dbc.Row(
            [
                dbc.Col([
                    dbc.Card(
                        dcc.Loading(
                            id='label-loading',
                            type='circle',
                            children=
                            [
                                html.Div(id='label-loading-output'),
                                dbc.CardBody(
                                    [
                                        html.Div(id='score_lbl'),
                                        html.Div(id='rmse_lbl'),
                                    ]
                                )
                            ]
                        )
                    )
                ], md=4, style={'margin':'0px','padding':'0px 4px'}),

                dbc.Col([
                    dbc.Card(
                        dcc.Loading(
                            id='forecast-loading',
                            type='circle',
                            children=[
                                html.Div(id='forecast-loading-output'),
                                dcc.Graph(id='forecast-graph')
                            ]
                        )
                    )
                ], style={'margin':'0px','padding':'0px 4px'})
            ], style={'padding':'0px 4px'}
        ),
        dbc.Row([
            dbc.Col([
                dbc.Card(
                    dbc.CardBody(
                        dbc.Row([
                            dbc.Col([
                                dcc.Loading(
                                    id='dist-loading',
                                    type='circle',
                                    children=[
                                        html.Div(id='dist-loading-output'),
                                        dcc.Graph(
                                            id='distribution-graph',
                                            style={'height': 300, 'margin': 0, 'padding': 0}
                                        )
                                    ]
                                )
                            ]),
                            dbc.Col([
                                dcc.Loading(
                                    id='reg-loading',
                                    type='circle',
                                    children=[
                                        html.Div(id='reg-loading-output'),
                                        dcc.Graph(
                                            id='regression-graph',
                                            style={'height': 300, 'margin': 0, 'padding': 0}
                                        )
                                    ]
                                )
                            ])
                        ])
                    )
                )
            ], style={'margin':'8px 4px 4px','padding':'0px 4px'})
        ]),
        dbc.Row([
            dbc.Col([
                create_accordion()
            ], style={'margin':'4px','padding':'0px 4px'})
        ])
    ])
    return body

def footer():
    footer = dbc.Row([])
    return footer

def make_item(i):
    # we use this function to make the example items to avoid code duplication
    return dbc.Card(
        [
            dbc.CardBody(
                html.H2(
                    dbc.Button(
                        f"{'Forecast Table' if i == 1 else 'Main Data Table'}",
                        color="link",
                        id=f"group-{i}-toggle",
                    )
                )
            ),
            dbc.Collapse(
                id=f"collapse-{i}"
            ),

        ]
    )

def create_accordion():
    accordion = html.Div(
        [make_item(1),make_item(2)], className="accordion"
    )
    return accordion