import flask
import sys
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Output, Input
import plotly
import plotly.graph_objs as go
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from os import environ

app = flask.Flask(__name__, instance_relative_config=False)
app.config.from_object(environ.get('APP_SETTINGS'))
db = SQLAlchemy(app)

def start_app(app):

    with app.app_context():

        from . import routes

        from .dash.dashboard import init_dashboard
        app = init_dashboard(app)

        return app

def init_app():
    return start_app(app)

def def_app():
    return app

def def_db():
    return db