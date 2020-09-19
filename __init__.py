import flask
from werkzeug.middleware.dispatcher import DispatcherMiddleware
from werkzeug.serving import run_simple
import sys
import dash
import dash_core_components as dcc
import dash_html_components as html



server = flask.Flask(__name__)

graph = dash.Dash(__name__,server=server, url_base_pathname='/')

graph.layout = html.Div(children=[html.H1('Dashboard'),
                                  dcc.Graph(id='example',
                                            figure={
                                                'data':[
                                                    {'x':[1,2,3,4],'y':[5,6,7,8],'type':'line','name':'boats'},
                                                    {'x':[1,2,3,4],'y':[5,6,7,8],'type':'bar','name':'cars'},
                                                ],
                                                'layout':{
                                                    'title':'Basic Dash'
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