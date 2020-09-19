import flask
import dash
import dash_core_components as dcc
import dash_html_components as html
import sys

app = flask.Flask(__name__)
graph = dash.Dash(__name__,server=app, url_base_pathname='/graph')

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


@app.route('/')
def homepage():
    return flask.redirect('/graph')

@app.route('/about')
def about():
    return flask.render_template('about.html')

if __name__ == '__main__':
    app.run(debug=True)