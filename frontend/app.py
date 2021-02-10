import numpy as np
from textblob import TextBlob
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import plotly.graph_objs as go
import itertools
import math
import base64
from flask import request, Flask, jsonify
import os
import psycopg2
import datetime
import time
import re
import requests
import pandas as pd
import json
import dash_table

external_stylesheets = [
    'https://codepen.io/chriddyp/pen/bWLwgP.css',
    {
        'href': 'https://stackpath.bootstrapcdn.com/bootstrap/4.1.3/css/bootstrap.min.css',
        'rel': 'stylesheet',
        'integrity': 'sha384-MCw98/SFnGE8fJT3GXwEOngsV7Zt27NXFoaoApmYm81iuXoPkFOJwJ8ERdknLPMO',
        'crossorigin': 'anonymous'
    }
]

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
app.title = 'Real-Time Twitter Monitor'

server = app.server

app.layout = html.Div(children=[
    html.H2('Twitter Sentiment Analysis Application', style={
        'textAlign': 'center'
    }),
    html.H3(id='button-clicks'),
    html.Label('Search Keywords'),
    dcc.Input(id='search_words'),

    # html.Label('Date Since'),
    # dcc.Input(id='date_since'),

    # html.Label('Number of Tweets'),
    # dcc.Input(id='tweets_num'),

    html.Button('Run', id='btn_run'),
    html.Hr(),
    html.Div(id='text-classification-dashborad'),
    dcc.Interval(
        id='interval-component-slow',
        interval=1*10000*60,  # in milliseconds
        n_intervals=0
    )
], style={'padding': '20px'})
from datetime import datetime
# Multiple components can update everytime interval gets fired.
# @app.callback(Output('text-classification-dashborad', 'children'),
#             [Input('btn_run', 'n_clicks')],
#             state=[State('search_words', 'value'),
#                     State('date_since', 'value'),
#                     State('tweets_num', 'value')])
@app.callback(Output('text-classification-dashborad', 'children'),
            [Input('btn_run', 'n_clicks')],
            state=[State('search_words', 'value')])
# def load_textClassification_dashboard(n_clicks, search_words, date_since, tweets_num):
def load_textClassification_dashboard(n_clicks, search_words):
    children = []
    
    if(n_clicks != None):
        form = {
            'search_words' : search_words,
            # 'date_since' : date_since,
            # 'tweets_num' : tweets_num
        }
        response = requests.post("http://127.0.0.1:5000/analytics", data = form)
        app.logger.info(f"response: \n {response.json()}")
        df_tweets = pd.DataFrame(response.json(), columns=['Time','Text','polarity'])
        app.logger.info(f"df_tweets: \n {df_tweets}")
        df_tweets['Time'] = pd.to_datetime(df_tweets['Time']).apply(lambda x: datetime.strptime(str(x), "%Y-%m-%d %H:%M:%S"))
        # df_tweets = df_tweets.set_index(['Time'])
        # df_tweets.index = pd.to_datetime(df_tweets.index, unit='s')
        # app.logger.info(df_tweets.groupby([pd.Grouper(key='Time', freq='10s'), 'polarity']))
        result = df_tweets.groupby([pd.Grouper(key='Time', freq='10s'), 'polarity'])['polarity'].count().unstack(fill_value=0).stack().reset_index()
        result = result.rename(columns={0: "Quantity"})  
        app.logger.info(f"result222: \n {result['Quantity'][result['polarity']==0]}")
        time_series = result["Time"][result['polarity']==0].reset_index(drop=True)
        app.logger.info(f"timeseries: \n {time_series}")
        children = [
                    html.Div([
                        html.Div([
                            dcc.Graph(
                                id='crossfilter-indicator-scatter',
                                figure={
                                    'data': [
                                        go.Scatter(
                                            x=time_series,
                                            y= result['Quantity'][result['polarity']==0].reset_index(drop=True),
                                            name="Neutrals",
                                            opacity=0.8,
                                            mode='lines',
                                            line=dict(width=0.5, color='rgb(131, 90, 241)'),
                                            stackgroup='one' 
                                        ),
                                        go.Scatter(
                                            x=time_series,
                                            y=result['Quantity'][result['polarity']==-1].reset_index(drop=True).apply(lambda x: -x),
                                            name="Negatives",
                                            opacity=0.8,
                                            mode='lines',
                                            line=dict(width=0.5, color='rgb(255, 50, 50)'),
                                            stackgroup='two' 
                                        ),
                                        go.Scatter(
                                            x=time_series,
                                            y=result['Quantity'][result['polarity']==1].reset_index(drop=True),
                                            name="Positives",
                                            opacity=0.8,
                                            mode='lines',
                                            line=dict(width=0.5, color='rgb(184, 247, 212)'),
                                            stackgroup='three' 
                                        )
                                    ]
                                }
                            )
                        ], style={'width': '73%', 'display': 'inline-block', 'padding': '0 0 0 20'}),
                        
                        html.Div([dash_table.DataTable(
                            style_cell={
                                'whiteSpace': 'normal',
                                'height': 'auto',
                            },
                            style_cell_conditional=[
                                {
                                    'textAlign': 'left'
                                }
                            ],
                            data=df_tweets.to_dict('records'),
                            columns=[{'id': c, 'name': c} for c in df_tweets.columns],
                            page_size=10,
                            style_table={'height': '300px','overflowY': 'auto'}
                        )
                        ],style = {
                                'font-size': '12px',
                            }
                        )
                        # html.Div([
                        #     html.Table(className="responsive-table",
                        #     children=[
                        #         html.Thead(
                        #             html.Tr(
                        #                 children=[
                        #                     html.Th(col.title()) for col in df_tweets.columns.values
                        #                 ]
                        #                 # style={'color':'#FFFFFF'}
                        #                 )
                        #             ),
                        #         html.Tbody(
                        #             [
                                        
                        #             html.Tr(
                        #                 children=[
                        #                     html.Td(data) for data in d
                        #                     ]
                        #                     # style={'color':'#FFFFFF','background-color':'#0C0F0A'}
                        #                 )
                        #             for d in df_tweets.values.tolist()
                        #             ],style = {
                        #                 'height': '10px',
                        #                 'overflowY': 'scroll'
                        #             }
                        #         )
                        #     ]
                        #     )
                        # ])
                    ])
        ]
        
    return children

if __name__ == '__main__':
    app.run_server(debug=True)
