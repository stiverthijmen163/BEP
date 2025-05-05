import pandas as pd
import dash
from dash import dcc, html, callback, Output, Input, State
import plotly.express as px
import plotly.graph_objects as go
import cv2
from PIL import Image
import numpy as np
from functions import *


class Clusteror(html.Div):
    def __init__(self, name, df, df_faces):
        self.html_id = name
        self.df = df
        self.df_faces = df_faces

        # Initialize Detector as empty Div
        super().__init__(
            className="graph_card", id=self.html_id,
            children=[
                html.H2("Face Clustering", style={"textAlign": "center"}),
                html.Hr(),
                html.Div(
                    style={
                        'backgroundColor': '#dbeafe',
                        # 'padding': '20px',
                        'margin': '20px auto',
                        'width': '99vw',
                        'borderRadius': '12px',
                        'boxShadow': '0px 2px 5px rgba(0,0,0,0.1)',
                        'textAlign': 'center'
                    },
                    children=[
                        html.Br(),
                        html.Div(
                            style={
                                'display': 'flex',  # Enable horizontal layout
                                'justifyContent': 'space-between',  # Optional: adds spacing between elements
                                'alignItems': 'flex-start',  # Optional: align tops
                                'gap': '10px',  # Optional: space between items
                                'width': '99%',
                                'margin': '10px auto',
                                # "marginTop": "5px"
                            },
                            children=[
                                dcc.Graph(
                                    id=f"{self.html_id}",
                                    style={'flex': '1',
                                           "height": "20vw"}  # Takes available space
                                ),
                                html.Div(
                                    style={
                                        'backgroundColor': 'white',
                                        'width': '50%',  # Adjust width as needed
                                        'borderRadius': '12px',
                                        'border': '2px black solid',
                                        'textAlign': 'center',
                                        "height": "20vw"
                                    },
                                )
                            ]
                        ),
                        html.Br(),
                        html.Button(
                            "Save to Database",
                            disabled=False,
                            style={
                                'padding': '10px 20px',
                                'fontSize': '16pt',
                                'borderRadius': '12px',
                                'border': 'none',
                                'backgroundColor': '#2196F3',
                                'color': 'white',
                                'cursor': 'pointer',
                                "width": "20vw",
                                "marginBottom": "20px",
                            },
                            id="button3"
                        ),
                        html.Br()
                    ]
                )
            ]
        )