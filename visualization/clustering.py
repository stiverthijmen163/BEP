import pandas as pd
import dash
from dash import dcc, html, callback, Output, Input, State
import plotly.express as px
import plotly.graph_objects as go
import cv2
from PIL import Image
import numpy as np
from functions import *
from sklearn.cluster import DBSCAN
import sqlite3
import json


class Clusteror(html.Div):
    def __init__(self, name, df, df_faces):
        self.html_id = name
        self.df = df
        self.df_faces = df_faces
        self.fig = None
        self.ebs = 5.4

        self.df_faces["face"] = self.df_faces["face"].apply(json.dumps)
        self.df_faces["img"] = self.df_faces["img"].apply(lambda x: json.dumps(x.tolist()))
        # Save the embeddings
        conn = sqlite3.connect("temp.db")
        self.df_faces.to_sql("faces", conn, if_exists="replace")

        # Load the embeddings
        # conn = sqlite3.connect("temp.db")
        # query = """SELECT * FROM faces"""
        # self.df_faces = pd.read_sql_query(query, conn)
        # self.df_faces["embedding_tsne"] = self.df_faces["embedding_tsne"].apply(lambda x: np.fromstring(x, sep=","))
        # self.df_faces["embedding"] = self.df_faces["embedding"].apply(lambda x: np.fromstring(x, sep=","))
        # self.df_faces["face"] = self.df_faces["face"].apply(json.loads)
        # self.df_faces["img"] = self.df_faces["img"].apply(json.loads)
        # self.df_faces["img"] = self.df_faces["img"].apply(np.array)


        # STILL NEEDED EVEN WHEN NOT LOADING FROM DB
        self.df_faces["embedding_tsne"] = self.df_faces["embedding_tsne"].apply(lambda x: np.fromstring(x, sep=","))
        self.df_faces["embedding"] = self.df_faces["embedding"].apply(lambda x: np.fromstring(x, sep=","))

        # Collect clusters
        labels = DBSCAN(eps=self.ebs, min_samples=1, n_jobs=-1, metric="euclidean").fit_predict(
            self.df_faces["embedding_tsne"].tolist())
        labels = [str(i) for i in labels]
        print(labels)
        self.df_faces["cluster"] = labels
        self.df_faces["name"] = labels

        self.fig = px.scatter(self.df_faces, x="tsne_x", y="tsne_y", color="name")
        self.fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0}, showlegend=True)


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
                                    figure=self.fig,
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