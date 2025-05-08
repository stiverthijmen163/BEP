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
import dash_bootstrap_components as dbc
import seaborn as sns
import matplotlib.colors as mcolors

# custom_palette = ['#3333ff', '#8080ff', '#9999ff', '#ccccff', '#ff3333', '#ff8080', '#ff9999', '#ffcccc', '#ffb733', '#ffd280', '#ffdb99', '#ffedcc', '#339933', '#80c080', '#99cc99', '#cce6cc', '#ff87c3', '#ffb4da', '#ffc3e1', '#ffe1f0', '#993399', '#c080c0', '#cc99cc', '#e6cce6', '#ffff33', '#ffff80', '#ffff99', '#ffffcc', '#b75555', '#d29494', '#dbaaaa', '#edd4d4']
palette = sns.color_palette("tab20", n_colors=20)
custom_palette = [mcolors.to_hex(color) for color in palette]


class Clusteror(html.Div):
    def __init__(self, name, df, df_faces):
        self.html_id = name
        self.df = df
        self.df_faces = df_faces
        self.fig = None
        self.eps = 5.4
        self.min_samples = 1
        self.min_size = (0, 0)
        self.selected_index = 0

        # self.df_faces["face"] = self.df_faces["face"].apply(json.dumps)
        # self.df_faces["img"] = self.df_faces["img"].apply(lambda x: json.dumps(x.tolist()))
        # # Save the embeddings
        # conn = sqlite3.connect("temp.db")
        # self.df_faces.to_sql("faces", conn, if_exists="replace")

        # Load the embeddings
        conn = sqlite3.connect("temp.db")
        query = """SELECT * FROM faces"""
        self.df_faces = pd.read_sql_query(query, conn)
        self.df_faces["embedding_tsne"] = self.df_faces["embedding_tsne"].apply(lambda x: np.fromstring(x, sep=","))
        self.df_faces["embedding"] = self.df_faces["embedding"].apply(lambda x: np.fromstring(x, sep=","))
        self.df_faces["face"] = self.df_faces["face"].apply(json.loads)
        self.df_faces["img"] = self.df_faces["img"].apply(json.loads)
        self.df_faces["img"] = self.df_faces["img"].apply(lambda x: np.array(x, dtype=np.uint8))
        self.df_faces["width"] = self.df_faces["face"].apply(lambda x: list(x)[2])
        self.df_faces["height"] = self.df_faces["face"].apply(lambda x: list(x)[3])


        # STILL NEEDED EVEN WHEN NOT LOADING FROM DB
        # self.df_faces["embedding_tsne"] = self.df_faces["embedding_tsne"].apply(lambda x: np.fromstring(x, sep=","))
        # self.df_faces["embedding"] = self.df_faces["embedding"].apply(lambda x: np.fromstring(x, sep=","))

        # Collect clusters
        labels = DBSCAN(eps=self.eps, min_samples=self.min_samples, n_jobs=-1, metric="euclidean").fit_predict(
            self.df_faces["embedding_tsne"].tolist())
        labels = [str(i) for i in labels]
        print(labels)
        self.df_faces["cluster"] = labels
        self.df_faces["name"] = labels

        self.fig = px.scatter(self.df_faces, x="tsne_x", y="tsne_y", color="name", color_discrete_sequence=custom_palette)
        self.fig.update_layout(margin=dict(l=0, r=0, b=0), showlegend=True,
                               title_text=f"Resulting clusters with eps={self.eps}, min_samples={self.min_samples}, min_size={self.min_size}", title_x=0.5)

        maximum = round(max(max(self.df_faces["tsne_x"]) - min(self.df_faces["tsne_x"]),
                      max(self.df_faces["tsne_y"]) - min(self.df_faces["tsne_y"])), 1)

        # Set the images to display as example
        self.images = self.df_faces[self.df_faces["cluster"] == "0"]["img"].copy().to_list()

        if len(self.images) > 10:
            nr = 10
            right_disabled = False
        else:
            nr = len(self.images)
            right_disabled = True
        children = []
        for i in range(nr):
            img = self.images[i].copy()
            image = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            children.append(
                html.Div(
                    id={"type": "image-click1", "index": i},  # Pattern-matching id
                    style={"display": "inline-block", "width": "9%", "padding": "0.5%"},
                    n_clicks=0,  # Enables click tracking
                    children=html.Img(src=image, style={
                        "width": "100%",
                        "cursor": "pointer",
                        "maxHeight": "8vw",
                        "objectFit": "contain",
                        # "height": "8vw",
                    })
                )
            )


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
                                           "height": "22vw"}  # Takes available space
                                ),
                                html.Div(
                                    style={
                                        'backgroundColor': 'white',
                                        'width': '50%',  # Adjust width as needed
                                        'borderRadius': '12px',
                                        'border': '2px black solid',
                                        'textAlign': 'center',
                                        "height": "22vw"
                                    },
                                    children=[
                                        html.P(
                                            "First set the parameters such that the graph shows clusters you are satisfied with. Then, you can edit any cluster.",
                                            style={
                                                "fontSize": "16pt",
                                                "marginBottom": "5px",
                                                "textAlign": "center",
                                                "margin": "5px",
                                                "fontWeight": "bold"
                                            }
                                        ),
                                        html.Div([
                                            html.P(
                                                [
                                                    "eps:",
                                                    html.I(className="fa-solid fa-circle-question", id="info_icon_eps",
                                                           style={"cursor": "pointer", "color": "#0d6efd",
                                                                  "marginLeft": "5px",
                                                                  "position": "relative",
                                                                  "top": "-3px"
                                                                  }),
                                                    " ",
                                                    dcc.Input(
                                                        id="eps_input",
                                                        type="number",
                                                        min=0.1,
                                                        max=maximum,
                                                        step=0.1,
                                                        value=self.eps,
                                                        style={"marginLeft": "10px", "width": "8%"}
                                                    )
                                                ],
                                                style={
                                                    "fontSize": "16pt",
                                                    "marginBottom": "5px",
                                                    "textAlign": "center",
                                                    "margin": "0"
                                                },
                                            ),
                                            dbc.Tooltip(
                                                "The maximum distance between two samples for one to be considered as in the neighborhood of the other. This is not a maximum bound on the distances of points within a cluster.",
                                                target="info_icon_eps",
                                                placement="top"
                                            )
                                        ]),
                                        html.Br(),
                                        html.Div([
                                            html.P(
                                                [
                                                    "min_samples:",
                                                    html.I(className="fa-solid fa-circle-question", id="info_icon_min_samples",
                                                           style={"cursor": "pointer", "color": "#0d6efd",
                                                                  "marginLeft": "5px",
                                                                  "position": "relative",
                                                                  "top": "-3px"
                                                                  }),
                                                    " ",
                                                    dcc.Input(
                                                        id="min_samples_input",
                                                        type="number",
                                                        min=1,
                                                        max=20,
                                                        step=1,
                                                        value=self.min_samples,
                                                        style={"marginLeft": "10px", "width": "8%"}
                                                    )
                                                ],
                                                style={
                                                    "fontSize": "16pt",
                                                    "marginBottom": "5px",
                                                    "textAlign": "center",
                                                    "margin": "0"
                                                },
                                            ),
                                            dbc.Tooltip(
                                                "The number of samples (or total weight) in a neighborhood for a point to be considered as a core point. This includes the point itself. If min_samples is set to a higher value, DBSCAN will find denser clusters, whereas if it is set to a lower value, the found clusters will be more sparse.",
                                                target="info_icon_min_samples",
                                                placement="top"
                                            )
                                        ]),
                                        html.Br(),
                                        html.Div([
                                            html.P(
                                                [
                                                    "min_size:",
                                                    html.I(className="fa-solid fa-circle-question",
                                                           id="info_icon_min_size",
                                                           style={"cursor": "pointer", "color": "#0d6efd",
                                                                  "marginLeft": "5px",
                                                                  "position": "relative",
                                                                  "top": "-3px"
                                                                  }),
                                                    " width:",
                                                    dcc.Input(
                                                        id="min_width_cls_input",
                                                        type="number",
                                                        min=0,
                                                        max=max(self.df_faces["width"]),
                                                        step=1,
                                                        value=self.min_size[0],
                                                        style={"marginLeft": "10px", "width": "8%"}
                                                    ),
                                                    " height:",
                                                    dcc.Input(
                                                        id="min_height_cls_input",
                                                        type="number",
                                                        min=0,
                                                        max=max(self.df_faces["height"]),
                                                        step=1,
                                                        value=self.min_size[1],
                                                        style={"marginLeft": "10px", "width": "8%"}
                                                    )
                                                ],
                                                style={
                                                    "fontSize": "16pt",
                                                    "marginBottom": "5px",
                                                    "textAlign": "center",
                                                    "margin": "0"
                                                },
                                            ),
                                            dbc.Tooltip(
                                                "Select the minimum size of a face to be considered for clustering, faces that are smaller will be put in the 'unknown' cluster.",
                                                target="info_icon_min_size",
                                                placement="top"
                                            )
                                        ]),
                                        html.Br(),
                                        html.Div(
                                            style={
                                                'display': 'flex',
                                                'justifyContent': 'center',
                                                'gap': '5%',
                                                # 'padding': '20px',
                                                # 'backgroundColor': '#e0e0e0',
                                            },
                                            children=[
                                                html.Button(
                                                    "Update Clusters",
                                                    disabled=False,
                                                    style={
                                                        'padding': '10px 20px',
                                                        'fontSize': '16pt',
                                                        'borderRadius': '12px',
                                                        'border': 'none',
                                                        'backgroundColor': '#2196F3',
                                                        'color': 'white',
                                                        'cursor': 'pointer',
                                                        "width": "10vw",
                                                        # "marginBottom": "20px",
                                                    },
                                                    id="button_update_clusters"
                                                ),
                                                html.Div(children=[
                                                    html.Button(
                                                        "Continue",
                                                        disabled=False,
                                                        style={
                                                            'padding': '10px 0px',
                                                            'fontSize': '16pt',
                                                            'borderRadius': '12px',
                                                            'border': 'none',
                                                            'backgroundColor': '#2196F3',
                                                            'color': 'white',
                                                            'cursor': 'pointer',
                                                            "width": "10vw",
                                                            # "marginBottom": "20px",
                                                        },
                                                        id="button_continue_clusters"
                                                    ),
                                                    html.I(
                                                        className="fa-solid fa-circle-exclamation",
                                                        id="exclamation_button",
                                                        style={"cursor": "pointer", "color": "red",
                                                              "marginLeft": "5px",
                                                              "position": "relative",
                                                              "top": "-8pt"
                                                        }
                                                    ),
                                                    dbc.Tooltip(
                                                        "When this button is pressed, the parameters can't be changed anymore.",
                                                        target="exclamation_button",
                                                        placement="top"
                                                    )
                                                ], style={
                                                    "fontSize": "16pt"
                                                })
                                            ]
                                        ),
                                        html.Br(),
                                        html.Label(
                                            "Select cluster to inspect: ",
                                            style={
                                                "fontSize": "16pt",
                                                "marginBottom": "5px",
                                                "textAlign": "center",
                                                "margin": "0"
                                            },
                                        ),
                                        # dropdown for attribute
                                        html.Div(
                                            id="dropdown_change",
                                            children=
                                            [dcc.Dropdown(
                                                id="dropdown_cls",
                                                options=self.df_faces["name"].unique(),
                                                value="0",
                                                clearable=False,
                                                className="dropdown",
                                                searchable=True,
                                                # style={
                                                #     "width": "80%",
                                                #     "textAlign": "center",
                                                #     "margin": "0 auto"
                                                # }
                                            )],
                                            style={
                                                "width": "80%",
                                                "textAlign": "center",
                                                "margin": "0 auto"
                                            }
                                        )
                                    ]
                                )
                            ]
                        ),
                        # html.Br(),
                        html.Div([
                            html.Div("", style={"flex": 1}),  # Left spacer

                            html.Div(html.P(f"Showing cluster '0'.",  # Select a face to move to another cluster",
                                id="showing_cluster0",
                                style={
                                    "fontSize": "16pt",
                                    "marginBottom": "5px",
                                    "textAlign": "center",
                                    "margin": "0"
                            }), style={"flex": 1, "display": "flex", "justifyContent": "center"}),  # Center content

                            html.Div(html.P(f"Showing 1 - {nr} out of {len(self.images)}",
                                id="images_showing_txt1", style={
                                    "fontSize": "16pt",
                                    "marginBottom": "5px",
                                    "textAlign": "center",
                                    "margin": "0"
                            }), style={"flex": 0.995, "display": "flex", "justifyContent": "flex-end",
                                       "paddingRight": "0.5%"})  # Right content
                        ], style={
                            "display": "flex",
                            "alignItems": "center",
                            "width": "100%"
                        }),
                        html.Div(
                            style={
                                'backgroundColor': 'white',
                                # 'padding': '20px',
                                'margin': '5px auto',
                                'width': '99%',
                                'borderRadius': '12px',
                                # 'boxShadow': '0px 2px 5px rgba(0,0,0,0.1)',
                                "border": "2px black solid",
                                'textAlign': 'center',
                                # "maxHeight": "8vw"
                            },
                            # children=[
                            #     html.H2("Face Detection"),
                            #     html.Hr(),
                            #     # children,
                            # ].append(children)
                            children=children,
                            id="box_images1"
                        ),
                        html.Div(
                            style={
                                'display': 'flex',
                                'justifyContent': 'space-between',
                                'width': '99%',
                                'margin': '10px auto'
                            },
                            children=[
                                html.Button("⬅️ Back", id="box_images1_left", style={
                                    'width': '5%',
                                    "opacity": 0.5
                                }, disabled=True),
                                html.Button("Next ➡️", id="box_images1_right", style={
                                    'width': '5%',
                                    "opacity": 0.5 if right_disabled else 1.0
                                }, disabled=right_disabled)
                            ]
                        ),
                        # html.Br(),
                        html.Button(
                            "Save to Database",
                            disabled=True,
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
                                "opacity": 0.5
                            },
                            id="button3"
                        ),
                        html.Br()
                    ]
                )
            ]
        )


    def next_button(self, n_clicks):
        if n_clicks is not None and n_clicks > 0:
            self.selected_index += 10
            children = []

            if len(self.images[self.selected_index:]) > 10:
                nr = 10
                disabled_right = False
            else:
                nr = len(self.images[self.selected_index:])
                disabled_right = True

            for i in range(nr):
                img = self.images[i + self.selected_index].copy()

                image = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                children.append(
                    html.Div(
                        id={"type": "image-click1", "index": i},  # Pattern-matching id
                        style={"display": "inline-block", "width": "9%", "padding": "0.5%"},
                        n_clicks=0,  # Enables click tracking
                        children=html.Img(src=image, style={
                            "width": "100%",
                            "cursor": "pointer",
                            "maxHeight": "8vw",
                            "objectFit": "contain",
                            # "height": "8vw",
                        })
                    )
                )

            style_left = {"width": "5%"}
            style_right = {"width": "5%", "opacity": 0.5 if disabled_right else 1.0}

            new_txt = f"Showing {self.selected_index + 1} - {self.selected_index + nr} out of {len(self.images)}"

            return children, False, disabled_right, style_left, style_right, new_txt
        else:
            return dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update


    def previous_button(self, n_clicks):
        if n_clicks is not None and n_clicks > 0:
            self.selected_index -= 10
            children = []

            if self.selected_index == 0:
                disabled_left = True
            else:
                disabled_left = False

            for i in range(10):
                img = self.images[i + self.selected_index].copy()

                image = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                children.append(
                    html.Div(
                        id={"type": "image-click1", "index": i},  # Pattern-matching id
                        style={"display": "inline-block", "width": "9%", "padding": "0.5%"},
                        n_clicks=0,  # Enables click tracking
                        children=html.Img(src=image, style={
                            "width": "100%",
                            "cursor": "pointer",
                            "maxHeight": "8vw",
                            "objectFit": "contain",
                            # "height": "8vw",
                        })
                    )
                )

            style_left = {"width": "5%", "opacity": 0.5 if disabled_left else 1.0}
            style_right = {"width": "5%"}

            new_txt = f"Showing {self.selected_index + 1} - {self.selected_index + 10} out of {len(self.images)}"

            return children, disabled_left, False, style_left, style_right, new_txt


    def update_clusters(self, data, eps_input, min_samples_input, min_width_cls_input, min_height_cls_input):
        # if n_clicks is not None and n_clicks > 0:
        if data is not None and data:
            print(eps_input, min_samples_input, min_width_cls_input, min_height_cls_input)
            # Update the self values with new inputted values
            if eps_input is not None:
                self.eps = eps_input
            if min_samples_input is not None:
                self.min_samples = min_samples_input
            if min_width_cls_input is not None:
                self.min_size = (min_width_cls_input, self.min_size[1])
            if min_height_cls_input is not None:
                self.min_size = (self.min_size[0], min_height_cls_input)
            print(self.eps, self.min_samples, self.min_size)

            # Create new dataframe taking min size of face into account
            df_temp0 = self.df_faces[(self.df_faces["width"] >= self.min_size[0]) & (self.df_faces["height"] >= self.min_size[1])].copy()
            df_temp1 = self.df_faces[(self.df_faces["width"] < self.min_size[0]) | (self.df_faces["height"] < self.min_size[1])].copy()

            # Collect new clusters
            labels = DBSCAN(eps=self.eps, min_samples=self.min_samples, n_jobs=-1, metric="euclidean").fit_predict(
                df_temp0["embedding_tsne"].tolist())
            labels = [str(i) for i in labels]
            print(labels)
            df_temp0["cluster"] = labels
            df_temp0["name"] = labels
            df_temp0.loc[df_temp0["cluster"] == "-1", "name"] = "unknown"

            df_temp1["cluster"] = "-1"
            df_temp1["name"] = "unknown"

            print(df_temp0)
            print(df_temp1)
            print(pd.concat([df_temp0, df_temp1], ignore_index=True))
            self.df_faces = pd.concat([df_temp0, df_temp1], ignore_index=True)

            # Update the figure
            self.fig = px.scatter(self.df_faces, x="tsne_x", y="tsne_y", color="name", color_discrete_sequence=custom_palette)
            self.fig.update_layout(margin=dict(l=0, r=0, b=0), showlegend=True,
                                   title_text=f"Resulting clusters with eps={self.eps}, min_samples={self.min_samples}, min_size={self.min_size}",
                                   title_x=0.5)
            # Hide the "unknown" cluster initially, but keep it in the legend
            self.fig.for_each_trace(
                lambda t: t.update(visible='legendonly') if t.name == 'unknown' else None
            )

            # Set the images to display as example
            self.images = self.df_faces[self.df_faces["cluster"] == "0"]["img"].copy().to_list()

            # Reset the index tracker
            self.selected_index = 0

            if len(self.images) > 10:
                nr = 10
                right_disabled = False
            else:
                nr = len(self.images)
                right_disabled = True
            children = []
            for i in range(nr):
                img = self.images[i].copy()
                image = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                children.append(
                    html.Div(
                        id={"type": "image-click1", "index": i},  # Pattern-matching id
                        style={"display": "inline-block", "width": "9%", "padding": "0.5%"},
                        n_clicks=0,  # Enables click tracking
                        children=html.Img(src=image, style={
                            "width": "100%",
                            "cursor": "pointer",
                            "maxHeight": "8vw",
                            "objectFit": "contain",
                            # "height": "8vw",
                        })
                    )
                )

            new_txt = f"Showing 1 - {nr} out of {len(self.images)}"
            showing_cls = "Showing cluster '0'."

            style_left = {"width": "5%"}
            style_right = {"width": "5%", "opacity": 0.5 if right_disabled else 1.0}

            # options = self.df_faces["name"].unique(),

            children = [dcc.Dropdown(
                id="dropdown_cls",
                options=self.df_faces["name"].unique(),
                value="0",
                clearable=False,
                className="dropdown",
                searchable=True,
            )]

            style = {
                'padding': '10px 20px',
                'fontSize': '16pt',
                'borderRadius': '12px',
                'border': 'none',
                'backgroundColor': '#2196F3',
                'color': 'white',
                'cursor': 'pointer',
                "width": "10vw"
            }

            return self.fig, children, new_txt, showing_cls, True, right_disabled, style_left, style_right, children, False, style, False, False
        else:
            return dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update


    def update_image_box(self, value):
        if value is not None:
            # Update the images
            self.images = self.df_faces[self.df_faces["name"] == value]["img"].copy().to_list()

            # Reset the index tracker
            self.selected_index = 0

            if len(self.images) > 10:
                nr = 10
                right_disabled = False
            else:
                nr = len(self.images)
                right_disabled = True
            children = []
            for i in range(nr):
                img = self.images[i].copy()
                image = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                children.append(
                    html.Div(
                        id={"type": "image-click1", "index": i},  # Pattern-matching id
                        style={"display": "inline-block", "width": "9%", "padding": "0.5%"},
                        n_clicks=0,  # Enables click tracking
                        children=html.Img(src=image, style={
                            "width": "100%",
                            "cursor": "pointer",
                            "maxHeight": "8vw",
                            "objectFit": "contain",
                            # "height": "8vw",
                        })
                    )
                )

            new_txt = f"Showing 1 - {nr} out of {len(self.images)}"
            showing_cls = f"Showing cluster '{value}'."

            style_left = {"width": "5%"}
            style_right = {"width": "5%", "opacity": 0.5 if right_disabled else 1.0}

            return children, new_txt, showing_cls, True, right_disabled, style_left, style_right
        else:
            return dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update

