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
        self.df = df.copy()
        self.df_faces = df_faces.copy()
        self.fig = None
        self.eps = 5.4
        self.min_samples = 1
        self.min_size = (0, 0)
        self.selected_index = 0
        self.selected_cluster = None

        # Saving some data for hard-coded loading
        # self.df_faces["face"] = self.df_faces["face"].apply(json.dumps)
        # self.df_faces["img"] = self.df_faces["img"].apply(lambda x: json.dumps(x.tolist()))
        # # Save the embeddings
        # conn = sqlite3.connect("temp.db")
        # self.df_faces.to_sql("faces", conn, if_exists="replace")

        # Load the embeddings (hard-coded)
        # conn = sqlite3.connect("temp.db")
        # query = """SELECT * FROM faces"""
        # self.df_faces = pd.read_sql_query(query, conn)
        # self.df_faces["embedding_tsne"] = self.df_faces["embedding_tsne"].apply(lambda x: np.fromstring(x, sep=","))
        # self.df_faces["embedding"] = self.df_faces["embedding"].apply(lambda x: np.fromstring(x, sep=","))
        # self.df_faces["face"] = self.df_faces["face"].apply(json.loads)
        # self.df_faces["img"] = self.df_faces["img"].apply(json.loads)
        # self.df_faces["img"] = self.df_faces["img"].apply(lambda x: np.array(x, dtype=np.uint8))
        # self.df_faces["width"] = self.df_faces["face"].apply(lambda x: list(x)[2])
        # self.df_faces["height"] = self.df_faces["face"].apply(lambda x: list(x)[3])


        # STILL NEEDED EVEN WHEN NOT LOADING FROM DB
        self.df_faces["embedding_tsne"] = self.df_faces["embedding_tsne"].apply(lambda x: np.fromstring(x, sep=","))
        self.df_faces["embedding"] = self.df_faces["embedding"].apply(lambda x: np.fromstring(x, sep=","))

        # Collect clusters
        labels = DBSCAN(eps=self.eps, min_samples=self.min_samples, n_jobs=-1, metric="euclidean").fit_predict(
            self.df_faces["embedding_tsne"].tolist())

        # Set a counter to know what value to use when adding a new cluster
        self.counter = max(labels)

        # Update all labels to being strings
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
                                    id="div_cls_right",
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
                                                options=sort_items(self.df_faces["name"].unique()),
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
                        html.Div([
                                html.P([
                                    "Name of the database:",
                                    dcc.Input(
                                        id="name_of_db",
                                        type="text",
                                        placeholder="Only letters and '_' allowed",
                                        value="",
                                        style={"width": "30vw", "marginLeft": "1vw"}
                                    )],
                                    style={
                                        "fontSize": "16pt",
                                        # "gap": "1vw",
                                        # "marginBottom": "5px",
                                        "textAlign": "center",
                                        # "margin": "5px",
                                        # "fontWeight": "bold"
                                    }
                                )
                                # dcc.Input(
                                #     id="name_of_db",
                                #     type="text",
                                #     placeholder="Only letters, spaces and '_' allowed",
                                #     value="",
                                #     style={"width": "50%"}
                                # )
                            ],
                            style={
                                "display": "flex",
                                "alignItems": "center",
                                "justifyContent": "center",
                                "margin": "0 auto",
                                # "padding": "1vw",
                                "gap": "1vw"  # Optional: adds spacing between items
                            }
                        ),
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
                        html.P(
                            "",
                            style = {
                                "fontSize": "16pt",
                                "marginBottom": "5px",
                                "textAlign": "center",
                                "margin": "5px",
                                "fontWeight": "bold"
                            },
                            id="successful_database"
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
            self.counter = max(labels)
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
            # self.fig = px.scatter(self.df_faces, x="tsne_x", y="tsne_y", color="name", color_discrete_sequence=custom_palette)
            # self.fig.update_layout(margin=dict(l=0, r=0, b=0), showlegend=True,
            #                        title_text=f"Resulting clusters with eps={self.eps}, min_samples={self.min_samples}, min_size={self.min_size}",
            #                        title_x=0.5)
            # # Hide the "unknown" cluster initially, but keep it in the legend
            # self.fig.for_each_trace(
            #     lambda t: t.update(visible='legendonly') if t.name == 'unknown' else None
            # )
            self.update_fig()

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
            children_img = []
            for i in range(nr):
                img = self.images[i].copy()
                image = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                children_img.append(
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
                options=sort_items(self.df_faces["name"].unique()),
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

            return self.fig, children_img, new_txt, showing_cls, True, right_disabled, style_left, style_right, children, False, style, False, False
        else:
            return dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update


    def update_image_box(self, value, data):
        if value is not None:
            print(f"Selected cluster '{value}'")
            # Update the images
            self.images = self.df_faces[self.df_faces["name"] == value]["img"].copy().to_list()

            # Reset the index tracker
            self.selected_index = 0
            self.selected_cluster = value

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

            # children_change_name = dash.no_update

            # In case we are in the edit mode for clustering
            # if n_clicks is not None and n_clicks > 0:
            #     # Update value for changing name to be empty
            #     children_change_name = dcc.Input(
            #                     id="name_of_cluster",
            #                     type="text",
            #                     placeholder="Only letters, spaces and '_' allowed",
            #                     value="",
            #                     # style={"marginLeft": "10px", "width": "50%"}

            #                 )
            if data is None:
                d = 0
            else:
                d = data + 1

            return children, new_txt, showing_cls, True, right_disabled, style_left, style_right, d
        else:
            return dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update


    def update_components_on_cls_change(self):
        # children_change_name = dcc.Input(
        #                     id="name_of_cluster",
        #                     type="text",
        #                     placeholder="Only letters, spaces and '_' allowed",
        #                     value="",
        #                     # style={"marginLeft": "10px", "width": "50%"}
        #
        #                 )
        value = ""
        _, children_merge = self.update_merge_dropdown()
        # items = np.delete(self.df_faces["name"].copy().unique(),
        #                   np.where(self.df_faces["name"].copy().unique() == self.selected_cluster))
        # val = items[0] if len(items) > 0 else None
        # children_merge = dcc.Dropdown(
        #     id="dropdown_cls_merge",
        #     options=items,
        #     value=val,
        #     clearable=False,
        #     className="dropdown",
        #     searchable=True,
        # )

        return value, children_merge, []


    def continue_to_edit(self, n_clicks):
        if n_clicks is not None and n_clicks > 0:
            items = sort_items(np.delete(self.df_faces["name"].copy().unique(), np.where(self.df_faces["name"].copy().unique() == "0")))
            val = items[0]
            self.selected_cluster = "0"
            # print(np.delete(self.df_faces["name"].copy().unique(), np.where(self.df_faces["name"].copy().unique() == "0")))
            children = [
                html.P(
                    "Edit any cluster, you can move faces to other clusters, merge clusters and rename them. Empty clusters will be removed.",
                    style={
                        "fontSize": "16pt",
                        "marginBottom": "5px",
                        "textAlign": "center",
                        "margin": "5px",
                        "fontWeight": "bold"
                    }
                ),
                # html.Br(),
                html.Label(
                    "Select cluster to inspect/edit: ",
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
                        options=sort_items(self.df_faces["name"].unique()),
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
                ),
                html.Div([
                    html.P(
                        [
                            "Cluster Name: ",
                            dcc.Input(
                                id="name_of_cluster",
                                type="text",
                                placeholder="Only letters, spaces and '_' allowed",
                                value="",
                                style={"marginLeft": "10px", "width": "50%"}
                            ),
                            html.Button(
                                "Change Name",
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
                                    "marginLeft": "1vw",
                                    "marginTop": "5px"
                                },
                                id="button_change_name"
                            )
                        ],
                        style={
                            "fontSize": "16pt",
                            "marginBottom": "5px",
                            "textAlign": "center",
                            "margin": "0"
                        },
                    ),
                ]),
                html.Div([
                    html.P(style={
                            'display': 'flex',
                            'justifyContent': 'center',
                            'gap': '1vw',
                            "marginTop": "5px",
                            'alignItems': 'center',
                            # 'padding': '20px',
                            # 'backgroundColor': '#e0e0e0',
                        },
                        children=[
                            html.Div([
                                html.P(
                                    [
                                        "Merge with cluster:",
                                        html.I(className="fa-solid fa-circle-question", id="info_icon_merge_cls",
                                               style={"cursor": "pointer", "color": "#0d6efd",
                                                      "marginLeft": "5px",
                                                      "position": "relative",
                                                      "top": "-3px"
                                                      }),
                                        " ",
                                    ],
                                    style={
                                        "fontSize": "16pt",
                                        "marginBottom": "5px",
                                        "textAlign": "center",
                                        "margin": "0"
                                    },
                                ),
                                dbc.Tooltip(
                                    "The name of the currently selected cluster is kept.",
                                    target="info_icon_merge_cls",
                                    placement="top"
                                )
                            ]),
                            html.Div(id="dropdown_merge_change", children=dcc.Dropdown(
                                id="dropdown_cls_merge",
                                options=items,
                                value=val,
                                clearable=False,
                                className="dropdown",
                                searchable=True,
                            ), style={"width": "50%"}),
                            html.Button(
                                "Merge clusters",
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
                                },
                                id="button_merge_clusters"
                            )
                        ],
                    )],
                    style={
                        "fontSize": "16pt",
                        "marginBottom": "5px",
                        "textAlign": "center",
                        "margin": "0"
                    },
                    # ),
                ),
                html.Div(
                    children=[],
                    id="selected_image_div_cls",
                    style={
                        "display": "flex",
                        "width": "99%",
                        "height": "35%",  # optional, depending on parent container
                        "margin": "0 auto",
                        "gap": "1vw"  # spacing between the image and div
                    }
                )
                # html.Div([
                #     html.Img(
                #         src="data:image/png;base64,...",  # or external URL / `app.get_asset_url(...)`
                #         style={
                #             "height": "100%",  # or a fixed size like "20vw"
                #             "width": "auto",
                #             "maxHeight": "100%",  # Ensure it scales nicely
                #             "objectFit": "contain"
                #         }
                #     ),
                #     html.Div(
                #         "Your content here",
                #         style={
                #             "flex": "1",
                #             "padding": "1vw",
                #             "border": "1px solid #ccc",
                #             "backgroundColor": "#f8f8f8"
                #         }
                #     )
                # ], style={
                #     "display": "flex",
                #     "height": "100%",  # optional, depending on parent container
                #     "gap": "1vw"  # spacing between the image and div
                # })
            ]

            style = {
                'padding': '10px 20px',
                'fontSize': '16pt',
                'borderRadius': '12px',
                'border': 'none',
                'backgroundColor': '#2196F3',
                'color': 'white',
                'cursor': 'pointer',
                "width": "20vw",
                "marginBottom": "20px"
            }

            return children, True, False, style
        else:
            return dash.no_update, dash.no_update, dash.no_update, dash.no_update


    def change_name(self, n_clicks, name):
        if n_clicks is not None and n_clicks > 0:
            if name == "":
                name = "None"

            print(f"Update name of cluster '{self.selected_cluster}' to '{name}'")
            # print(f"selected cluster: {self.selected_cluster}")
            self.df_faces.loc[self.df_faces["name"] == self.selected_cluster, "name"] = f"{name}"
            self.update_fig()

            children = [dcc.Dropdown(
                id="dropdown_cls",
                options=sort_items(self.df_faces["name"].unique()),
                value=f"{name}",
                clearable=False,
                className="dropdown",
                searchable=True,
            )]

            _, children_merge = self.update_merge_dropdown()
            # items = np.delete(self.df_faces["name"].copy().unique(),
            #                   np.where(self.df_faces["name"].copy().unique() == f"{name}"))
            # val = items[0]
            # children_merge = dcc.Dropdown(
            #     id="dropdown_cls_merge",
            #     options=items,
            #     value=val,
            #     clearable=False,
            #     className="dropdown",
            #     searchable=True,
            # )

            # Update selected cluster to new name
            self.selected_cluster = f"{name}"
            return self.fig, children, children_merge, []
        else:
            print("no_update")
            # self.update_fig()
            return dash.no_update, dash.no_update, dash.no_update, dash.no_update


    def update_fig(self):
        # Update the figure
        self.fig = px.scatter(self.df_faces, x="tsne_x", y="tsne_y", color="name",
                              color_discrete_sequence=custom_palette)
        self.fig.update_layout(margin=dict(l=0, r=0, b=0), showlegend=True,
                               title_text=f"Resulting clusters with eps={self.eps}, min_samples={self.min_samples}, min_size={self.min_size}",
                               title_x=0.5)

        # Hide the "unknown" cluster initially, but keep it in the legend
        self.fig.for_each_trace(
            lambda t: t.update(visible='legendonly') if t.name == 'unknown' else None
        )


    def merge_clusters(self, n_clicks, cluster):
        if n_clicks is not None and n_clicks > 0:
            print(f"Merge cluster '{self.selected_cluster}' with cluster '{cluster}'")

            # Collect the name
            name = self.df_faces[self.df_faces["name"] == self.selected_cluster]["name"].values[0]

            # Merge the clusters by sharing the name
            self.df_faces.loc[self.df_faces["name"] == cluster, "name"] = f"{name}"
            self.update_fig()

            # Update the dropdown manu for selecting a cluster to inspect
            children = [dcc.Dropdown(
                id="dropdown_cls",
                options=sort_items(self.df_faces["name"].unique()),
                value=f"{name}",
                clearable=False,
                className="dropdown",
                searchable=True,
            )]

            items, children_merge = self.update_merge_dropdown()

            # Set the standard style of the button
            style = {
                'padding': '10px 0px',
                'fontSize': '16pt',
                'borderRadius': '12px',
                'border': 'none',
                'backgroundColor': '#2196F3',
                'color': 'white',
                'cursor': 'pointer',
                "width": "10vw",
                "opacity": 1.0 if len(items) > 0 else 0.5
            }
            disabled = False if len(items) > 0 else True

            # If no items are left, the button disables and the dropdown empties
            # if len(items) > 0:  # Items left
            #     val = items[0]
            #     disabled = False
            # else:  # No items left
            #     val = None
            #     disabled = True
            #     style.update({"opacity": 0.5})

            # val = items[0] if len(items) > 0 else None

            # children_merge = dcc.Dropdown(
            #                     id="dropdown_cls_merge",
            #                     options=items,
            #                     value=val,
            #                     clearable=False,
            #                     className="dropdown",
            #                     searchable=True,
            #                 )

            return self.fig, children, children_merge, disabled, style, []
        else:
            print("no_update")
            self.update_fig()
            return dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update


    def update_merge_dropdown(self):
        items = sort_items(np.delete(self.df_faces["name"].copy().unique(),
                          np.where(self.df_faces["name"].copy().unique() == self.selected_cluster)))

        val = items[0] if len(items) > 0 else None
        c = dcc.Dropdown(
            id="dropdown_cls_merge",
            options=items,
            value=val,
            clearable=False,
            className="dropdown",
            searchable=True,
        )

        return items, c


    def add_new_cluster(self, selected_data):
        if selected_data is not None and len(selected_data["points"]) > 0:
            selected_points = selected_data["points"]
            selected_coords = [(point["x"], point["y"]) for point in selected_points]
            print(selected_points)
            print(selected_coords)
            # print(self.df_faces.iloc[indices]["tsne_x"])
            # df_temp = self.df_faces.copy()
            #
            # mask = df_temp[
            #     df_temp.apply(lambda row: (row["tsne_x"], row["tsne_y"]) in selected_coords, axis=1)
            # ]

            # Update counter
            self.counter += 1

            self.df_faces['name'] = self.df_faces.apply(
                lambda row: f"{self.counter}" if (row['tsne_x'], row['tsne_y']) in selected_coords else row['name'],
                axis=1
            )

            # Update counter
            # self.counter += 1
            # self.df_faces.loc[mask, "name"] = f"{self.counter}"

            self.update_fig()
            # print(selected_rows.)
            # print(selected_rows["tsne_x"])
            # print(selected_rows["tsne_y"])
            children = [dcc.Dropdown(
                id="dropdown_cls",
                options=sort_items(self.df_faces["name"].unique()),
                value=f"{self.counter}",
                clearable=False,
                className="dropdown",
                searchable=True,
            )]

            return self.fig, children, []

        return dash.no_update, dash.no_update, dash.no_update


    def select_image(self, data):
        index = self.selected_index + data
        img = Image.fromarray(cv2.cvtColor(self.images[index].copy(), cv2.COLOR_BGR2RGB))

        items = sort_items(np.delete(self.df_faces["name"].copy().unique(),
                                     np.where(self.df_faces["name"].copy().unique() == self.selected_cluster)))

        val = items[0] if len(items) > 0 else None
        disabled = False if len(items) > 0 else True

        children = [
            html.Img(
                src=img,  # or external URL / `app.get_asset_url(...)`
                style={
                    "maxWidth": "10vw",
                    "maxHeight": "100%",
                    "width": "auto",  # Fill available horizontal space
                    "height": "100%",  # Prevents vertical stretching
                    "display": "block",
                    "marginLeft": "auto",
                    "marginRight": "auto",
                    "objectFit": "contain"
                }
            ),
            html.Div(
                [
                    html.Label("Move to cluster:",
                               style={"fontSize": "16pt", "alignSelf": "center"}),
                    dcc.Dropdown(
                        id="dropdown_move_to_cls",
                        options=items,
                        value=val,
                        clearable=False,
                        className="dropdown",
                        searchable=True,
                        style={"width": "20vw"}  #, "marginRight": "10px"}
                    ),
                    html.Button(
                        "Move",
                        disabled=disabled,
                        id="button_move",
                        style={
                            'padding': '10px 0px',
                            'fontSize': '16pt',
                            'borderRadius': '12px',
                            'border': 'none',
                            'backgroundColor': '#2196F3',
                            'color': 'white',
                            'cursor': 'pointer',
                            "width": "5vw",
                            "marginTop": "0px"
                        }
                    )
                ],
                style={
                    "display": "flex",
                    "alignItems": "center",
                    "justifyContent": "center",
                    "margin": "0 auto",
                    # "padding": "1vw",
                    "gap": "1vw"  # Optional: adds spacing between items
                }
            )
        ]

        return children


    def move_face(self, n_clicks, value, index):
        if n_clicks is not None and n_clicks > 0:
            i = self.selected_index + index
            # img = self.images[i]

            img_id = self.df_faces[self.df_faces["name"] == self.selected_cluster]["id"].copy().to_list()[i]
            self.df_faces.loc[self.df_faces["id"] == img_id, "name"] = f"{value}"

            self.images = self.df_faces[self.df_faces["name"] == self.selected_cluster]["img"].copy().to_list()

            self.update_fig()

            print(len(self.images), self.selected_index)

            if len(self.images) > 0:
                if len(self.images) == self.selected_index:
                    self.selected_index -= 10
            # if len(self.images) == 0:
            #     self.selected_index = 0

                if len(self.images[self.selected_index:]) > 10:
                    nr = 10
                    right_disabled = False
                else:
                    nr = len(self.images[self.selected_index:])
                    right_disabled = True

                children = []
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
                            })
                        )
                    )

                new_txt = f"Showing {self.selected_index + 1} - {nr + self.selected_index} out of {len(self.images)}"
                # showing_cls = f"Showing cluster '{value}'."

                style_left = {"width": "5%", "opacity": 0.5 if self.selected_index == 0 else 1.0}
                style_right = {"width": "5%", "opacity": 0.5 if right_disabled else 1.0}

                left_disabled = True if self.selected_index == 0 else False
            else:
                children = []
                new_txt = f"Showing 0 - 0 out of {len(self.images)}"
                left_disabled = True
                right_disabled = True
                style_left = {"width": "5%", "opacity": 0.5}
                style_right = {"width": "5%", "opacity": 0.5}


            # img_id = self.df_faces[self.df_faces["face"] == img]
            return self.fig, [], children, new_txt, left_disabled, right_disabled, style_left, style_right
        else:
            return dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update
