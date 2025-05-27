import dash
import pandas as pd
from dash import html, dcc
from functions import *
import plotly.express as px
from PIL import Image


class Results(html.Div):
    """
    Contains all functionalities of the 'results' section for the 'visualization' page.
    """
    def __init__(self, name):
        """
        Initialize the Results object.

        :param name: id of the Results object
        """
        # Initialize the object's parameters
        self.html_id = name  # id of the object
        self.df_main = None  # Dataframe containing all info regarding the main images
        self.df_faces = None  # Dataframe containing all faces in all images
        self.df_rel_faces = None  # Dataframe containing all faces which exists at least ones in the same image as the poi
        self.selected_cluster = None  # Keeps track of the currently selected cluster
        self.selected_index = 0  # Keeps track of the index of the first images shown as example
        self.h_fig = None  # Horizontal bar chart showing the people with whom the poi exists in some images
        self.v_fig = None  # Bar chart showing the extra information
        self.selected_bar = None


        # Initialize ChooseFaceSection as an empty Div
        super().__init__(className="graph_card", id=self.html_id, children=[])


    def initialize_res(self, df_main, df_faces, selected_cluster):
        print("Initializing Results object")

        # Update the object's parameters
        self.df_main = df_main.copy()
        self.df_faces = df_faces.copy()
        self.selected_cluster = selected_cluster
        self.selected_index = 0
        self.selected_bar = None

        new_txt = "Showing 0 - 0 out of 0"
        images_to_display = html.Div(
            html.P(  # Display text saying that no poi has been selected yet
                f"Click on a bar in the figure on the left to display all images where both {self.selected_cluster} and the person corresponding to that bar appear in.",
                style={"fontSize": "25pt", "textAlign": "center", "opacity": 0.3}
            ),
            style={  # Center the text
                "display": "flex",
                "justifyContent": "center",
                "alignItems": "center",
                "height": "100%"
            }
        )
        right_disabled = True

        self.df_rel_faces = self.df_faces[self.df_faces["img_id"].isin(self.df_main["id"].to_list())].copy()
        self.df_rel_faces = self.df_rel_faces[~self.df_rel_faces["name"].isin([self.selected_cluster, "Unknown", "unknown"])].copy()
        self.df_rel_faces["count"] = self.df_rel_faces["name"].copy().map(self.df_rel_faces["name"].value_counts())

        df_face_counts = pd.DataFrame({"name": self.df_rel_faces["name"].unique()})
        df_face_counts["count"] = df_face_counts["name"].map(self.df_rel_faces["name"].value_counts())
        df_face_counts = df_face_counts.sort_values(by="count", ascending=True)

        bar_height = 50  # px per bar
        num_bars = len(df_face_counts)
        self.h_fig = px.bar(df_face_counts, x="count", y="name", orientation="h")

        # Update the layout of the figure
        self.h_fig.update_layout(
            height=num_bars * bar_height + 100,
            margin=dict(l=0, r=10, b=0),  # Use all available space, leave space at the top for title
            showlegend=True,  # Show the legend
            title_text=f"Number of times a person occurs with '{self.selected_cluster}' in the images", # Add a title
            title_x=0.5,  # Centre the title
            font = dict(size=14)
        )

        options = sort_items([i for i in self.df_main.columns if i not in ["index", "id", "url", "img"]])
        # print([i for i in self.df_main.columns if i not in ["index", "id", "url", "img"]])

        if len(options) > 0:  # Additional information should be plotted
            values = [i for _, row in df_main.iterrows() for i in row[options[0]]]
            # print(values)
            df_add_info = pd.DataFrame({options[0]: values, "count": 1})
            # df_add_info["count"] = 1
            # print(df_add_info)
            df_add_info = df_add_info.groupby(options[0]).sum().reset_index()
            df_add_info = df_add_info.sort_values(by="count", ascending=False)
            # print(df_add_info)

            bar_width = 70
            num_bars1 = len(df_add_info)
            self.v_fig = px.bar(df_add_info, x=options[0], y="count", orientation="v")

            # Update the layout of the figure
            self.v_fig.update_layout(
                height=num_bars1 * bar_width + 100,
                margin=dict(l=0, r=10, b=0),  # Use all available space, leave space at the top for title
                showlegend=True,  # Show the legend
                title_text=f"TITLE",
                # Add a title
                title_x=0.5,  # Centre the title
                font=dict(size=14)
            )

            children_add_info = html.Div([
                html.Br(),
                html.Hr(),
                html.Br(),
                html.Div(  # Allows components to be next to each other
                    style={
                        "display": "flex",
                        "justifyContent": "center",
                        "alignItems": "center",
                        "gap": "1vw"
                    },
                    children=[  # Choose additional data to visualize
                        html.Label("Choose additional information to visualize:", style={"fontSize": "16pt"}),
                        dcc.Dropdown(  # All options of additional data
                            id="dropdown_choose_add_data",
                            options=options,
                            value=options[0],
                            clearable=False,
                            className="dropdown",
                            searchable=True,
                            style={"width": "40vw"}
                        ),
                    ]
                ),
                html.Br(),
                html.Div(
                    style={
                        "height": "22vw",  # Fixed visible area
                        "width": "100%",
                        "overflowX": "auto",  # Enables scroll if needed
                        "border": "1px solid #ccc",
                        "borderRadius": "8px"
                    },
                    children=dcc.Graph(
                        id=f"{self.html_id}_v_fig",
                        figure=self.v_fig,
                        style={"width": f"{num_bars1 * bar_width + 100}px",
                               "minWidth": "100%",
                               "height": "100%"}
                        # Full height of all bars
                    )
                ),
            ])
        else:
            children_add_info = html.Div([])

        # Update the layout of the
        children = [
            html.Div(
                style={"textAlign": "center"},
                children=[
                    html.H2("Results", style={"marginTop": "20px"}),  # Header
                    html.Hr(),
                    html.Div(  # Blue box for the cluster exploration section
                        style={
                            "backgroundColor": "#dbeafe",
                            "padding": "10px",
                            "margin": "20px auto",
                            "width": "100%",
                            "borderRadius": "12px",
                            "boxShadow": "0px 2px 5px rgba(0,0,0,0.1)",
                            "textAlign": "center"
                        },
                        children=[
                            html.Div(
                                style={  # Work within a slightly smaller space, allows items to be next to each other
                                    "display": "flex",
                                    "justifyContent": "center",
                                    "alignItems": "flex-start",
                                    "gap": "10px",
                                    "width": "100%",
                                    "margin": "0 auto",
                                    "textAlign": "center",
                                    "marginTop": "20px",
                                    "textFont": "16pt"
                                },
                                children=[
                                    html.Div(
                                        style={
                                            "height": "22vw",  # Fixed visible area
                                            "width": "50%",
                                            "overflowY": "auto",  # Enables scroll if needed
                                            "border": "1px solid #ccc",
                                            "borderRadius": "8px"
                                        },
                                        children=dcc.Graph(
                                            id=f"{self.html_id}_h_fig",
                                            figure=self.h_fig,
                                            style={"height": f"{num_bars * bar_height + 100}px",
                                                   "minHeight": "100%"}
                                            # Full height of all bars
                                        )
                                    ),
                                    # dcc.Graph(  # The horizontal bar chart (left half)
                                    #     id=f"{self.html_id}_h_fig",
                                    #     figure=self.h_fig,
                                    #     style={"flex": "1",
                                    #            "height": "22vw",
                                    #            "width": "50%",
                                    #            }
                                    # ),
                                    html.Div([  # Right half, show example images
                                        html.P(new_txt, style={"textAlign": "right", "fontSize": "16pt"},
                                               id="images_showing_res_cls_txt"),
                                        html.Div(
                                            images_to_display,
                                            style={
                                                "backgroundColor": "white",
                                                "borderRadius": "12px",
                                                "border": "2px black solid",
                                                "textAlign": "center",
                                                "height": "80%"
                                            },
                                            id="example_images_res_box"
                                        ),
                                        html.Div(  # Allows content to be next to each other
                                            style={
                                                "display": "flex",
                                                "justifyContent": "space-between",
                                                "width": "99%",
                                                "margin": "10px auto"
                                            },
                                            children=[
                                                html.Button(  # Button to show the previous 8 images
                                                    "⬅️ Back", id="box_res_images_left", style={
                                                        "width": "10%",
                                                        "opacity": 0.5
                                                    }, disabled=True),
                                                html.Button(  # Button to show the next 8 images
                                                    "Next ➡️", id="box_res_images_right", style={
                                                        "width": "10%",
                                                        "opacity": 0.5 if right_disabled else 1.0
                                                    }, disabled=right_disabled)
                                            ]
                                        ),
                                    ], style={"height": "22vw", "width": "50%"})
                                ]
                            ),
                            children_add_info
                        ]
                    )
                ]
            )
        ]

        return children