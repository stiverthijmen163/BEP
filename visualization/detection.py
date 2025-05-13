import pandas as pd
import dash
from dash import dcc, html, callback, Output, Input, State
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
import cv2
from PIL import Image
import numpy as np
from functions import *


class Detector(html.Div):
    def __init__(self, name, df):
        self.html_id = name.lower().replace(" ", "-")
        self.fig = None
        self.selected_index = 0
        self.df_faces = None
        self.show_nrs = False
        self.selected_image = None
        self.df_detected = None
        self.min_size = (0,0)
        if df is None:
            self.df = None
        else:
            self.df = df.copy()

        # Initialize Detector as empty Div
        super().__init__(className="graph_card", id=self.html_id, children=[])

    def initialize_detector(self, df):
        self.df = df.copy()
        length = len(self.df)
        right_disabled = False
        # images_to_show = self.df["img"].tolist()

        # Detect all faces
        detected = []
        images_faces = []

        df_faces_id = []
        df_faces_img_id = []
        df_faces_face = []
        df_faces_nr = []
        for _, row in self.df.copy().iterrows():
            img_w_faces, detections = detect_faces(row["img"].copy())
            images_faces.append(cv2.cvtColor(img_w_faces, cv2.COLOR_BGR2RGB))
            detected.append(detections)

            count = 0
            for face in detections:
                df_faces_id.append(f"{row['id']}_{count}")
                df_faces_img_id.append(row["id"])
                df_faces_face.append(face)
                df_faces_nr.append(count)


                count += 1

        self.df_faces = pd.DataFrame({
            "id": df_faces_id,
            "img_id": df_faces_img_id,
            "face": df_faces_face,
            "nr": df_faces_nr,
            "use": True
        })

        print(self.df_faces)
        print(detected)

        self.df["img_w_faces"] = images_faces
        self.df["faces"] = detected

        self.df_faces["width"] = self.df_faces["face"].apply(lambda x: list(x)[2])
        self.df_faces["height"] = self.df_faces["face"].apply(lambda x: list(x)[3])
        # self.df_faces["detect"] = True

        self.df_detected = self.df_faces.copy()

        if length > 10:
            nr = 10
        else:
            nr = length
            right_disabled = True

        children = [
            # html.H2("Face Detection"),
            # html.Hr(),
        ]
        images = self.df["img_w_faces"].to_list()
        names = self.df["id"].to_list()
        for i in range(nr):
            # img = self.df["img"][0]
            # name = self.df["id"][0]
            img = images[i]
            name = names[i]

            # print(img)
            # print(name)

            # Convert from BGR (OpenCV format) to RGB (Pillow format)
            # img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(img)
            # image = cv2.imread(img)
            # cv2.imshow()

            # children.append(html.Img(src=image, style={"width": "9%", "padding": "0.5%"}))

            children.append(
                html.Div(
                    id={"type": "image-click", "index": i},  # Pattern-matching id
                    style={"display": "inline-block", "width": "9%", "padding": "0.5%"},
                    n_clicks=0,  # Enables click tracking
                    children=html.Img(src=image, style={"width": "100%", "cursor": "pointer"})
                )
            )

            # cv2.imshow("Face Detection", img)
            # cv2.waitKey(0)




        result = html.Div(
            children = [
                html.H2("Face Detection", style={"textAlign": "center"}),
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
                        html.Div([
                            html.Div("", style={"flex": 1}),  # Left spacer

                            html.Div(html.P("Select an image to edit or Continue", style={
                                "fontSize": "16pt",
                                "marginBottom": "5px",
                                "textAlign": "center",
                                "margin": "0"
                            }), style={"flex": 1, "display": "flex", "justifyContent": "center"}),  # Center content

                            html.Div(html.P(f"Showing 1 - {nr} out of {len(self.df)}",id="images_showing_txt", style={
                                "fontSize": "16pt",
                                "marginBottom": "5px",
                                "textAlign": "center",
                                "margin": "0"
                            }), style={"flex": 0.995, "display": "flex", "justifyContent": "flex-end", "paddingRight": "0.5%"})  # Right content
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
                                'textAlign': 'center'
                            },
                            # children=[
                            #     html.H2("Face Detection"),
                            #     html.Hr(),
                            #     # children,
                            # ].append(children)
                            children=children,
                            id="box_images0"
                        ),
                        dcc.Store(id="box_images0_index", data=0),
                        html.Div(
                            style={
                                'display': 'flex',
                                'justifyContent': 'space-between',
                                'width': '99%',
                                'margin': '10px auto'
                            },
                            children=[
                                html.Button("⬅️ Back", id="box_images0_left", style={
                                    'width': '5%',
                                    "opacity": 0.5
                                }, disabled=True),
                                html.Button("Next ➡️", id="box_images0_right", style={
                                    'width': '5%',
                                    "opacity": 0.5 if right_disabled else 1.0
                                }, disabled=right_disabled)
                            ]
                        ),
                        # html.Br(),
                        html.P("", id="selected_image_txt", style={
                            "fontSize": "16pt",
                            "marginBottom": "5px",
                            "textAlign": "center",
                            "margin": "0"
                        }),
                        html.Div(
                            style={
                                'display': 'flex',  # Enable horizontal layout
                                'justifyContent': 'space-between',  # Optional: adds spacing between elements
                                'alignItems': 'flex-start',  # Optional: align tops
                                'gap': '10px',  # Optional: space between items
                                'width': '99%',
                                'margin': '10px auto'
                            },
                            children=[
                                dcc.Graph(
                                    id=f"{self.html_id}_fig",
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
                                        html.P("Select which detections you want to use, or draw your own detection in the figure on the left.",
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
                                                    "min_size:",
                                                    html.I(className="fa-solid fa-circle-question",
                                                           id="info_icon_min_size0",
                                                           style={"cursor": "pointer", "color": "#0d6efd",
                                                                  "marginLeft": "5px",
                                                                  "position": "relative",
                                                                  "top": "-3px"
                                                                  }),
                                                    " width:",
                                                    dcc.Input(
                                                        id="min_width_det_input",
                                                        type="number",
                                                        min=0,
                                                        max=max(self.df_faces["width"]) if len(self.df_faces) > 0 else 1000,
                                                        step=1,
                                                        value=0,
                                                        style={"marginLeft": "10px", "width": "8%"}
                                                    ),
                                                    " height:",
                                                    dcc.Input(
                                                        id="min_height_det_input",
                                                        type="number",
                                                        min=0,
                                                        max=max(self.df_faces["height"]) if len(self.df_faces) > 0 else 1000,
                                                        step=1,
                                                        value=0,
                                                        style={"marginLeft": "10px", "width": "8%"}
                                                    ),
                                                    html.Button(
                                                        "Update detections",
                                                        disabled=False,
                                                        style={
                                                            'padding': '10px 20px',
                                                            'fontSize': '16pt',
                                                            'borderRadius': '12px',
                                                            'border': 'none',
                                                            'backgroundColor': '#2196F3',
                                                            'color': 'white',
                                                            'cursor': 'pointer',
                                                            "width": "12vw",
                                                            "marginLeft": "1vw",
                                                        },
                                                        id="button_update_detections"
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
                                                "Select the minimum size of a face to be detected, faces that are smaller will be discarded.",
                                                target="info_icon_min_size0",
                                                placement="top"
                                            )
                                        ]),
                                        html.Div(
                                            id="selection_of_faces",
                                            style={
                                                "textAlign": "left",
                                                "margin": "0.5%",
                                            },
                                            children=[]
                                        )
                                        # Add more children as needed
                                    ]
                                )
                            ]
                        ),
                        html.Br(),
                        html.Button(
                            "Continue to Face Clustering",
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
                            id="button2"
                        ),
                        html.P(
                            "",
                            style={
                                "fontSize": "16pt",
                                "marginBottom": "20px",
                                "textAlign": "center",
                                "margin": "5px",
                                "fontWeight": "bold"
                            },
                            id="successful_detection"
                        ),
                        # html.Br()
                        # html.Button(id="box_images0_left", style={}),
                        # html.Button(id="box_images0_right", style={})
                    ]
                )
            ]
        )

        existing_children = []
        # print(existing_children)
        updated_children = list(existing_children) + [result]

        self.children = updated_children

        return self.children


    # @callback(
    #     Output("box_images0_index", "data"),
    #     Output("box_images0", "children"),
    #     Output("box_images0_left", "disabled"),
    #     Output('box_images0_right', 'disabled'),
    #     Output("box_images0_left", "style"),
    #     Output("box_images0_right", "style"),
    #     Input('box_images0_right', 'n_clicks'),
    #     State("box_images0_index", "data"),
    #     prevent_initial_call=True
    # )
    def next_button(self, n_clicks, current_index):
        if n_clicks is not None and n_clicks > 0:
            print(current_index)
            new_index = current_index + 10
            self.selected_index = new_index
            children = []

            images = self.df["img_w_faces"].to_list()
            names = self.df["id"].to_list()

            if len(images[new_index:]) > 10:
                nr = 10
                disabled0 = False
            else:
                nr = len(images[new_index:])
                disabled0 = True

            for i in range(nr):
                img = images[i + new_index]
                name = names[i + new_index]

                # img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(img)

                # children.append(html.Img(src=image, style={"width": "9%", "padding": "0.5%"}))
                children.append(
                    html.Div(
                        id={"type": "image-click", "index": i},  # Pattern-matching id
                        style={"display": "inline-block", "width": "9%", "padding": "0.5%"},
                        n_clicks=0,  # Enables click tracking
                        children=html.Img(src=image, style={"width": "100%", "cursor": "pointer"})
                    )
                )

            style_left = {"width": "5%"}
            style_right = {"width": "5%", "opacity": 0.5 if disabled0 else 1.0}

            new_txt = f"Showing {new_index + 1} - {new_index + nr} out of {len(self.df)}"

            return new_index, children, False, disabled0, style_left, style_right, new_txt
        else:
            return dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update


    def previous_button(self, n_clicks, current_index):
        if n_clicks is not None and n_clicks > 0:
            print(current_index)
            new_index = current_index - 10
            self.selected_index = new_index
            children = []

            images = self.df["img_w_faces"].to_list()
            names = self.df["id"].to_list()

            if new_index == 0:
                disabled0 = True
            else:
                disabled0 = False

            for i in range(10):
                img = images[i + new_index]
                name = names[i + new_index]

                # img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(img)

                # children.append(html.Img(src=image, style={"width": "9%", "padding": "0.5%"}))
                children.append(
                    html.Div(
                        id={"type": "image-click", "index": i},  # Pattern-matching id
                        style={"display": "inline-block", "width": "9%", "padding": "0.5%"},
                        n_clicks=0,  # Enables click tracking
                        children=html.Img(src=image, style={"cursor": "pointer", "width": "100%"})
                    )
                )

            style_left = {"width": "5%", "opacity": 0.5 if disabled0 else 1.0}
            style_right = {"width": "5%"}

            new_txt = f"Showing {new_index + 1} - {new_index + 10} out of {len(self.df)}"

            return new_index, children, disabled0, False, style_left, style_right, new_txt
        else:
            return dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update


    def init_picture_fig(self, image_nr):
        # Collect the image
        index = self.selected_index + image_nr

        id = self.df["id"].to_list()[index]
        self.selected_image = id
        print(id)

        self.df_detected = self.df_faces[
            (self.df_faces["width"] >= self.min_size[0]) & (self.df_faces["height"] >= self.min_size[1])].copy()
        temp_df = self.df_detected[self.df_detected["img_id"] == id].copy()
        # print(temp_df)

        img = self.df["img"].to_list()[index]
        # img = img["img"]

        img = plot_faces_on_img_opacity(img.copy(), temp_df.copy(), self.show_nrs)

        # print(img)

        fig = px.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        fig.update_layout(
            margin=dict(l=0, r=0, t=0, b=0),
            xaxis=dict(showticklabels=False),
            yaxis=dict(showticklabels=False),
            dragmode="select"
        )

        children = []
        for _, row in temp_df.iterrows():
            children.append(dcc.Checklist(
                options=[{"label": f" Face {row['nr']}", "value": "keep"}],
                value=["keep"] if row["use"] else [],
                id={"type": "keep-face", "index": row["nr"]},
                style={
                    "fontSize": "16pt",
                    "marginBottom": "5px",
                    # "textAlign": "center",
                    "margin": "0"
                }
            ))

        scrollable_column = html.Div(
            id="scrollable_column",
            children=children,
            style={
                "maxHeight": "16vw",  # adjust as needed
                "height": "16vw",
                "overflowY": "scroll",
                "padding": "10px",
                "border": "1px solid lightgray",
                "borderRadius": "8px",
                "marginTop": "10px",
                "width": "15%"
            }
        )

        layout = html.Div(
            style={"display": "flex", "gap": "20px"},  # horizontal layout
            children=[
                # Left column: scrollable list
                scrollable_column,
                # Right column: checklist + image stacked vertically
                html.Div(
                    style={
                        "display": "flex",
                        "flexDirection": "column",
                        "gap": "10px",
                        "textAlign": "center",
                        "width": "80%",
                    },
                    children=[
                        # Checklist (you can add options dynamically)
                        dcc.Checklist(
                            id="show_nrs",
                            options=[
                                {"label": " Show face numbers in figure on the left", "value": "boxes"}
                                # {"label": "Highlight faces", "value": "highlight"}
                            ],
                            value=["boxes"] if self.show_nrs else [],  # start unselected
                            style={"fontSize": "16pt", "marginTop": "10px", "textAlign": "left"}
                        ),

                        # Image (replace `src` with base64 string or Dash figure)
                        html.P("",
                            style={
                                "fontSize": "16pt",
                                "marginBottom": "5px",
                                "textAlign": "center",
                                "margin": "0"
                            },
                            id="text_for_image"
                        ),
                        html.Img(
                            id="image-preview", src="",
                            style={
                                "maxWidth": "30vw",
                                "maxHeight": "12vw",
                                "width": "auto",              # Fill available horizontal space
                                "height": "100%",   # Prevents vertical stretching
                                "display": "block",
                                "marginLeft": "auto",
                                "marginRight": "auto"
                            }
                        )
                    ]
                )
            ]
        )

        return fig, layout, False


    def update_picture_fig(self, show_nrs, checklist, activated_by, selection, children, n_clicks, w, h):
        self.show_nrs = False if show_nrs == [] else True

        # Update the faces to show
        if activated_by == f"{self.html_id}_fig":
            if selection:
                points = selection.get("range", {})
                x0, x1 = points.get("x", [None, None])
                y0, y1 = points.get("y", [None, None])
                print(f"Selected area: x=({x0}, {x1}), y=({y0}, {y1})")

                if x0 and x1 and y0 and y1:

                    box = [int(x0), int(y0), int(x1) - int(x0), int(y1) - int(y0)]
                    print(box)
                    nr = max(self.df_faces[self.df_faces["img_id"] == self.selected_image]["nr"].to_list()) + 1
                    # self.df_faces = pd.DataFrame({
                    #     "id": df_faces_id,
                    #     "img_id": df_faces_img_id,
                    #     "face": df_faces_face,
                    #     "nr": df_faces_nr,
                    #     "use": True
                    # })
                    print(self.df_faces.columns)
                    self.df_faces.loc[len(self.df_faces)] = [
                        f"{self.selected_image}_{nr}", self.selected_image, box, nr, True, box[2], box[3]
                    ]

                    img = self.df[self.df["id"] == self.selected_image]["img"].copy().to_list()[0]
                    x, y, w, h = box
                    face = img[y:y + h, x:x + w]


                    # result_txt = f"Created and added 'Face {nr}' to the set of faces:"

                    result = Image.fromarray(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))

                    if w >= self.min_size[0] and h >= self.min_size[1]:
                        children.append(dcc.Checklist(
                            options=[{"label": f" Face {nr}", "value": "keep"}],
                            value=["keep"],
                            id={"type": "keep-face", "index": nr},
                            style={
                                "fontSize": "16pt",
                                "marginBottom": "5px",
                                # "textAlign": "center",
                                "margin": "0"
                            }
                        ))
                        result_txt = f"Created and added 'Face {nr}' to the set of faces:"
                    else:
                        result_txt = f"Created 'Face {nr}', but it is smaller than min_size"


                else:
                    return dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update

                # result = dash.no_update
                # result_txt = dash.no_update
            else:
                return dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update
        elif activated_by == "button_update_detections" and n_clicks is not None and n_clicks > 0:
            # Update the min sizes
            self.update_min_size(w, h)
            # if w is not None:
            #     self.min_size = (w, self.min_size[1])
            # if h is not None:
            #     self.min_size = (self.min_size[0], h)
            # self.df_detected = self.df_faces[(self.df_faces["width"] >= self.min_size[0]) & (self.df_faces["height"] >= self.min_size[1])].copy()
            # print(self.df_detected)
            result = dash.no_update
            result_txt = dash.no_update
            children = []
            for _, row in self.df_detected[self.df_detected["img_id"] == self.selected_image].copy().iterrows():
                # print(row)
                children.append(dcc.Checklist(
                    options=[{"label": f" Face {row['nr']}", "value": "keep"}],
                    value=["keep"] if row["use"] else [],
                    id={"type": "keep-face", "index": row["nr"]},
                    style={
                        "fontSize": "16pt",
                        "marginBottom": "5px",
                        # "textAlign": "center",
                        "margin": "0"
                    }
                ))

        elif activated_by != "show_nrs":
            id = f"{self.selected_image}_{activated_by['index']}"
            changed = checklist[activated_by['index']]
            replace_val = False if changed == [] else True
            print(replace_val)
            print(id)

            self.df_faces.loc[self.df_faces["id"] == id, "use"] = replace_val

            face_box = self.df_faces[self.df_faces["id"] == id]["face"].copy().to_list()[0]
            x, y, w, h = face_box
            img = self.df[self.df["id"] == self.selected_image]["img"].copy().to_list()[0]
            face = img[y:y+h, x:x+w]

            result = Image.fromarray(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))

            if replace_val:
                result_txt = f"Added 'Face {activated_by['index']}' to the set of faces:"
            else:
                result_txt = f"Removed 'Face {activated_by['index']}' from the set of faces:"
        else:
            result = dash.no_update
            result_txt = dash.no_update
        self.df_detected = self.df_faces[
            (self.df_faces["width"] >= self.min_size[0]) & (self.df_faces["height"] >= self.min_size[1])].copy()
        img = self.df[self.df["id"] == self.selected_image]["img"].to_list()[0]
        temp_df = self.df_detected[self.df_detected["img_id"] == self.selected_image].copy()
        print(temp_df)

        img = plot_faces_on_img_opacity(img.copy(), temp_df.copy(), self.show_nrs)

        fig = px.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        fig.update_layout(
            margin=dict(l=0, r=0, t=0, b=0),
            xaxis=dict(showticklabels=False),
            yaxis=dict(showticklabels=False),
            dragmode="select"
        )

        return fig, True, result, result_txt, children, dash.no_update


    def update_min_size(self, w, h):
        if w is not None:
            self.min_size = (w, self.min_size[1])
        if h is not None:
            self.min_size = (self.min_size[0], h)

        self.df_detected = self.df_faces[
            (self.df_faces["width"] >= self.min_size[0]) & (self.df_faces["height"] >= self.min_size[1])].copy()



