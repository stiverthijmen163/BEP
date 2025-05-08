import dash
from dash import html, register_page, callback, Output, Input, dcc, State, ctx, ALL
import dash_bootstrap_components as dbc
import tkinter as tk
from tkinter import filedialog
import threading
import datetime
import base64
import io
import numpy as np
import cv2
import pandas as pd
import time
import face_recognition
from sklearn.manifold import TSNE
from visualization.functions import *

from sympy.codegen.fnodes import allocatable

from visualization.detection import Detector
from visualization.clustering import Clusteror

df0 = None
df1 = None
cls0 = None
register_page(__name__, path="/")

# # Folder picker function
# def select_folder():
#     root = tk.Tk()
#     root.withdraw()
#     folder_path = filedialog.askopenfilename()
#     return folder_path

det0 = Detector("det0", df0)
# cls0 = Clusteror("cls0", df0)


layout = html.Div(
    children=[
        html.Div(
            style={'textAlign': 'center'},
            children=[
                html.H2("Data Selection"),
                html.Hr(),
                html.Div(
                    style={
                        'backgroundColor': '#dbeafe',
                        'padding': '20px',
                        'margin': '20px auto',
                        'width': '60%',
                        'borderRadius': '12px',
                        'boxShadow': '0px 2px 5px rgba(0,0,0,0.1)',
                        'textAlign': 'center'
                    },
                    children=[
                        html.P("Click the button to select a folder: You can either select any number of .jpg and .png files, or one .csv file. The csv-file should contain the following:", style={'fontSize': '14pt'}),
                        html.P("- The unique name of the image as 'id'", style={'fontSize': '14pt'}),
                        html.P("- The image in numpy format as 'img'", style={'fontSize': '14pt'}),
                        html.P("- Any extra information you may be interested in.", style={'fontSize': '14pt'}),
                        dcc.Upload(
                            id='upload-folder',
                            children=html.Button("📁 Select File(s)"),
                            multiple=True,  # Allow multiple files to simulate folder
                        ),
                        html.Div(id="selected-folder-path", style={'marginTop': '20px', 'fontWeight': 'bold'})
                    ]
                )
            ]
        ),
        det0,
        html.Div(id="placeholder_cls0", children=[], style={"textAlign": "center"}),
        dcc.Interval(id='progress_timer', interval=500, disabled=True),
    ]
)


def images_to_db(contents, filename):
    """

    """
    global df0
    images = []
    file_names = []
    for c, f in zip(contents, filename):
        # Decode the image
        content_type, content_string = c.split(',')
        decoded = base64.b64decode(content_string)

        # Convert the image into cv2-format
        nparr = np.frombuffer(decoded, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # Append results
        images.append(img)
        file_names.append(f[:-4])

    df0 = pd.DataFrame({"id": file_names, "img": images})

    return len(df0)


def db0_to_db1():
    """

    """
    print("Transforming database...")
    df_use_faces = det0.df_detected[det0.df_detected["use"] == True].copy()
    faces = []
    for _, row in df_use_faces.iterrows():
        id = row["img_id"]
        img = det0.df[det0.df["id"] == id]["img"].copy().to_list()[0]
        box = row["face"]
        x, y, w, h = box
        # print(img)
        # print(box)
        face = img[y:y + h, x:x + w]
        faces.append(face)

    df_use_faces["img"] = faces
    print("DONE")

    return df_use_faces.reset_index()




@callback(
    Output('selected-folder-path', 'children'),
    Input('upload-folder', 'contents'),
    Input('upload-folder', 'filename')
)
def update_output(contents, filename):
    global df0
    if contents is not None:
        # Initialize variables
        jpg_or_png = True
        csv = True
        length = 0
        for content, file_name in zip(contents, filename):
            # Check if all files are images
            if not file_name.endswith('.jpg') or file_name.endswith('.png'):
                jpg_or_png = False

            # Check if all files are csv files
            if not file_name.endswith('.csv'):
                csv = False

        # Use if and only if one csv file was uploaded and nothing else
        if len(filename) == 1 and csv:
            pass
        # Use if and only if 1 or more images were uploaded and nothing else
        elif len(filename) > 0 and jpg_or_png:
            length = images_to_db(contents, filename)
        else:
            return html.Div(html.P("The upload requirements are not met..."))

        # Return successfully and provide button to start face detection
        return html.Div(children=[
            html.P(f"Successfully uploaded {length} images."),
            html.Button(
                "Start Face Detection",
                disabled=False,
                style={
                    'padding': '10px 20px',
                    'fontSize': '16pt',
                    'borderRadius': '12px',
                    'border': 'none',
                    'backgroundColor': '#2196F3',
                    'color': 'white',
                    'cursor': 'pointer',
                    "width": "20vw"
                },
                id="button1"
            ),
            dcc.Store(id="trigger-computation"),
            dcc.Store(id="trigger-next-button0"),
            dcc.Store(id="change_or_clicked_image"),
            dcc.Store(id="progressbar_cls"),
            dcc.Store(id="trigger_update"),
            dcc.Store(id="disable_update_cls_btn")
        ])

    return "No files uploaded yet."


@callback(
    # Output('selected-folder-path', 'children', allow_duplicate=True),
    Output('button1', 'disabled', allow_duplicate=True),
    Output("trigger-computation", "data", allow_duplicate=True),
    Output("button1", "style", allow_duplicate=True),
    Input('button1', 'n_clicks'),
    prevent_initial_call=True
)
def react_on_button_click_start_detection(n_clicks):
    # global df0
    # disabled = True
    #
    # if df0 is not None:
    #     print(df0)
    #     # for ids, images in zip(df0["id"].tolist(), df0["img"]):
    #     #     pass
    #
    # return None, disabled
    print(n_clicks)
    if n_clicks is not None and n_clicks > 0:
        style = {
            'padding': '10px 20px',
            'fontSize': '16pt',
            'borderRadius': '12px',
            'border': 'none',
            'backgroundColor': '#2196F3',
            'color': 'white',
            'cursor': 'pointer',
            "width": "20vw",
            "opacity": 0.5
        }
        # return dcc.Loading(
        #     id="loading1",
        #     type="circle",
        #     children=html.Div(id="selected-folder-path")
        # ), True
        return True, {"trigger": True}, style
    else:
        # return None, False
        return False, {"trigger": False}, dash.no_update


@callback(
    Output("selected-folder-path", "children", allow_duplicate=True),
    Output("button1", "disabled", allow_duplicate=True),
    Output("trigger-computation", "data", allow_duplicate=True),
    Output("button1", "style", allow_duplicate=True),
    Output(det0.html_id, "children"),
    Input("trigger-computation", "data"),
    prevent_initial_call=True
)
def start_detection(data):
    global df0
    print(data)
    if not data or "trigger" not in data:
        return dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update

    if data["trigger"]:
        style = {
            'padding': '10px 20px',
            'fontSize': '16pt',
            'borderRadius': '12px',
            'border': 'none',
            'backgroundColor': '#2196F3',
            'color': 'white',
            'cursor': 'pointer',
            "width": "20vw"
        }

        return dash.no_update, False, {"trigger": False}, style, det0.initialize_detector(df0)
    else:
        return dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update


@callback(
    Output("box_images0_index", "data", allow_duplicate=True),
    Output("box_images0", "children", allow_duplicate=True),
    Output("box_images0_left", "disabled", allow_duplicate=True),
    Output('box_images0_right', 'disabled', allow_duplicate=True),
    Output("box_images0_left", "style", allow_duplicate=True),
    Output("box_images0_right", "style", allow_duplicate=True),
    Output("images_showing_txt", "children", allow_duplicate=True),
    Input('box_images0_right', 'n_clicks'),
    State("box_images0_index", "data"),
    prevent_initial_call=True
)
def callback_next_button(n_clicks, data):
    return det0.next_button(n_clicks, data)


@callback(
    Output("box_images0_index", "data", allow_duplicate=True),
    Output("box_images0", "children", allow_duplicate=True),
    Output("box_images0_left", "disabled", allow_duplicate=True),
    Output('box_images0_right', 'disabled', allow_duplicate=True),
    Output("box_images0_left", "style", allow_duplicate=True),
    Output("box_images0_right", "style", allow_duplicate=True),
    Output("images_showing_txt", "children", allow_duplicate=True),
    Input('box_images0_left', 'n_clicks'),
    State("box_images0_index", "data"),
    prevent_initial_call=True
)
def callback_previous_button(n_clicks, data):
    return det0.previous_button(n_clicks, data)


@callback(
    Output("det0_fig", "figure", allow_duplicate=True),
    Output("selection_of_faces", "children", allow_duplicate=True),
    Output("change_or_clicked_image", "data", allow_duplicate=True),
    Input({"type": "image-click", "index": ALL}, "n_clicks"),
    # State("change_or_clicked_image", "data"),
    prevent_initial_call=True
)
def on_image_click(n_clicks):
    if not ctx.triggered_id or all(click is None or click == 0 for click in n_clicks):
        return dash.no_update
    # ctx.triggered_id gives the id of the clicked image
    index = ctx.triggered_id["index"]
    print(f" you clicked on image {index}")
    # det0.update_figure(index)
    return det0.init_picture_fig(index)


@callback(
    Input("button_update_detections", "n_clicks"),
    State("min_width_det_input", "value"),
    State("min_height_det_input", "value"),
    State("change_or_clicked_image", "data"),
)
def update_min_size_det(n_clicks, w, h, data):
    print("ACTIVITY")
    print(data)
    if data is None or not data:
        if n_clicks is not None and n_clicks > 0 and ctx.triggered_id == "button_update_detections":
            print("UPDATED MIN SIZE")
            det0.update_min_size(w, h)


@callback(
    Output("det0_fig", "figure", allow_duplicate=True),
    # Output("show_nrs", "value", allow_duplicate=True),  # Replace with your actual output
    Output("change_or_clicked_image", "data", allow_duplicate=True),
    Output("image-preview", "src", allow_duplicate=True),
    Output("text_for_image", "children", allow_duplicate=True),
    Output("scrollable_column", "children", allow_duplicate=True),
    Output("selected_image_txt", "children", allow_duplicate=True),
    Input("show_nrs", "value"),  # Trigger this callback with a button or other input
    Input({"type": "keep-face", "index": ALL}, "value"),
    Input(f"det0_fig", "selectedData"),
    State("change_or_clicked_image", "data"),
    State("scrollable_column", "children"),
    Input("button_update_detections", "n_clicks"),
    State("min_width_det_input", "value"),
    State("min_height_det_input", "value"),
    prevent_initial_call=True
)
def process_checklist_values(show_nrs_val, checklist_values, selected_data, data, current_children, n_clicks, w, h):
    # Activated due to change in selecting image
    # print("ACTIVITY")
    if data is None or not data:
        print("SELECTED NEW IMAGE, NO BUTTON PRESSED")
        selected_image = f"Selected image: {det0.selected_image}"
        return dash.no_update, True, dash.no_update, "", dash.no_update, selected_image


    # checklist_values will be a list like [['keep'], [], ['keep'], ...]
    print("Checklist states:", checklist_values)
    print(f"Show numbers?: {show_nrs_val}")
    trigger_id = ctx.triggered_id
    print("Triggered by:", trigger_id)

    # You can interpret them like this:
    # keep_flags = [("keep" in value) for value in checklist_values]
    return det0.update_picture_fig(show_nrs_val, checklist_values, trigger_id, selected_data, current_children, n_clicks, w, h)

@callback(
    Output("placeholder_cls0", "children", allow_duplicate=True),
    Output("progressbar_cls", "data", allow_duplicate=True),
    Output("progress_timer", "disabled", allow_duplicate=True),
    Output("button2", "disabled", allow_duplicate=True),
    Output("button2", "style", allow_duplicate=True),
    Input("button2", "n_clicks"),
    prevent_initial_call=True
)
def react_on_button2(n_clicks):
    if n_clicks is not None and n_clicks > 0:
        global df1
        df1 = db0_to_db1().copy()
        # global cls0
        # cls0 = Clusteror("cls0", det0.df, det0.df_faces)
        size = len(df1)
        result = html.Div([
            html.H2("Face Clustering", style={"textAlign": "center"}),
            html.Hr(),
            # dcc.Interval(id='progress_timer', interval=500, disabled=False),
            dbc.Progress(
                id="progressbar0",
                label="0%",
                value=0,
                striped=True,
                animated=True,
                color="#2196F3",
                style={
                    "width": "99%",
                    "height": "30px",
                    "margin": "10px auto",
                    "marginBottom": "10px"
                }
            )
        ])
        data = {"mode": "running", "size": size, "n": 0, "embeddings": []}
        style = {
            'padding': '10px 20px',
            'fontSize': '16pt',
            'borderRadius': '12px',
            'border': 'none',
            'backgroundColor': '#2196F3',
            'color': 'white',
            'cursor': 'pointer',
            "width": "20vw",
            "opacity": 0.5,
            "marginBottom": "20px"
        }

        return result, data, False, True, style
    else:
        return dash.no_update, dash.no_update, False, False, dash.no_update


@callback(
    Output("progressbar0", "value", allow_duplicate=True),
    Output("progressbar0", "label", allow_duplicate=True),
    Output("progressbar_cls", "data", allow_duplicate=True),
    Output("progress_timer", "disabled", allow_duplicate=True),
    Output("trigger_update", "data", allow_duplicate=True),
    Output("button2", "disabled", allow_duplicate=True),
    Output("button2", "style", allow_duplicate=True),
    Output("placeholder_cls0", "children", allow_duplicate=True),
    # Input("progress_timer", "n_intervals"),
    Input("trigger_update", "data"),
    State("progressbar_cls", "data"),
    # Input("trigger_update", "data"),
    prevent_initial_call=True
)
def update_progress_bar(trigger, data):
    if data is not None and "embeddings" in data:
        if data["mode"] == "running":
            global df0
            global df1
            global cls0
            # if "embeddings" in data:
            n = data["n"]
            print(n)
            # if n >= data["size"]:
            #     data["mode"] = "DONE"
            #     return dash.no_update, dash.no_update, data, True, dash.no_update

            image = np.ascontiguousarray(df1.iloc[n]["img"])
            height, width = image.shape[:2]
            # print(image)
            # print(image.shape[:2])

            encoding = face_recognition.face_encodings(image, num_jitters=2, model="large", known_face_locations=[(0, width, height, 0)])
            # encoding = face_encoder.compute_face_descriptor(face_image, raw_landmark_set, num_jitters)
            data["embeddings"].append(encoding[0])
            # data["embeddings_tsne"].append(tsne.fit_transform(encoding[0].reshape(1, -1)))
            data["n"] += 1
            new_val = 100 * (n + 1) / data['size']

            if data["n"] >= data["size"]:
                data["mode"] = "DONE"
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

                df1["embedding"] = [",".join(map(str, e.tolist())) for e in np.array(data["embeddings"])]
                # df1["embedding"] = data["embeddings"]
                # save_embeddings(data["embeddings"], [str(i) for i in range(len(data["embeddings"]))], filename=f"temp_embeddings.npz")
                # emb, paths = load_embeddings(f"temp_embeddings.npz")
                tsne = TSNE(n_components=2, perplexity=min(30.0, len(data["embeddings"]) - 1))
                emb_tsne = tsne.fit_transform(np.array(data["embeddings"]))
                df1["tsne_x"] = emb_tsne[:, 0]
                df1["tsne_y"] = emb_tsne[:, 1]

                df1["embedding_tsne"] = [",".join(map(str, e.tolist())) for e in emb_tsne]
                print(df1)

                cls0 = Clusteror("cls0", df0, df1)
                return new_val, f"{new_val:.1f}%", data, True, dash.no_update, False, style, cls0
            return new_val, f"{new_val:.1f}%", data, False, False, True, dash.no_update, dash.no_update
    else:
        return dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update


@callback(
    Output("progress_timer", "disabled", allow_duplicate=True),
    Output("trigger_update", "data", allow_duplicate=True),
    Input("progress_timer", "n_intervals"),
    prevent_initial_call=True
)
def disabled_interval(_):
    return True, True


@callback(
    Output("box_images1", "children", allow_duplicate=True),
    Output("box_images1_left", "disabled", allow_duplicate=True),
    Output('box_images1_right', 'disabled', allow_duplicate=True),
    Output("box_images1_left", "style", allow_duplicate=True),
    Output("box_images1_right", "style", allow_duplicate=True),
    Output("images_showing_txt1", "children", allow_duplicate=True),
    Input('box_images1_right', 'n_clicks'),
    prevent_initial_call=True
)
def callback_next_button_cls(n_clicks):
    return cls0.next_button(n_clicks)


@callback(
    Output("box_images1", "children", allow_duplicate=True),
    Output("box_images1_left", "disabled", allow_duplicate=True),
    Output('box_images1_right', 'disabled', allow_duplicate=True),
    Output("box_images1_left", "style", allow_duplicate=True),
    Output("box_images1_right", "style", allow_duplicate=True),
    Output("images_showing_txt1", "children", allow_duplicate=True),
    Input('box_images1_left', 'n_clicks'),
    prevent_initial_call=True
)
def callback_previous_button_cls(n_clicks):
    return cls0.previous_button(n_clicks)


@callback(
    Output("button_update_clusters", "disabled", allow_duplicate=True),
    Output("button_update_clusters", "style", allow_duplicate=True),
    Output("disable_update_cls_btn", "data", allow_duplicate=True),
    Output("button_continue_clusters", "disabled", allow_duplicate=True),
    Input("button_update_clusters", "n_clicks"),
    prevent_initial_call=True
)
def disable_button_update_clusters(n_clicks):
    if n_clicks is not None and n_clicks > 0:
        style = {
            'padding': '10px 20px',
            'fontSize': '16pt',
            'borderRadius': '12px',
            'border': 'none',
            'backgroundColor': '#2196F3',
            'color': 'white',
            'cursor': 'pointer',
            "width": "10vw",
            "opacity": 0.5
        }
        print("running")
        return True, style, True, True
    else:
        return dash.no_update, dash.no_update, dash.no_update, dash.no_update


@callback(
    Output("cls0", "figure", allow_duplicate=True),
    Output("box_images1", "children", allow_duplicate=True),
    Output("images_showing_txt1", "children", allow_duplicate=True),
    Output("showing_cluster0", "children", allow_duplicate=True),
    Output("box_images1_left", "disabled", allow_duplicate=True),
    Output("box_images1_right", "disabled", allow_duplicate=True),
    Output("box_images1_left", "style", allow_duplicate=True),
    Output("box_images1_right", "style", allow_duplicate=True),
    Output("dropdown_change", "children", allow_duplicate=True),
    Output("button_update_clusters", "disabled", allow_duplicate=True),
    Output("button_update_clusters", "style", allow_duplicate=True),
    Output("disable_update_cls_btn", "data", allow_duplicate=True),
    Output("button_continue_clusters", "disabled", allow_duplicate=True),
    Input("disable_update_cls_btn", "data"),
    State("eps_input", "value"),
    State("min_samples_input", "value"),
    State("min_width_cls_input", "value"),
    State("min_height_cls_input", "value"),
    prevent_initial_call=True
)
def update_clusters_cls(data, eps_input, min_samples_input, min_width_cls_input, min_height_cls_input):
    return cls0.update_clusters(data, eps_input, min_samples_input, min_width_cls_input, min_height_cls_input)


@callback(
    Output("box_images1", "children", allow_duplicate=True),
    Output("images_showing_txt1", "children", allow_duplicate=True),
    Output("showing_cluster0", "children", allow_duplicate=True),
    Output("box_images1_left", "disabled", allow_duplicate=True),
    Output("box_images1_right", "disabled", allow_duplicate=True),
    Output("box_images1_left", "style", allow_duplicate=True),
    Output("box_images1_right", "style", allow_duplicate=True),
    Input("dropdown_cls", "value"),
    prevent_initial_call=True
)
def update_image_box(value):
    return cls0.update_image_box(value)



# @callback(
#     Output("trigger_update", "data"),
#     Input("progressbar0", "value"),
#     State("trigger_update", "data"),
# )
# def trigger_update(value, data):
#     if data:
#         data = False
#     else:
#         data = True
#
#     return data
# for _, row in df1:
#     image = np.ascontiguousarray(row["img"])
#     height, width = image.shape[:2]
#     encoding = face_recognition.face_encodings(image, num_jitters=2, model="large",
#                                                known_face_locations=[(0, width, height, 0)])

# @callback(
#     Output('selected-folder-path', 'children', allow_duplicate=True),
#     Output('button1', 'disabled', allow_duplicate=True),
#     Input('button1', 'n_clicks'),
#     prevent_initial_call=True
# )
# def start_detection(n_clicks):
#     global df0
#     if n_clicks is not None and n_clicks > 0:
#         i = 0
#         while i < 100:
#             print(f"i: {i}")
#             i += 1
#         return dash.no_update, False
#     else:
#         return dash.no_update, False



# # ✅ Callback defined in the same file
# @callback(
#     Output("selected-folder-path", "children"),
#     Input("select-folder-btn", "n_clicks"),
#     prevent_initial_call=True
# )
# def update_folder_path(n_clicks):
#     # root = tk.Tk()
#     # root.withdraw()  # Hide the main window
#     #
#     # path = filedialog.askopenfilename()  # Show file picker
#     # # path = select_folder()
#     # print(path)
#
#     file_dialog_thread = threading.Thread(target=select_folder)
#     file_dialog_thread.start()
#
#     return "ygqwedoygweoywgefowegyf"
    # return f"Selected folder path: {path}" if path else "No folder selected."
