import dash
from dash import html, register_page, callback, Output, Input, dcc, State, ctx, ALL
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
from visualization.detection import Detector

df0 = None
register_page(__name__, path="/")

# # Folder picker function
# def select_folder():
#     root = tk.Tk()
#     root.withdraw()
#     folder_path = filedialog.askopenfilename()
#     return folder_path

det0 = Detector("det0", df0)


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
        det0
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
            dcc.Store(id="change_or_clicked_image")
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
    prevent_initial_call=True
)
def process_checklist_values(show_nrs_val, checklist_values, selected_data, data, current_children):
    # Activated due to change in selecting image
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
    return det0.update_picture_fig(show_nrs_val, checklist_values, trigger_id, selected_data, current_children)  # or however you want to use them


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
