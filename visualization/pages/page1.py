import dash
from dash import html, register_page, callback, Output, Input, dcc, State
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

df0 = None
register_page(__name__, path="/")

# # Folder picker function
# def select_folder():
#     root = tk.Tk()
#     root.withdraw()
#     folder_path = filedialog.askopenfilename()
#     return folder_path

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
        )
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
        file_names.append(f)

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
            dcc.Store(id="trigger-computation")
        ])

    return "No files uploaded yet."


@callback(
    # Output('selected-folder-path', 'children', allow_duplicate=True),
    Output('button1', 'disabled', allow_duplicate=True),
    Output("trigger-computation", "data", allow_duplicate=True),
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
        # return dcc.Loading(
        #     id="loading1",
        #     type="circle",
        #     children=html.Div(id="selected-folder-path")
        # ), True
        return True, {"trigger": True}
    else:
        # return None, False
        return False, {"trigger": False}


@callback(
    Output("selected-folder-path", "children", allow_duplicate=True),
    Output("button1", "disabled", allow_duplicate=True),
    Output("trigger-computation", "data", allow_duplicate=True),
    # Output("button1", "style", allow_duplicate=True),
    Input("trigger-computation", "data"),
    prevent_initial_call=True
)
def start_detection(data):
    print(data)
    if not data or "trigger" not in data:
        return dash.no_update, dash.no_update, dash.no_update

    if data["trigger"]:
        time.sleep(10)

        return dash.no_update, False, {"trigger": False}
    else:
        return dash.no_update, dash.no_update, dash.no_update



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
