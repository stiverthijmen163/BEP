import dash
from dash import html, register_page, callback, Output, Input, dcc, State, ctx, ALL
import dash_bootstrap_components as dbc
import base64
import io
import face_recognition
from sklearn.manifold import TSNE
from visualization.functions import *
import re
from visualization.detection import Detector
from visualization.clustering import Clusteror
import ast

# Initialize dataframes and the placeholder for the clustering section
df0 = None
df1 = None
cls0 = None

# Set page
register_page(__name__, path="/")

# Initialize Detector object (detection section)
det0 = Detector("det0", df0)

# Initialize the layout for this page
layout = html.Div(
    children=[
        html.Div(
            style={"textAlign": "center"},
            children=[
                html.H2("Data Selection", style={"marginTop": "20px"}),  # Header
                html.Hr(),
                html.Div(  # Box where you can upload data
                    style={
                        "backgroundColor": "#dbeafe",
                        "padding": "20px",
                        "margin": "20px auto",
                        "width": "60%",
                        "borderRadius": "12px",
                        "boxShadow": "0px 2px 5px rgba(0,0,0,0.1)",
                        "textAlign": "center"
                    },
                    children=[  # Text explaining what you can upload and the expected format
                        html.P("Click the button to select files: You can either select any number of .jpg and .png files, or one .csv file. The csv-file should contain the following:", style={"fontSize": "14pt"}),
                        html.P("- The unique name of the image as 'id'", style={"fontSize": "14pt"}),
                        html.P("- The image as lists containing BGR-values as 'img'", style={"fontSize": "14pt"}),
                        html.P([
                            "- Any extra information you may be interested in. This may only contain lists of strings: '['item1', 'item2']', url's must be in the 'url' column. Example: ",
                            html.A("GitHub", href="https://github.com/stiverthijmen163/BEP/blob/main/examples/example_csv.csv", target="_blank", style={"color": "#1d4ed8", "textDecoration": "underline"}),
                        ],
                        style={"fontSize": "14pt"}),
                        dcc.Upload(  # Button to upload data
                            id="upload-folder",
                            children=html.Button("📁 Select File(s)"),
                            multiple=True,  # Allow multiple files to be uploaded
                        ),  # Empty section to provide feedback to the user's uploaded data
                        html.Div(id="selected-folder-path", style={"marginTop": "20px", "fontWeight": "bold"})
                    ]
                )
            ]
        ),
        det0,  # Detection section
        html.Div(id="placeholder_cls0", children=[], style={"textAlign": "center"}),  # Placeholder for clustering section
        dcc.Interval(id="progress_timer", interval=500, disabled=True),  # Timer at which rate the progressbar is updated
    ]
)


def images_to_db(contents, filename) -> int:
    """
    Created a dataframe (df0) from a given set of images and unique file names.

    :param contents: set of images
    :param filename: set of unique file names

    :return: number of images processed
    """
    # Make df0 accessible and editable
    global df0

    # Initialize list of images and file-names
    images = []
    file_names = []
    for c, f in zip(contents, filename):
        # Decode the image
        content_type, content_string = c.split(",")
        decoded = base64.b64decode(content_string)

        # Convert the image into cv2-format
        nparr = np.frombuffer(decoded, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # Append results
        images.append(img)
        file_names.append(f[:-4])

    # Create dataframe and assign it to the global variable df0
    df0 = pd.DataFrame({"id": file_names, "img": images})

    print(f"(Page 1)              - Uploaded {len(df0)} images successfully")

    return len(df0)


def csv_to_db(contents, filename) -> Tuple[int, [List[str]]]:
    """
    Creates a dataframe (df0) from a single csv-file.

    :param contents: csv file contents, must contain at least 'id' & 'img' columns
    :param filename: csv file name

    :return: number of images processed and a list of errors occurred during processing the csv-file
    """
    # Make df0 accessible and editable
    global df0

    # Decode the content of the csv-file
    content_string = contents[0].split(",")[1]
    decoded = base64.b64decode(content_string)

    # Initialize list of errors and nr of images
    error_txt = []
    length = 0
    try:  # Try to read the csv-file
        # Read the data
        df0 = pd.read_csv(io.StringIO(decoded.decode("utf-8")))
        print(f"(Page 1)              - CSV file '{filename[0]}' uploaded successfully. Shape: {df0.shape}")

        # Check if expected columns are present
        if {"id", "img"}.issubset(df0.columns):
            for column in df0.columns:
                if column == "url":  # Make sure format is string
                    df0["url"] = df0[column].apply(lambda x: f"{x}")
                elif column == "img":  # Convert img into correct format
                    try:  # Convert to numpy format
                        df0["img"] = df0["img"].apply(base64_to_img)
                    except Exception as e:  # Input format is incorrect
                        print(f"(Page 1)              - Column 'img' is not in the expected format: {e}")
                        error_txt.append("Column 'img' is not in the expected format.")
                elif column == "id":  # Make sure format is tring
                    df0["id"] = df0["id"].apply(lambda x: str(x))
                else: # For all remaining columns
                    try: # Convert string of list into list
                        df0[column] = df0[column].apply(lambda x: json.loads(f"{x}"))
                        df0[column] = df0[column].apply(json.dumps)
                    except Exception as e:  # Input not in expected format
                        print(f"(Page 1)              - Error trying to load '{column}': {e}")
                        error_txt.append(f"Column '{column}' is not in the expected format.")
            if len(error_txt) > 0:  # Remove df0 is errors occurred
                df0 = None
            else:  # Set length otherwise
                length = len(df0)
        else:  # Check which columns are missing
            missing_cols = [col for col in ["id", "img"] if col not in df0.columns]
            error_txt.append(f"Missing columns: {', '.join(missing_cols)}")
    except Exception as e:  # Can't read csv-file
        print(f"(Page 1)              - Error reading CSV file: {str(e)}")
        error_txt.append(f"Error reading CSV file: {str(e)}")

    print(f"(Page 1)              - Processed {length} images successfully")

    return length, error_txt


def db0_to_db1():
    """
    Transitions from the detection dataset to the clustering dataset.
    """
    print("(Page 1)              - Transforming database to contain faces")
    # Collect the detection dataset only containing the selected images
    df_use_faces = det0.df_detected[det0.df_detected["use"] == True].copy()

    # Initialize the list of faces
    faces = []

    # Collect all faces
    for _, row in df_use_faces.iterrows():
        # Get the id and bounding box (face) from the current row
        id = row["img_id"]
        box = row["face"]
        x, y, w, h = box

        # Collect its corresponding image
        img = det0.df[det0.df["id"] == id]["img"].copy().to_list()[0]

        # Crop the image to the selected face and append the result
        face = img[y:y + h, x:x + w]
        faces.append(face)

    # Append all faces to the dataset
    df_use_faces["img"] = faces

    return df_use_faces.reset_index()


@callback(
    Output("selected-folder-path", "children"),
    Input("upload-folder", "contents"),
    Input("upload-folder", "filename")
)
def process_upload(contents, filename):
    """
    Callback function for processing all uploaded files.

    :param contents: set of file contents
    :param filename: set of unique file names
    """
    # Make df0 accessible and editable
    global df0

    # Process the uploaded files
    if contents is not None:
        # Initialize variables
        jpg_or_png = True
        csv = True

        # Check what type of files are uploaded
        for content, file_name in zip(contents, filename):
            # Check if all files are images
            if not (file_name.endswith(".jpg") or file_name.endswith(".png")):
                jpg_or_png = False

            # Check if all files are csv-files
            if not file_name.endswith(".csv"):
                csv = False

        # Use if and only if one csv-file was uploaded and nothing else
        if len(filename) == 1 and csv:
            # Process the csv-file
            length, errors = csv_to_db(contents, filename)
            if len(errors) > 0:  # Return errors is there are any
                children = []
                for e in errors:
                    children.append(html.P(e))
                return html.Div(children=children)

        # Use if and only if 1 or more images (jpg or png) were uploaded and nothing else
        elif len(filename) > 0 and jpg_or_png:
            # Process all images
            length = images_to_db(contents, filename)
        else:  # No files of the expected type
            return html.Div(html.P("The upload requirements are not met..."))

        # If no images were uploaded/processed
        if length == 0:
            return html.Div(html.P("Uploaded 0 images, you should upload at least one image!"))

        # Return successfully and provide button to start face detection
        return html.Div(children=[
            html.P(f"Successfully uploaded {length} images."),
            html.Button(  # Button to load face detection section
                "Start Face Detection",
                disabled=False,
                style={
                    "padding": "10px 20px",
                    "fontSize": "16pt",
                    "borderRadius": "12px",
                    "border": "none",
                    "backgroundColor": "#2196F3",
                    "color": "white",
                    "cursor": "pointer",
                    "width": "20vw"
                },
                id="button1"
            ),  # Store objects, used to trigger updates
            dcc.Store(id="trigger_start_detection"),  # Trigger detection initialization
            dcc.Store(id="change_or_clicked_image"),  # Keeps track of whether an image was changed or not in the detection section image box
            dcc.Store(id="progressbar_cls"),  # Keep track of the variables used for processing the faces and updating the progressbar when moving on to clustering
            dcc.Store(id="trigger_update"),  # Trigger to update the progressbar
            dcc.Store(id="disable_update_cls_btn"),  # To disable the update_clusters button
            dcc.Store(id="enable_click_on_image"),  # When cluster mode is in edit True to allow clickable images
            dcc.Store(id="update_components"),  # Triggers update on components in edit mode
            dcc.Store(id="add_cluster"),  # Trigger update to create a new cluster
            dcc.Store(id="update_click_on_image"),  # Trigger update when clicking on image in clustering section
            dcc.Store(id="trigger_button3")  # To disable button3
        ])

    # Standard text
    return "No files uploaded yet."


@callback(
    Output("button1", "disabled", allow_duplicate=True),
    Output("trigger_start_detection", "data", allow_duplicate=True),
    Output("button1", "style", allow_duplicate=True),
    Input("button1", "n_clicks"),
    prevent_initial_call=True
)
def disable_button1(n_clicks):
    """
    Callback function that disables button1 (button to start face detection) when clicked.

    :param n_clicks: number of clicks on button1
    """
    if n_clicks is not None and n_clicks > 0:
        # Update style to show the user it's not clickable
        style = {
            "padding": "10px 20px",
            "fontSize": "16pt",
            "borderRadius": "12px",
            "border": "none",
            "backgroundColor": "#2196F3",
            "color": "white",
            "cursor": "pointer",
            "width": "20vw",
            "opacity": 0.5
        }

        # Update outputs
        return True, {"trigger": True}, style

    # No update, trigger = False provides no further updates
    return False, {"trigger": False}, dash.no_update


@callback(
    Output("button1", "disabled", allow_duplicate=True),
    Output("trigger_start_detection", "data", allow_duplicate=True),
    Output("button1", "style", allow_duplicate=True),
    Output(det0.html_id, "children", allow_duplicate=True),
    Input("trigger_start_detection", "data"),
    prevent_initial_call=True
)
def start_detection(data):
    """
    Callback function that initializes the detection section as a non-empty html div.

    :param data: trigger data, True when initialization is needed
    """
    # Make df0 accessible and editable
    global df0

    # If no updated data
    if not data or "trigger" not in data:
        return dash.no_update, dash.no_update, dash.no_update, dash.no_update

    # If triggered by data update output
    if data["trigger"]:
        # Update style of button to show user it is clickable
        style = {
            "padding": "10px 20px",
            "fontSize": "16pt",
            "borderRadius": "12px",
            "border": "none",
            "backgroundColor": "#2196F3",
            "color": "white",
            "cursor": "pointer",
            "width": "20vw"
        }

        # Update outputs
        return False, {"trigger": False}, style, det0.initialize_detector(df0)

    # No update
    return dash.no_update, dash.no_update, dash.no_update, dash.no_update


@callback(
    Output("box_images0_index", "data", allow_duplicate=True),
    Output("box_images0", "children", allow_duplicate=True),
    Output("box_images0_left", "disabled", allow_duplicate=True),
    Output("box_images0_right", "disabled", allow_duplicate=True),
    Output("box_images0_left", "style", allow_duplicate=True),
    Output("box_images0_right", "style", allow_duplicate=True),
    Output("images_showing_txt", "children", allow_duplicate=True),
    Input("box_images0_right", "n_clicks"),
    State("box_images0_index", "data"),
    prevent_initial_call=True
)
def callback_next_button_detection(n_clicks, data):
    """
    Callback function that processes the click on the next button for showing the next 10 images
    in the detection section.

    :param n_clicks: number of clicks on 'next' button
    :param data: the current index of the first shown image
    """
    # Update output
    return det0.next_button(n_clicks, data)


@callback(
    Output("box_images0_index", "data", allow_duplicate=True),
    Output("box_images0", "children", allow_duplicate=True),
    Output("box_images0_left", "disabled", allow_duplicate=True),
    Output("box_images0_right", "disabled", allow_duplicate=True),
    Output("box_images0_left", "style", allow_duplicate=True),
    Output("box_images0_right", "style", allow_duplicate=True),
    Output("images_showing_txt", "children", allow_duplicate=True),
    Input("box_images0_left", "n_clicks"),
    State("box_images0_index", "data"),
    prevent_initial_call=True
)
def callback_back_button_detection(n_clicks, data):
    """
    Callback function that processes the click on the 'back' button for showing the previous 10 images
    in the detection section.

    :param n_clicks: number of clicks on 'back' button
    :param data: the current index of the first shown image
    """
    # Update output
    return det0.previous_button(n_clicks, data)


@callback(
    Output("det0_fig", "figure", allow_duplicate=True),
    Output("selection_of_faces", "children", allow_duplicate=True),
    Output("change_or_clicked_image", "data", allow_duplicate=True),
    Input({"type": "image-click", "index": ALL}, "n_clicks"),
    prevent_initial_call=True
)
def on_image_click_detection(n_clicks):
    """
    Callback function that shows an image and its corresponding faces when clicked on in the detection section.

    :param n_clicks: number of clicks on an image
    """
    # If none of the images was clicked on, no update
    if not ctx.triggered_id or all(click is None or click == 0 for click in n_clicks):
        return dash.no_update, dash.no_update, dash.no_update

    # Check which image was clicked on
    index = ctx.triggered_id["index"]

    # Update output
    return det0.init_picture_fig(index)


@callback(
    Input("button_update_detections", "n_clicks"),
    State("min_width_det_input", "value"),
    State("min_height_det_input", "value"),
    State("change_or_clicked_image", "data"),
)
def update_min_size_detection(n_clicks, w, h, data):
    """
    Callback function that updates the dataset in the Detector object to all faces larger than the input,
    in case no image has been clicked on yet.

    :param n_clicks: number of clicks on the update_detections button
    :param w: minimum width of a face
    :param h: minimum height of a face
    :param data: whether a new image has been clicked on or not (None is no image clicked on yet)
    """
    # In case no image has been clicked on yet
    if data is None or not data:
        # In case button has been clicked and callback is trigger by button click
        if n_clicks is not None and n_clicks > 0 and ctx.triggered_id == "button_update_detections":
            # Update dataset Detector object
            det0.update_min_size(w, h)


@callback(
    Output("det0_fig", "figure", allow_duplicate=True),
    Output("change_or_clicked_image", "data", allow_duplicate=True),
    Output("image-preview", "src", allow_duplicate=True),
    Output("text_for_image", "children", allow_duplicate=True),
    Output("scrollable_column", "children", allow_duplicate=True),
    Output("selected_image_txt", "children", allow_duplicate=True),
    Input("show_nrs", "value"),
    Input({"type": "keep-face", "index": ALL}, "value"),
    Input(f"det0_fig", "selectedData"),
    State("change_or_clicked_image", "data"),
    State("scrollable_column", "children"),
    Input("button_update_detections", "n_clicks"),
    State("min_width_det_input", "value"),
    State("min_height_det_input", "value"),
    prevent_initial_call=True
)
def process_updates_detection(show_nrs_val, checklist_values, selected_data, data, current_children, n_clicks, w, h):
    """
    Callback function that processes various checklist clicks or adds a face to the checklist when
    'box select' is used in the figure and updates the dataset in the Detector object to all faces
    larger than the input, in case an image has been clicked on already.

    :param show_nrs_val: whether to show the face number in the shown image
    :param checklist_values: list of checklist values, each value denotes whether to keep the face or not
    :param selected_data: the data from selecting something in the figure (image) using 'box select'
    :param data: whether a new image has been clicked on or not (None is no image clicked on yet)
    :param current_children: the current children (items) in the checklist
    :param n_clicks: number of clicks on the update_detections button
    :param w: minimum width of a face
    :param h: minimum height of a face
    """
    # If new image has been selected, update the display text
    if data is None or not data:
        selected_image = f"Selected image: {det0.selected_image}"
        return dash.no_update, True, dash.no_update, "", dash.no_update, selected_image

    # Collect the id of the object that triggered this callback
    trigger_id = ctx.triggered_id

    # Update the output
    return det0.update_picture_fig(show_nrs_val, checklist_values, trigger_id, selected_data, current_children, n_clicks, w, h)


@callback(
    Output("placeholder_cls0", "children", allow_duplicate=True),
    Output("progressbar_cls", "data", allow_duplicate=True),
    Output("progress_timer", "disabled", allow_duplicate=True),
    Output("button2", "disabled", allow_duplicate=True),
    Output("button2", "style", allow_duplicate=True),
    Output("successful_detection", "children", allow_duplicate=True),
    Input("button2", "n_clicks"),
    prevent_initial_call=True
)
def react_on_button2(n_clicks):
    """
    Callback function that disabled button2 (continue to clustering) and initializes the progress bar
    for calculating the embeddings after updating the dataset to contain all selected faces only.

    :param n_clicks: number of clicks on button2
    """
    # If the button has been clicked on
    if n_clicks is not None and n_clicks > 0:
        # Makes the new dataset (df1) accessible and editable
        global df1

        # Update the dataset to contain only the selected faces
        df1 = db0_to_db1().copy()

        if len(df1) > 1:  # Clustering needs at least 2 faces
            # Collect the number of faces
            size = len(df1)

            # Initialize the progressbar
            result = html.Div([  # Header
                html.H2("Face Clustering", style={"textAlign": "center"}),
                html.Hr(),
                dbc.Progress(  # Progressbar
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

            # Initialize the variables used for processing the faces and updating the progressbar
            data = {"mode": "running", "size": size, "n": 0, "embeddings": []}

            # Update style of the button to show the user it is disabled
            style = {
                "padding": "10px 20px",
                "fontSize": "16pt",
                "borderRadius": "12px",
                "border": "none",
                "backgroundColor": "#2196F3",
                "color": "white",
                "cursor": "pointer",
                "width": "20vw",
                "opacity": 0.5,
                "marginBottom": "20px"
            }

            # Update output
            return result, data, False, True, style, ""
        else:  # Not enough faces selected
            # Output error text
            txt = "At least two faces should be selected!"
            return dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, txt

    # No update and make sure nothing is disabled
    return dash.no_update, dash.no_update, False, False, dash.no_update, dash.no_update


@callback(
    Output("progressbar0", "value", allow_duplicate=True),
    Output("progressbar0", "label", allow_duplicate=True),
    Output("progressbar_cls", "data", allow_duplicate=True),
    Output("progress_timer", "disabled", allow_duplicate=True),
    Output("trigger_update", "data", allow_duplicate=True),
    Output("button2", "disabled", allow_duplicate=True),
    Output("button2", "style", allow_duplicate=True),
    Output("placeholder_cls0", "children", allow_duplicate=True),
    Input("trigger_update", "data"),
    State("progressbar_cls", "data"),
    prevent_initial_call=True
)
def update_progress_bar(_, data):
    """
    Callback function which calculates the embeddings for all faces and updates the progressbar
    accordingly, provides the user with the clustering section when all faces are processed.

    :param data: Keeps track of the variables used for processing the faces and updating the progressbar
    """
    if data is not None and "embeddings" in data:  # Data is initialized
        if data["mode"] == "running":  # Correct mode
            # Make all datasets and the clustering section placeholder accessible and editable
            global df0
            global df1
            global cls0

            # Collect the current index to process
            n = data["n"]

            # Collect the face and its shape to process
            image = np.ascontiguousarray(df1.iloc[n]["img"])
            height, width = image.shape[:2]

            # Embed the face
            encoding = face_recognition.face_encodings(image, num_jitters=2, model="large", known_face_locations=[(0, width, height, 0)])

            # Append the embedding
            data["embeddings"].append(encoding[0])

            # Update the index
            data["n"] += 1

            # Update the progress made so far
            new_val = 100 * (n + 1) / data["size"]

            # In case all faces are processed
            if data["n"] >= data["size"]:
                # Update mode
                data["mode"] = "DONE"

                # Update button style to show it is enabled
                style = {
                    "padding": "10px 20px",
                    "fontSize": "16pt",
                    "borderRadius": "12px",
                    "border": "none",
                    "backgroundColor": "#2196F3",
                    "color": "white",
                    "cursor": "pointer",
                    "width": "20vw",
                    "marginBottom": "20px"
                }

                # Append the embeddings to the dataset
                df1["embedding"] = [",".join(map(str, e.tolist())) for e in np.array(data["embeddings"])]

                # Use t-SNE to reduce the size of the embeddings
                tsne = TSNE(n_components=2, perplexity=min(30.0, len(data["embeddings"]) - 1))
                emb_tsne = tsne.fit_transform(np.array(data["embeddings"]))

                # Append the x and y coordinates of t-SNE to the dataset for easy plotting
                df1["tsne_x"] = emb_tsne[:, 0]
                df1["tsne_y"] = emb_tsne[:, 1]

                # Append the t-SNE reduced embeddings to the dataset
                df1["embedding_tsne"] = [",".join(map(str, e.tolist())) for e in emb_tsne]

                # Initialize the clustering section by using the Clusteror object
                cls0 = Clusteror("cls0", df0, df1)

                # Update output with the new clustering section
                return new_val, f"{new_val:.1f}%", data, True, dash.no_update, False, style, cls0
            # Update progressbar and activate Interval for the next iteration
            return new_val, f"{new_val:.1f}%", data, False, False, True, dash.no_update, dash.no_update
    # No updates
    return dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update


@callback(
    Output("progress_timer", "disabled", allow_duplicate=True),
    Output("trigger_update", "data", allow_duplicate=True),
    Input("progress_timer", "n_intervals"),
    prevent_initial_call=True
)
def disabled_interval(_):
    """
    Callback function to disable the Interval when it has triggered ones to prevent loops,
    updates trigger to update progressbar.
    """
    # Disable Interval and update trigger
    return True, True


@callback(
    Output("box_images1", "children", allow_duplicate=True),
    Output("box_images1_left", "disabled", allow_duplicate=True),
    Output("box_images1_right", "disabled", allow_duplicate=True),
    Output("box_images1_left", "style", allow_duplicate=True),
    Output("box_images1_right", "style", allow_duplicate=True),
    Output("images_showing_txt1", "children", allow_duplicate=True),
    Input("box_images1_right", "n_clicks"),
    prevent_initial_call=True
)
def callback_next_button_cls(n_clicks):
    """
    Callback function that processes the click on the next button for showing the next 10 images
    in the clustering section.

    :param n_clicks: number of clicks on 'next' button
    """
    # Update output
    return cls0.next_button(n_clicks)


@callback(
    Output("box_images1", "children", allow_duplicate=True),
    Output("box_images1_left", "disabled", allow_duplicate=True),
    Output("box_images1_right", "disabled", allow_duplicate=True),
    Output("box_images1_left", "style", allow_duplicate=True),
    Output("box_images1_right", "style", allow_duplicate=True),
    Output("images_showing_txt1", "children", allow_duplicate=True),
    Input("box_images1_left", "n_clicks"),
    prevent_initial_call=True
)
def callback_back_button_cls(n_clicks):
    """
    Callback function that processes the click on the 'back' button for showing the previous 10 images
    in the clustering section.

    :param n_clicks: number of clicks on 'back' button
    """
    # Update output
    return cls0.previous_button(n_clicks)


@callback(
    Output("button_update_clusters", "disabled", allow_duplicate=True),
    Output("button_update_clusters", "style", allow_duplicate=True),
    Output("disable_update_cls_btn", "data", allow_duplicate=True),
    Output("button_continue_clusters", "disabled", allow_duplicate=True),
    Output("enable_click_on_image", "data", allow_duplicate=True),
    Input("button_update_clusters", "n_clicks"),
    prevent_initial_call=True
)
def disable_buttons_update_clusters(n_clicks):
    """
    Callback function that disables both the 'update clusters' and 'continue' button when the
    'update clusters' button is clicked, updates trigger to update the clusters.

    :param n_clicks: number of clicks on 'update cluster' button
    """
    # If the button is clicked
    if n_clicks is not None and n_clicks > 0:
        # Update style of button to show user it has been disabled
        style = {
            "padding": "10px 20px",
            "fontSize": "16pt",
            "borderRadius": "12px",
            "border": "none",
            "backgroundColor": "#2196F3",
            "color": "white",
            "cursor": "pointer",
            "width": "10vw",
            "opacity": 0.5
        }

        # Update outputs
        return True, style, True, True, False
    # No update
    return dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update


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
    """
    Callback function that updates the clusters in the clustering section according to the inputted
    parameters.

    :param data: trigger that states whether to update the clusters or not
    :param eps_input: eps input value, parameter of DBSCAN method
    :param min_samples_input: min_samples input value, parameter of DBSCAN method
    :param min_width_cls_input: minimum width of a face to be considered for clustering
    :param min_height_cls_input: minimum height of a face to be considered for clustering
    """
    # Update outputs
    return cls0.update_clusters(data, eps_input, min_samples_input, min_width_cls_input, min_height_cls_input)


@callback(
    Output("box_images1", "children", allow_duplicate=True),
    Output("images_showing_txt1", "children", allow_duplicate=True),
    Output("showing_cluster0", "children", allow_duplicate=True),
    Output("box_images1_left", "disabled", allow_duplicate=True),
    Output("box_images1_right", "disabled", allow_duplicate=True),
    Output("box_images1_left", "style", allow_duplicate=True),
    Output("box_images1_right", "style", allow_duplicate=True),
    Output("update_components", "data", allow_duplicate=True),
    Input("dropdown_cls", "value"),
    State("enable_click_on_image", "data"),
    prevent_initial_call=True
)
def update_image_box_cls(value, data):
    """
    Callback function that updates the images shown in the clustering section in accordance with
    newly the selected cluster.

    :param value: selected cluster
    :param data: whether to enable clickable images
    """
    # Update outputs
    return cls0.update_image_box(value, data)


@callback(
    Output("name_of_cluster", "value", allow_duplicate=True),
    Output("dropdown_merge_change", "children", allow_duplicate=True),
    Output("selected_image_div_cls", "children", allow_duplicate=True),
    Input("update_components", "data"),
    prevent_initial_call=True
)
def update_components_change_cls(_):
    """
    Callback function that updates the components of editing clusters when a new cluster is selected.
    """
    # Update outputs
    return cls0.update_components_on_cls_change()


@callback(
    Output("div_cls_right", "children", allow_duplicate=True),
    Output("enable_click_on_image", "data", allow_duplicate=True),
    Output("button3", "disabled", allow_duplicate=True),
    Output("button3", "style", allow_duplicate=True),
    Output("showing_cluster0", "children", allow_duplicate=True),
    Input("button_continue_clusters", "n_clicks"),
    prevent_initial_call=True
)
def continue_button_cls(n_clicks):
    """
    Callback function that update the layout to start editing clusters when clicked
    on the 'continue' button.

    :param n_clicks: number of clicks on 'continue' button
    """
    # Update outputs
    return cls0.continue_to_edit(n_clicks)


@callback(
    Output("cls0", "figure", allow_duplicate=True),
    Output("dropdown_change", "children", allow_duplicate=True),
    Output("dropdown_merge_change", "children", allow_duplicate=True),
    Output("selected_image_div_cls", "children", allow_duplicate=True),
    Input("button_change_name", "n_clicks"),
    State("name_of_cluster", "value"),
    prevent_initial_call=True
)
def update_name_cls(n_clicks, name):
    """
    Callback function that updates the name of the currently selected cluster
    when clicked on the 'change name' button.

    :param n_clicks: number of clicks on 'change name' button
    :param name: name of to change to
    """
    # If the input matches the expected format
    if re.fullmatch(r"[A-Za-z_ ]*", name):
        # Update outputs
        return cls0.change_name(n_clicks, name)
    # No update
    return dash.no_update, dash.no_update, dash.no_update, dash.no_update


@callback(
    Output("cls0", "figure", allow_duplicate=True),
    Output("dropdown_change", "children", allow_duplicate=True),
    Output("dropdown_merge_change", "children", allow_duplicate=True),
    Output("button_merge_clusters", "disabled", allow_duplicate=True),
    Output("button_merge_clusters", "style", allow_duplicate=True),
    Output("selected_image_div_cls", "children", allow_duplicate=True),
    Input("button_merge_clusters", "n_clicks"),
    State("dropdown_cls_merge", "value"),
    prevent_initial_call=True
)
def merge_cls(n_clicks, cls):
    """
    Callback function that merges two clusters when the 'merge' button is clicked on.

    :param n_clicks: number of clicks on 'merge' button
    :param cls: cluster to merge the currently selected cluster with
    """
    # Update outputs
    return cls0.merge_clusters(n_clicks, cls)


@callback(
    Output("add_cluster", "data", allow_duplicate=True),
    Input("cls0", "selectedData"),
    State("enable_click_on_image", "data"),
    prevent_initial_call=True
)
def callback_new_cluster(selected_data, data):
    """
    Callback function that updates a trigger to create a new cluster when 'box select' or 'lasso select'
    is used within the figure in the clustering section.

    :param selected_data: contains the selected points to create a new cluster with.
    :param data: whether clickable images are enabled (thus in edit mode)
    """
    # Checks if we are in edit mode
    if data is not None and data == True:
        # Update trigger with selected data
        return {"selectedData": selected_data}
    # No update
    return dash.no_update


@callback(
    Output("cls0", "figure", allow_duplicate=True),
    Output("dropdown_change", "children", allow_duplicate=True),
    Output("selected_image_div_cls", "children", allow_duplicate=True),
    Input("add_cluster", "data"),
    prevent_initial_call=True
)
def new_cluster(data):
    """
    Callback function that creates a new cluster when the add_cluster trigger is updated.

    :param data: the selected points to create a new cluster with
    """
    # Add a new cluster only if the data contains the selected points
    if data is not None:
        # Update outputs
        return cls0.add_new_cluster(data["selectedData"])
    # No update
    return dash.no_update, dash.no_update, dash.no_update


@callback(
    Output("update_click_on_image", "data", allow_duplicate=True),
    Input({"type": "image-click1", "index": ALL}, "n_clicks"),
    State("enable_click_on_image", "data"),
    prevent_initial_call=True
)
def on_image_click_cls(n_clicks, data):
    """
    Callback function that updates the trigger to update the layout such that the user can move
    the face to any other cluster when an image has been clicked on.

    :param n_clicks: number of clicks on a specific image
    :param data: whether clickable images are enabled (thus in edit mode)
    """
    # No image was clicked, just initialized
    if not ctx.triggered_id or all(click is None or click == 0 for click in n_clicks):
        # No update
        return dash.no_update

    # Checks if we are in edit mode
    if data is not None and data == True:
        # Collect the image on which was clicked
        index = ctx.triggered_id["index"]
        # Update output
        return index
    # No update
    return dash.no_update


@callback(
    Output("selected_image_div_cls", "children", allow_duplicate=True),
    Input("update_click_on_image", "data"),
    prevent_initial_call=True
)
def update_click_on_image_cls(data):
    """
    Callback function that updates the layout such that the clicked image can be moved to another
    cluster when the update_click_on_image trigger is updated.

    :param data: the index of the image clicked on
    """
    return cls0.select_image(data)


@callback(
    Output("cls0", "figure", allow_duplicate=True),
    Output("selected_image_div_cls", "children", allow_duplicate=True),
    Output("box_images1", "children", allow_duplicate=True),
    Output("images_showing_txt1", "children", allow_duplicate=True),
    Output("box_images1_left", "disabled", allow_duplicate=True),
    Output("box_images1_right", "disabled", allow_duplicate=True),
    Output("box_images1_left", "style", allow_duplicate=True),
    Output("box_images1_right", "style", allow_duplicate=True),
    Input("button_move", "n_clicks"),
    State("dropdown_move_to_cls", "value"),
    State("update_click_on_image", "data"),
    prevent_initial_call=True
)
def move_face_to_cls(n_clicks, value, index):
    """
    Callback function to move the currently selected face to the inputted cluster when the 'move'
    button is clicked on.

    :param n_clicks: number of clicks on 'move' button
    :param value: selected cluster to move the face to
    :param index: index of the selected face to move
    """
    # Update outputs
    return cls0.move_face(n_clicks, value, index)


@callback(
    Output("button3", "style", allow_duplicate=True),
    Output("trigger_button3", "data", allow_duplicate=True),
    Input("button3", "n_clicks"),
    State("trigger_button3", "data"),
    prevent_initial_call=True
)
def disable_btn3(n_clicks, data):
    """
    Callback function to disable button3 (save to database) when clicked on and updates the trigger
    to save the database.

    :param n_clicks: number of clicks on 'save to database' button
    :param data: trigger data to update
    """
    # If button3 is clicked on
    if n_clicks is not None and n_clicks > 0:
        # Update style to show the user that the button is disabled
        style = {
            "padding": "10px 20px",
            "fontSize": "16pt",
            "borderRadius": "12px",
            "border": "none",
            "backgroundColor": "#2196F3",
            "color": "white",
            "cursor": "pointer",
            "width": "20vw",
            "marginBottom": "20px",
            "opacity": 0.5
        }

        # Update the trigger data
        if data is None:
            data = 0
        else:
            data += 1

        # Update outputs
        return style, data
    # No update
    return dash.no_update, dash.no_update


@callback(
    Output("button3", "style", allow_duplicate=True),
    Output("successful_database", "children", allow_duplicate=True),
    Input("trigger_button3", "data"),
    State("name_of_db", "value"),
    prevent_initial_call=True
)
def btn_click_save_to_db(data, value):
    """
    Callback function to save the datasets to the database when the trigger_button3 trigger is updated.

    :param data: trigger data that trigger this callback
    :param value: inputted name of the database to save to
    """
    # If the trigger was updated
    if data is not None:
        # Check if inputted name matches the expected format
        if re.fullmatch(r"[A-Za-z_]*", f"{value}"):
            # Save the datasets to the database
            name = save_to_db(df0.copy(), cls0.df_faces.copy(), f"{value}")

            # Update the style of button3 to show the user it has been enabled again
            style = {
                "padding": "10px 20px",
                "fontSize": "16pt",
                "borderRadius": "12px",
                "border": "none",
                "backgroundColor": "#4CAF50",
                "color": "white",
                "cursor": "pointer",
                "width": "20vw",
                "marginBottom": "20px"
            }

            # Feedback to show to the user
            txt = f"Success! Database saved as '{name}'"

            # Update outputs
            return style, txt
        else:  # Inputted name does not match the expected format
            # Update the style of button3 to show the user it has been enabled again (red)
            style = {
                "padding": "10px 20px",
                "fontSize": "16pt",
                "borderRadius": "12px",
                "border": "none",
                "backgroundColor": "#F94449",
                "color": "white",
                "cursor": "pointer",
                "width": "20vw",
                "marginBottom": "20px"
            }

            # Feedback to show to the user
            txt = "The inputted name is not in the expected format."

            # Update outputs
            return style, txt
    # No update
    return dash.no_update, dash.no_update
