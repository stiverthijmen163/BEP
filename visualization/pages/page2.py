import dash
from dash import html, register_page, dcc, callback, Input, Output, State
from visualization.functions import *
from visualization.choose_face import ChooseFaceSection
from visualization.explore_selected_cluster import ExploreSelectedCluster

# Initialize global dataframe
data_main = None
data_face = None

# Initialize the section where the user can choose a person of interest (poi)
cfs0 = ChooseFaceSection("cfs0")
esc0 = ExploreSelectedCluster("esc0")

# Set page
register_page(__name__, path="/page2")

# Initialize the layout for this page
layout = html.Div([
    html.Div(
        style={"textAlign": "center"},
        children=[
            html.H2("Face of Interest", style={"marginTop": "20px"}),  # Header
            html.Hr(),
            html.Div(  # Blue box for the first section (choose poi)
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
                    html.Div(  # Allows components to be next to each other
                        style={
                            "display": "flex",
                            "justifyContent": "center",
                            "alignItems": "center",
                            "gap": "1vw"
                        },
                        children=[  # Choose database to load in
                            html.Label("Choose database:", style={"fontSize": "16pt"}),
                            dcc.Dropdown(  # All db options
                                id="dropdown_choose_database",
                                options=sort_items(os.listdir("databases")),
                                value=None,
                                clearable=False,
                                className="dropdown",
                                searchable=True,
                                style={"width": "40vw"}
                            ),
                            html.Button(  # Button to trigger loading the db
                                "Load",
                                disabled=False,
                                id="button_load_db",
                                style={
                                    "padding": "10px 20px",
                                    "fontSize": "16pt",
                                    "borderRadius": "12px",
                                    "border": "none",
                                    "backgroundColor": "#2196F3",
                                    "color": "white",
                                    "cursor": "pointer",
                                    "width": "10vw"
                                }
                            )
                        ]
                    ),
                    html.Div(id="feedback_load_data"),  # Placeholder for feedback in case of errors
                    cfs0  # the section to choose the poi
                ]
            )
        ]
    ),
    dcc.Store(id="trigger_read_database"),  # Triggers loading the database when 'button_load_db' disables
    esc0  # Section where you can explore the selected cluster (poi)
])


def read_database(database_name: str) -> List[str]:
    """
    Reads a database and checks if all expected columns and types are present,
    assumes that 'main' and 'faces' tables are present.

    :param database_name: name of the database

    :return: list of errors occurred while trying to read the database
    """
    print(f"Loading database: '{database_name}'")

    # Make the datasets accessible and editable
    global data_main, data_face

    # Connect to the database
    conn = sqlite3.connect(f"databases/{database_name}")

    # Set the queries to read the data
    q_main = """SELECT * FROM main"""
    q_face = """SELECT * FROM faces"""

    # Load the data
    data_main = pd.read_sql_query(q_main, conn, index_col="index")
    data_face = pd.read_sql_query(q_face, conn, index_col="index")

    # Initialize list of errors
    error_txt = []

    # ---------------------------------------------------- MAIN DATA ---------------------------------------------------
    # Check if expected columns are present
    if {"id", "img"}.issubset(data_main.columns):
        # Make sure 'id' column is in the right format
        data_main["id"] = data_main["id"].apply(str)

        # Check if 'img' column is in the expected format
        try:
            data_main["img"] = data_main["img"].apply(base64_to_img)
        except Exception as e:
            print(f"Error trying to convert images from base64 to img: {e}")
            error_txt.append("Column 'img' is not in the expected format in 'main'.")

        # Remaining columns
        for column in data_main.columns:
            if column == "url":  # Make sure 'url' column is in the right format
                data_main[column] = data_main[column].apply(str)
            elif column not in ["id", "img"]:
                # Check if column is in the expected format
                try:
                    data_main[column] = data_main[column].apply(json.loads)
                except Exception as e:
                    print(f"Error trying to load '{column}': {e}")
                    error_txt.append(f"Column '{column}' is not in the expected format in 'main'.")
    else:  # Check which columns are missing
        missing_cols = [col for col in ["id", "img"] if col not in data_main.columns]
        error_txt.append(f"Missing column(s) in 'main': {', '.join(missing_cols)}")

    # ---------------------------------------------------- FACE DATA ---------------------------------------------------
    # Convert columns to expected format
    data_face["embedding_tsne"] = data_face["embedding_tsne"].apply(lambda x: np.fromstring(x, sep=","))
    data_face["embedding"] = data_face["embedding"].apply(lambda x: np.fromstring(x, sep=","))
    data_face["face"] = data_face["face"].apply(json.loads)
    data_face["img"] = data_face["img"].apply(base64_to_img)

    # Return the list of errors
    return error_txt


@callback(
    Output("trigger_read_database", "data", allow_duplicate=True),
    Output("button_load_db", "disabled", allow_duplicate=True),
    Output("button_load_db", "style", allow_duplicate=True),
    Input("button_load_db", "n_clicks"),
    State("trigger_read_database", "data"),
    prevent_initial_call=True
)
def trigger_select_new_database(n_clicks, data):
    """
    Callback function that disables the 'button_load_db' button when pressed
    and updates the trigger to read the database.

    :param n_clicks: number of clicks on the 'button_load_db' button
    :param data: the data of the trigger to update
    """
    # Check if function call is triggered by clicking on 'button_load_db' button
    if n_clicks is not None and n_clicks > 0:
        # Update the trigger's data
        if data is None:
            data = 0
        else:
            data += 1

        # Update the style of the button to show the user it has been disabled
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
        return data, True, style
    # No update
    return dash.no_update, dash.no_update


@callback(
    Output("button_load_db", "disabled", allow_duplicate=True),
    Output("button_load_db", "style", allow_duplicate=True),
    Output("feedback_load_data", "children", allow_duplicate=True),
    Output(cfs0.html_id, "children", allow_duplicate=True),
    Input("trigger_read_database", "data"),
    State("dropdown_choose_database", "value"),
    prevent_initial_call=True
)
def select_new_database(data, value):
    """
    Callback function that reads date from the selected database and enables the 'button_load_db' button.

    :param data: the data of the trigger that calls this function
    :param value: the name of the database to read
    """
    # Update the style of the 'button_load_db' button to show the user it is enabled
    style = {
        "padding": "10px 20px",
        "fontSize": "16pt",
        "borderRadius": "12px",
        "border": "none",
        "backgroundColor": "#2196F3",
        "color": "white",
        "cursor": "pointer",
        "width": "10vw"
    }

    # Check if a database has been selected
    if data is not None and value is not None and value != "":
        # Make the datasets accessible and editable
        global data_face, data_main

        # Read the database and collect the errors
        errors = read_database(value)

        # Return errors if there are any
        if len(errors) > 0:
            # Show all errors
            children = []
            for e in errors:
                children.append(html.P(e))

            # Reset the datasets
            data_face = None
            data_main = None
        else:  # No errors
            # No errors shown
            children = []

        # Update outputs
        return False, style, children, cfs0.initialize_options(data_main, data_face)
    # No update
    return False, style, dash.no_update, dash.no_update


@callback(
    Output(f"{cfs0.html_id}_fig", "figure", allow_duplicate=True),
    Output("radio_selected_face", "options", allow_duplicate=True),
    Output("radio_selected_face", "value", allow_duplicate=True),
    Input("upload_image", "contents"),
    Input("upload_image", "filename"),
    prevent_initial_call=True
)
def uploaded_image(contents, filename):
    """
    Callback function that updates the image figure to correspond with the uploaded image,
    and updates the radio menu with the newly detected faces.

    :param contents: the contents of the uploaded image
    :param filename: the name of the uploaded image
    """
    # Check if uploaded file is an image in .jpg or .png format
    if filename.endswith(".jpg") or filename.endswith(".png"):
        # Update outputs
        return cfs0.update_uploaded_image(contents)
    # No update
    return dash.no_update, dash.no_update, dash.no_update


@callback(
    Output(f"{cfs0.html_id}_fig", "figure", allow_duplicate=True),
    Output("radio_selected_face", "options", allow_duplicate=True),
    Output("radio_selected_face", "value", allow_duplicate=True),
    Input("show_face_nrs", "value"),
    prevent_initial_call=True
)
def update_show_nrs(value):
    """
    Callback function that updates the image figure to enable/disabled showing the face numbers.

    :param value: the value of the show_face_nrs checkbox
    """
    # Update outputs
    return cfs0.update_show_nrs_val(value)


@callback(
    Output("div_detect_right", "children", allow_duplicate=True),
    Input("radio_selected_face", "value"),
    prevent_initial_call=True
)
def update_poi(value):
    """
    Callback function that updates the selected poi according to the selected value in the radio menu,
    this may be a cluster (use cluster) or a face (predict the corresponding cluster).

    :param value: the selected poi from the radio menu
    """
    # Update outputs
    return cfs0.update_right_half(value)


@callback(
    Output(f"{cfs0.html_id}_fig", "figure", allow_duplicate=True),
    Output("radio_selected_face", "options", allow_duplicate=True),
    Output("radio_selected_face", "value", allow_duplicate=True),
    Input(f"{cfs0.html_id}_fig", "selectedData"),
    prevent_initial_call=True
)
def new_face_selected(selectedData):
    """
    Callback function that updates the image figure to correspond with the uploaded image,
    and updates the radio menu with the newly selected face when a selection is made in the figure.
    """
    # Update outputs
    return cfs0.add_face(selectedData)


@callback(
    Output("button_continue_to_results", "disabled", allow_duplicate=True),
    Output("button_continue_to_results", "style", allow_duplicate=True),
    Input("button_continue_to_results", "n_clicks"),
    prevent_initial_call=True
)
def disable_to_results_button(n_clicks):
    """
    Callback function that disables the 'show results' button when clicked.

    :param n_clicks: the number of clicks of the 'show results' button
    """
    # Check if button click triggered this callback
    if n_clicks is not None and n_clicks > 0:
        # Update style of the 'show results' button to show the user it is disabled
        style = {
            "padding": "10px 20px",
            "fontSize": "16pt",
            "borderRadius": "12px",
            "border": "none",
            "backgroundColor": "#2196F3",
            "color": "white",
            "cursor": "pointer",
            "width": "10vw",
            "marginTop": "1vw",
            "opacity": 0.5
        }

        # Update outputs
        return True, style
    # No update
    return dash.no_update, dash.no_update


@callback(
    Output(esc0.html_id, "children", allow_duplicate=True),
    Output("button_continue_to_results", "disabled", allow_duplicate=True),
    Output("button_continue_to_results", "style", allow_duplicate=True),
    Input("button_continue_to_results", "disabled"),
    prevent_initial_call=True
)
def initialize_results(disabled):
    """
    Callback function that initializes the results sections when
    the 'show results' button is disabled (thus clicked on)

    :param disabled: whether the 'show results' button is disabled or not
    """
    # Check if the 'show results' button is disabled
    if disabled:
        print("initializing")

        # Make the datasets accessible
        global data_face, data_main

        # Update the style of the 'show results' button to show the user it is enabled
        style = {
            "padding": "10px 20px",
            "fontSize": "16pt",
            "borderRadius": "12px",
            "border": "none",
            "backgroundColor": "#2196F3",
            "color": "white",
            "cursor": "pointer",
            "width": "10vw",
            "marginTop": "1vw"
        }

        # Update outputs
        return esc0.initialize_esc(data_main, data_face, cfs0.selected_cluster), False, style
    # No update
    return dash.no_update, dash.no_update, dash.no_update


@callback(
    Output("example_images_cls_box", "children", allow_duplicate=True),
    Output("box_ex_images_left", "disabled", allow_duplicate=True),
    Output("box_ex_images_right", "disabled", allow_duplicate=True),
    Output("box_ex_images_left", "style", allow_duplicate=True),
    Output("box_ex_images_right", "style", allow_duplicate=True),
    Output("images_showing_ex_cls_txt", "children", allow_duplicate=True),
    Input("box_ex_images_right", "n_clicks"),
    prevent_initial_call=True
)
def next_button_examples_cls(n_clicks):
    """
    Callback function that processes the click on the next button for showing the next 8 images
    in the 'explore cluster' section.

    :param n_clicks: number of clicks on 'next' button
    """
    # Update output
    return esc0.show_next_example_images_cls(n_clicks)


@callback(
    Output("example_images_cls_box", "children", allow_duplicate=True),
    Output("box_ex_images_left", "disabled", allow_duplicate=True),
    Output("box_ex_images_right", "disabled", allow_duplicate=True),
    Output("box_ex_images_left", "style", allow_duplicate=True),
    Output("box_ex_images_right", "style", allow_duplicate=True),
    Output("images_showing_ex_cls_txt", "children", allow_duplicate=True),
    Input("box_ex_images_left", "n_clicks"),
    prevent_initial_call=True
)
def previous_button_examples_cls(n_clicks):
    """
    Callback function that processes the click on the 'back' button for showing the previous 8 images
    in the 'explore cluster' section.

    :param n_clicks: number of clicks on 'back' button
    """
    # Update output
    return esc0.show_previous_example_images_cls(n_clicks)
