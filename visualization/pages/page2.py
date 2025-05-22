import dash
from dash import html, register_page, dcc, callback, Input, Output, State
from visualization.functions import *
from visualization.choose_face import ChooseFaceSection

# Initialize global dataframe
data_main = None
data_face = None

cfs0 = ChooseFaceSection("cfs0")

# Set page
register_page(__name__, path="/page2")

# Initialize the layout for this page
layout = html.Div([
    html.Div(
        style={"textAlign": "center"},
        children=[
            html.H2("Face of Interest", style={"marginTop": "20px"}),  # Header
            html.Hr(),
            html.Div(
                style={
                    "backgroundColor": "#dbeafe",
                    "padding": "10px",
                    "margin": "20px auto",
                    "width": "100vw",
                    "borderRadius": "12px",
                    "boxShadow": "0px 2px 5px rgba(0,0,0,0.1)",
                    "textAlign": "center"
                },
                children=[
                    html.Div(
                        style={
                            "display": "flex",
                            "justifyContent": "center",
                            "alignItems": "center",
                            "gap": "1vw",
                            # "marginBottom": "10px"
                        },
                        children=[
                            html.Label("Choose database:", style={"fontSize": "16pt"}),
                            dcc.Dropdown(
                                id="dropdown_choose_database",
                                options=sort_items(os.listdir("databases")),
                                value=None,
                                clearable=False,
                                className="dropdown",
                                searchable=True,
                                style={"width": "40vw"}
                            ),
                            html.Button(
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
                    html.Div(id="feedback_load_data"),#, style={"marginTop": "20px", "fontWeight": "bold", "textFont": "16pt"}),
                    cfs0
                ]
            )
        ]
    ),
    dcc.Store(id="trigger_read_database"),
])


def read_database(database_name):
    print(database_name)

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
        data_main["id"] = data_main["id"].apply(str)
        try:
            data_main["img"] = data_main["img"].apply(base64_to_img)
        except Exception as e:
            print(f"Error trying to convert images from base64 to img: {e}")
            error_txt.append("Column 'img' is not in the expected format in 'main'.")
        for column in data_main.columns:
            if column == "url":
                data_main[column] = data_main[column].apply(str)
            elif column not in ["id", "img"]:
                try:
                    data_main[column] = data_main[column].apply(json.loads)
                except Exception as e:
                    print(f"Error trying to load '{column}': {e}")
                    error_txt.append(f"Column '{column}' is not in the expected format in 'main'.")
    else:  # Check which columns are missing
        missing_cols = [col for col in ["id", "img"] if col not in data_main.columns]
        error_txt.append(f"Missing column(s) in 'main': {', '.join(missing_cols)}")

    # ---------------------------------------------------- FACE DATA ---------------------------------------------------
    data_face["embedding_tsne"] = data_face["embedding_tsne"].apply(lambda x: np.fromstring(x, sep=","))
    data_face["embedding"] = data_face["embedding"].apply(lambda x: np.fromstring(x, sep=","))
    data_face["face"] = data_face["face"].apply(json.loads)
    data_face["img"] = data_face["img"].apply(base64_to_img)

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
    if n_clicks is not None and n_clicks > 0:
        if data is None:
            data = 0
        else:
            data += 1

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

        return data, True, style
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

    if data is not None and value is not None and value != "":
        global data_face, data_main
        errors = read_database(value)

        if len(errors) > 0:  # Return errors is there are any
            children = []
            for e in errors:
                children.append(html.P(e))

            # Reset the datasets
            data_face = None
            data_main = None
        else:  # No errors
            children = []

        return False, style, children, cfs0.initialize_options(data_main, data_face)
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
    if filename.endswith(".jpg") or filename.endswith(".png"):
        return cfs0.update_uploaded_image(contents)
    return dash.no_update, dash.no_update, dash.no_update


@callback(
    Output(f"{cfs0.html_id}_fig", "figure", allow_duplicate=True),
    Output("radio_selected_face", "options", allow_duplicate=True),
    Output("radio_selected_face", "value", allow_duplicate=True),
    Input("show_face_nrs", "value"),
    prevent_initial_call=True
)
def update_show_nrs(value):
    return cfs0.update_show_nrs_val(value)