from dash import Dash
import dash_bootstrap_components as dbc


# Set the necessary styles
external_stylesheets = [
    {
        "href": "https://fonts.googleapis.com/css2?family=Roboto&display=swap",
        "rel": "stylesheet",
    },
    {
        "rel": "stylesheet",
        "href": "https://cdnjs.cloudflare.com/ajax/libs/normalize/8.0.1/normalize.min.css",
    },
    "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css",  # For icons
    dbc.themes.BOOTSTRAP
]

# Initialize the app and server
app = Dash(__name__, use_pages=True, external_stylesheets=external_stylesheets)
server = app.server