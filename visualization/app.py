from dash import Dash

external_stylesheets = [
    {
        "href": "https://fonts.googleapis.com/css2?family=Roboto&display=swap",
        "rel": "stylesheet",
    },
    {
        "rel": "stylesheet",
        "href": "https://cdnjs.cloudflare.com/ajax/libs/normalize/8.0.1/normalize.min.css",
    },
]

app = Dash(__name__, use_pages=True, external_stylesheets=external_stylesheets)
server = app.server  # For deployment