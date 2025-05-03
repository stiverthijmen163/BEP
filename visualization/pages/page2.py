from dash import html, register_page

register_page(__name__, path="/page2")

layout = html.Div([
    html.H2("Page 2"),
    html.P("Welcome to the second page!")
])