from dash import html, dcc
from app import app
import dash


# The initial layout of the app
app.layout = html.Div(
    style={  # Set the background color for all pages
        "backgroundColor": "#f0f8ff",
        "minHeight": "100vh",
        "margin": "0",
        "padding": "0",
        "boxSizing": "border-box",
    },
    children=[
        dcc.Location(id="url"),  # Needed for navigation

        # Navigation bar
        html.Div(
            style={
                "display": "flex",
                  "justifyContent": "center",
                  "gap": "5%",
                  "padding": "20px",
                  "backgroundColor": "#e0e0e0"
            },
            children=[  # Buttons to pages
                dcc.Link(
                    html.Button(  # First page
                        "New Data",
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
                    ),
                    href="/",
                    refresh=False
                ),
                dcc.Link(
                    html.Button(  # Second page
                        "Visualization",
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
                    ),
                    href="/page2",
                    refresh=False
                )
            ]
        ),
        html.Div(
            children=[
                dash.page_container
            ]
        )
    ]
)


if __name__ == "__main__":
    # Start running the app
    app.run(debug=False)