from dash import html, dcc
from app import app
import dash

app.layout = html.Div(
    style={
        'backgroundColor': '#f0f8ff',
        'minHeight': '100vh',
        'margin': '0',
        'padding': '0',
        'boxSizing': 'border-box',
    },
    children=[
        dcc.Location(id='url'),  # Needed for navigation

        # Navigation bar
        html.Div(
            style={
                'display': 'flex',
                'justifyContent': 'center',
                'gap': '5%',
                'padding': '20px',
                'backgroundColor': '#e0e0e0',
            },
            children=[
                dcc.Link(
                    html.Button(
                        "New Data",
                        style={
                            'padding': '10px 20px',
                            'fontSize': '16pt',
                            'borderRadius': '12px',
                            'border': 'none',
                            'backgroundColor': '#2196F3',
                            'color': 'white',
                            'cursor': 'pointer',
                            'width': "10vw"
                        }
                    ),
                    href='/',
                    refresh=True
                ),
                dcc.Link(
                    html.Button(
                        "Visualization",
                        style={
                            'padding': '10px 20px',
                            'fontSize': '16pt',
                            'borderRadius': '12px',
                            'border': 'none',
                            'backgroundColor': '#2196F3',
                            'color': 'white',
                            'cursor': 'pointer',
                            "width": "10vw"
                        }
                    ),
                    href='/page2',
                    refresh=True
                )
            ]
        ),

        html.Div(
            # style={
            #     # 'padding': '40px',
            #     'textAlign': 'center',
            # },
            children=[
                dash.page_container
            ]
        )
    ]
)


if __name__ == "__main__":
    app.run(debug=False)