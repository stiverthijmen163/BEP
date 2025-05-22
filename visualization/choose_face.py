import plotly.express as px
import dash
from dash import dcc, html
from visualization.functions import *


class ChooseFaceSection(html.Div):
    """
    Contains all functionalities of the 'face of interest' section for the 'visualization' page.
    """
    def __init__(self, name):
        """
        Initializes the ChooseFaceSection object (no database has been loaded).

        :param name: id of the ChooseFaceSection object
        """
        # Initialize the object's parameters
        self.html_id = name  # Id
        self.df_main = None  # Dataframe containing all info regarding the main images
        self.df_face = None  # Dataframe containing all faces in all images
        self.df_faces_uploaded = None  # Dataframe containing all faces in a uploaded image
        self.img = None  # The currently loaded image
        self.show_nrs = False  # Whether to show face number in the currently loaded image or not

        # Initialize ChooseFaceSection as an empty Div
        super().__init__(className="graph_card", id=self.html_id, children=[])


    def initialize_options(self, main, faces):
        """
        Initializes the ChooseFaceSection object as a non-empty div (database has been loaded).

        :param main: dataframe containing all info regarding the main images
        :param faces: dataframe containing all faces in all images
        """
        # Update the dataframes
        self.df_main = main.copy()
        self.df_face = faces.copy()

        # Update the layout to allow the user to select a face
        res = [
            html.Div(
                style={  # Work within a slightly smaller space
                    "display": "flex",
                    "justifyContent": "center",
                    "alignItems": "flex-start",
                    "gap": "10px",
                    "width": "99vw",
                    "margin": "0 auto",
                    "textAlign": "center",
                    "marginTop": "20px",
                    "textFont": "16pt"
                },
                children = [
                    html.Div(  # The left half
                        id="div_detect_left",
                        style={
                            "backgroundColor": "white",
                            "width": "50%",
                            "borderRadius": "12px",
                            "border": "2px black solid",
                            "textAlign": "center",
                            "height": "23vw"
                        },
                        children=[
                            html.P(  # Explanatory text
                                "Select a person of interest (cluster) from the database, or upload an image to select a person from.",
                                style={"fontSize": "16pt", "fontWeight": "bold"}),
                            html.Div([  # Allows components to be next to each other
                                dcc.Upload(  # Button to upload images
                                    id="upload_image",
                                    children=html.Button("📁 Select Image (png/jpg)")
                                ),
                                dcc.Checklist(  # Checkbox to show face numbers or not
                                    id="show_face_nrs",
                                    options=[
                                        {"label": " Show face numbers",
                                         "value": "boxes"}
                                    ],
                                    value=["boxes"] if self.show_nrs else [],
                                    style={"fontSize": "16pt"}
                                )],
                                style={
                                    "display": "flex",
                                    "justifyContent": "center",
                                    "gap": "1vw"
                                }
                            ),
                            html.Div(  # Allows components to be next to each other
                                style={
                                    "display": "flex",
                                    "justifyContent": "center",
                                    "gap": "1%",
                                    "height": "17vw",
                                    "width": "99%",
                                    "marginTop": "20px",
                                },
                                children=[
                                    dcc.Graph(  # Figure to display selected image in
                                        id=f"{self.html_id}_fig",
                                        style={"flex": "1",
                                               "height": "100%",
                                               "width": "70%",}
                                    ),
                                    html.Div(  # Scrollable column for radio menu to select the face you want to use
                                        id="div_radio_selected_face",
                                        style={
                                            "textAlign": "left",
                                            "margin": "0.5%",
                                            "width": "30%",
                                            "overflowY": "scroll",
                                            "border": "1px solid lightgray",
                                            "borderRadius": "8px",
                                            "padding": "10px",
                                        },
                                        children=[
                                            html.Div(
                                                dcc.RadioItems(  # Radio menu to select at most one face
                                                    id="radio_selected_face",
                                                    options=[f" {i}" for i in
                                                             sort_items(self.df_face["name"].unique())],
                                                    labelStyle={"fontSize": "16pt"},
                                                )
                                            )
                                        ]
                                    )
                                ]
                            )
                        ]
                    ),
                    html.Div(  # Right half placeholder
                        id="div_detect_right",
                        style={
                            "backgroundColor": "white",
                            "width": "50%",
                            "borderRadius": "12px",
                            "border": "2px black solid",
                            "textAlign": "center",
                            "height": "23vw"
                        },
                    )
                ]
            )
        ]

        return res


    def update_uploaded_image(self, c):
        """
        Updates the image figure to contain a newly uploaded image and
        updates the other component in the left half of this object accordingly.
        """
        # Decode the image
        content_type, content_string = c.split(",")
        decoded = base64.b64decode(content_string)

        # Convert the image into cv2-format
        nparr = np.frombuffer(decoded, np.uint8)
        self.img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # Load in the model to detect faces with
        model = YOLO("../face_detection/yolo_v12s/yolov12s-face/weights/epoch60.pt")

        # Detect faces in the image
        _, detections = detect_faces(self.img.copy(), model)

        # Created a dataframe containing those faces
        self.df_faces_uploaded = pd.DataFrame({"face": detections, "use": True})
        self.df_faces_uploaded["nr"] = self.df_faces_uploaded.index

        # Update outputs (components left half of ChooseFaceSection object)
        return self.update_left_half()


    def update_show_nrs_val(self, value):
        """
        Updates whether to plot the face numbers or not, updates the other components accordingly.
        """
        # Update the value whether to show face numbers or not
        self.show_nrs = True if value != [] else False

        # Check if an images is loaded or not
        if self.img is not None:  # Loaded
            # Update the components
            fig, options, _ = self.update_left_half()

            # Update outputs
            return fig, options, dash.no_update
        # No update
        return dash.no_update, dash.no_update, dash.no_update


    def update_left_half(self):
        """
        Updates the left half of the ChooseFaceSection object
        such that the image figure and the radio options are up to date.
        """
        # Plot all faces on the image
        img_w_faces = plot_faces_on_img_opacity(self.img.copy(), self.df_faces_uploaded.copy(), self.show_nrs)

        # Update the figure
        fig = px.imshow(cv2.cvtColor(img_w_faces, cv2.COLOR_BGR2RGB))
        fig.update_layout(
            margin=dict(l=0, r=0, t=0, b=0),
            xaxis=dict(showticklabels=False),
            yaxis=dict(showticklabels=False),
            dragmode="select"
        )

        # Update the options in the radio menu
        options = ([f" Face {row['nr']}" for _, row in self.df_faces_uploaded.iterrows()]
                   + [f" {i}" for i in sort_items(self.df_face["name"].unique())])

        # Update outputs
        return fig, options, None
