import plotly.express as px
import dash
from dash import dcc, html
from functions import *
import re
from PIL import Image
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC, LinearSVC
import face_recognition


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
        self.df_faces_uploaded = None  # Dataframe containing all faces in an uploaded image
        self.df_face_selected_cluster = None  # Subset of df_face containing only those within the selected cluster
        self.img = None  # The currently loaded image
        self.show_nrs = False  # Whether to show face number in the currently loaded image or not
        self.selected_poi = None  # Keeps track of the selected person of interest (poi)
        self.selected_cluster = None  # Keeps track of the cluster corresponding to the poi
        self.pipe_svc = None  # Face recognition model

        # Initialize ChooseFaceSection as an empty Div
        super().__init__(className="graph_card", id=self.html_id, children=[])


    def initialize_options(self, main, faces):
        """
        Initializes the ChooseFaceSection object as a non-empty div (database has been loaded).

        :param main: dataframe containing all info regarding the main images
        :param faces: dataframe containing all faces in all images
        """
        print("(Choose POI)          - Initializing ChooseFaceSection object")

        # Update the dataframes
        self.df_main = main.copy()
        self.df_face = faces.copy()

        # Set the nr of pixels for each face, used to find the largest face to display as example
        self.df_face["nr_of_pixels"] = self.df_face["width"] * self.df_face["height"]

        # Initialize a simple face recognition model
        self.pipe_svc = Pipeline(
            [("scaler", StandardScaler()), ("svc_rbf", SVC(kernel="rbf", random_state=7, probability=True))])

        # Set the training data, embeddings as data and cluster names as classes
        X = self.df_face[self.df_face["name"] != "unknown"]["embedding"].copy().to_list()
        y = self.df_face[self.df_face["name"] != "unknown"]["name"].copy().to_list()

        # Train a simple face recognition model
        self.pipe_svc.fit(X, y)

        # Update the layout to allow the user to select a face
        res = [
            html.Div(
                style={  # Work within a slightly smaller space
                    "display": "flex",
                    "justifyContent": "center",
                    "alignItems": "flex-start",
                    "gap": "10px",
                    "width": "100%",
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
                                                             sort_items([j for j in self.df_face["name"].unique() if j.lower() not in ["unknown"]])],
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
                        children=html.Div(
                            html.P(  # Display text saying that no poi has been selected yet
                                "No face of interest has been selected yet",
                                style={"fontSize": "25pt", "textAlign": "center", "opacity": 0.3}
                            ),
                            style={  # Center the text
                                "display": "flex",
                                "justifyContent": "center",
                                "alignItems": "center",
                                "height": "100%"
                            }
                        )
                    )
                ]
            )
        ]

        # Output layout
        return res


    def update_uploaded_image(self, c):
        """
        Updates the image figure to contain a newly uploaded image and
        updates the other component in the left half of this object accordingly.

        :param c: uploaded image content
        """
        print("(Choose POI)          - Uploaded image, detecting faces")

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
        self.df_faces_uploaded["nr"] = self.df_faces_uploaded["nr"].apply(str)

        # Update outputs (components left half of ChooseFaceSection object)
        return self.update_left_half()


    def update_show_nrs_val(self, value):
        """
        Updates whether to plot the face numbers or not, updates the other components accordingly.

        :param value: value of show_nrs checkbox
        """
        print(f"(Choose POI)          - Updating show_nrs parameter to {True if value != [] else False}")

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


    def update_left_half(self, selected_val = None):
        """
        Updates the left half of the ChooseFaceSection object
        such that the image figure and the radio options are up to date.

        :param selected_val: the selected value of the radio menu
        """
        print(f"(Choose POI)          - Updating image figure and radio menu")

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
                   + [f" {i}" for i in sort_items([j for j in self.df_face["name"].unique() if j.lower() not in ["unknown"]])])

        # Update outputs
        return fig, options, selected_val


    def update_right_half(self, radio_value):
        """
        Updates the right half of the ChooseFaceSection object, containing the chosen/predicted poi (cluster).

        :param radio_value: the selected value of the radio menu, this can be a face from the image or a cluster
        """
        print(f"(Choose POI)          - Updating selected POI to '{radio_value}' (right half)")

        # If no poi has been selected
        if radio_value is None or radio_value == "":
            # Update selected poi to empty
            self.selected_poi = None
            self.selected_cluster = None

            # Update the layout of the right half
            children = html.Div(
                html.P(  # Display text saying that no poi has been selected yet
                    "No face of interest has been selected yet",
                    style={"fontSize": "25pt", "textAlign": "center", "opacity": 0.3}
                ),
                style={  # Center the text
                    "display": "flex",
                    "justifyContent": "center",
                    "alignItems": "center",
                    "height": "100%"
                }
            )

            # Update output
            return children
        elif re.fullmatch(r" Face \d+", radio_value):  # Poi is a selected face from the uploaded image
            # Update the selected poi to match the face in the uploaded image
            self.selected_poi = radio_value[1:]

            # Collect the bounding box of the selected face
            bounding_box = self.df_faces_uploaded[self.df_faces_uploaded["nr"] == self.selected_poi.split(" ")[1]]["face"].tolist()[0]
            x, y, w, h = bounding_box

            # Collect the image corresponding to that face
            face_img = self.img[y:y + h, x:x + w].copy()

            # Encode the selected face
            encoding = face_recognition.face_encodings(face_img, num_jitters=2, model="large",
                                                       known_face_locations=[(0, w, h, 0)])

            # Use classifier to find matching cluster
            probs = self.pipe_svc.predict_proba([encoding[0]])
            predicted_class = self.pipe_svc.classes_[np.argmax(probs)]

            # Find the largest face to display as example
            self.df_face_selected_cluster = self.df_face[self.df_face["name"] == predicted_class].copy()
            predicted_face_img = self.df_face_selected_cluster[
                self.df_face_selected_cluster["nr_of_pixels"] == max(self.df_face_selected_cluster["nr_of_pixels"])][
                "img"].copy().tolist()[0]

            # Collect the maximum probability
            certainty = np.max(probs)

            # Check if certainty meets threshold value
            if certainty >= 0.5:  # More likely to be this cluster than another
                # Update the selected cluster
                self.selected_cluster = predicted_class

                # Update layout
                children = [
                    html.Div(  # Allows components to be next to each other
                        style={
                            "display": "flex",
                            "justifyContent": "center",
                            "gap": "1%",
                            "height": "18vw",
                            "width": "99%",
                            "marginTop": "20px",
                        },
                        children=[  # Show the selected face
                            html.Div(
                                style={
                                    "width": "48%"
                                },
                                children=[
                                    html.P(f"Selected '{self.selected_poi}':", style={"fontSize": "16pt"}),
                                    html.Img(src=Image.fromarray(cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)), style={
                                        "width": "100%",
                                        "cursor": "pointer",
                                        "maxHeight": "16vw",
                                        "objectFit": "contain",
                                    }),
                                ]
                            ),
                            html.Div(  # Show an example of the predicted poi
                                style={
                                    "width": "48%",
                                    "overflow": "visible"
                                },
                                children=[
                                    html.P(f"Predicted '{predicted_class}' with {certainty:.2f} certainty:",
                                        style={
                                            "fontSize": "16pt", "overflow": "visible",
                                            "position": "relative", "whiteSpace": "nowrap"
                                        }),
                                    html.Img(src=Image.fromarray(cv2.cvtColor(predicted_face_img, cv2.COLOR_BGR2RGB)),
                                        style={
                                            "width": "100%",
                                            "cursor": "pointer",
                                            "maxHeight": "16vw",
                                            "objectFit": "contain",
                                        }
                                    ),
                                ]
                            )
                        ]
                    ),
                    html.Button(  # Button to continue to the resulting visualization
                        "Show Results",
                        disabled=False,
                        id="button_continue_to_results",
                        style={
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
                    )
                ]
            else:  # Likely that the model has not been trained on the input face
                # Update selected cluster to None
                self.selected_cluster = None

                children = html.Div(
                html.P(  # Display text saying that predicted class if likely to be wrong
                    f"Certainty of prediction was below the threshold ({certainty:.2f} < 0.50). Therefore it is likely that there exists no cluster for the selected person of interest.",
                    style={"fontSize": "25pt", "textAlign": "center", "opacity": 0.3}
                ),
                style={  # Center the text
                    "display": "flex",
                    "justifyContent": "center",
                    "alignItems": "center",
                    "height": "100%"
                }
            )

            # Update output
            return children
        else:  # Poi is a cluster from the database
            # Update the selected poi to the selected cluster
            self.selected_poi = radio_value[1:]
            self.selected_cluster = radio_value[1:]

            # Find the largest face to display as example
            self.df_face_selected_cluster = self.df_face[self.df_face["name"] == self.selected_poi].copy()
            face_img = self.df_face_selected_cluster[self.df_face_selected_cluster["nr_of_pixels"] == max(self.df_face_selected_cluster["nr_of_pixels"])]["img"].copy().tolist()[0]

            children = [  # Show the selected poi (cluster)
                html.P(f"Selected cluster '{self.selected_poi}'", style={"fontSize": "16pt"}),
                html.Img(src=Image.fromarray(cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)), style={
                    "width": "100%",
                    "cursor": "pointer",
                    "maxHeight": "16vw",
                    "objectFit": "contain",
                }),
                html.Button(  # Button to continue to the resulting visualization
                    "Show Results",
                    disabled=False,
                    id="button_continue_to_results",
                    style={
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
                )
            ]

            # Update output
            return children


    def add_face(self, selection):
        """
        Adds a face to the dataframe of the uploaded image when something is selected in that image.

        :param selection: the selection in the image
        """
        if selection:  # Something is selected
            # Get the points of the selected area
            points = selection.get("range", {})
            x0, x1 = points.get("x", [None, None])
            y0, y1 = points.get("y", [None, None])
            print(f"(Choose POI)          - Manually added face, selected area: x=({x0}, {x1}), y=({y0}, {y1})")

            # If all coordinates exist
            if x0 and x1 and y0 and y1:
                # Get the bounding box of the manually added face
                box = [int(x0), int(y0), int(x1) - int(x0), int(y1) - int(y0)]

                # Get the face number for the new face
                nr = len(self.df_faces_uploaded)

                # Add face to the dataset
                self.df_faces_uploaded.loc[len(self.df_faces_uploaded)] = [
                    box, True, f"{nr}"
                ]

                # Update outputs (components left half of ChooseFaceSection object)
                return self.update_left_half(f" Face {nr}")
        # No update
        return dash.no_update, dash.no_update, dash.no_update
