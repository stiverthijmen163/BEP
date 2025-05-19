import dash
from dash import dcc, html
import dash_bootstrap_components as dbc
import plotly.express as px
from PIL import Image
from functions import *


class Detector(html.Div):
    """
    Contains all functionalities of the detection section for the 'New Data' page.
    """
    def __init__(self, name, df):
        """
        Initializes the Clusteror object as an empty layout.

        :param name: name (id) of the Detector object
        :param df: dataframe containing all images and their corresponding id's
        """
        # Initialize all variables for the Detector object
        self.html_id = name.lower().replace(" ", "-")
        self.fig = None  # Figure to display an image and allow selection
        self.selected_index = 0  # Currently selected index of first image shown in image box
        self.df_faces = None  # dataframe where all detected faces will be stored
        self.show_nrs = False  # Whether to show the face numbers in the figure or not
        self.selected_image = None  # The currently selected image
        self.df_detected = None  # Copy of df_faces but has the width and height restrictions
        self.min_size = (0,0)  # Minimum size of a face to be detected
        self.images = []  # All images to display

        # Copy input dataframe
        if df is None:  # No data inputted so far
            self.df = None
        else:  # Data exists
            self.df = df.copy()

        # Initialize Detector as an empty Div
        super().__init__(className="graph_card", id=self.html_id, children=[])


    def initialize_detector(self, df):
        """
        Initializes the Detector object with an updated layout for the detection section.

        :param df: dataframe containing all images and their corresponding id's
        """
        print(f"Initializing the Detector object: '{self.html_id}'")

        # Update the current dataset
        self.df = df.copy()

        # Initialize lists for all data
        detected = []  # Detected faces
        images_faces = []  # image with all faces plotted on top of it

        # Rows for dataframe
        df_faces_id = []  # id of face
        df_faces_img_id = []  # id of image
        df_faces_face = []  # bounding boxes for face
        df_faces_nr = []  # face number

        # Load in the model to detect faces with
        model = YOLO("../face_detection/yolo_v12s/yolov12s-face/weights/epoch60.pt")

        # Iterate over all uploaded images
        for _, row in self.df.copy().iterrows():
            # Detect faces and plot them on the image
            img_w_faces, detections = detect_faces(row["img"].copy(), model)

            # Append face image and detected faces to the lists
            images_faces.append(cv2.cvtColor(img_w_faces, cv2.COLOR_BGR2RGB))
            detected.append(detections)

            # Initialize face counter
            count = 0

            # Create row for every detected face
            for face in detections:
                df_faces_id.append(f"{row['id']}_{count}")  # face id
                df_faces_img_id.append(row["id"])  # image id
                df_faces_face.append(face)  # face bounding box
                df_faces_nr.append(count)  # face number

                # Update face counter
                count += 1

        # Create dataframe from results containing all faces
        self.df_faces = pd.DataFrame({
            "id": df_faces_id,
            "img_id": df_faces_img_id,
            "face": df_faces_face,
            "nr": df_faces_nr,
            "use": True
        })

        # Append all faces to the main dataset
        self.df["img_w_faces"] = images_faces
        self.df["faces"] = detected

        # Update the images to display
        self.images = images_faces

        # Get the width and height for every image
        self.df_faces["width"] = self.df_faces["face"].apply(lambda x: list(x)[2])
        self.df_faces["height"] = self.df_faces["face"].apply(lambda x: list(x)[3])

        # Copy faces dataset to dataframe with size restrictions
        self.df_detected = self.df_faces.copy()

        # Collect the updated images to display and its corresponding components
        children, _, right_disabled, _, _, _, nr = self.update_displayed_images()

        # Update the layout to contain the detection section
        result = html.Div(
            children = [  # Header
                html.H2("Face Detection", style={"textAlign": "center"}),
                html.Hr(),
                html.Div(  # Blue box containing detection section
                    style={
                        "backgroundColor": "#dbeafe",
                        "margin": "20px auto",
                        "width": "99vw",
                        "borderRadius": "12px",
                        "boxShadow": "0px 2px 5px rgba(0,0,0,0.1)",
                        "textAlign": "center"
                    },
                    children=[
                        html.Div([  # Allows component to be next to each other
                            html.Div("", style={"flex": 1}),  # Left spacer
                            html.Div(html.P(  # Show user displayed images are selectable
                                "Select an image to edit or Continue", style={
                                "fontSize": "16pt",
                                "marginBottom": "5px",
                                "textAlign": "center",
                                "margin": "0"
                            }), style={"flex": 1, "display": "flex", "justifyContent": "center"}),
                            html.Div(html.P(  # Shows user what images are displayed (index)
                                f"Showing 1 - {nr} out of {len(self.df)}",id="images_showing_txt", style={
                                "fontSize": "16pt",
                                "marginBottom": "5px",
                                "textAlign": "center",
                                "margin": "0"
                            }), style={"flex": 0.995, "display": "flex", "justifyContent": "flex-end", "paddingRight": "0.5%"})  # Right content
                        ], style={
                            "display": "flex",
                            "alignItems": "center",
                            "width": "100%"
                        }),
                        html.Div(  # box to display images in
                            style={
                                "backgroundColor": "white",
                                "margin": "5px auto",
                                "width": "99%",
                                "borderRadius": "12px",
                                "border": "2px black solid",
                                "textAlign": "center"
                            },
                            children=children,  # Images to show
                            id="box_images0"
                        ),
                        dcc.Store(id="box_images0_index", data=0),  # Keeps track of the index of the first shown image
                        html.Div(  # Allows components to be next to each other
                            style={
                                "display": "flex",
                                "justifyContent": "space-between",
                                "width": "99%",
                                "margin": "10px auto"
                            },
                            children=[
                                html.Button(  # Button to display previous 10 images
                                    "⬅️ Back", id="box_images0_left", style={
                                    "width": "5%",
                                    "opacity": 0.5
                                }, disabled=True),
                                html.Button(# Button to display next 10 images
                                    "Next ➡️", id="box_images0_right", style={
                                    "width": "5%",
                                    "opacity": 0.5 if right_disabled else 1.0
                                }, disabled=right_disabled)
                            ]
                        ),
                        html.P(  # Tex placeholder for the image selected
                            "", id="selected_image_txt", style={
                            "fontSize": "16pt",
                            "marginBottom": "5px",
                            "textAlign": "center",
                            "margin": "0"
                        }),
                        html.Div(  # Allows components to be next to each other
                            style={
                                "display": "flex",
                                "justifyContent": "space-between",
                                "alignItems": "flex-start",
                                "gap": "10px",
                                "width": "99%",
                                "margin": "10px auto"
                            },
                            children=[
                                dcc.Graph(  # Figure to display selected image in (left half)
                                    id=f"{self.html_id}_fig",
                                    style={"flex": "1",
                                           "height": "22vw"}
                                ),
                                html.Div(  # White box to edit the faces in (right half)
                                    style={
                                        "backgroundColor": "white",
                                        "width": "50%",
                                        "borderRadius": "12px",
                                        "border": "2px black solid",
                                        "textAlign": "center",
                                        "height": "22vw"
                                    },
                                    children=[
                                        html.P(  # Explanation text
                                            "Select which detections you want to use, or draw your own detection in the figure on the left.",
                                           style={
                                               "fontSize": "16pt",
                                               "marginBottom": "5px",
                                               "textAlign": "center",
                                               "margin": "5px",
                                               "fontWeight": "bold"
                                           }
                                        ),
                                        html.Div([  # Allows components to be next to each other
                                            html.P(  # Set min_size variable
                                                [
                                                    "min_size:",
                                                    html.I(  # Info icon
                                                        className="fa-solid fa-circle-question",
                                                        id="info_icon_min_size0",
                                                        style={"cursor": "pointer", "color": "#0d6efd",
                                                            "marginLeft": "5px",
                                                            "position": "relative",
                                                            "top": "-3px"
                                                        }
                                                    ),
                                                    " width:",
                                                    dcc.Input(  # Input for minimum width of face to be detected
                                                        id="min_width_det_input",
                                                        type="number",
                                                        min=0,
                                                        max=max(self.df_faces["width"]) if len(self.df_faces) > 0 else 1000,
                                                        step=1,
                                                        value=0,
                                                        style={"marginLeft": "10px", "width": "8%"}
                                                    ),
                                                    " height:",
                                                    dcc.Input(  # Input for minimum height for face to be detected
                                                        id="min_height_det_input",
                                                        type="number",
                                                        min=0,
                                                        max=max(self.df_faces["height"]) if len(self.df_faces) > 0 else 1000,
                                                        step=1,
                                                        value=0,
                                                        style={"marginLeft": "10px", "width": "8%"}
                                                    ),
                                                    html.Button(  # Button to updates to face larger than min_size
                                                        "Update detections",
                                                        disabled=False,
                                                        style={
                                                            "padding": "10px 20px",
                                                            "fontSize": "16pt",
                                                            "borderRadius": "12px",
                                                            "border": "none",
                                                            "backgroundColor": "#2196F3",
                                                            "color": "white",
                                                            "cursor": "pointer",
                                                            "width": "12vw",
                                                            "marginLeft": "1vw"
                                                        },
                                                        id="button_update_detections"
                                                    )
                                                ],
                                                style={
                                                    "fontSize": "16pt",
                                                    "marginBottom": "5px",
                                                    "textAlign": "center",
                                                    "margin": "0"
                                                },
                                            ),
                                            dbc.Tooltip(  # When hovering over the info icon, display some explanation
                                                "Select the minimum size of a face to be detected, faces that are smaller will be discarded.",
                                                target="info_icon_min_size0",
                                                placement="top"
                                            )
                                        ]),
                                        html.Div(  # Placeholder for dropdown menu to select the faces you want to use
                                            id="selection_of_faces",
                                            style={
                                                "textAlign": "left",
                                                "margin": "0.5%",
                                            },
                                            children=[]
                                        )
                                    ]
                                )
                            ]
                        ),
                        html.Br(),
                        html.Button(  # Button to continue to the clustering section
                            "Continue to Face Clustering",
                            disabled=False,
                            style={
                                "padding": "10px 20px",
                                "fontSize": "16pt",
                                "borderRadius": "12px",
                                "border": "none",
                                "backgroundColor": "#2196F3",
                                "color": "white",
                                "cursor": "pointer",
                                "width": "20vw",
                                "marginBottom": "20px"
                            },
                            id="button2"
                        ),
                        html.P(  # Placeholder for feedback to the user in case too small dataset
                            "",
                            style={
                                "fontSize": "16pt",
                                "marginBottom": "20px",
                                "textAlign": "center",
                                "margin": "5px",
                                "fontWeight": "bold"
                            },
                            id="successful_detection"
                        ),
                    ]
                )
            ]
        )

        # Update layout
        return result


    def next_button(self, n_clicks, current_index):
        """
        Updates the image box to display the next 10 images.

        :param n_clicks: number of clicks on 'next' button
        :param current_index: index of the currently first displayed image
        """
        # If the 'next' button has called this function
        if n_clicks is not None and n_clicks > 0:
            # Update index
            new_index = current_index + 10
            self.selected_index = new_index

            # Collect the updated images to display and its corresponding components
            children, _, right_disabled, style_left, style_right, new_txt, nr = self.update_displayed_images()

            print(f"Displaying the next {nr} images")
             # Update outputs
            return new_index, children, False, right_disabled, style_left, style_right, new_txt
        # No update
        return dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update


    def previous_button(self, n_clicks, current_index):
        """
        Updates the image box to display the previous 10 images.

        :param n_clicks: number of clicks on 'back' button
        :param current_index: index of the currently first displayed image
        """
        # If the 'back' button has triggered this function
        if n_clicks is not None and n_clicks > 0:
            print("Displaying the previous 10 images")

            # Update index
            new_index = current_index - 10
            self.selected_index = new_index

            # Collect the updated images to display and its corresponding components
            children, left_disabled, _, style_left, style_right, new_txt, _ = self.update_displayed_images()

            # Update outputs
            return new_index, children, left_disabled, False, style_left, style_right, new_txt
        # No update
        return dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update


    def init_picture_fig(self, image_nr):
        """
        Initializes the picture figure when a new image has been clicked,

        :param image_nr: index of the image clicked on
        """
        print(f"Initialize picture fig to contain image {image_nr}")

        # Collect the image index
        index = self.selected_index + image_nr

        # Update the currently selected image
        id = self.df["id"].to_list()[index]
        self.selected_image = id

        # Take only the faces of the currently selected image
        temp_df = self.df_detected[self.df_detected["img_id"] == id].copy()

        # Collect the image to display
        img = self.df["img"].to_list()[index]

        # Plot the faces on the image
        img = plot_faces_on_img_opacity(img.copy(), temp_df.copy(), self.show_nrs)

        # Update the figure
        fig = px.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        fig.update_layout(
            margin=dict(l=0, r=0, t=0, b=0),
            xaxis=dict(showticklabels=False),
            yaxis=dict(showticklabels=False),
            dragmode="select"
        )

        # Initialize the list of checkbox
        children = []

        # Create a checkbox for each face
        for _, row in temp_df.iterrows():
            children.append(dcc.Checklist(
                options=[{"label": f" Face {row['nr']}", "value": "keep"}],
                value=["keep"] if row["use"] else [],
                id={"type": "keep-face", "index": row["nr"]},
                style={
                    "fontSize": "16pt",
                    "marginBottom": "5px",
                    "margin": "0"
                }
            ))

        # Create scrollable column containing the checkboxes
        scrollable_column = html.Div(
            id="scrollable_column",
            children=children,
            style={
                "maxHeight": "16vw",
                "height": "16vw",
                "overflowY": "scroll",
                "padding": "10px",
                "border": "1px solid lightgray",
                "borderRadius": "8px",
                "marginTop": "10px",
                "width": "15%"
            }
        )

        # Update the layout
        layout = html.Div(  # Allow components to be next to each other
            style={"display": "flex", "gap": "20px"},
            children=[
                # Left column: scrollable list
                scrollable_column,
                # Right column: checklist + image
                html.Div(
                    style={
                        "display": "flex",
                        "flexDirection": "column",
                        "gap": "10px",
                        "textAlign": "center",
                        "width": "80%",
                    },
                    children=[
                        # Checklist for showing the face numbers
                        dcc.Checklist(
                            id="show_nrs",
                            options=[
                                {"label": " Show face numbers in figure on the left", "value": "boxes"}
                            ],
                            value=["boxes"] if self.show_nrs else [],
                            style={"fontSize": "16pt", "marginTop": "10px", "textAlign": "left"}
                        ),
                        html.P(  # Text placeholder showing the action that just happened
                            "",
                            style={
                                "fontSize": "16pt",
                                "marginBottom": "5px",
                                "textAlign": "center",
                                "margin": "0"
                            },
                            id="text_for_image"
                        ),
                        html.Img(  # Image placeholder showing the face which has been edited
                            id="image-preview", src="",
                            style={
                                "maxWidth": "30vw",
                                "maxHeight": "12vw",
                                "width": "auto",
                                "height": "100%",
                                "display": "block",
                                "marginLeft": "auto",
                                "marginRight": "auto"
                            }
                        )
                    ]
                )
            ]
        )

        # Update outputs
        return fig, layout, False


    def update_picture_fig(self, show_nrs, checklist, activated_by, selection, children, n_clicks, w, h):
        """
        Updates the picture figure when checkboxes are clicked or when a new face is drawn in the figure.

        :param show_nrs: whether to show the face number in the shown image
        :param checklist: list of checklist values, each value denotes whether to keep the face or not
        :param activated_by: the data from selecting something in the figure (image) using 'box select'
        :param selection: whether a new image has been clicked on or not (None is no image clicked on yet)
        :param children: the current children (items) in the checklist
        :param n_clicks: number of clicks on the update_detections button
        :param w: minimum width of a face
        :param h: minimum height of a face
        """
        # Update show face numbers variable
        self.show_nrs = False if show_nrs == [] else True

        # Update the faces to show
        if activated_by == f"{self.html_id}_fig":  # Function called by face drawn in figure
            if selection:  # Something is selected
                # Get the points of the selected area
                points = selection.get("range", {})
                x0, x1 = points.get("x", [None, None])
                y0, y1 = points.get("y", [None, None])
                print(f"Manually added face, selected area: x=({x0}, {x1}), y=({y0}, {y1})")

                # If all coordinates exist
                if x0 and x1 and y0 and y1:
                    # Get the bounding box of the manually added face
                    box = [int(x0), int(y0), int(x1) - int(x0), int(y1) - int(y0)]

                    # Get the face number for the new face
                    try:  # Non-empty set of faces
                        nr = max(self.df_faces[self.df_faces["img_id"] == self.selected_image]["nr"].to_list()) + 1
                    except ValueError:  # Empty set of faces
                        nr = 0

                    # Add face to the dataset
                    self.df_faces.loc[len(self.df_faces)] = [
                        f"{self.selected_image}_{nr}", self.selected_image, box, nr, True, box[2], box[3]
                    ]

                    # Collect the image of the face
                    img = self.df[self.df["id"] == self.selected_image]["img"].copy().to_list()[0]
                    x, y, w, h = box
                    face = img[y:y + h, x:x + w]

                    # Plot the face
                    result = Image.fromarray(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))

                    # In case the face is larger than or equal to min_size
                    if w >= self.min_size[0] and h >= self.min_size[1]:
                        # Add checkbox for the new face
                        children.append(dcc.Checklist(
                            options=[{"label": f" Face {nr}", "value": "keep"}],
                            value=["keep"],
                            id={"type": "keep-face", "index": nr},
                            style={
                                "fontSize": "16pt",
                                "marginBottom": "5px",
                                "margin": "0"
                            }
                        ))

                        # Update text
                        result_txt = f"Created and added 'Face {nr}' to the set of faces:"
                    else:  # Smaller than min_size
                        # Update text
                        result_txt = f"Created 'Face {nr}', but it is smaller than min_size"
                else:
                    # No update
                    return dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update
            else:
                # No update
                return dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update
        # Min_size updated, only if function is activated by the 'update detections' button
        elif activated_by == "button_update_detections" and n_clicks is not None and n_clicks > 0:
            print(f"Updated min_size to (f{w}, f{h})")

            # Update the min sizes
            self.update_min_size(w, h)

            # Do not update shown face and text
            result = dash.no_update
            result_txt = dash.no_update

            # Initialize list of checkboxes
            children = []

            # Update checkboxes to only those who fulfill min_size requirement
            for _, row in self.df_detected[self.df_detected["img_id"] == self.selected_image].copy().iterrows():
                children.append(dcc.Checklist(
                    options=[{"label": f" Face {row['nr']}", "value": "keep"}],
                    value=["keep"] if row["use"] else [],
                    id={"type": "keep-face", "index": row["nr"]},
                    style={
                        "fontSize": "16pt",
                        "marginBottom": "5px",
                        "margin": "0"
                    }
                ))
        elif activated_by != "show_nrs":  # If function is called by checkbox whether to show face or not
            # Collect face id of which the checkbox has changed
            id = f"{self.selected_image}_{activated_by['index']}"

            # Get the updated value of whether to use face or not
            try:
                changed = checklist[activated_by["index"]]
                replace_val = False if changed == [] else True
            except IndexError:  # In case the checklist hasn't been updated correctly (glitch)
                # Update the value to the opposite
                if self.df_faces[self.df_faces["id"] == id]["use"].to_list()[0]:
                    replace_val = False
                else:
                    replace_val = True

            print(f"{'Checked' if replace_val else 'Unchecked'} checkbox of face '{id}'")

            # Update the dataset
            self.df_faces.loc[self.df_faces["id"] == id, "use"] = replace_val

            # Collect the face to show that has been updated
            face_box = self.df_faces[self.df_faces["id"] == id]["face"].copy().to_list()[0]
            x, y, w, h = face_box
            img = self.df[self.df["id"] == self.selected_image]["img"].copy().to_list()[0]
            face = img[y:y+h, x:x+w]

            # Plot the updated face
            result = Image.fromarray(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))

            # Update the text
            if replace_val:
                result_txt = f"Added 'Face {activated_by['index']}' to the set of faces:"
            else:
                result_txt = f"Removed 'Face {activated_by['index']}' from the set of faces:"
        else:  # No change in selected faces
            print(f"{'Checked' if self.show_nrs else 'Unchecked'} checkbox to show face numbers")
            result = dash.no_update
            result_txt = dash.no_update

        # Update faces to only those that meet the min_size requirement
        self.df_detected = self.df_faces[
            (self.df_faces["width"] >= self.min_size[0]) & (self.df_faces["height"] >= self.min_size[1])].copy()

        # Collect the entire image
        img = self.df[self.df["id"] == self.selected_image]["img"].to_list()[0]

        # Contain only the faces within that image
        temp_df = self.df_detected[self.df_detected["img_id"] == self.selected_image].copy()

        # Plot all faces on the image
        img = plot_faces_on_img_opacity(img.copy(), temp_df.copy(), self.show_nrs)

        # Update the figure
        fig = px.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        fig.update_layout(
            margin=dict(l=0, r=0, t=0, b=0),
            xaxis=dict(showticklabels=False),
            yaxis=dict(showticklabels=False),
            dragmode="select"
        )

        # Update outputs
        return fig, True, result, result_txt, children, dash.no_update


    def update_min_size(self, w, h):
        """
        Updates the min_sice requirement.

        :param w: minimum width of face to be detected
        :param h: minimum height of face to be detected
        """
        if w is not None:  # Update minimum width
            self.min_size = (w, self.min_size[1])
        if h is not None:  # Update minimum height
            self.min_size = (self.min_size[0], h)

        # Update dataset with only those faces that meet the min_size requirement
        self.df_detected = self.df_faces[
            (self.df_faces["width"] >= self.min_size[0]) & (self.df_faces["height"] >= self.min_size[1])].copy()


    def update_displayed_images(self):
        """
        Updates the images to display in the image box, and its correspond buttons (disable/enable).

        :return: children to display (images), whether to disable buttons and their corresponding style,
        text displaying the indexes of the shown images and the nr of displayed images
        """
        # Checks how many images to display
        if len(self.images[self.selected_index:]) > 10:  # Show 10 images ('next' button enabled)
            nr = 10
            right_disabled = False
        else:  # Shows the remaining images ('next' button disables)
            nr = len(self.images[self.selected_index:])
            right_disabled = True

        # Initialize the list of images to display
        children = []

        # Set images to display
        for i in range(nr):
            # Read the image
            img = self.images[i + self.selected_index].copy()
            image = Image.fromarray(img)

            # Display img as html.Image and append to the list
            children.append(
                html.Div(
                    id={"type": "image-click", "index": i},
                    style={"display": "inline-block", "width": "9%", "padding": "0.5%"},
                    n_clicks=0,
                    children=html.Img(src=image, style={"cursor": "pointer", "width": "100%"})
                )
            )

        # Update the text showing what images (indexes) are displayed
        new_txt = f"Showing {self.selected_index + 1} - {nr + self.selected_index} out of {len(self.images)}"

        # Whether the 'back' button is disabled or not
        left_disabled = True if self.selected_index == 0 else False

        # Update the styles of the buttons, both conditionally on whether the button is disabled or not
        style_left = {"width": "5%", "opacity": 0.5 if left_disabled else 1.0}
        style_right = {"width": "5%", "opacity": 0.5 if right_disabled else 1.0}

        return children, left_disabled, right_disabled, style_left, style_right, new_txt, nr
