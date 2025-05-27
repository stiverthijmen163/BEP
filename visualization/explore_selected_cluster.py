import dash
from dash import html, dcc
from functions import *
import plotly.express as px
from PIL import Image


class ExploreSelectedCluster(html.Div):
    """
    Contains all functionalities of the 'explore cluster' section for the 'visualization' page.
    """
    def __init__(self, name):
        """
        Initialize the ExploreSelectedCluster object.

        :param name: id of the ExploreSelectedCluster object
        """
        # Initialize the object's parameters
        self.html_id = name  # id of the object
        self.df_main = None  # Dataframe containing all info regarding the main images
        self.df_faces = None  # Dataframe containing all faces in all images
        self.df_selected_cluster = None  # Dataframe containing all faces within the selected cluster
        self.selected_cluster = None  # Keeps track of the currently selected cluster
        self.selected_index = 0  # Keeps track of the index of the first images shown as example
        self.images = []  # List of images to display as example
        self.image_links = []  # List of links corresponding the example images
        self.fig = None  # Scatterplot figure to display all face points in

        # Initialize ChooseFaceSection as an empty Div
        super().__init__(className="graph_card", id=self.html_id, children=[])


    def initialize_esc(self, df_main, df_faces, selected_cluster):
        """
        Initializes the ExploreSelectedCluster object as a non-empty div.

        :param df_main: dataframe containing all info regarding the main images
        :param df_faces: dataframe containing all faces in all images
        :param selected_cluster: the cluster corresponding to the selected person of interest
        """
        print("Initializing ExploreSelectedCluster object")

        # Update the object's parameters
        self.df_main = df_main.copy()
        self.df_faces = df_faces.copy()
        self.selected_cluster = selected_cluster
        self.selected_index = 0

        # Collect all faces from the dataset within the selected cluster
        self.df_selected_cluster = self.df_faces[self.df_faces["name"] == self.selected_cluster].copy()

        # Collect only the relevant main information
        self.df_main = self.df_main[self.df_main["id"].isin(self.df_selected_cluster["img_id"].to_list())].copy()

        # Initialize list or example images and url's
        self.images = []
        self.image_links = []

        # Collect images and url's from dataset
        for _, row in self.df_selected_cluster.iterrows():
            # Get the full image from the main dataset
            id = row["img_id"]
            img = self.df_main[self.df_main["id"] == id]["img"].to_list()[0]

            # Plot the face from the person of interest on the full image
            self.images.append(plot_faces_on_img(img.copy(), [row["face"]], 8))

            # Collect the corresponding url
            if "url" in self.df_main.columns:
                self.image_links.append(self.df_main[self.df_main["id"] == id]["url"].to_list()[0])

        # Introduce a new column which helps to define the color for the scatterplot
        self.df_faces["color"] = self.df_faces["name"].apply(lambda x: self.selected_cluster if x == self.selected_cluster else "Other")

        # Set the colors for the scatter plot
        custom_palette = {
            self.selected_cluster: "red",
            "Other": "lightgray"
        }

        # Update the scatterplot/figure
        self.fig = px.scatter(self.df_faces, x="tsne_x", y="tsne_y", color="color", hover_data=["name"],
                              color_discrete_map=custom_palette)

        # Update the layout of the figure
        self.fig.update_layout(
            margin=dict(l=0, r=0, b=0),  # Use all available space, leave space at the top for title
            showlegend=True,  # Show the legend
            title_text=f"Selected cluster among all detected faces", # Add a title
            title_x=0.5  # Centre the title
        )

        # Update the example images to show
        images_to_display, _, right_disabled, _, _, new_txt, nr = self.update_example_images_cls()

        # Update the layout of this section
        children = [
            html.Div(
                style={"textAlign": "center"},
                children=[
                    html.H2("Explore Cluster", style={"marginTop": "20px"}),  # Header
                    html.Hr(),
                    html.Div(  # Blue box for the cluster exploration section
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
                            html.Div(
                                style={  # Work within a slightly smaller space, allows items to be next to each other
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
                                children=[
                                    dcc.Graph(  # The scatter plot (left half)
                                        id=f"{self.html_id}_fig",
                                        figure=self.fig,
                                        style={"flex": "1",
                                               "height": "22vw",
                                               "width": "50%"}
                                    ),
                                    html.Div([  # Right half, show example images
                                        html.P(new_txt, style={"textAlign": "right", "fontSize": "16pt"}, id="images_showing_ex_cls_txt"),
                                        html.Div(
                                            images_to_display,
                                            style={
                                                "backgroundColor": "white",
                                                "borderRadius": "12px",
                                                "border": "2px black solid",
                                                "textAlign": "center",
                                                "height": "80%"
                                            },
                                            id="example_images_cls_box"
                                        ),
                                        html.Div(  # Allows content to be next to each other
                                            style={
                                                "display": "flex",
                                                "justifyContent": "space-between",
                                                "width": "99%",
                                                "margin": "10px auto"
                                            },
                                            children=[
                                                html.Button(  # Button to show the previous 8 images
                                                    "⬅️ Back", id="box_ex_images_left", style={
                                                        "width": "10%",
                                                        "opacity": 0.5
                                                    }, disabled=True),
                                                html.Button(  # Button to show the next 8 images
                                                    "Next ➡️", id="box_ex_images_right", style={
                                                        "width": "10%",
                                                        "opacity": 0.5 if right_disabled else 1.0
                                                    }, disabled=right_disabled)
                                            ]
                                        ),
                                    ], style={"height": "22vw", "width": "50%"})
                                ]
                            )
                        ]
                    )
                ]
            )
        ]

        # Update outputs
        return children


    def show_next_example_images_cls(self, n_clicks):
        """
        Updates the image box to display the next 8 images.

        :param n_clicks: number of clicks on 'next' button
        """
        # If the button triggered this callback
        if n_clicks is not None and n_clicks > 0:
            # Update the index
            self.selected_index += 8

            # Collect the updated images to display and its corresponding components
            children, _, right_disabled, style_left, style_right, new_txt, nr = self.update_example_images_cls()

            print(f"Displaying the next {nr} images")

            # Update outputs
            return children, False, right_disabled, style_left, style_right, new_txt
        # No update
        return dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update


    def show_previous_example_images_cls(self, n_clicks):
        """
        Updates the image box to display the previous 8 images.

        :param n_clicks: number of clicks on 'back' button
        """
        # If this function is trigger by a button click
        if n_clicks is not None and n_clicks > 0:
            print(f"Displaying the previous 8 images")

            # Update the index
            self.selected_index -= 8

            # Collect the updated images to display and its corresponding components
            children, left_disabled, _, style_left, style_right, new_txt, _ = self.update_example_images_cls()

            # Update outputs
            return children, left_disabled, False, style_left, style_right, new_txt
        # No update
        return dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update


    def update_example_images_cls(self):
        """
        Updates the images to display in the image box, and its correspond buttons (disable/enable).

        :return: children to display (images), whether to disable buttons and their corresponding style,
        text displaying the indexes of the shown images and the nr of displayed images
        """
        # Checks how many images to display
        if len(self.images[self.selected_index:]) > 8:  # Show 8 images ('next' button enabled)
            nr = 8
            right_disabled = False
        else:  # Shows the remaining images ('next' button disables)
            nr = len(self.images[self.selected_index:])
            right_disabled = True

        # Initialize the list of images to display
        images_to_display = []

        # Set images to display
        for i in range(nr):
            # Read the image
            img = self.images[i + self.selected_index].copy()
            image = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

            # Display img as html.Image and append to the list
            if len(self.image_links) == 0:  # No links were added to the dataset
                images_to_display.append(
                    html.Div(
                        id={"type": "image_ex_cls_click", "index": i},  # Pattern-matching id
                        style={"display": "inline-flex", "width": "24%", "padding": "0.5%", "height": "50%",
                               "alignItems": "center"},
                        n_clicks=0,  # Enables click tracking
                        children=html.Img(src=image, style={
                            "width": "100%",
                            "cursor": "pointer",
                            "maxHeight": "100%",
                            "objectFit": "contain",
                        })
                    )
                )
            else:  # Make images clickable to send user to new url
                images_to_display.append(
                    html.Div(
                        id={"type": "image_ex_cls_click", "index": i},  # Pattern-matching id
                        style={"display": "inline-flex", "width": "24%", "padding": "0.5%", "height": "50%",
                               "alignItems": "center"},
                        n_clicks=0,  # Enables click tracking
                        children=html.A(
                            href=self.image_links[i + self.selected_index],  # Add url
                            target="_blank",  # Open in new tab
                            children=html.Img(
                                src=image,
                                style={
                                    "width": "100%",
                                    "cursor": "pointer",
                                    "maxHeight": "100%",
                                    "objectFit": "contain",
                                }
                            )
                        )
                    )
                )

        # Update the text showing what images (indexes) are displayed
        new_txt = f"Showing {self.selected_index + 1} - {nr + self.selected_index} out of {len(self.images)}"

        # Whether the 'back' button is disabled or not
        left_disabled = True if self.selected_index == 0 else False

        # Update the styles of the buttons, both conditionally on whether the button is disabled or not
        style_left = {"width": "10%", "opacity": 0.5 if left_disabled else 1.0}
        style_right = {"width": "10%", "opacity": 0.5 if right_disabled else 1.0}

        return images_to_display, left_disabled, right_disabled, style_left, style_right, new_txt, nr