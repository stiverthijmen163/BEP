import dash
from dash import html, dcc
from functions import *
import plotly.express as px
from PIL import Image


class Results(html.Div):
    """
    Contains all functionalities of the 'results' section for the 'visualization' page.
    """
    def __init__(self, name):
        """
        Initialize the Results object.

        :param name: id of the Results object
        """
        # Initialize the object's parameters
        self.html_id = name  # id of the object
        self.df_main = None  # Dataframe containing all info regarding the main images with the poi in them
        self.df_faces = None  # Dataframe containing all faces in all images
        self.df_selected_cluster = None  # Dataframe containing all faces within the selected cluster
        self.df_rel_faces = None  # Dataframe containing all faces which exists at least ones in the same image as the poi
        self.selected_cluster = None  # Keeps track of the currently selected cluster
        self.selected_index = 0  # Keeps track of the index of the first images shown as example
        self.h_fig = None  # Horizontal bar chart showing the people with whom the poi exists in some images
        self.v_fig = None  # Bar chart showing the extra information
        self.selected_bar = None  # Keeps track of the currently selected bar in the horizontal bar chart
        self.options = []  # List of all options to choose as additional data
        self.value = None  # Keeps track of the additional data to use
        self.images = []  # List of all images to show where both the poi and a selected person occur in
        self.image_links = []  # List of url's corresponding to the self.images list

        # Initialize ChooseFaceSection as an empty Div
        super().__init__(className="graph_card", id=self.html_id, children=[])


    def initialize_res(self, df_main, df_faces, selected_cluster, df_selected_cluster):
        """
        Initializes the Results object as a non-empty div.

        :param df_main: dataframe containing all info regarding the main images with the poi in them
        :param df_faces: dataframe containing all faces in all images
        :param selected_cluster: the cluster corresponding to the selected person of interest
        :param df_selected_cluster: dataframe containing all faces within the selected cluster
        """
        print("(Results)             - Initializing Results object")

        # Update the object's parameters
        self.df_main = df_main.copy()
        self.df_faces = df_faces.copy()
        self.df_selected_cluster = df_selected_cluster.copy()
        self.selected_cluster = selected_cluster
        self.selected_index = 0
        self.selected_bar = None

        # Set the initial text for the example images
        new_txt = "Showing 0 - 0 out of 0"

        # Initially do not show images, but tell the user to select a bar
        images_to_display = html.Div(
            html.P(  # Display text saying that no poi has been selected yet
                f"Click on a bar in the figure on the left to display all images where both {self.selected_cluster} and the person corresponding to that bar appear in.",
                style={"fontSize": "25pt", "textAlign": "center", "opacity": 0.3}
            ),
            style={
                "display": "flex",
                "justifyContent": "center",
                "alignItems": "center",
                "height": "100%"
            }
        )

        # Disable 'next' button
        right_disabled = True

        # Creat dataframe containing all faces which exists at least ones in the same image as the poi
        self.df_rel_faces = self.df_faces[self.df_faces["img_id"].isin(self.df_main["id"].to_list())].copy()
        self.df_rel_faces = self.df_rel_faces[~self.df_rel_faces["name"].isin([self.selected_cluster, "Unknown", "unknown"])].copy()

        # Count the number of occurrences for each face with the poi, and sort them
        df_face_counts = pd.DataFrame({"name": self.df_rel_faces["name"].unique()})
        df_face_counts["count"] = df_face_counts["name"].map(self.df_rel_faces["name"].value_counts())
        df_face_counts = df_face_counts.sort_values(by="count", ascending=True)

        # Set the settings for the bars in the horizontal bar chart
        bar_height = 50  # px per bar
        num_bars = len(df_face_counts)

        # Plot the counts per face as a horizontal bar chart
        self.h_fig = px.bar(df_face_counts, x="count", y="name", orientation="h")

        # Update the layout of the h_bar figure
        self.h_fig.update_layout(
            # height=num_bars * bar_height + 100,
            margin=dict(l=0, r=10, b=0),  # Use all available space, leave space at the top for title, add space to the right
            showlegend=True,  # Show the legend
            title_text=f"Number of times a person occurs with '{self.selected_cluster}' in the images", # Add a title
            title_x=0.5,  # Centre the title
            font = dict(size=14)
        )

        # Set the options for the dropdown menu (additional information)
        self.options = sort_items([i for i in self.df_main.columns if i not in ["index", "id", "url", "img"]])

        # Check if there exists additional data in the database
        if len(self.options) > 0:  # Additional information should be plotted
            # Update/Initialize the vertical bar chart
            title_v_bar = f"Distribution of '{self.options[0]}' for those images containing '{self.selected_cluster}'"
            children_v_fig = self.update_v_fig(self.options[0], title_v_bar)

            # Update the layout to insert the vertical bar chart into
            children_add_info = html.Div([
                html.Br(),  # Create space between the previous section
                html.Hr(),
                html.Br(),
                html.Div(  # Allows components to be next to each other
                    style={
                        "display": "flex",
                        "justifyContent": "center",
                        "alignItems": "center",
                        "gap": "1vw"
                    },
                    children=[  # Dropdown menu to choose additional data to visualize
                        html.Label("Choose additional information to visualize:", style={"fontSize": "16pt"}),
                        dcc.Dropdown(  # All options of additional data
                            id="dropdown_choose_add_data",
                            options=self.options,
                            value=self.options[0],
                            clearable=False,
                            className="dropdown",
                            searchable=True,
                            style={"width": "40vw"}
                        ),
                    ]
                ),
                html.Br(),
                html.Div(  # Show the vertical bar chart
                    id=f"div_{self.html_id}_v_fig",
                    style={
                        "height": "22vw",
                        "width": "100%",
                        "overflowX": "auto",  # Enables scroll if needed (horizontal direction)
                        "border": "1px solid #ccc",
                        "borderRadius": "8px"
                    },
                    children=children_v_fig
                ),
            ])
        else:  # No additional data exists, thus no additional layout (empty)
            children_add_info = html.Div([])

        # Update the layout of the results section (before the additional data)
        children = [
            html.Div(
                style={"textAlign": "center"},
                children=[
                    html.H2("Results", style={"marginTop": "20px"}),  # Header
                    html.Hr(),
                    html.Div(  # Blue box for the results section
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
                                    html.Div(  # Left half, horizontal bar chart
                                        style={
                                            "height": "22vw",
                                            "width": "50%",
                                            "overflowY": "auto",  # Enables scroll if needed (vertical direction)
                                            "border": "1px solid #ccc",
                                            "borderRadius": "8px"
                                        },
                                        children=dcc.Graph(  # Plot the horizontal bar chart
                                            id=f"{self.html_id}_h_fig",
                                            figure=self.h_fig,
                                            style={"height": f"{num_bars * bar_height + 100}px",
                                                   "minHeight": "100%"}  # Height is at least height of div
                                        )
                                    ),
                                    html.Div([  # Right half, show example images
                                        html.Div(  # Allows content to be next to each other
                                            style={
                                                "display": "flex",
                                                "justifyContent": "space-between",
                                                "width": "100%",
                                                "margin": "0 auto"
                                            },
                                            children=[  # Text displaying which images are shown
                                                html.P("", style={"textAlign": "left", "fontSize": "16pt"},
                                                       id="persons_showing_res_cls_txt"),
                                                html.P(new_txt, style={"textAlign": "right", "fontSize": "16pt"},
                                                id="images_showing_res_cls_txt")
                                            ]
                                        ),
                                        html.Div(  # Box to display the images in
                                            images_to_display,
                                            style={
                                                "backgroundColor": "white",
                                                "borderRadius": "12px",
                                                "border": "2px black solid",
                                                "textAlign": "center",
                                                "height": "80%"
                                            },
                                            id="example_images_res_box"
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
                                                    "⬅️ Back", id="box_res_images_left", style={
                                                        "width": "10%",
                                                        "opacity": 0.5
                                                    }, disabled=True),
                                                html.Button(  # Button to show the next 8 images
                                                    "Next ➡️", id="box_res_images_right", style={
                                                        "width": "10%",
                                                        "opacity": 0.5 if right_disabled else 1.0
                                                    }, disabled=right_disabled)
                                            ]
                                        ),
                                    ], style={"height": "22vw", "width": "50%"})
                                ]
                            ),
                            children_add_info  # Additional information figure section
                        ]
                    )
                ]
            )
        ]

        # Update output
        return children


    def update_v_fig(self, value, title):
        """
        Updates the vertical bar chart (additional information section) with a (new) column to display and a title.

        :param value: name of column to display information from
        :param title: title of the figure
        """
        # Update the additional information to display
        self.value = value

        # Check if a bar is selected and collect all values of the additional data to display
        if self.selected_bar is None:  # Use all information with poi in the images
            values = [i for _, row in self.df_main.iterrows() for i in row[self.value]]
        else:  # Use only information with poi and the selected bar (= a person) in the images
            list_ids = self.df_faces[self.df_faces["name"] == self.selected_bar].copy()["img_id"].to_list()
            values = [i for _, row in self.df_main.iterrows() if row["id"] in list_ids for i in row[self.value]]

        # Count the number of occurrences per value and sort them
        df_add_info = pd.DataFrame({self.value: values, "count": 1})
        df_add_info = df_add_info.groupby(self.value).sum().reset_index()
        df_add_info = df_add_info.sort_values(by="count", ascending=False)

        # Set the settings for the bars in the vertical bar chart
        bar_width = 70
        num_bars1 = len(df_add_info)

        # Plot the counts per <additional information value>
        self.v_fig = px.bar(df_add_info, x=self.value, y="count", orientation="v")

        # Update the layout of the figure
        self.v_fig.update_layout(
            margin=dict(l=0, r=10, b=0),
            showlegend=True,
            title_text=title,
            title_x=0.5,
            font=dict(size=14)
        )

        # Convert figure to html, make sure size is always at least the full width
        children = dcc.Graph(
            id=f"{self.html_id}_v_fig",
            figure=self.v_fig,
            style={"width": f"{num_bars1 * bar_width + 100}px",  # If larger than full width, use the bar size as measurement
                   "minWidth": "100%",  # At least full width
                   "height": "100%"}
        )

        # Return html figure
        return children


    def bar_clicked_update(self ,value, dropdown_val):
        """
        Updates the vertical bar chart and the displayed example images to
        the newly selected bar (person) or dropdown value (additional information name).

        :param value: name of the selected bar
        :param dropdown_val: name of column to display additional information from
        """
        print(f"(Results)             - Clicked on bar '{self.value}'")

        # Update self values
        self.selected_bar = value
        self.selected_index = 0
        self.images = []  # Images to show
        self.image_links = []  # Their corresponding url's (if exists)

        # Collect ids of images to display
        list_ids = self.df_faces[self.df_faces["name"] == self.selected_bar].copy()["img_id"].to_list()  # Contains selected bar
        list_ordered_ids = [row["id"] for _, row in self.df_main.iterrows() if row["id"] in list_ids]  # Also contains poi

        # Collect the images and corresponding url's to display
        for id in list_ordered_ids:
            # Collect the bounding box of the face from the selected bar
            box = self.df_faces[(self.df_faces["name"] == self.selected_bar) & (self.df_faces["img_id"] == id)]["face"].copy().to_list()[0]

            # Collect the image to display the face on and save it in the list
            image = self.df_selected_cluster[self.df_selected_cluster["img_id"] == id]["img_w_poi"].copy().to_list()[0]

            # Plot the face on the image in another color
            self.images.append(plot_faces_on_img(image.copy(), [box], 8, (255, 0, 255)))

            # If the images should be clickable to follow a link, add the url to the corresponding list
            if "url" in self.df_main.columns:
                self.image_links.append(self.df_main[self.df_main["id"] == id]["url"].copy().to_list()[0])

        # Update the images box and the corresponding components
        images_to_display, _, right_disabled, style_left, style_right, new_txt, _ = self.update_example_images_res()

        # Text displaying what images are shown, using colors to display the persons corresponding to their plotted color
        persons_showing_txt = [
            "Images with ",
            html.Span(self.selected_cluster, style={"color": "#00b400"}),  # Green for poi
            " and ",
            html.Span(self.selected_bar, style={"color": "#ff00ff"})  # Purple/pink for the selected bar
        ]

        # Update the vertical bar chart
        title_v_bar = f"Distribution of '{dropdown_val}' for all images containing '{self.selected_cluster}' and '{self.selected_bar}'"
        children = self.update_v_fig(dropdown_val, title_v_bar)

        # Update outputs
        return images_to_display, True, right_disabled, style_left, style_right, new_txt, persons_showing_txt, children


    def change_add_info(self, value):
        """
        Updates the vertical bar chart with a (new) column to display as additional information.

        :param value: name of column to display additional information from
        """
        print(f"(Results)             - Updated additional information to display to '{value}'")

        # Update title of the vertical bar chart in accordance with the selected bar
        if self.selected_bar is None:  # No bar (person) selected
            title_v_bar = f"Distribution of '{value}' for those images containing '{self.selected_cluster}'"
        else:  # Selected a bar
            title_v_bar = f"Distribution of '{value}' for all images containing '{self.selected_cluster}' and '{self.selected_bar}'"

        # Update the vertical bar chart
        children = self.update_v_fig(value, title_v_bar)

        # Update output
        return children


    def show_next_example_images_res(self, n_clicks):
        """
        Updates the image box to display the next 8 images.

        :param n_clicks: number of clicks on 'next' button
        """
        # If the button triggered this callback
        if n_clicks is not None and n_clicks > 0:
            # Update the index
            self.selected_index += 8

            # Collect the updated images to display and its corresponding components
            children, _, right_disabled, style_left, style_right, new_txt, nr = self.update_example_images_res()

            print(f"(Results)             - Displaying the next {nr} images")

            # Update outputs
            return children, False, right_disabled, style_left, style_right, new_txt
        # No update
        return dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update


    def show_previous_example_images_res(self, n_clicks):
        """
        Updates the image box to display the previous 8 images.

        :param n_clicks: number of clicks on 'back' button
        """
        # If this function is trigger by a button click
        if n_clicks is not None and n_clicks > 0:
            print(f"(Results)             - Displaying the previous 8 images")

            # Update the index
            self.selected_index -= 8

            # Collect the updated images to display and its corresponding components
            children, left_disabled, _, style_left, style_right, new_txt, _ = self.update_example_images_res()

            # Update outputs
            return children, left_disabled, False, style_left, style_right, new_txt
        # No update
        return dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update


    def update_example_images_res(self):
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
