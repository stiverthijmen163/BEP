import dash
from dash import dcc, html
import plotly.express as px
from PIL import Image
from functions import *
from sklearn.cluster import DBSCAN
import dash_bootstrap_components as dbc
import seaborn as sns
import matplotlib.colors as mcolors


# Set the colors used for plotting the cluster
palette = sns.color_palette("tab20", n_colors=20)
custom_palette = [mcolors.to_hex(color) for color in palette]


class Clusteror(html.Div):
    """
    Contains all functionalities of the clustering section for the 'New Data' page.
    """
    def __init__(self, name, df, df_faces):
        """
        Initializes the Clusteror object.

        :param name: name (id) of the Clusteror object
        :param df: dataframe containing the data of all images
        :param df_faces: dataframe containing the faces of all images
        """
        print(f"Initializing the Clusteror object: '{name}'")

        # Initialize all variables for the Clusteror object
        self.html_id = name
        self.df = df.copy()
        self.df_faces = df_faces.copy()
        self.fig = None  # The scatter plot for clustering visualization
        self.eps = 5.4  # Eps parameter DBSCAN
        self.min_samples = 1  # Min_samples parameter DBSCAN
        self.min_size = (0, 0)  # Min size of face to be considered for clustering
        self.selected_index = 0  # Currently selected index of first image shown in image box
        self.selected_cluster = None  # The currently selected cluster

        # --------------------- <Hard-code the dataset used for demonstrating the visualization> -----------------------

        # # - Saving some data (hard-coded)
        # # Change format of columns to meet de SQL database requirements
        # self.df_faces["face"] = self.df_faces["face"].apply(json.dumps)
        # self.df_faces["img"] = self.df_faces["img"].apply(lambda x: json.dumps(x.tolist()))
        # # Save the embeddings
        # conn = sqlite3.connect("temp.db")
        # self.df_faces.to_sql("faces", conn, if_exists="replace")
        #
        # - Load the embeddings (hard-coded)
        # Load the database
        # conn = sqlite3.connect("temp.db")
        # query = """SELECT * FROM faces"""
        # self.df_faces = pd.read_sql_query(query, conn)
        #
        # # Change the format of columns such that they can be read without further preprocessing
        # self.df_faces["embedding_tsne"] = self.df_faces["embedding_tsne"].apply(lambda x: np.fromstring(x, sep=","))
        # self.df_faces["embedding"] = self.df_faces["embedding"].apply(lambda x: np.fromstring(x, sep=","))
        # self.df_faces["face"] = self.df_faces["face"].apply(json.loads)
        # self.df_faces["img"] = self.df_faces["img"].apply(json.loads)
        # self.df_faces["img"] = self.df_faces["img"].apply(lambda x: np.array(x, dtype=np.uint8))

        # --------------------------------------------------- <END> ----------------------------------------------------

        # Change the format of columns such that they can be read without further preprocessing
        # (comment when hard-coded data is used)
        self.df_faces["embedding_tsne"] = self.df_faces["embedding_tsne"].apply(lambda x: np.fromstring(x, sep=","))
        self.df_faces["embedding"] = self.df_faces["embedding"].apply(lambda x: np.fromstring(x, sep=","))

        # Collect clusters
        labels = DBSCAN(eps=self.eps, min_samples=self.min_samples, n_jobs=-1, metric="euclidean").fit_predict(
            self.df_faces["embedding_tsne"].tolist())

        # Set a counter to know what value to use when adding a new cluster
        self.counter = max(labels)

        # Update all labels to be formatted as strings
        labels = [str(i) for i in labels]

        # Append clusters to dataset
        self.df_faces["cluster"] = labels  # Original clusters (can't change)
        self.df_faces["name"] = labels  # Name of the cluster (can change)

        # Rename the unknown cluster (='-1')
        self.df_faces.loc[self.df_faces["cluster"] == "-1", "name"] = "unknown"

        # Initialize the scatter plot
        self.update_fig()

        # Find the maximum value for the eps parameter for DBSCAN
        maximum = round(max(max(self.df_faces["tsne_x"]) - min(self.df_faces["tsne_x"]),
                      max(self.df_faces["tsne_y"]) - min(self.df_faces["tsne_y"])), 1)

        # Set the images to display as example
        self.images = self.df_faces[self.df_faces["cluster"] == "0"]["img"].copy().to_list()

        # Initialize the images to display, whether the 'next' buttons is disabled or not and
        # the number of images displayed
        children, _, right_disabled, _, _, _, nr = self.update_displayed_images()

        # Initialize the Clusteror object
        super().__init__(
            className="graph_card", id=self.html_id,
            children=[  # Header
                html.H2("Face Clustering", style={"textAlign": "center"}),
                html.Hr(),
                html.Div(
                    style={  # Blue box to work in
                        "backgroundColor": "#dbeafe",
                        "margin": "20px auto",
                        "width": "99vw",
                        "borderRadius": "12px",
                        "boxShadow": "0px 2px 5px rgba(0,0,0,0.1)",
                        "textAlign": "center"
                    },
                    children=[
                        html.Br(),
                        html.Div(
                            style={  # Work within a slightly smaller space
                                "display": "flex",
                                "justifyContent": "space-between",
                                "alignItems": "flex-start",
                                "gap": "10px",
                                "width": "99%",
                                "margin": "10px auto"
                            },
                            children=[
                                dcc.Graph(  # The scatter plot (left)
                                    id=f"{self.html_id}",
                                    figure=self.fig,
                                    style={"flex": "1",
                                           "height": "22vw"}
                                ),
                                html.Div(  # A white box on the right half
                                    id="div_cls_right",
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
                                            "First set the parameters such that the graph shows clusters you are satisfied with. Then, you can edit any cluster.",
                                            style={
                                                "fontSize": "16pt",
                                                "marginBottom": "5px",
                                                "textAlign": "center",
                                                "margin": "5px",
                                                "fontWeight": "bold"
                                            }
                                        ),
                                        html.Div([  # Change eps parameter for DBSCAN
                                            html.P(
                                                [
                                                    "eps:",
                                                    html.I(  # Info icon
                                                        className="fa-solid fa-circle-question", id="info_icon_eps",
                                                        style={"cursor": "pointer", "color": "#0d6efd",
                                                            "marginLeft": "5px",
                                                            "position": "relative",
                                                            "top": "-3px"
                                                        }
                                                    ),
                                                    " ",
                                                    dcc.Input(  # Input eps parameter
                                                        id="eps_input",
                                                        type="number",
                                                        min=0.1,
                                                        max=maximum,
                                                        step=0.1,
                                                        value=self.eps,
                                                        style={"marginLeft": "10px", "width": "8%"}
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
                                                "The maximum distance between two samples for one to be considered as in the neighborhood of the other. This is not a maximum bound on the distances of points within a cluster.",
                                                target="info_icon_eps",
                                                placement="top"
                                            )
                                        ]),
                                        html.Br(),
                                        html.Div([  # Change min_samples parameter for DBSCAN
                                            html.P(
                                                [
                                                    "min_samples:",
                                                    html.I(  # Info icon
                                                        className="fa-solid fa-circle-question", id="info_icon_min_samples",
                                                        style={"cursor": "pointer", "color": "#0d6efd",
                                                            "marginLeft": "5px",
                                                            "position": "relative",
                                                            "top": "-3px"
                                                        }
                                                    ),
                                                    " ",
                                                    dcc.Input(  # Input for min_samples parameter
                                                        id="min_samples_input",
                                                        type="number",
                                                        min=1,
                                                        max=20,
                                                        step=1,
                                                        value=self.min_samples,
                                                        style={"marginLeft": "10px", "width": "8%"}
                                                    )
                                                ],
                                                style={
                                                    "fontSize": "16pt",
                                                    "marginBottom": "5px",
                                                    "textAlign": "center",
                                                    "margin": "0"
                                                },
                                            ),
                                            dbc.Tooltip(  # When hovering over info icon display some explanation
                                                "The number of samples (or total weight) in a neighborhood for a point to be considered as a core point. This includes the point itself. If min_samples is set to a higher value, DBSCAN will find denser clusters, whereas if it is set to a lower value, the found clusters will be more sparse.",
                                                target="info_icon_min_samples",
                                                placement="top"
                                            )
                                        ]),
                                        html.Br(),
                                        html.Div([  # Change minimum size of faces to be considered for clustering
                                            html.P(
                                                [
                                                    "min_size:",
                                                    html.I(  # Info icon
                                                        className="fa-solid fa-circle-question",
                                                        id="info_icon_min_size",
                                                        style={"cursor": "pointer", "color": "#0d6efd",
                                                            "marginLeft": "5px",
                                                            "position": "relative",
                                                            "top": "-3px"
                                                        }
                                                    ),
                                                    " width:",
                                                    dcc.Input(  # Input for minimum width
                                                        id="min_width_cls_input",
                                                        type="number",
                                                        min=0,
                                                        max=max(self.df_faces["width"]),
                                                        step=1,
                                                        value=self.min_size[0],
                                                        style={"marginLeft": "10px", "width": "8%"}
                                                    ),
                                                    " height:",
                                                    dcc.Input(  # Inout for minimum height
                                                        id="min_height_cls_input",
                                                        type="number",
                                                        min=0,
                                                        max=max(self.df_faces["height"]),
                                                        step=1,
                                                        value=self.min_size[1],
                                                        style={"marginLeft": "10px", "width": "8%"}
                                                    )
                                                ],
                                                style={
                                                    "fontSize": "16pt",
                                                    "marginBottom": "5px",
                                                    "textAlign": "center",
                                                    "margin": "0"
                                                },
                                            ),
                                            dbc.Tooltip(  # When hovering over info icon display some explanation
                                                "Select the minimum size of a face to be considered for clustering, faces that are smaller will be put in the 'unknown' cluster.",
                                                target="info_icon_min_size",
                                                placement="top"
                                            )
                                        ]),
                                        html.Br(),
                                        html.Div(  # Allows buttons to be displayed next to each other
                                            style={
                                                "display": "flex",
                                                "justifyContent": "center",
                                                "gap": "5%"
                                            },
                                            children=[
                                                html.Button(  # Button that updates cluster using the updated parameters
                                                    "Update Clusters",
                                                    disabled=False,
                                                    style={
                                                        "padding": "10px 20px",
                                                        "fontSize": "16pt",
                                                        "borderRadius": "12px",
                                                        "border": "none",
                                                        "backgroundColor": "#2196F3",
                                                        "color": "white",
                                                        "cursor": "pointer",
                                                        "width": "10vw"
                                                    },
                                                    id="button_update_clusters"
                                                ),
                                                html.Div(children=[
                                                    html.Button(  # Button that continues to 'edit' mode
                                                        "Continue",
                                                        disabled=False,
                                                        style={
                                                            "padding": "10px 0px",
                                                            "fontSize": "16pt",
                                                            "borderRadius": "12px",
                                                            "border": "none",
                                                            "backgroundColor": "#2196F3",
                                                            "color": "white",
                                                            "cursor": "pointer",
                                                            "width": "10vw"
                                                        },
                                                        id="button_continue_clusters"
                                                    ),
                                                    html.I(  # Warning icon
                                                        className="fa-solid fa-circle-exclamation",
                                                        id="exclamation_button",
                                                        style={"cursor": "pointer", "color": "red",
                                                              "marginLeft": "5px",
                                                              "position": "relative",
                                                              "top": "-8pt"
                                                        }
                                                    ),
                                                    dbc.Tooltip(  # When hovering over the warning icon display some warning
                                                        "When this button is pressed, the parameters can't be changed anymore.",
                                                        target="exclamation_button",
                                                        placement="top"
                                                    )
                                                ], style={
                                                    "fontSize": "16pt"
                                                })
                                            ]
                                        ),
                                        html.Br(),
                                        html.Label(  # Display explanation for user what to do
                                            "Select cluster to inspect: ",
                                            style={
                                                "fontSize": "16pt",
                                                "marginBottom": "5px",
                                                "textAlign": "center",
                                                "margin": "0"
                                            },
                                        ),
                                        html.Div(  # Place holder for dropdown menu
                                            id="dropdown_change",
                                            children=
                                            [dcc.Dropdown(  # Dropdown menu for selecting a different cluster
                                                id="dropdown_cls",
                                                options=sort_items(self.df_faces["name"].unique()),
                                                value="0",
                                                clearable=False,
                                                className="dropdown",
                                                searchable=True,
                                            )],
                                            style={
                                                "width": "80%",
                                                "textAlign": "center",
                                                "margin": "0 auto"
                                            }
                                        )
                                    ]
                                )
                            ]
                        ),
                        html.Div([  # Allows multiple text components to be displayed next to each other
                            html.Div("", style={"flex": 1}),  # Left spacer
                            html.Div(html.P(  # Text showing what cluster is currently displayed/selected
                                f"Showing cluster '0'.",
                                id="showing_cluster0",
                                style={
                                    "fontSize": "16pt",
                                    "marginBottom": "5px",
                                    "textAlign": "center",
                                    "margin": "0"
                            }), style={"flex": 1, "display": "flex", "justifyContent": "center"}),  # Centered content

                            html.Div(html.P(  # Text displaying what range of images is displayed
                                f"Showing 1 - {nr} out of {len(self.images)}",
                                id="images_showing_txt1", style={
                                    "fontSize": "16pt",
                                    "marginBottom": "5px",
                                    "textAlign": "center",
                                    "margin": "0"
                            }), style={"flex": 0.995, "display": "flex", "justifyContent": "flex-end",
                                       "paddingRight": "0.5%"})  # Right content
                        ], style={
                            "display": "flex",
                            "alignItems": "center",
                            "width": "100%"
                        }),
                        html.Div(  # White box for displaying example images
                            style={
                                "backgroundColor": "white",
                                "margin": "5px auto",
                                "width": "99%",
                                "borderRadius": "12px",
                                "border": "2px black solid",
                                "textAlign": "center"
                            },
                            children=children,  # Images to display
                            id="box_images1"
                        ),
                        html.Div(  # Allows  content to be next to each other
                            style={
                                "display": "flex",
                                "justifyContent": "space-between",
                                "width": "99%",
                                "margin": "10px auto"
                            },
                            children=[
                                html.Button(  # Button to show the previous 10 images
                                    "⬅️ Back", id="box_images1_left", style={
                                    "width": "5%",
                                    "opacity": 0.5
                                }, disabled=True),
                                html.Button(  # Button to show the next 10 images
                                    "Next ➡️", id="box_images1_right", style={
                                    "width": "5%",
                                    "opacity": 0.5 if right_disabled else 1.0
                                }, disabled=right_disabled)
                            ]
                        ),
                        html.Div([  # Allows components to be next to each other
                                html.P([
                                    "Name of the database:",
                                    dcc.Input(  # Input for the name of the database to save to
                                        id="name_of_db",
                                        type="text",
                                        placeholder="Only letters and '_' allowed",
                                        value="",
                                        style={"width": "30vw", "marginLeft": "1vw"}
                                    )],
                                    style={
                                        "fontSize": "16pt",
                                        "textAlign": "center"
                                    }
                                )
                            ],
                            style={
                                "display": "flex",
                                "alignItems": "center",
                                "justifyContent": "center",
                                "margin": "0 auto",
                                "gap": "1vw"
                            }
                        ),
                        html.Button(  # Button to trigger saving the datasets to the database
                            "Save to Database",
                            disabled=True,  # Disabled initially
                            style={
                                "padding": "10px 20px",
                                "fontSize": "16pt",
                                "borderRadius": "12px",
                                "border": "none",
                                "backgroundColor": "#2196F3",
                                "color": "white",
                                "cursor": "pointer",
                                "width": "20vw",
                                "marginBottom": "20px",
                                "opacity": 0.5
                            },
                            id="button3"
                        ),
                        html.P(  # Text to show feedback to the user regarding their attempt at saving the data
                            "",
                            style = {
                                "fontSize": "16pt",
                                "marginBottom": "5px",
                                "textAlign": "center",
                                "margin": "5px",
                                "fontWeight": "bold"
                            },
                            id="successful_database"
                        ),
                        html.Br()
                    ]
                )
            ]
        )


    def next_button(self, n_clicks):
        """
        Updates the image box to display the next 10 images.

        :param n_clicks: number of clicks on 'next' button
        """
        # If the button triggered this callback
        if n_clicks is not None and n_clicks > 0:
            # Update the index
            self.selected_index += 10

            # Collect the updated images to display and its corresponding components
            children, _, right_disabled, style_left, style_right, new_txt, nr = self.update_displayed_images()

            print(f"Displaying the next {nr} images")

            # Update outputs
            return children, False, right_disabled, style_left, style_right, new_txt
        # No update
        return dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update


    def previous_button(self, n_clicks):
        """
        Updates the image box to display the previous 10 images.

        :param n_clicks: number of clicks on 'back' button
        """
        # If this function is trigger by a button click
        if n_clicks is not None and n_clicks > 0:
            print(f"Displaying the previous 10 images")

            # Update the index
            self.selected_index -= 10

            # Collect the updated images to display and its corresponding components
            children, left_disabled, _, style_left, style_right, new_txt, _ = self.update_displayed_images()

            # Update outputs
            return children, left_disabled, False, style_left, style_right, new_txt
        # No update
        return dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update


    def update_clusters(self, data, eps_input, min_samples_input, min_width_cls_input, min_height_cls_input):
        """
        Updates the cluster in accordance with the updated parameters.

        :param data: trigger that states whether to update the clusters or not
        :param eps_input: eps input value, parameter of DBSCAN method
        :param min_samples_input: min_samples input value, parameter of DBSCAN method
        :param min_width_cls_input: minimum width of a face to be considered for clustering
        :param min_height_cls_input: minimum height of a face to be considered for clustering
        """
        # Check if clusters should be updated
        if data is not None and data:
            print(f"Updating clusters using the following parameters: eps: {eps_input}, min_samples:"
                  f" {min_samples_input}, min_size: ({min_width_cls_input}, {min_height_cls_input})")

            # Update the self values with new inputted values
            if eps_input is not None:  # Eps parameter
                self.eps = eps_input
            if min_samples_input is not None:  # Min_samples parameter
                self.min_samples = min_samples_input
            if min_width_cls_input is not None:  # Min width of face
                self.min_size = (min_width_cls_input, self.min_size[1])
            if min_height_cls_input is not None:  # Min height of face
                self.min_size = (self.min_size[0], min_height_cls_input)

            # Create new dataframes taking min size of face into account
            df_temp0 = self.df_faces[(self.df_faces["width"] >= self.min_size[0]) & (self.df_faces["height"] >= self.min_size[1])].copy()
            df_temp1 = self.df_faces[(self.df_faces["width"] < self.min_size[0]) | (self.df_faces["height"] < self.min_size[1])].copy()

            # Collect new clusters
            labels = DBSCAN(eps=self.eps, min_samples=self.min_samples, n_jobs=-1, metric="euclidean").fit_predict(
                df_temp0["embedding_tsne"].tolist())

            # Set a counter to know what value to use when adding a new cluster
            self.counter = max(labels)

            # Convert labels to strings
            labels = [str(i) for i in labels]

            # Append labels to dataset
            df_temp0["cluster"] = labels  # Original cluster (can't change)
            df_temp0["name"] = labels  # Name of cluster (Can change)

            # Rename the 'unknown' cluster ('-1')
            df_temp0.loc[df_temp0["cluster"] == "-1", "name"] = "unknown"

            # All faces smaller than the minimum size are put in the 'unknown' cluster
            df_temp1["cluster"] = "-1"
            df_temp1["name"] = "unknown"

            # Combine the 2 dataframes
            self.df_faces = pd.concat([df_temp0, df_temp1], ignore_index=True)

            # Update the figure
            self.update_fig()

            # Set the images to display as example
            self.images = self.df_faces[self.df_faces["cluster"] == "0"]["img"].copy().to_list()

            # Reset the index tracker
            self.selected_index = 0

            # Collect the updated images to display and its corresponding components
            children_img, _, right_disabled, style_left, style_right, new_txt, _ = self.update_displayed_images()

            # Text displaying what cluster is currently selected
            showing_cls = "Showing cluster '0'."

            # Update the dropdown menu to select a cluster with, with the updated list of clusters
            children = [dcc.Dropdown(
                id="dropdown_cls",
                options=sort_items(self.df_faces["name"].unique()),
                value="0",
                clearable=False,
                className="dropdown",
                searchable=True,
            )]

            # Update style of the 'update cluster' button to show the user it is enabled
            style = {
                "padding": "10px 20px",
                "fontSize": "16pt",
                "borderRadius": "12px",
                "border": "none",
                "backgroundColor": "#2196F3",
                "color": "white",
                "cursor": "pointer",
                "width": "10vw"
            }

            # Update outputs
            return self.fig, children_img, new_txt, showing_cls, True, right_disabled, style_left, style_right, children, False, style, False, False
        # No update
        return dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update


    def update_image_box(self, value, data):
        """
        Updates the image box in accordance with the newly selected cluster.

        :param value: selected cluster
        :param data: whether to enable clickable images
        """
        if value is not None:
            print(f"Selected cluster '{value}'")
            # Update the images
            self.images = self.df_faces[self.df_faces["name"] == value]["img"].copy().to_list()

            # Reset the index tracker
            self.selected_index = 0

            # Update the currently selected cluster
            self.selected_cluster = value

            # Collect the updated images to display and its corresponding components
            children, _, right_disabled, style_left, style_right, new_txt, _ = self.update_displayed_images()

            # Update the text displaying which cluster is currently selected
            showing_cls = f"Showing cluster '{value}'."

            # Trigger update on remaining components in the clustering section
            if data is None:
                d = 0
            else:
                d = data + 1
                if data:  # Update text to allow for selecting images to move to another cluster
                    showing_cls = f"Showing cluster '{value}'. Select a face to move to another cluster"

            # Update outputs
            return children, new_txt, showing_cls, True, right_disabled, style_left, style_right, d
        # No update
        return dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update


    def update_components_on_cls_change(self):
        """
        Updates the remaining components in the clustering section:
        -   value of name of cluster (empty)
        -   dropdown containing the clusters with which you can merge
        -   the image clicked on (empty).
        """
        print(f"Updating remaining components in the clustering section in accordance with the selected cluster")
        # Reset the name of the cluster input
        value = ""

        # Update the dropdown of cluster with which you can merge
        _, children_merge = self.update_merge_dropdown()

        # Update outputs
        return value, children_merge, []


    def continue_to_edit(self, n_clicks):
        """
        Updates the layout of the clustering section to allow the user to edit clusters.

        :param n_clicks: number of clicks on 'continue' button
        """
        # Check if the 'continue' button triggered this function
        if n_clicks is not None and n_clicks > 0:
            print("Continuing to 'edit' mode")
            # Sort all clusters except for the currently selected cluster (items for dropdown)
            items = sort_items(np.delete(self.df_faces["name"].copy().unique(), np.where(self.df_faces["name"].copy().unique() == "0")))

            # Collect the value to display as standard value in the dropdown
            val = items[0] if len(items) > 0 else None

            # Reset the currently selected cluster
            self.selected_cluster = "0"

            # Update the layout
            children = [
                html.P(  # Explanatory text
                    "Edit any cluster, you can move faces to other clusters, merge clusters and rename them. Empty clusters will be removed.",
                    style={
                        "fontSize": "16pt",
                        "marginBottom": "5px",
                        "textAlign": "center",
                        "margin": "5px",
                        "fontWeight": "bold"
                    }
                ),
                html.Label(  # Text that tells the user what to select
                    "Select cluster to inspect/edit: ",
                    style={
                        "fontSize": "16pt",
                        "marginBottom": "5px",
                        "textAlign": "center",
                        "margin": "0"
                    },
                ),
                html.Div(  # Placeholder for dropdown menu
                    id="dropdown_change",
                    children=
                        [dcc.Dropdown(  # Dropdown containing all cluster to select
                        id="dropdown_cls",
                        options=sort_items(self.df_faces["name"].unique()),
                        value="0",
                        clearable=False,
                        className="dropdown",
                        searchable=True
                    )],
                    style={
                        "width": "80%",
                        "textAlign": "center",
                        "margin": "0 auto"
                    }
                ),
                html.Div([  # Allows components to be next to each other
                    html.P(
                        [
                            "Cluster Name: ",
                            dcc.Input(  # Input for setting a new name for the currently selected cluster
                                id="name_of_cluster",
                                type="text",
                                placeholder="Only letters, spaces and '_' allowed",
                                value="",
                                style={"marginLeft": "10px", "width": "50%"}
                            ),
                            html.Button(  # Button to change the name
                                "Change Name",
                                disabled=False,
                                style={
                                    "padding": "10px 0px",
                                    "fontSize": "16pt",
                                    "borderRadius": "12px",
                                    "border": "none",
                                    "backgroundColor": "#2196F3",
                                    "color": "white",
                                    "cursor": "pointer",
                                    "width": "10vw",
                                    "marginLeft": "1vw",
                                    "marginTop": "5px"
                                },
                                id="button_change_name"
                            )
                        ],
                        style={
                            "fontSize": "16pt",
                            "marginBottom": "5px",
                            "textAlign": "center",
                            "margin": "0"
                        },
                    ),
                ]),
                html.Div([  # Allows components to be next to each other
                    html.P(style={
                            "display": "flex",
                            "justifyContent": "center",
                            "gap": "1vw",
                            "marginTop": "5px",
                            "alignItems": "center"
                        },
                        children=[
                            html.Div([
                                html.P(
                                    [
                                        "Merge with cluster:",
                                        html.I(  # Info icon
                                            className="fa-solid fa-circle-question", id="info_icon_merge_cls",
                                            style={"cursor": "pointer", "color": "#0d6efd",
                                                "marginLeft": "5px",
                                                "position": "relative",
                                                "top": "-3px"
                                            }
                                        ),
                                        " ",
                                    ],
                                    style={
                                        "fontSize": "16pt",
                                        "marginBottom": "5px",
                                        "textAlign": "center",
                                        "margin": "0"
                                    },
                                ),
                                dbc.Tooltip(  # When hovering over the info icon, display some explanation
                                    "The name of the currently selected cluster is kept.",
                                    target="info_icon_merge_cls",
                                    placement="top"
                                )
                            ]),
                            html.Div(  # Placeholder for dropdown containing the merge possibilities
                                id="dropdown_merge_change",
                                children=dcc.Dropdown(  # Dropdown menu with all possible cluster to merge with
                                    id="dropdown_cls_merge",
                                    options=items,
                                    value=val,
                                    clearable=False,
                                    className="dropdown",
                                    searchable=True,
                                ),
                                style={"width": "50%"}
                            ),
                            html.Button(  # Button to merge the selected clusters
                                "Merge clusters",
                                disabled=False,
                                style={
                                    "padding": "10px 0px",
                                    "fontSize": "16pt",
                                    "borderRadius": "12px",
                                    "border": "none",
                                    "backgroundColor": "#2196F3",
                                    "color": "white",
                                    "cursor": "pointer",
                                    "width": "10vw"
                                },
                                id="button_merge_clusters"
                            )
                        ]
                    )],
                    style={
                        "fontSize": "16pt",
                        "marginBottom": "5px",
                        "textAlign": "center",
                        "margin": "0"
                    }
                ),
                html.Div(  # placeholder for the selected image to move to another cluster
                    children=[],
                    id="selected_image_div_cls",
                    style={
                        "display": "flex",
                        "width": "99%",
                        "height": "35%",
                        "margin": "0 auto",
                        "gap": "1vw"
                    }
                )
            ]

            # Update the style of the 'save to database' button so that the user knows it is enabled
            style = {
                "padding": "10px 20px",
                "fontSize": "16pt",
                "borderRadius": "12px",
                "border": "none",
                "backgroundColor": "#2196F3",
                "color": "white",
                "cursor": "pointer",
                "width": "20vw",
                "marginBottom": "20px"
            }

            # Update the text which displays the currently selected cluster and an instruction
            new_txt = f"Showing cluster '{self.selected_cluster}'. Select a face to move to another cluster"

            return children, True, False, style, new_txt
        # No update
        return dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update


    def change_name(self, n_clicks, name):
        """
        Changes the name of the currently selected cluster.

        :param n_clicks: the number of clicks on the 'change name' button
        :param name: the name to change the currently selected cluster to
        """
        # If the 'change name' button triggered this function
        if n_clicks is not None and n_clicks > 0:
            # In case of an empty name, name becomes 'None'
            if name == "":
                name = "None"

            print(f"Update name of cluster '{self.selected_cluster}' to '{name}'")
            # Change the name in the dataset
            self.df_faces.loc[self.df_faces["name"] == self.selected_cluster, "name"] = f"{name}"

            # Update the figure
            self.update_fig()

            # Update the dropdown placeholder with the new list of clusters
            children = [dcc.Dropdown(
                id="dropdown_cls",
                options=sort_items(self.df_faces["name"].unique()),
                value=f"{name}",
                clearable=False,
                className="dropdown",
                searchable=True,
            )]

            # Update the merge dropdown placeholder with the new list of clusters to merge with
            _, children_merge = self.update_merge_dropdown()

            # Update selected cluster to new name
            self.selected_cluster = f"{name}"

            # Update outputs
            return self.fig, children, children_merge, []
        # No update
        return dash.no_update, dash.no_update, dash.no_update, dash.no_update


    def update_fig(self):
        """
        Updates the figure (scatter plot) to correspond with the current dataset.
        """
        # Update the figure
        self.fig = px.scatter(self.df_faces, x="tsne_x", y="tsne_y", color="name",
                              color_discrete_sequence=custom_palette)

        # Update the layout
        self.fig.update_layout(
            margin=dict(l=0, r=0, b=0),  # Use all available space, leave space at the top for title
            showlegend=True,  # Show the legend
            # Add a title
            title_text=f"Resulting clusters with eps={self.eps}, min_samples={self.min_samples}, min_size={self.min_size}",
            title_x=0.5  # Centre the title
        )

        # Hide the "unknown" cluster initially, but keep it in the legend so that you can enable it
        self.fig.for_each_trace(
            lambda t: t.update(visible="legendonly") if t.name == "unknown" else None
        )


    def merge_clusters(self, n_clicks, cluster):
        """
        Merges the currently selected cluster with another one,
        keeps the name of the currently selected cluster.

        :param n_clicks: the number of clicks on the 'merge' button
        :param cluster: cluster to merge with
        """
        # If 'merge' button triggered this function and the currently selected cluster is non-empty
        if n_clicks is not None and n_clicks > 0 and self.selected_cluster in self.df_faces["name"].unique():
            print(f"Merge cluster '{self.selected_cluster}' with cluster '{cluster}'")

            # Collect the name of the currently selected cluster
            name = self.df_faces[self.df_faces["name"] == self.selected_cluster]["name"].values[0]

            # Merge the clusters by sharing the name
            self.df_faces.loc[self.df_faces["name"] == cluster, "name"] = f"{name}"
            self.update_fig()

            # Update the dropdown menu for selecting a cluster to inspect
            children = [dcc.Dropdown(
                id="dropdown_cls",
                options=sort_items(self.df_faces["name"].unique()),
                value=f"{name}",
                clearable=False,
                className="dropdown",
                searchable=True,
            )]

            # Update the dropdown for clusters to merge with
            items, children_merge = self.update_merge_dropdown()

            # Update the style of the 'merge' button
            style = {
                "padding": "10px 0px",
                "fontSize": "16pt",
                "borderRadius": "12px",
                "border": "none",
                "backgroundColor": "#2196F3",
                "color": "white",
                "cursor": "pointer",
                "width": "10vw",
                "opacity": 1.0 if len(items) > 0 else 0.5  # Enable only if there are clusters to merge with
            }

            # Disable button if no cluster to merge with are left
            disabled = False if len(items) > 0 else True

            # Update outputs
            return self.fig, children, children_merge, disabled, style, []
        # No update
        return dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update


    def update_merge_dropdown(self):
        """
        Updates the dropdown containing all clusters you can merge with the currently selected cluster.
        """

        # Collect all sorted names of cluster except for the currently selected cluster
        items = sort_items(np.delete(self.df_faces["name"].copy().unique(),
                          np.where(self.df_faces["name"].copy().unique() == self.selected_cluster)))

        # Set the standard value to use if one exists
        val = items[0] if len(items) > 0 else None

        # Update the dropdown
        c = dcc.Dropdown(
            id="dropdown_cls_merge",
            options=items,
            value=val,
            clearable=False,
            className="dropdown",
            searchable=True,
        )

        return items, c


    def add_new_cluster(self, selected_data):
        """
        Adds a new cluster containing the selected points.

        :param selected_data: the selected points
        """
        # Check if the data contains any points
        if selected_data is not None and len(selected_data["points"]) > 0:
            # Get the selected coordinates
            selected_points = selected_data["points"]
            selected_coords = [(point["x"], point["y"]) for point in selected_points]

            # Update counter
            self.counter += 1
            print(f"Adding a new cluster: {self.counter}")

            # Change name of faces where the coordinates match the input coordinates
            self.df_faces["name"] = self.df_faces.apply(
                lambda row: f"{self.counter}" if (row["tsne_x"], row["tsne_y"]) in selected_coords else row["name"],
                axis=1
            )

            # Update the figure
            self.update_fig()

            # Update the dropdown containing the selectable clusters
            children = [dcc.Dropdown(
                id="dropdown_cls",
                options=sort_items(self.df_faces["name"].unique()),
                value=f"{self.counter}",
                clearable=False,
                className="dropdown",
                searchable=True,
            )]

            # Update outputs
            return self.fig, children, []
        # No update
        return dash.no_update, dash.no_update, dash.no_update


    def select_image(self, data):
        """
        Updates the selected image placeholder when an image is clicked on.

        :param data: index of image which has been clicked on
        """
        print(f"Selected face: {data}")
        # Get the index of the image
        index = self.selected_index + data

        # Read the image
        img = Image.fromarray(cv2.cvtColor(self.images[index].copy(), cv2.COLOR_BGR2RGB))

        # Get all clusters to which this image can be moved
        items = sort_items(np.delete(self.df_faces["name"].copy().unique(),
                                     np.where(self.df_faces["name"].copy().unique() == self.selected_cluster)))

        # Set the initial value for the dropdown
        val = items[0] if len(items) > 0 else None

        # Disable the 'move' button if no other clusters exist
        disabled = False if len(items) > 0 else True

        # Update the layout of the selected image placeholder
        children = [
            html.Img(  # Plot the face
                src=img,
                style={
                    "maxWidth": "10vw",
                    "maxHeight": "100%",
                    "width": "auto",
                    "height": "100%",
                    "display": "block",
                    "marginLeft": "auto",
                    "marginRight": "auto",
                    "objectFit": "contain"
                }
            ),
            html.Div(  # Allows components to be next to each other
                [
                    html.Label("Move to cluster:",
                               style={"fontSize": "16pt", "alignSelf": "center"}),
                    dcc.Dropdown(  # Dropdown with all options to move the face to
                        id="dropdown_move_to_cls",
                        options=items,
                        value=val,
                        clearable=False,
                        className="dropdown",
                        searchable=True,
                        style={"width": "20vw"}
                    ),
                    html.Button(  # The button which moves the face to another cluster
                        "Move",
                        disabled=disabled,
                        id="button_move",
                        style={
                            "padding": "10px 0px",
                            "fontSize": "16pt",
                            "borderRadius": "12px",
                            "border": "none",
                            "backgroundColor": "#2196F3",
                            "color": "white",
                            "cursor": "pointer",
                            "width": "5vw",
                            "marginTop": "0px"
                        }
                    )
                ],
                style={
                    "display": "flex",
                    "alignItems": "center",
                    "justifyContent": "center",
                    "margin": "0 auto",
                    "gap": "1vw"
                }
            )
        ]

        # Update outputs (layout)
        return children


    def move_face(self, n_clicks, value, index):
        """
        Moves an image to another cluster when the 'move' button is pressed.

        :param n_clicks: number of clicks on the 'move' button
        :param value: cluster to move the image to
        :param index: index of image which has been clicked on
        """
        if n_clicks is not None and n_clicks > 0:
            print(f"Move selected face to cluster '{value}'")
            # Get the index of the image
            i = self.selected_index + index

            # Get the id corresponding to the image
            img_id = self.df_faces[self.df_faces["name"] == self.selected_cluster]["id"].copy().to_list()[i]

            # Move the image to the new cluster
            self.df_faces.loc[self.df_faces["id"] == img_id, "name"] = f"{value}"

            # update the self images which keeps track of all images in the currently selectyed cluster
            self.images = self.df_faces[self.df_faces["name"] == self.selected_cluster]["img"].copy().to_list()

            # Update the fig
            self.update_fig()

            # If there are still images in the currently selected cluster
            if len(self.images) > 0:
                # If there are no images to display at the current index
                if len(self.images) == self.selected_index:
                    # Update index
                    self.selected_index -= 10

                # Collect the updated images to display and its corresponding components
                children, left_disabled, right_disabled, style_left, style_right, new_txt, _ = self.update_displayed_images()
            else:  # Currently selected cluster is empty
                children = []  # No images to display
                new_txt = f"Showing 0 - 0 out of {len(self.images)}"  # No images to display
                left_disabled = True  # Disable 'back' button
                right_disabled = True  # Disable 'next' button

                # Show the user the buttons are disabled
                style_left = {"width": "5%", "opacity": 0.5}
                style_right = {"width": "5%", "opacity": 0.5}

            # update outputs
            return self.fig, [], children, new_txt, left_disabled, right_disabled, style_left, style_right
        # No update
        return dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update


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
            image = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

            # Display img as html.Image and append to the list
            children.append(
                html.Div(
                    id={"type": "image-click1", "index": i},  # Pattern-matching id
                    style={"display": "inline-block", "width": "9%", "padding": "0.5%"},
                    n_clicks=0,  # Enables click tracking
                    children=html.Img(src=image, style={
                        "width": "100%",
                        "cursor": "pointer",
                        "maxHeight": "8vw",
                        "objectFit": "contain",
                    })
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
