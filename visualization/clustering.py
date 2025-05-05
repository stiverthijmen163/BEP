import pandas as pd
import dash
from dash import dcc, html, callback, Output, Input, State
import plotly.express as px
import plotly.graph_objects as go
import cv2
from PIL import Image
import numpy as np
from functions import *


class Clusteror(html.Div):
    def __init__(self, name, df, df_faces):
        self.html_id = name
        self.df = df
        self.df_faces = df_faces

        # Initialize Detector as empty Div
        super().__init__(className="graph_card", id=self.html_id, children=[])