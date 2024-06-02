import pickle
import numpy as np
import matplotlib.pyplot as plt

# open the data using pickle

with open('rtvslo_keywords.pkl', 'rb') as f:
	data = pickle.load(f)

sorted_keywords = data['keywords']
tfidf = data['tfidf']
tfidf_pca = data['tfidf_pca']
pca = data['pca']


# plot 3d data usig vispy

from vispy import scene
from vispy.scene import visuals

canvas = scene.SceneCanvas(keys='interactive', show=True)
view = canvas.central_widget.add_view()

scatter = visuals.Markers()
scatter.set_data(tfidf_pca, edge_color=None, face_color=(1, 1, 1, .5), size=5)

view.add(scatter)

# view.camera = 'arcball'
