import numpy as np
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
from skimage import data
from skimage.filters import gaussian
from skimage.segmentation import active_contour


img = r"C:\Users\Cullen\Downloads\iris_matlab\iris1.jpg"

s = np.linspace(0, 2*np.pi,100)
init = 50*np.array([np.cos(s), np.sin(s)]).T+50

snake = active_contour(img, init, w_edge=0, w_line=1)
