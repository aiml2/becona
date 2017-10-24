import sys
import os
import numpy as np

from keras.preprocessing import image as im
from keras.models import Model,load_model
from keras import activations
from keras.backend import image_data_format,image_dim_ordering
from keras.applications.inception_v3 import InceptionV3

input_shape = (299, 299, 3)
model = InceptionV3(input_shape=input_shape, weights='imagenet', include_top=True) 

from keras.optimizers import SGD
model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy')


from vis.utils import utils as vizut
# Swap softmax with linear
layer_idx=-1
model.layers[layer_idx].activation = activations.linear
model = vizut.apply_modifications(model)

from matplotlib import pyplot as plt
from skimage import io, transform
#%matplotlib inline
plt.rcParams['figure.figsize'] = (18, 6)

img1 = vizut.load_img('ouzel.jpg', target_size=(299, 299))
img2 = vizut.load_img("tench2.jpg", target_size=(299, 299))

from vis.visualization import visualize_saliency, overlay
from vis.utils import utils
from keras import activations

import numpy as np
import matplotlib.cm as cm
from vis.visualization import visualize_cam
from vis.visualization import overlay
layer_idx = -1 

#plt.figure()
for modifier in [None, 'guided', 'relu']:
    f, ax = plt.subplots(2, 2)
    plt.suptitle("vanilla" if modifier is None else modifier)
    for i, img in enumerate([img1, img2]):    
        grads = visualize_cam(model, layer_idx, filter_indices=20, seed_input=img, backprop_modifier=modifier)        
        print(grads)
        print(grads.shape)
        # Lets overlay the heatmap onto original image.    
        cmjet=cm.jet(grads)
        print(cmjet)
        print(cmjet.shape)
        jet_heatmap = np.uint8(cm.jet(grads)[..., :3] * 255)
        print(jet_heatmap)
        print(jet_heatmap.shape)
        print(img.shape)
        ax[i][0].imshow(img)
        ax[i][1].imshow(grads, cmap='jet')
    plt.show()
