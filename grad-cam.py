import sys
import os
import numpy as np
sys.path.append("code/utils")
import utils
#command line args as follows:
print("Command line args: model image")

modelArg = sys.argv[1]
imgArg = sys.argv[2]

from keras.preprocessing import image as im
from keras.models import Model,load_model
from keras import activations
from keras.backend import image_data_format,image_dim_ordering
from keras.applications.inception_v3 import InceptionV3

#For testing
sys.path.append("code")
import FTMC_InceptionV3_v3
import FTMC_InceptionV3_v4
import FTMC_Xception_v3
import FTMC_Xception_v4

# oldmodel=load_model(modelArg)
# # print(oldmodel.summary())
# # print(image_data_format())
# # print(image_dim_ordering())
# # print(oldmodel.inputs)
# # print(oldmodel.layers[0])
# # print(type(oldmodel.layers[0]))
# # layer_idx = -1 
# #
# #
# #model = FTMC_InceptionV3_v3.FTMC_InceptionV3_v3().model
# model = FTMC_Xception_v3.FTMC_Xception_v3().model
#
# assert len(oldmodel.layers) == len(model.layers)
#
# for idx,layer in enumerate(oldmodel.layers):
#     assert type(oldmodel.layers[idx]) == type(model.layers[idx])
#     model.layers[idx].set_weights(layer.get_weights())
# #
# # print('##############new model#############')
# # print(model.summary())
# # print(image_data_format())
# # print(image_dim_ordering())
# # print(model.inputs)
# # print(model.layers[0])
# # print(type(model.layers[0]))
# #
# from keras.optimizers import SGD
# model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy')
# model.save(modelArg+"FIX")
model=load_model(modelArg)

# input_shape = (299, 299, 3)
#model = InceptionV3(input_shape=input_shape, weights='imagenet', include_top=True) 


#TODO:
## setting input_shape on model initialisation is absolutely required for keras-vis
## not sure why figuring this out might be intresting for contributing to keras-vis :-)
## workaround instantiate new identical model as the trained model
## iterate over the layers of both the trained and new model
#   newmodellayer.set_weights(oldmodellayer.get_weights)
# do initially with Inceptionv3_v3 
# use a model-file-naming -> class mapping

from vis.utils import utils as vizut
# Swap softmax with linear
layer_idx=-1
model.layers[layer_idx].activation = activations.linear
model = vizut.apply_modifications(model)

from matplotlib import pyplot as plt
from skimage import io, transform
#%matplotlib inline
plt.rcParams['figure.figsize'] = (18, 6)

#img1 = vizut.load_img('ouzel.jpg', target_size=(299, 299))
#img2 = vizut.load_img("tench2.jpg", target_size=(299, 299))
#img1 = vizut.load_img('B2split_0_Val/1_Spanklem/IMG_20170904_142748285.jpg', target_size=(299, 299))
img1 = vizut.load_img('B2split_2_Val/3_VleugelMoerOpleg_Rond/P1090190.JPG', target_size=(299, 299))
img2 = vizut.load_img("B2split_2_Val/4.0_Variable_Spanklem_Kort/P1090420.JPG", target_size=(299, 299))
#img2 = vizut.load_img("B2split_0_Val/1_Spanklem/IMG_20170904_154745119.jpg", target_size=(299, 299))
#img2 = vizut.load_img('ouzel.jpg', target_size=(299, 299))

# print(vizut.get_img_shape(img1))
# f, ax = plt.subplots(2, 2)
# ax[0][0].imshow(img1)
# ax[0][1].imshow(img1)
# ax[1][0].imshow(img2)
# ax[1][1].imshow(img2)
# plt.show()



from vis.visualization import visualize_saliency, overlay
from vis.utils import utils
from keras import activations

# Utility to search for layer index by name. 
# Alternatively we can specify this as -1 since it corresponds to the last layer.
#final_layer_idx = utils.find_final_layer_idx(model, 'predictions')
final_layer_idx = -1 

filter_index=3

f, ax = plt.subplots(2, 2)
for i, img in enumerate([img1, img2]):    
    # 20 is the imagenet index corresponding to `ouzel`
    grads = visualize_saliency(model, final_layer_idx, filter_indices=filter_index, seed_input=img)

    # visualize grads as heatmap
    ax[i][0].imshow(img)
    ax[i][1].imshow(grads, cmap='jet')
plt.show()

for modifier in ['guided', 'relu']:
    f, ax = plt.subplots(2, 2)
    for i, img in enumerate([img1, img2]):    
        plt.suptitle(modifier)
        # 0 i assume shoudl be the filter index for spanklem1.0
        grads = visualize_saliency(model, final_layer_idx, filter_indices=filter_index, seed_input=img, backprop_modifier=modifier)
        ax[i][0].imshow(img)
        ax[i][1].imshow(grads, cmap='jet')
    plt.show()

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
        grads = visualize_cam(model, layer_idx, filter_indices=filter_index, seed_input=img, backprop_modifier=modifier)        
        print(grads)
        print(grads.shape)
        # Lets overlay the heatmap onto original image.    
        #jet_heatmap = np.uint8(cm.jet(grads) * 255)[:, : , :, 0]
        cmjet=cm.jet(grads)
        print(cmjet)
        print(cmjet.shape)
        print("HELI")
        cmheli=cm.cubehelix(grads)
        print(cmheli)
        print(cmheli.shape)
        # print(cmjet[0][0][0])
        # print(cmjet[0][0][1])
        print("[..., :3]")
        print(cmjet[..., :3])
        print(cmjet[..., :3].shape)
        print("[:,:,:,:3]")
        print(cmjet[:,:,:,:3])
        print(cmjet[:,:,:,:3].shape)
        jet_heatmap = np.uint8(cm.jet(grads)[..., :3] * 255)
        print(jet_heatmap)
        print(jet_heatmap.shape)
        print(img.shape)
        ax[i][0].imshow(img)
        ax[i][1].imshow(grads, cmap='jet')
        #ax[i].imshow(overlay(jet_heatmap, img))
        #ax[i][1].imshow(overlay(jet_heatmap, img))
    plt.show()
