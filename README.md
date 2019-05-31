# Context-Aggregation-Network-for-Semantic-Labeling-in-Aerial-Images
Python implementation of Convolutional Neural Network (CNN) proposed in academia

This repository includes functions to preprocess the input images and their respective polygons so as to create the input image patches 
and mask patches to be used for model training. The CNN used here is the Context Aggregation Network model implemented in the paper 
'Context Aggregation Network for Semantic Labeling in Aerial Images' by Cheng W., Yang W., Wang M., Wang G., Chen J. (2019).

The main differences between the implementations in the paper and the implementation in this repository is as follows:

- Group Normalization is used instead of Batch Normalization, since it is envisaged that very small batch sizes would be used for 
  training this model with consumer - level Graphics Processing Unit (GPU) in view of memory constraints, and it has been shown in 
  academia that Group Normalization outperforms Batch Normalization for very small batch sizes.
  
- Transpose convolutions are used instead of bilinear interpolation to produce the final resolution classification map, as it is believed
  that such convolutions would produce a smoother result than bilinear interpolation of class probabilities
  
The group normalization implementation in Keras used in this repository is the exact same class object defined in the group_norm.py 
file located in titu1994's Keras-Group-Normalization github repository at https://github.com/titu1994/Keras-Group-Normalization. 
Please ensure that the group_norm.py file is placed in the correct directory before use.

Requirements:
- cv2
- glob
- json
- numpy
- rasterio
- group_norm (downloaded from https://github.com/titu1994/Keras-Group-Normalization)
- keras (tensorflow backend)
