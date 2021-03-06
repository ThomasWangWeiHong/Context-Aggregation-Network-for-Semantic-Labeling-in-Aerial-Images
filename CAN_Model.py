import cv2
import glob
import json
import numpy as np
import rasterio
from group_norm import GroupNormalization
from keras import backend as K
from keras.applications.resnet50 import ResNet50
from keras.models import Input, Model
from keras.layers import Activation, Add, concatenate, Conv2D, Conv2DTranspose, Lambda, Multiply, UpSampling2D
from keras.optimizers import Adam



def training_mask_generation(input_image_filename, input_geojson_filename, labels):
    """ 
    This function is used to create a binary raster mask from polygons in a given geojson file, so as to label the pixels 
    in the image as either background or target.
    
    Inputs:
    - input_image_filename: File path of georeferenced image file to be used for model training
    - input_geojson_filename: File path of georeferenced geojson file which contains the polygons drawn over the targets
    - labels: List of labels for multi - class semantic segmentation of image 
    
    Outputs:
    - mask: Numpy array representing the training mask, with values of 0 for background pixels, and value of 1 for target 
            pixels.
    
    """
    
    with rasterio.open(input_image_filename) as f:
        metadata = f.profile
    
    mask = np.zeros((metadata['height'], metadata['width'], len(labels)))
    
    xres = metadata['transform'][0]
    ulx = metadata['transform'][2]
    yres = metadata['transform'][4]
    uly = metadata['transform'][5]
    
    lrx = ulx + (metadata['width'] * xres)                                                         
    lry = uly - (metadata['height'] * abs(yres))

    polygons = json.load(open(input_geojson_filename))
    
    for polygon in range(len(polygons['features'])):
        layer_num = labels.index(str(polygons['features'][polygon]['properties']['Label']))
        mask_required = mask[:, :, layer_num].copy()
        coords = np.array(polygons['features'][polygon]['geometry']['coordinates'][0][0])                      
        xf = ((metadata['width']) ** 2 / (metadata['width'] + 1)) / (lrx - ulx)
        yf = ((metadata['height']) ** 2 / (metadata['height'] + 1)) / (lry - uly)
        coords[:, 1] = yf * (coords[:, 1] - uly)
        coords[:, 0] = xf * (coords[:, 0] - ulx)                                       
        position = np.round(coords).astype(np.int32)
        cv2.fillConvexPoly(mask_required, position, 1)
        mask[:, :, layer_num] = mask_required
    
    mask[:, :, -1] = np.sum(mask[:, :, : -1], axis = 2) == 0
    
    return mask



def image_clip_to_segment_and_convert(image_array, mask_array, image_height_size, image_width_size, mode, percentage_overlap, 
                                      buffer):
    """ 
    This function is used to cut up images of any input size into segments of a fixed size, with empty clipped areas 
    padded with zeros to ensure that segments are of equal fixed sizes and contain valid data values. The function then 
    returns a 4 - dimensional array containing the entire image and its mask in the form of fixed size segments. 
    
    Inputs:
    - image_array: Numpy array representing the image to be used for model training (channels last format)
    - mask_array: Numpy array representing the binary raster mask to mark out background and target pixels
    - image_height_size: Height of image segments to be used for model training
    - image_width_size: Width of image segments to be used for model training
    - mode: Integer representing the status of image size
    - percentage_overlap: Percentage of overlap between image patches extracted by sliding window to be used for model 
                          training
    - buffer: Percentage allowance for image patch to be populated by zeros for positions with no valid data values
    
    Outputs:
    - image_segment_array: 4 - Dimensional numpy array containing the image patches extracted from input image array
    - mask_segment_array: 4 - Dimensional numpy array containing the mask patches extracted from input binary raster mask
    
    """
    
    y_size = ((image_array.shape[0] // image_height_size) + 1) * image_height_size
    x_size = ((image_array.shape[1] // image_width_size) + 1) * image_width_size
    
    if mode == 0:
        img_complete = np.zeros((y_size, image_array.shape[1], image_array.shape[2]))
        mask_complete = np.zeros((y_size, mask_array.shape[1], mask_array.shape[2]))
        img_complete[0 : image_array.shape[0], 0 : image_array.shape[1], 0 : image_array.shape[2]] = image_array
        mask_complete[0 : mask_array.shape[0], 0 : mask_array.shape[1], 0 : mask_array.shape[2]] = mask_array
    elif mode == 1:
        img_complete = np.zeros((image_array.shape[0], x_size, image_array.shape[2]))
        mask_complete = np.zeros((image_array.shape[0], x_size, mask_array.shape[2]))
        img_complete[0 : image_array.shape[0], 0 : image_array.shape[1], 0 : image_array.shape[2]] = image_array
        mask_complete[0 : mask_array.shape[0], 0 : mask_array.shape[1], 0 : mask_array.shape[2]] = mask_array
    elif mode == 2:
        img_complete = np.zeros((y_size, x_size, image_array.shape[2]))
        mask_complete = np.zeros((y_size, x_size, mask_array.shape[2]))
        img_complete[0 : image_array.shape[0], 0 : image_array.shape[1], 0 : image_array.shape[2]] = image_array
        mask_complete[0 : mask_array.shape[0], 0 : mask_array.shape[1], 0 : mask_array.shape[2]] = mask_array
    elif mode == 3:
        img_complete = image_array
        mask_complete = mask_array
        
    img_list = []
    mask_list = [[] for i in range(mask_array.shape[2])]
    
    
    for i in range(0, int(img_complete.shape[0] - (2 - buffer) * image_height_size), 
                   int((1 - percentage_overlap) * image_height_size)):
        for j in range(0, int(img_complete.shape[1] - (2 - buffer) * image_width_size), 
                       int((1 - percentage_overlap) * image_width_size)):
            M_90 = cv2.getRotationMatrix2D((image_width_size / 2, image_height_size / 2), 90, 1.0)
            M_180 = cv2.getRotationMatrix2D((image_width_size / 2, image_height_size / 2), 180, 1.0)
            M_270 = cv2.getRotationMatrix2D((image_width_size / 2, image_height_size / 2), 270, 1.0)
            
            img_original = img_complete[i : i + image_height_size, j : j + image_width_size, 0 : image_array.shape[2]]
            img_rotate_90 = cv2.warpAffine(img_original, M_90, (image_height_size, image_width_size))
            img_rotate_180 = cv2.warpAffine(img_original, M_180, (image_width_size, image_height_size))
            img_rotate_270 = cv2.warpAffine(img_original, M_270, (image_height_size, image_width_size))
            img_flip_hor = cv2.flip(img_original, 0)
            img_flip_vert = cv2.flip(img_original, 1)
            img_flip_both = cv2.flip(img_original, -1)
            img_list.extend([img_original, img_rotate_90, img_rotate_180, img_rotate_270, img_flip_hor, img_flip_vert, 
                             img_flip_both])
            for label in range(mask_array.shape[2]):
                mask_original = mask_complete[i : i + image_height_size, j : j + image_width_size, label]
                mask_rotate_90 = cv2.warpAffine(mask_original, M_90, (image_height_size, image_width_size))
                mask_rotate_180 = cv2.warpAffine(mask_original, M_180, (image_width_size, image_height_size))
                mask_rotate_270 = cv2.warpAffine(mask_original, M_270, (image_height_size, image_width_size))
                mask_flip_hor = cv2.flip(mask_original, 0)
                mask_flip_vert = cv2.flip(mask_original, 1)
                mask_flip_both = cv2.flip(mask_original, -1)
                mask_list[label].extend([mask_original, mask_rotate_90, mask_rotate_180, mask_rotate_270, mask_flip_hor, 
                                         mask_flip_vert, mask_flip_both])
    
    image_segment_array = np.zeros((len(img_list), image_height_size, image_width_size, image_array.shape[2]))
    mask_segment_array = np.zeros((len(img_list), image_height_size, image_width_size, mask_array.shape[2]))
    
    for index in range(len(img_list)):
        image_segment_array[index] = img_list[index]
        for label in range(len(mask_list)):
            mask_segment_array[index, :, :, label] = mask_list[label][index]
        
    return image_segment_array, mask_segment_array



def training_data_generation(DATA_DIR, img_height_size, img_width_size, perc, buff, label_list):
    """ 
    This function is used to convert image files and their respective polygon training masks into numpy arrays, so as to 
    facilitate their use for model training.
    
    Inputs:
    - DATA_DIR: File path of folder containing the image files, and their respective polygons in a subfolder
    - img_height_size: Height of image patches to be used for model training
    - img_width_size: Width of image patches to be used for model training
    - perc: Percentage of overlap between image patches extracted by sliding window to be used for model training
    - buff: Percentage allowance for image patch to be populated by zeros for positions with no valid data values
    - label_list: List containing all the labels to be used for multi - class semantic segmentation (label for background
                  should be in the last position of the list)
    
    Outputs:
    - img_full_array: 4 - Dimensional numpy array containing image patches extracted from all image files for model training
    - mask_full_array: 4 - Dimensional numpy array containing binary raster mask patches extracted from all polygons for 
                       model training
    """
    
    if perc < 0 or perc > 1:
        raise ValueError('Please input a number between 0 and 1 (inclusive) for perc.')
        
    if buff < 0 or buff > 1:
        raise ValueError('Please input a number between 0 and 1 (inclusive) for buff.')
    
    img_files = glob.glob(DATA_DIR + '\\Train_MS' + '\\Train_*.tif')
    polygon_files = glob.glob(DATA_DIR + '\\Train_Polygons' + '\\Train_*.geojson')
    
    img_array_list = []
    mask_array_list = []
    
    for file in range(len(img_files)):
        with rasterio.open(img_files[file]) as f:
            metadata = f.profile
            img = np.transpose(f.read(tuple(np.arange(metadata['count']) + 1)), [1, 2, 0])
        mask = training_mask_generation(img_files[file], polygon_files[file], labels = label_list)
    
        if (img.shape[0] % img_height_size != 0) and (img.shape[1] % img_width_size == 0):
            img_array, mask_array = image_clip_to_segment_and_convert(img, mask, img_height_size, img_width_size, mode = 0, 
                                                                      percentage_overlap = perc, buffer = buff)
        elif (img.shape[0] % img_height_size == 0) and (img.shape[1] % img_width_size != 0):
            img_array, mask_array = image_clip_to_segment_and_convert(img, mask, img_height_size, img_width_size, mode = 1, 
                                                                      percentage_overlap = perc, buffer = buff)
        elif (img.shape[0] % img_height_size != 0) and (img.shape[1] % img_width_size != 0):
            img_array, mask_array = image_clip_to_segment_and_convert(img, mask, img_height_size, img_width_size, mode = 2, 
                                                                      percentage_overlap = perc, buffer = buff)
        else:
            img_array, mask_array = image_clip_to_segment_and_convert(img, mask, img_height_size, img_width_size, mode = 3, 
                                                                      percentage_overlap = perc, buffer = buff)
        
        img_array_list.append(img_array)
        mask_array_list.append(mask_array)
        
    img_full_array = np.concatenate(img_array_list, axis = 0)
    mask_full_array = np.concatenate(mask_array_list, axis = 0)
    
    return img_full_array, mask_full_array



def CAN_model(img_height_size, img_width_size, n_bands, num_classes, group_filters, rcm_4_filters = 512, cfm_gp_filters = 256, 
              cfm_conv_1_filters = 512, cfm_conv_2_filters = 256, amm_3_filters = 256, l_r = 0.0001):
    """
    This function is used to generate the Context Aggregation Network (CAN) architecture as described in the paper 'Context 
    Aggregation Network for Semantic Labeling in Aerial Images' by Cheng W., Yang W., Wang M., Wang G., Chen J. (2019).
    
    Inputs:
    - img_height_size: Height of image patches to be used for model training
    - img_width_size: Width of image patches to be used for model training
    - n_bands: Number of channels contained in the image patches to be used for model training
    - num_classes: Number of classes to be used for semantic segmentation
    - group_filters: Number of groups to be used for group normalization
    - rcm_4_filters: Number of filters to be used for Residual Convolutional Module (RCM) on Res - 4 output
    - cfm_gp_filters: Number of filters to be used for global pooling component of Context Fuse Module (CFM)
    - cfm_conv_1_filters: Number of filters to be used for first convolutions in parallel convolutional block of CFM
    - cfm_conv_2_filters: Number of filters to be used for second convolutions in parallel convolutional block of CFM
    - amm_3_filters: Number of filters to be used for convolutional layer in Attention Mix Module (AMM) on Res - 3 output
    - l_r: Learning rate for the Adam optimizer
    
    Outputs:
    - can_model: CAN model to be trained using input parameters and network architecture
    
    """
    
    rcm_3_filters = int(rcm_4_filters / 2)
    rcm_2_filters = int(rcm_3_filters / 2)
    rcm_1_filters = int(rcm_2_filters / 2)
    rcm_base_filters = int(rcm_1_filters / 2)
    
    amm_2_filters = int(amm_3_filters / 2)
    amm_1_filters = int(amm_2_filters / 2)
    
    
    resnet50_model = ResNet50(include_top = False, input_shape = (img_height_size, img_width_size, n_bands))
    
    
    
    res_1 = resnet50_model.get_layer('activation_10').output
    res_2 = resnet50_model.get_layer('activation_22').output
    res_3 = resnet50_model.get_layer('activation_40').output
    res_4 = resnet50_model.get_layer('activation_49').output
    
    
    
    rcm_4_1_1_1 = Conv2D(rcm_4_filters, (1, 1), padding = 'same')(res_4)
    rcm_4_1_3_3 = Conv2D(rcm_4_filters, (3, 3), padding = 'same')(rcm_4_1_1_1)
    rcm_4_1_3_3_gn = GroupNormalization(groups = group_filters, axis = -1, epsilon = 0.1)(rcm_4_1_3_3)
    rcm_4_1_3_3_act = Activation('relu')(rcm_4_1_3_3_gn)
    rcm_4_1_2nd_3_3 = Conv2D(rcm_4_filters, (3, 3), padding = 'same')(rcm_4_1_3_3_act)
    rcm_4_1_out = Add()([rcm_4_1_1_1, rcm_4_1_2nd_3_3])
    
    
    
    cfm_b_gp = Lambda(lambda x: K.mean(x, axis = 3)[:, :, :, None])(rcm_4_1_out)
    cfm_b_1_1 = Conv2D(cfm_gp_filters, (1, 1), padding = 'same')(cfm_b_gp)
    cfm_b_gn = GroupNormalization(groups = group_filters, axis = -1, epsilon = 0.1)(cfm_b_1_1)
    
    cfm_a_3_1 = Conv2D(cfm_conv_1_filters, (3, 3), padding = 'same')(rcm_4_1_out)
    cfm_a_3_1_gn = GroupNormalization(groups = group_filters, axis = -1, epsilon = 0.1)(cfm_a_3_1)
    cfm_a_3_1_act = Activation('relu')(cfm_a_3_1_gn)
    cfm_a_3_2 = Conv2D(cfm_conv_2_filters, (3, 3), padding = 'same')(cfm_a_3_1_act)
    cfm_a_3_2_gn = GroupNormalization(groups = group_filters, axis = -1, epsilon = 0.1)(cfm_a_3_2)
    
    cfm_a_7_1 = Conv2D(cfm_conv_1_filters, (7, 7), padding = 'same')(rcm_4_1_out)
    cfm_a_7_1_gn = GroupNormalization(groups = group_filters, axis = -1, epsilon = 0.1)(cfm_a_7_1)
    cfm_a_7_1_act = Activation('relu')(cfm_a_7_1_gn)
    cfm_a_7_2 = Conv2D(cfm_conv_2_filters, (7, 7), padding = 'same')(cfm_a_7_1_act)
    cfm_a_7_2_gn = GroupNormalization(groups = group_filters, axis = -1, epsilon = 0.1)(cfm_a_7_2)
    
    cfm_a_11_1 = Conv2D(cfm_conv_1_filters, (11, 11), padding = 'same')(rcm_4_1_out)
    cfm_a_11_1_gn = GroupNormalization(groups = group_filters, axis = -1, epsilon = 0.1)(cfm_a_11_1)
    cfm_a_11_1_act = Activation('relu')(cfm_a_11_1_gn)
    cfm_a_11_2 = Conv2D(cfm_conv_2_filters, (11, 11), padding = 'same')(cfm_a_11_1_act)
    cfm_a_11_2_gn = GroupNormalization(groups = group_filters, axis = -1, epsilon = 0.1)(cfm_a_11_2)
    
    cfm_a_15_1 = Conv2D(cfm_conv_1_filters, (15, 15), padding = 'same')(rcm_4_1_out)
    cfm_a_15_1_gn = GroupNormalization(groups = group_filters, axis = -1, epsilon = 0.1)(cfm_a_15_1)
    cfm_a_15_1_act = Activation('relu')(cfm_a_15_1_gn)
    cfm_a_15_2 = Conv2D(cfm_conv_2_filters, (15, 15), padding = 'same')(cfm_a_15_1_act)
    cfm_a_15_2_gn = GroupNormalization(groups = group_filters, axis = -1, epsilon = 0.1)(cfm_a_15_2)
    
    cfm_out = concatenate([cfm_b_gn, cfm_a_3_2_gn, cfm_a_7_2_gn, cfm_a_11_2_gn, cfm_a_15_2_gn])
    
    
    
    rcm_4_2_1_1 = Conv2D(rcm_3_filters, (1, 1), padding = 'same')(cfm_out)
    rcm_4_2_3_3 = Conv2D(rcm_3_filters, (3, 3), padding = 'same')(rcm_4_2_1_1)
    rcm_4_2_3_3_gn = GroupNormalization(groups = group_filters, axis = -1, epsilon = 0.1)(rcm_4_2_3_3)
    rcm_4_2_3_3_act = Activation('relu')(rcm_4_2_3_3_gn)
    rcm_4_2_2nd_3_3 = Conv2D(rcm_3_filters, (3, 3), padding = 'same')(rcm_4_2_3_3_act)
    rcm_4_2_out = Add()([rcm_4_2_1_1, rcm_4_2_2nd_3_3])
    
    
    
    rcm_3_1_1_1 = Conv2D(rcm_3_filters, (1, 1), padding = 'same')(res_3)
    rcm_3_1_3_3 = Conv2D(rcm_3_filters, (3, 3), padding = 'same')(rcm_3_1_1_1)
    rcm_3_1_3_3_gn = GroupNormalization(groups = group_filters, axis = -1, epsilon = 0.1)(rcm_3_1_3_3)
    rcm_3_1_3_3_act = Activation('relu')(rcm_3_1_3_3_gn)
    rcm_3_1_2nd_3_3 = Conv2D(rcm_3_filters, (3, 3), padding = 'same')(rcm_3_1_3_3_act)
    rcm_3_1_out = Add()([rcm_3_1_1_1, rcm_3_1_2nd_3_3])
    
    
    
    upsam_rcm_4 = UpSampling2D(size = (2, 2), interpolation = 'bilinear')(rcm_4_2_out)
    amm_3_input = concatenate([rcm_3_1_out, upsam_rcm_4])
    amm_3_conv = Conv2D(amm_3_filters, (3, 3), padding = 'same')(amm_3_input)
    amm_3_conv_gn = GroupNormalization(groups = group_filters, axis = -1, epsilon = 0.1)(amm_3_conv)
    amm_3_conv_act = Activation('relu')(amm_3_conv_gn)
    amm_3_gp = Lambda(lambda x: K.mean(x, axis = 3)[:, :, :, None])(amm_3_conv_act)
    amm_3_mult = Multiply()([amm_3_conv_act, amm_3_gp])
    amm_3_out = Add()([amm_3_mult, rcm_3_1_out])
    
    
    
    rcm_3_2_1_1 = Conv2D(rcm_2_filters, (1, 1), padding = 'same')(amm_3_out)
    rcm_3_2_3_3 = Conv2D(rcm_2_filters, (3, 3), padding = 'same')(rcm_3_2_1_1)
    rcm_3_2_3_3_gn = GroupNormalization(groups = group_filters, axis = -1, epsilon = 0.1)(rcm_3_2_3_3)
    rcm_3_2_3_3_act = Activation('relu')(rcm_3_2_3_3_gn)
    rcm_3_2_2nd_3_3 = Conv2D(rcm_2_filters, (3, 3), padding = 'same')(rcm_3_2_3_3_act)
    rcm_3_2_out = Add()([rcm_3_2_1_1, rcm_3_2_2nd_3_3])
    
    
    
    rcm_2_1_1_1 = Conv2D(rcm_2_filters, (1, 1), padding = 'same')(res_2)
    rcm_2_1_3_3 = Conv2D(rcm_2_filters, (3, 3), padding = 'same')(rcm_2_1_1_1)
    rcm_2_1_3_3_gn = GroupNormalization(groups = group_filters, axis = -1, epsilon = 0.1)(rcm_2_1_3_3)
    rcm_2_1_3_3_act = Activation('relu')(rcm_2_1_3_3_gn)
    rcm_2_1_2nd_3_3 = Conv2D(rcm_2_filters, (3, 3), padding = 'same')(rcm_2_1_3_3_act)
    rcm_2_1_out = Add()([rcm_2_1_1_1, rcm_2_1_2nd_3_3])
    
    
    
    upsam_rcm_3 = UpSampling2D(size = (2, 2), interpolation = 'bilinear')(rcm_3_2_out)
    amm_2_input = concatenate([rcm_2_1_out, upsam_rcm_3])
    amm_2_conv = Conv2D(amm_2_filters, (3, 3), padding = 'same')(amm_2_input)
    amm_2_conv_gn = GroupNormalization(groups = group_filters, axis = -1, epsilon = 0.1)(amm_2_conv)
    amm_2_conv_act = Activation('relu')(amm_2_conv_gn)
    amm_2_gp = Lambda(lambda x: K.mean(x, axis = 3)[:, :, :, None])(amm_2_conv_act)
    amm_2_mult = Multiply()([amm_2_conv_act, amm_2_gp])
    amm_2_out = Add()([amm_2_mult, rcm_2_1_out])
    
    
    
    rcm_2_2_1_1 = Conv2D(rcm_1_filters, (1, 1), padding = 'same')(amm_2_out)
    rcm_2_2_3_3 = Conv2D(rcm_1_filters, (3, 3), padding = 'same')(rcm_2_2_1_1)
    rcm_2_2_3_3_gn = GroupNormalization(groups = group_filters, axis = -1, epsilon = 0.1)(rcm_2_2_3_3)
    rcm_2_2_3_3_act = Activation('relu')(rcm_2_2_3_3_gn)
    rcm_2_2_2nd_3_3 = Conv2D(rcm_1_filters, (3, 3), padding = 'same')(rcm_2_2_3_3_act)
    rcm_2_2_out = Add()([rcm_2_2_1_1, rcm_2_2_2nd_3_3])
    
    
    
    rcm_1_1_1_1 = Conv2D(rcm_1_filters, (1, 1), padding = 'same')(res_1)
    rcm_1_1_3_3 = Conv2D(rcm_1_filters, (3, 3), padding = 'same')(rcm_1_1_1_1)
    rcm_1_1_3_3_gn = GroupNormalization(groups = group_filters, axis = -1, epsilon = 0.1)(rcm_1_1_3_3)
    rcm_1_1_3_3_act = Activation('relu')(rcm_1_1_3_3_gn)
    rcm_1_1_2nd_3_3 = Conv2D(rcm_1_filters, (3, 3), padding = 'same')(rcm_1_1_3_3_act)
    rcm_1_1_out = Add()([rcm_1_1_1_1, rcm_1_1_2nd_3_3])
    
    
    
    upsam_rcm_2 = UpSampling2D(size = (2, 2), interpolation = 'bilinear')(rcm_2_2_out)
    amm_1_input = concatenate([rcm_1_1_out, upsam_rcm_2])
    amm_1_conv = Conv2D(amm_1_filters, (3, 3), padding = 'same')(amm_1_input)
    amm_1_conv_gn = GroupNormalization(groups = group_filters, axis = -1, epsilon = 0.1)(amm_1_conv)
    amm_1_conv_act = Activation('relu')(amm_1_conv_gn)
    amm_1_gp = Lambda(lambda x: K.mean(x, axis = 3)[:, :, :, None])(amm_1_conv_act)
    amm_1_mult = Multiply()([amm_1_conv_act, amm_1_gp])
    amm_1_out = Add()([amm_1_mult, rcm_1_1_out])
    
    
    
    rcm_1_2_1_1 = Conv2D(rcm_base_filters, (1, 1), padding = 'same')(amm_1_out)
    rcm_1_2_3_3 = Conv2D(rcm_base_filters, (3, 3), padding = 'same')(rcm_1_2_1_1)
    rcm_1_2_3_3_gn = GroupNormalization(groups = group_filters, axis = -1, epsilon = 0.1)(rcm_1_2_3_3)
    rcm_1_2_3_3_act = Activation('relu')(rcm_1_2_3_3_gn)
    rcm_1_2_2nd_3_3 = Conv2D(rcm_base_filters, (3, 3), padding = 'same')(rcm_1_2_3_3_act)
    rcm_1_2_out = Add()([rcm_1_2_1_1, rcm_1_2_2nd_3_3])
    
    
    
    deconv_2 = Conv2DTranspose(rcm_base_filters, (2, 2), strides = (2, 2), padding = 'same')(rcm_1_2_out)
    deconv_2_gn = GroupNormalization(groups = group_filters, axis = -1, epsilon = 0.1)(deconv_2)
    deconv_2_act = Activation('relu')(deconv_2_gn)
    
    
    
    deconv_1 = Conv2DTranspose(rcm_base_filters, (2, 2), strides = (2, 2), padding = 'same')(deconv_2_act)
    deconv_1_gn = GroupNormalization(groups = group_filters, axis = -1, epsilon = 0.1)(deconv_1)
    deconv_1_act = Activation('relu')(deconv_1_gn)
    
    
    
    pred_layer = Conv2D(num_classes, (1, 1), padding = 'same', activation = 'softmax')(deconv_1_act)
    
    
    can_model = Model(inputs = resnet50_model.input, outputs = pred_layer)
    can_model.compile(loss = 'categorical_crossentropy', optimizer = Adam(lr = l_r), metrics = ['categorical_crossentropy'])
    
    
    return can_model



def image_model_predict(input_image_filename, output_filename, img_height_size, img_width_size, n_classes, fitted_model, write):
    """ 
    This function cuts up an image into segments of fixed size, and feeds each segment to the model for prediction. The 
    output mask is then allocated to its corresponding location in the image in order to obtain the complete mask for the 
    entire image without being constrained by image size. 
    
    Inputs:
    - input_image_filename: File path of image file for which prediction is to be conducted
    - output_filename: File path of output predicted binary raster mask file
    - img_height_size: Height of image patches to be used for model prediction
    - img_height_size: Width of image patches to be used for model prediction
    - n_classes: Number of labesl for multi - class semantic segmentation
    - fitted_model: Trained keras model which is to be used for prediction
    - write: Boolean indicating whether to write predicted binary raster mask to file
    
    Output:
    - mask_complete: Numpy array of predicted binary raster mask for input image
    
    """
    
    with rasterio.open(input_image_filename) as f:
        metadata = f.profile
        img = np.transpose(f.read(tuple(np.arange(metadata['count']) + 1)), [1, 2, 0])
     
    y_size = ((img.shape[0] // img_height_size) + 1) * img_height_size
    x_size = ((img.shape[1] // img_width_size) + 1) * img_width_size
    
    if (img.shape[0] % img_height_size != 0) and (img.shape[1] % img_width_size == 0):
        img_complete = np.zeros((y_size, img.shape[1], img.shape[2]))
        img_complete[0 : img.shape[0], 0 : img.shape[1], 0 : img.shape[2]] = img
    elif (img.shape[0] % img_height_size == 0) and (img.shape[1] % img_width_size != 0):
        img_complete = np.zeros((img.shape[0], x_size, img.shape[2]))
        img_complete[0 : img.shape[0], 0 : img.shape[1], 0 : img.shape[2]] = img
    elif (img.shape[0] % img_height_size != 0) and (img.shape[1] % img_width_size != 0):
        img_complete = np.zeros((y_size, x_size, img.shape[2]))
        img_complete[0 : img.shape[0], 0 : img.shape[1], 0 : img.shape[2]] = img
    else:
         img_complete = img
            
    mask = np.zeros((img_complete.shape[0], img_complete.shape[1], n_classes))
    img_holder = np.zeros((1, img_height_size, img_width_size, img.shape[2]))
    
    for i in range(0, img_complete.shape[0], img_height_size):
        for j in range(0, img_complete.shape[1], img_width_size):
            img_holder[0] = img_complete[i : i + img_height_size, j : j + img_width_size, 0 : img.shape[2]]
            preds = fitted_model.predict(img_holder)
            mask[i : i + img_height_size, j : j + img_width_size, 0 : n_classes] = preds[0, :, :, 0 : n_classes]
            
    mask_complete = np.transpose(mask[0 : img.shape[0], 0 : img.shape[1], 0 : n_classes], [2, 0, 1])
    
    if write:
        metadata['count'] = n_classes
        
        with rasterio.open(output_filename, 'w', **metadata) as dst:
            dst.write(mask_complete)
    
    return mask_complete
