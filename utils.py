import numpy as np
import pandas as pd
import skimage.io as io
import PIL as Image
import glob
import gflags
import sys
import os
import errno
import warnings
warnings.simplefilter("ignore", UserWarning)

def rgb_to_mask(img, class_to_idx):

    """
        Parameters: img -> RGB segmentation mask
                    class_to_idx -> dictionary mapping pixel to class index
                                    for eg. {'[255,255,255]':0,
                                             '[0,0,0]':1'}
                                    can be generated like:
                                    class_to_idx  = {str(pixel):i for i,pixel in enumerate(PIXELS)}
                                    where PIXELS is list of pixels like [np.array([0,0,0]),np.array(255,255,255)]
        converts an rgb mask into multi channel one hot encoded mask
    """

    img_height = img.shape[0]
    img_width = img.shape[1]

    # Flatten imgage spatially so it becomes list of pixels([R,G,B])

    img = img.reshape(img_height*img_width,img.shape[2])

    # image to be returned

    ret_img = np.zeros((img_height*img_width, 9))

    for i,pixel in enumerate(img):
        global class_to_idx
        # idx of true class
        idx = class_to_idx[str(pixel)]
        # set thr idx of true class 1
        ret_img[i][idx] = 1
        
    # Reshape image back to its oiginal shape
    ret_img = ret_img.reshape(img_height, img_width,9)

    print(ret_img.shape)
    return ret_img

def mask_to_rgb(img, idx_to_class):
    """
        converts multi channel one-hot mask to 3 channel rgb mask

        Parameters: img -> RGB segmentation mask
                    idx_to_class -> dictionary mapping class index to pixel
                                    for eg. {0: [255,255,255],
                                             1: [0,0,0]}
                                    can be generated like:
                                    idx_to_class = {i: pixel for i, pixel in enumerate(PIXELS)}
                                    where PIXELS is list of pixels like [np.array([0,0,0]),np.array(255,255,255)]
    
    """
    img_height = img.shape[0]
    img_width = img.shape[1]
    ret_img = np.zeros((img_height, img_width, 3))
    for i in range(img_height):
        for j in range(img_width):
            pixel = idx_to_class[np.argmax(img[i][j])]
            ret_img[i][j] = pixel

    print(ret_img.shape)
    return ret_img

def create_patches(img, patch_height=256, patch_width=256, h_stride=0.5, w_stride=0.5):
    """ 
        Params: img -> Input image(numpy array)
                patch height -> height of patch to be cut
                patch width -> width of patch to be cut
                h_stride -> 1/overlap required among adjacent patch along height eg. 0.5 for twice overlap
                w_stride -> 1/overlap required among adjacent patch along width

        input -> image (height,width,channel), patch dimensions
    	outpout -> patches of desired dimensions (patch_height, patch_width, channel)
                   patch parameters dictionary containing:
                                                          --original image height
                                                          --original image width
                                                          --stride along height
                                                          --stride along width
                                                          --patch height
                                                          --patch width
    """
    h_stride = int(max(1, patch_height * h_stride))
    w_stride = int(max(1, patch_width * w_stride))

    patch_param = {}
    patch_param['image_height'] = img.shape[0]
    patch_param['image_width'] = img.shape[1]
    patch_param['h_stride'] = h_stride
    patch_param['w_stride'] = w_stride
    patch_param['patch_height'] = patch_height
    patch_param['patch_width'] = patch_width

    h = 0
    w = 0

    img = pad_image(img, patch_height, patch_width)

    patches = []

    while h <= img.shape[0] - patch_height:
        w = 0
        while w <= img.shape[1] - patch_width:
            patches.append(img[h:h+patch_height, w:w+patch_width, :])
            w = w + w_stride
        h = h+h_stride

    return patches, patch_param

def return_padding(img, height, width):
    " Return padding given image and height, width of patch"
    h = 0 if img.shape[0]%height == 0 else height - img.shape[0]%height
    w = 0 if img.shape[1]%width == 0 else width - img.shape[1]%width
    pad_shape = tuple(np.zeros((len(img.shape),2),dtype=np.uint16))
    pad_shape = [tuple(x) for x in pad_shape]
    h_left  = h//2
    h_right = h - h_left
    w_left  = w//2
    w_right = w - w_left
    pad_shape[0] = (int(h_left),int(h_right))
    pad_shape[1] = (int(w_left),int(w_right))
    
    print("pad shape is {}".format(pad_shape))
    return pad_shape

def pad_image(img, height, width, channels=4, mode='constant'):
    """ 
        Pads img to make it fit for extracting patches of 
        shape height X width from it
        mode -> constant, reflect 
        constant -> pads ith 0's
        symmetric -> pads with reflection of image borders
    """
    print('input shape {}'.format(img.shape))
    pad_shape = return_padding(img, height, width)
    img = np.pad(img,pad_shape,mode='constant')
    print('output shape {}'.format(img.shape))
    return img  

def create_binary_mask(source, destination, dict_map = None, type = 'png'):
    """
    Takes source folder contating rgb ground truth masks
    and converts them to binary(grayscale) mask of each class
    destination folder will look like :
    ./destination/
        class-1/
            img1/
            img2/
            ..
            ..
            img9/
        class-2/
            img1/
            ..
            ..
            img9/
        ..

        Params: source -> source folder containing RGB masks
                destination -> destination folder where binary masks are to be saved
                dict_map -> dictionary mapping class and pixel values
                            for eg. dict_map = {
                                                'class1' : [255, 255, 255],
                                                'class2':  [0, 0, 0]
                                                }
                type -> type of image png, jpg, tif .
    """
    if not dict_map:
        raise Exception('Dictionary mapping not provided')
    assert isinstance(dict_map, (dict))
    assert type in ['png', 'jpg', 'tif']

    if source[-1] != '/':
        source = source + '/'
    if destination[-1] != '/':
        destination = destination + '/'

    if not os.path.isdir(source):
        raise Exception('source file does not exist')
    if not os.path.isdir(destination):
        print('creating destination folder')
        os.mkdir(destination)

    masks = glob.glob(source + '*.'+type)
    for key, rgb in dict_map.items():
        path = destination + key
        if not os.path.isdir(path):
            os.mkdir(path)
        for mask in masks:
            if type == 'tif':
                io.use_plugin('tifffile')
            img = io.imread(mask)
            img_id = mask.split('/')[-1].split('.')[0]
            
            class_mask = img[:,:,:] == np.array(rgb)
            final_mask = class_mask[...,0] * class_mask[...,1] * class_mask[...,2]
            final_mask = final_mask.astype(np.uint8)
            io.imsave(path + '/' + str(img_id) + '.' + type, final_mask)


def make_divisor_mask(mask_height, mask_width, step):
    
    """ Create a mask array defining the overlap extent of patches"""
    mask = np.empty([mask_height, mask_width], dtype=np.uint16)
    for i in range(1,mask_height+1):
        for j in range(1,mask_width+1):
            mask[i-1][j-1] = min(i,mask_height-i+1,step)*min(j,mask_width-j+1,step)
    return mask

def sortKeyFunc(s):
    return int(os.path.basename(s).split('_')[2])

def stitch_patch(patch_path, recon_img_path, image_dict, h_stride, w_stride, channel=3, type = 'png'):
    """
        Takes source folder containing patches of Images and reconstruct the original image by recombining
        the patches using naive overlapping assumption without any smoothing and saves them in destination

        NOTE: Patch files should be named like patch_i_j where i = Image number eg. 1,2,3,4 and j = Patch number
              eg. 1,2,3,4 etc. i.e. patch_i_j represent jth patch of ith image
        Params: patch_path -> source folder of patches
                recon_img_path -> destination folder of reconstructed image
                image_dict -> dictionary having image height, image width, patch height, patch width
                            with keys- 'image_height', 'image_width', 'patch_height', 'patch_width'
                h_stride -> 1/overlap taken among adjacent patch along height eg. 0.5 for twice overlap
                w_stride -> 1/overlap taken among adjacent patch along width
                channel  -> number of channel in patches
                type     -> type of patch 'png', 'jpg', 'tif'
    """
    
    if patch_path[-1] != '/':
        patch_path += '/'

    if not os.path.isdir(patch_path):
        raise Exception('patch directory does not exist')
    if not os.path.isdir(recon_img_path):
        print('creating destination folder')
        os.makedirs(destination)

    assert type in ['png', 'jpg', 'tif']
        
    patch_list = []
    i=1
    while True:
        patches = sorted(glob.glob(patch_path+'/patch_{}_*.tif'.format(i)), key=sortKeyFunc)
        if not patches:
            break
        patch_list.append(patches)
    for files in patch_list:
        if not files:
            continue
        else:
            patch_height = int(image_dict['patch_height'])
            patch_width  = int(image_dict['patch_height'])
            img_id = files[0].split('/')[-1].split('_')[1]
            orig_img_height = int(image_dict['image_height'])
            orig_img_width  = int(image_dict['image_width'])
            h_stride = int(h_stride*patch_height)
            w_stride = int(w_stride*patch_width)

            img_dtype = np.uint16
            image     = np.zeros((orig_img_height, orig_img_width, channel), dtype = img_dtype)
            padding   = return_padding(image, patch_height, patch_width)
            image     = pad_zeros(image, patch_height, patch_width, channel)
            h = 0
            w = 0
            patches = []
            patch_id =0
            for name in files:
                try:
                    if type == 'tif':
                        io.use_plugin('tifffile')
                    patch = io.imread(name)
                    patches.append(patch)
                    if image.dtype != patch.dtype:
                        image = image.astype(patch.dtype, copy=False)                        
                except OSError as e:
                    print(e.errno)
                    print("Some of the patches are corrupted")

            while h <= image.shape[0]-patch_height:
                w = 0
                while w <= image.shape[1]-patch_width:
                    image[h:h+patch_height, w:w+patch_width, :] += patches[patch_id]
                    w = w + w_stride
                    patch_id+=1
                h = h+h_stride
            if(h_stride==w_stride):
                step = patch_height//h_stride
            else:
                print("Unequal strides are not yet suppported")

            mask_height = image.shape[0]//h_stride
            mask_width  = image.shape[1]//w_stride
            divisor_mask = make_divisor_mask(mask_height, mask_width, step)
            print("Divisor mask shape {}".format(divisor_mask.shape))

            h = 0
            w = 0
            mask_h = 0
            mask_w = 0
            print("Image shape {}".format(image.shape))
            while h <= image.shape[0] - h_stride:
                w = 0
                mask_w = 0
                while w <= image.shape[1] - w_stride:
                    image[h:h+h_stride, w:w+w_stride,:] /= divisor_mask[mask_h,mask_w]
                    w += w_stride
                    mask_w +=1
                h += h_stride
                mask_h +=1

            img = image[padding[0][0]:-padding[0][1], padding[1][0]:-padding[1][1],:]
            print("FinalImage shape{}".format(img.shape))
            assert img.shape == (orig_img_height, orig_img_width, channel)

            if not os.path.isdir(recon_img_path):
                os.mkdir(recon_img_path)

            io.imsave(recon_img_path + '/' + str(img_id) + '.' + type, img)