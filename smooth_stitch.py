import scipy.signal
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import gc

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    PLOT_PROGRESS = True
    # See end of file for the rest of the __main__.
else:
	PLOT_PROGRESS = False

def _spline_window(window_size, power=2):
    """
    Squared spline (power=2) window function:
    https://www.wolframalpha.com/input/?i=y%3Dx**2,+y%3D-(x-2)**2+%2B2,+y%3D(x-4)**2,+from+y+%3D+0+to+2
    """
    intersection = int(window_size/4)
    wind_outer = (abs(2*(scipy.signal.triang(window_size))) ** power)/2
    wind_outer[intersection:-intersection] = 0

    wind_inner = 1 - (abs(2*(scipy.signal.triang(window_size) - 1)) ** power)/2
    wind_inner[:intersection] = 0
    wind_inner[-intersection:] = 0

    wind = wind_inner + wind_outer
    wind = wind / np.average(wind)
    return wind


cached_2d_windows = dict()
def _window_2D(window_size, power=2):
    """
    Make a 1D window function, then infer and return a 2D window function.
    Done with an augmentation, and self multiplication with its transpose.
    Could be generalized to more dimensions.
    """
    # Memoization
    global cached_2d_windows
    key = "{}_{}".format(window_size, power)
    if key in cached_2d_windows:
        wind = cached_2d_windows[key]
    else:
        wind = _spline_window(window_size, power)
        wind = np.expand_dims(np.expand_dims(wind, 3), 3)
        wind = wind * wind.transpose(1, 0, 2)
        if PLOT_PROGRESS:
            # For demo purpose, let's look once at the window:
            plt.imshow(wind[:, :, 0], cmap="viridis")
            plt.title("2D Windowing Function for a Smooth Blending of "
                      "Overlapping Patches")
            plt.show()
        cached_2d_windows[key] = wind
    return wind

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

def pad_img(img, window_size, channels=3, mode='symmetric'):
    """Pads img to make it fit for extracting patches of 
    shape height*width from it
    mode -> constant, reflect 
    constant -> pads ith 0's
    reflect -> pads with reflection of image
    """
    height = width =  window_size
    print('input shape {}'.format(img.shape))
    pad_shape = return_padding(img, height, width)
    img = np.pad(img,pad_shape,mode=mode)
    print('output shape {}'.format(img.shape))
    if PLOT_PROGRESS:
        # For demo purpose, let's look once at the window:
        plt.imshow(img)
        plt.title("Padded Image for Using Tiled Prediction Patches\n"
                  "(notice the reflection effect on the padded borders)")
        plt.show()
    return img, pad_shape

def _pad_img(img, window_size, subdivisions):
    """
    Add borders to img for a "valid" border pattern according to "window_size" and
    "subdivisions".
    Image is an np array of shape (x, y, nb_channels).
    """
    aug = int(round(window_size * (1 - 1.0/subdivisions)))
    more_borders = ((aug, aug), (aug, aug), (0, 0))
    ret = np.pad(img, pad_width=more_borders, mode='reflect')
    # gc.collect()

    if PLOT_PROGRESS:
        # For demo purpose, let's look once at the window:
        plt.imshow(ret)
        plt.title("Padded Image for Using Tiled Prediction Patches\n"
                  "(notice the reflection effect on the padded borders)")
        plt.show()
    return ret


def _unpad_img(padded_img, padding):
    """
    Undo what's done in the `_pad_img` function.
    Image is an np array of shape (x, y, nb_channels).
    """
    if padding[0][1] == 0:
        img = padded_img[padding[0][0]:, padding[1][0]:-padding[1][1],:]
    elif padding[1][0] == 0:
        img = padded_img[padding[0][0]:-padding[0][1], padding[1][0]:,:]
    elif padding[0][1] == 0 and padding[1][0] == 0:
        img = padded_img[padding[0][0]:, padding[1][0]:,:]
    else:
        img = padded_img[padding[0][0]:-padding[0][1], padding[1][0]:-padding[1][1],:]
    return img


def _rotate_mirror_do(im):
    """
    Duplicate an np array (image) of shape (x, y, nb_channels) 8 times, in order
    to have all the possible rotations and mirrors of that image that fits the
    possible 90 degrees rotations.
    It is the D_4 (D4) Dihedral group:
    https://en.wikipedia.org/wiki/Dihedral_group
    """
    mirrs = []
    mirrs.append(np.array(im))
    mirrs.append(np.rot90(np.array(im), axes=(0, 1), k=1))
    mirrs.append(np.rot90(np.array(im), axes=(0, 1), k=2))
    mirrs.append(np.rot90(np.array(im), axes=(0, 1), k=3))
    im = np.array(im)[:, ::-1]
    mirrs.append(np.array(im))
    mirrs.append(np.rot90(np.array(im), axes=(0, 1), k=1))
    mirrs.append(np.rot90(np.array(im), axes=(0, 1), k=2))
    mirrs.append(np.rot90(np.array(im), axes=(0, 1), k=3))
    return mirrs


def _rotate_mirror_undo(im_mirrs):
    """
    merges a list of 8 np arrays (images) of shape (x, y, nb_channels) generated
    from the `_rotate_mirror_do` function. Each images might have changed and
    merging them implies to rotated them back in order and average things out.
    It is the D_4 (D4) Dihedral group:
    https://en.wikipedia.org/wiki/Dihedral_group
    """
    origs = []
    origs.append(np.array(im_mirrs[0]))
    origs.append(np.rot90(np.array(im_mirrs[1]), axes=(0, 1), k=3))
    origs.append(np.rot90(np.array(im_mirrs[2]), axes=(0, 1), k=2))
    origs.append(np.rot90(np.array(im_mirrs[3]), axes=(0, 1), k=1))
    origs.append(np.array(im_mirrs[4])[:, ::-1])
    origs.append(np.rot90(np.array(im_mirrs[5]), axes=(0, 1), k=3)[:, ::-1])
    origs.append(np.rot90(np.array(im_mirrs[6]), axes=(0, 1), k=2)[:, ::-1])
    origs.append(np.rot90(np.array(im_mirrs[7]), axes=(0, 1), k=1)[:, ::-1])
    return np.mean(origs, axis=0)


def _windowed_subdivs(padded_img, window_size, subdivisions, nb_classes, pred_func, OUT_OF_MEMORY = False):
    """
    Create tiled overlapping patches.
    Returns:
        5D numpy array of shape = (
            nb_patches_along_X,
            nb_patches_along_Y,
            patches_resolution_along_X,
            patches_resolution_along_Y,
            nb_output_channels
        )
    Note:
        patches_resolution_along_X == patches_resolution_along_Y == window_size
        if you get "Out of memory" error change OUT_OF_MEMORY = True
    """
    WINDOW_SPLINE_2D = _window_2D(window_size=window_size, power=2)

    step = int(window_size/subdivisions)
    padx_len = padded_img.shape[0]
    pady_len = padded_img.shape[1]
    subdivs = []

    for i in range(0, padx_len-window_size+1, step):
        subdivs.append([])
        for j in range(0, pady_len-window_size+1, step):
            patch = padded_img[i:i+window_size, j:j+window_size, :]
            subdivs[-1].append(patch)

    # Here, `gc.collect()` clears RAM between operations.
    # It should run faster if they are removed, if enough memory is available.
    gc.collect()
    subdivs = np.array(subdivs)
    gc.collect()
    a, b, c, d, e = subdivs.shape
    subdivs = subdivs.reshape(a * b, c, d, e)
    gc.collect()
    if OUT_OF_MEMORY:
        print("Optimizing memory compromizing on speed")
        pred_patches = []
        for patch in subdivs:
            pred_patches.append(pred_func(np.expand_dims(patch, axis=0)))
        subdivs = np.array(pred_patches)
    else:
        subdivs = pred_func(subdivs)

    gc.collect()
    subdivs = np.array([patch * WINDOW_SPLINE_2D for patch in subdivs])

    # Such 5D array:
    subdivs = subdivs.reshape(a, b, c, d, nb_classes)
    gc.collect()

    return subdivs


def _recreate_from_subdivs(subdivs, window_size, subdivisions, padded_out_shape):
    """
    Merge tiled overlapping patches smoothly.
    """
    step = int(window_size/subdivisions)
    padx_len = padded_out_shape[0]
    pady_len = padded_out_shape[1]

    y = np.zeros(padded_out_shape)

    a = 0
    for i in range(0, padx_len-window_size+1, step):
        b = 0
        for j in range(0, pady_len-window_size+1, step):
            windowed_patch = subdivs[a, b]
            y[i:i+window_size, j:j+window_size] = y[i:i+window_size, j:j+window_size] + windowed_patch
            b += 1
        a += 1
    return y / (subdivisions ** 2)


def predict_img_with_smooth_windowing(input_img, window_size, subdivisions, nb_classes, pred_func):
    """
    Apply the `pred_func` function to square patches of the image, and overlap
    the predictions to merge them smoothly.
    See 6th, 7th and 8th idea here:
    http://blog.kaggle.com/2017/05/09/dstl-satellite-imagery-competition-3rd-place-winners-interview-vladimir-sergey/
    """
    input_img_shape = input_img.shape
    pad, padding = pad_img(input_img, window_size)
    pads = _rotate_mirror_do(pad)

    # Note that the implementation could be more memory-efficient by merging
    # the behavior of `_windowed_subdivs` and `_recreate_from_subdivs` into
    # one loop doing in-place assignments to the new image matrix, rather than
    # using a temporary 5D array.

    # It would also be possible to allow different (and impure) window functions
    # that might not tile well. Adding their weighting to another matrix could
    # be done to later normalize the predictions correctly by dividing the whole
    # reconstructed thing by this matrix of weightings - to normalize things
    # back from an impure windowing function that would have badly weighted
    # windows.

    # For example, since the U-net of Kaggle's DSTL satellite imagery feature
    # prediction challenge's 3rd place winners use a different window size for
    # the input and output of the neural net's patches predictions, it would be
    # possible to fake a full-size window which would in fact just have a narrow
    # non-zero dommain. This may require to augment the `subdivisions` argument
    # to 4 rather than 2.

    res = []
    for pad in tqdm(pads):
        # For every rotation:
        sd = _windowed_subdivs(pad, window_size, subdivisions, nb_classes, pred_func)
        one_padded_result = _recreate_from_subdivs(
            sd, window_size, subdivisions,
            padded_out_shape=list(pad.shape[:-1])+[nb_classes])

        res.append(one_padded_result)

    # Merge after rotations:
    padded_results = _rotate_mirror_undo(res)

    prd = _unpad_img(padded_results, padding)

    prd = prd[:input_img_shape[0], :input_img_shape[1], :]

    if PLOT_PROGRESS:
        plt.imshow(prd.astype(np.int))
        plt.title("Smoothly Merged Patches that were Tiled Tighter")
        plt.show()
    return prd


def cheap_tiling_prediction(img, window_size, nb_classes, pred_func):
    """
    Does predictions on an image without tiling.
    """
    original_shape = img.shape
    full_borderx = img.shape[0] + (window_size - (img.shape[0] % window_size))
    full_bordery = img.shape[1] + (window_size - (img.shape[1] % window_size))
    prd = np.zeros((full_borderx, full_bordery, nb_classes))
    tmp = np.zeros((full_borderx, full_bordery, original_shape[-1]))
    tmp[:original_shape[0], :original_shape[1], :] = img
    img = tmp

    for i in tqdm(range(0, prd.shape[0], window_size)):
        for j in range(0, prd.shape[1], window_size):
            im = img[i:i+window_size, j:j+window_size]
            prd[i:i+window_size, j:j+window_size] = pred_func(np.array([im]))
    prd = prd[:original_shape[0], :original_shape[1]]
    if PLOT_PROGRESS:
        plt.imshow(prd.astype(np.int))
        plt.title("Cheaply Merged Patches")
        plt.show()
    return prd


def get_dummy_img(xy_size=128, nb_channels=3):
    """
    Create a random image with different luminosity in the corners.
    Returns an array of shape (xy_size, xy_size, nb_channels).
    """
    from sklearn.datasets import load_sample_image
    import cv2
    x = load_sample_image('china.jpg')
    x = cv2.resize(x, (x.shape[0],x.shape[0]))
    gc.collect()
    if PLOT_PROGRESS:
        plt.imshow(x)
        plt.title("Random image for a test")
        plt.show()
    return x


if __name__ == '__main__':


	###
	# Image:
	###
	PLOT_PROGRESS = True
	img_resolution = 600
	# 3 such as RGB, but there could be more in other cases:
	nb_channels_in = 3

	# Get an image
	input_img = get_dummy_img(img_resolution, nb_channels_in)
	# Normally, preprocess the image for input in the neural net:
	# input_img = to_neural_input(input_img)

	###
	# Neural Net predictions params:
	###

	# Number of output channels. E.g. a U-Net may output 10 classes, per pixel:
	nb_channels_out = 3
	# U-Net's receptive field border size, it does not absolutely
	# need to be a divisor of "img_resolution":
	window_size = 128

	# This here would be the neural network's predict function, to used below:
	def predict_for_patches(small_img_patches):
	    """
	    Apply prediction on images arranged in a 4D array as a batch.
	    Here, we use a random color filter for each patch so as to see how it
	    will blend.
	    Note that the np array shape of "small_img_patches" is:
	        (nb_images, x, y, nb_channels_in)
	    The returned arra should be of the same shape, except for the last
	    dimension which will go from nb_channels_in to nb_channels_out
	    """
	    small_img_patches = np.array(small_img_patches)
	    
	    rand_channel_color = np.random.random(size=(
	        small_img_patches.shape[0],
	        1,
	        1,
	        small_img_patches.shape[-1])
	    )
	    return (small_img_patches * rand_channel_color * 2).astype(np.int32)

	###
	# Doing cheap tiled prediction:
	###

	# Predictions, blending the patches:
	cheaply_predicted_img = cheap_tiling_prediction(
	    input_img, window_size, nb_channels_out, pred_func=predict_for_patches
	)

	###
	# Doing smooth tiled prediction:
	###

	# The amount of overlap (extra tiling) between windows. A power of 2, and is >= 2:
	subdivisions = 2

	# Predictions, blending the patches:
	smoothly_predicted_img = predict_img_with_smooth_windowing(
	    input_img, window_size, subdivisions,
	    nb_classes=nb_channels_out, pred_func=predict_for_patches
	)

	###
	# Demonstrating that the reconstruction is correct:
	###

	# No more plots from now on
	PLOT_PROGRESS = False

	# useful stats to get a feel on how high will be the error relatively
	print(
	    "Image's min and max pixels' color values:",
	    np.min(input_img),
	    np.max(input_img))

	# First, defining a prediction function that just returns the patch without
	# any modification:
	def predict_same(small_img_patches):
	    """
	    Apply NO prediction on images arranged in a 4D array as a batch.
	    This implies that nb_channels_in == nb_channels_out: dimensions
	    and contained values are unchanged.
	    """
	    return np.array(small_img_patches).astype(np.int32)

	same_image_reconstructed = predict_img_with_smooth_windowing(
	    input_img, window_size, subdivisions,
	    nb_classes=nb_channels_out, pred_func=predict_same
	)

	plt.imshow(same_image_reconstructed.astype(np.int32))
	plt.title("Same image reconstructed")
	plt.show()

	diff = np.mean(np.abs(same_image_reconstructed - input_img))
	print(
	    "Mean absolute reconstruction difference on pixels' color values:",
	    diff)
	print(
	    "Relative absolute mean error on pixels' color values:",
	    100*diff/(np.max(input_img)) - np.min(input_img),
	    "%")
	print(
	    "A low error confirms that the image is still "
	    "the same before and after reconstruction if no changes are "
	"made by the passed prediction function.")