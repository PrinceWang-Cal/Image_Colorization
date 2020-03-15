import numpy as np
import skimage as sk
import skimage.io as skio
import skimage.transform as sktr
import glob
import sys
from skimage.filters import roberts


# cut into three function
def split_im(image):
    height = np.floor(image.shape[0] / 3.0).astype(np.int)
    image1 = image[:height]
    image2 = image[height:2 * height]
    image3 = image[2 * height:3 * height]
    return image1, image2, image3


def show_im(im):
    skio.imshow(im)
    skio.show()

def stack_channel(B, G, R):
    im_stacked = np.stack(np.asarray([R, G, B]), axis = 2)
    return im_stacked


def crop_im(im, percentage):
    """
    crop the image in all four sides by <n> pixels
    """
    height = im.shape[0]
    width = im.shape[1]
    n = int(height * percentage)
    m = int(width * percentage)
    return im[n:-n, m:-m]


def crop_im_abs(im, dim):
    h = dim[0]
    w = dim[1]
    return im[h:-h, w:-w]


def resize_im(im, factor):
    size = (np.array(im.shape) * factor).astype(int)
    res_im = sktr.resize(im, size, order=3)
    return res_im


def shift_im(im, right, down):
    im = np.roll(im, right, axis=1)
    im = np.roll(im, down, axis=0)
    return im


# implement L2norm() for an image:
def L2norm(im1, im2):
    """
    inputs:
    im1 - image 1 in array representation
    im2 - image 2 in array representation

    return: the SSD of two images
    """
    im1 = crop_im(im1, 0.08)
    im2 = crop_im(im2, 0.08)
    difference = im1 - im2
    squared_diff = np.square(difference)
    ssd = np.sum(squared_diff)
    return ssd


def L2norm_optimization(disp_range, im1, im2):
    """
    disp_range: a possible window of displacements
    im1: image 1
    im2: image 2
    axis: shift along the x-axis or y-axis, axis = 0 is rolling vertically, axis = 1 is rolling horizontally
    """
    a = disp_range[0]
    b = disp_range[1]
    best_displacement = [0, 0]
    min_score = L2norm(im1, im2)
    for x in np.arange(a, b):
        for y in np.arange(a, b):
            shifted_x = np.roll(im2, x, axis=1)
            shifted_im = np.roll(shifted_x, y, axis=0)
            L2score = L2norm(im1, shifted_im)
            if L2score < min_score:
                min_score = L2score
                best_displacement = [x, y]
    return np.array(best_displacement), min_score


def pyramid_processing(im1, im2):
    """
    align the image using image pyramid procedure
    im1 - B
    im2 - R or G

    displacement

    """
    height = im1.shape[0]
    width = im2.shape[1]

    if (height < 500 and width < 500):

        displacement, ssd = L2norm_optimization([-15, 15], im1, im2)

        return displacement, [-20, 20]
    else:
        # assume we get the correct output of the imaged resized by 1/2
        im1_resized = resize_im(im1, 1 / 2)
        im2_resized = resize_im(im2, 1 / 2)

        displacement, displacement_range = pyramid_processing(im1_resized, im2_resized)
        shift = displacement * 2

        im2 = shift_im(im2, shift[0], shift[1])
        shift_this_level, ssd = L2norm_optimization(displacement_range, im1, im2)
        total_shift = shift + shift_this_level

        return total_shift, [-5, 5]


def colorize(im, with_edge):
    B, G, R = split_im(im)

    if with_edge:
        B_edge = roberts(B)
        G_edge = roberts(G)
        R_edge = roberts(R)
    else:
        B_edge = B
        G_edge = G
        R_edge = R

    im_height = B.shape[0]
    im_weight = B.shape[1]

    if max(im_height, im_weight) <= 500:
        G_displacement, _ = L2norm_optimization([-15, 15], B_edge, G_edge)
        R_displacement, _ = L2norm_optimization([-15, 15], B_edge, R_edge)
    else:
        G_displacement, _ = pyramid_processing(B_edge, G_edge)
        R_displacement, _ = pyramid_processing(B_edge, R_edge)

    G_aligned = shift_im(G, G_displacement[0], G_displacement[1])
    R_aligned = shift_im(R, R_displacement[0], R_displacement[1])

    im_stacked = stack_channel(B, G_aligned, R_aligned)
    displacement = [G_displacement, R_displacement]
    return im_stacked, displacement

def autocrop(im, displacement):
    G_disp = displacement[0]
    R_disp = displacement[1]
    height_crop = max(np.absolute(G_disp[0]),np.absolute(R_disp[0]))
    width_crop = max(np.absolute(G_disp[1]),np.absolute(R_disp[1]))
    im = crop_im(im, 0.08)
    im = crop_im_abs(im, [height_crop, width_crop])
    return im

# main function, have to add .jpg or .tif
def process_image(filename, with_edge, crop):
    im = skio.imread(filename)
    color_im, displacement = colorize(im, with_edge)
    if crop:
        color_im = autocrop(color_im, displacement)
    return color_im, displacement


def process_all(is_jpg, with_edge, crop):
    jpg_filename_list = glob.glob("./dataset/*.jpg")
    tif_filename_list = glob.glob("./dataset/*.tif")
    index = 0

    if is_jpg:
        for filename in jpg_filename_list:
            im, disp = process_image(filename, with_edge, crop)

            if with_edge:
                skio.imsave("./result/" + str(index) + str(disp) + "_with_edge.jpg", im)
                print("generated: " + filename)
            else:
                skio.imsave("./result/" + str(index) + str(disp) + "_without_edge.jpg", im)
                print("generated: " + filename)

            index += 1
    else:
        for filename in tif_filename_list:
            im, disp = process_image(filename, with_edge, crop)
            if with_edge:
                skio.imsave("./result/" + str(index) + str(disp) + "_tif_with_edge.jpg", im)
                print("generated: " + filename)
            else:
                skio.imsave("./result/" + str(index) + str(disp) + "_tif_without_edge.jpg", im)
                print("generated: " + filename)
            index += 1

if len(sys.argv) == 4 and sys.argv[0] == "main.py":

    bool_jpg = False
    bool_edge = False
    crop = False

    if sys.argv[1] == "true":
        bool_jpg = True
    if sys.argv[2] == "true":
        bool_edge = True
    if sys.argv[3] == "true":
        crop = True

    process_all(bool_jpg, bool_edge, crop)

