#!/usr/bin/env python

# A collection of methods for accomplishing mundain img processing tasks

import cv2
import pyautogui
from skimage.color import label2rgb
from scipy import ndimage
from typing import Union
import numpy as np


'''typical use would be to reduce the size of an img to fit the monitor'''
screen_width, screen_height = pyautogui.size()
auto_scale_target_pct = 0.7 # max percentage of screen area for img
target_width = auto_scale_target_pct * screen_width
target_height = auto_scale_target_pct * screen_height
def auto_scale_img(img):
    h,w = img.shape[:2]
    if h > target_height or w > target_width:
        sh = 1 - (h - target_height) / h
        sw = 1 - (w - target_width) / w
        alpha = min(sh, sw)
        print('image shape: ', img.shape)
        print('image type: ', img.dtype)
        print('height: ', h)
        print('width: ', w)
        return cv2.resize(img, (int(w*alpha), int(h*alpha)))
    return img

def expand_gray2color(img: np.ndarray) -> Union[np.ndarray, None]:
    ndims = len(img.shape)
    if ndims == 3:
        return img
    elif ndims == 2:
        return np.stack((img,)*3, axis=-1)
    return None

''' Shortcut for displaying an image (combines imshow and waitkey)
timeout=0 results in the image being displayed until the user pressed "q".
timeout is in ms'''
def disp_img(img, window_name="Image", timeout=0, auto_scale=True):
    if img.dtype == bool:
        img = 255*img.astype(np.uint8)
    if auto_scale:
        img = auto_scale_img(img)

    cv2.imshow(window_name, img)
    while True:
        key = cv2.waitKey(timeout)
        if key == -1 or key & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()

def disp_img_file(fname, timeout=0):
    img = cv2.imread(fname)
    disp_img(img, window_name=fname, timeout=timeout)


def disp_segmentation_labels(labels, img, timeout=0):
    image_label_overlay = label2rgb(labels, image=img, bg_label=0, alpha=0.5)
    disp_img(image_label_overlay, window_name="Labels", timeout=timeout)

'''For a binary image containing a single object, return the bounding contour
of the object'''
def get_bounding_contour(binary, return_longest=False):
    if binary.dtype == bool:
        binary = 255*binary.astype(np.uint8)
    contours = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]

    if not return_longest and len(contours) != 1:
        print(f"__FILE__: object does not have a single bounding box")
        return None

    longest_contour = contours[0]
    for cnt in contours:
        if len(cnt) > len(longest_contour):
            longest_contour = cnt
    return longest_contour

'''For a binary image containing a single object, return the bounding rectangle
of the object'''
def get_bounding_rect(binary, use_biggest_cnt=False):
    contour = get_bounding_contour(binary, return_longest=use_biggest_cnt)
    if contour is None:
        return None, None
    x,y,w,h = cv2.boundingRect(contour)
    return [y, x, y+h, x+w], contour

'''Finds the bounding box around an object in an image defined by
a binary mask and returns the subset of img pixels within the bounding
rectangle. The background pixels are white by default.
TODO make work for all data types'''
def extract_object_from_mask(img, mask, bg=(255,255,255)):
    img_in = expand_gray2color(img)
    if img_in is None:
        return None
    bb, _ = get_bounding_rect(mask)
    if bb is None:
        return None
    out = img_in.copy()
    out[~mask.astype(bool)] = bg
    return out[bb[0]:bb[2], bb[1]:bb[3]]

'''Rotates an ellipse like object in the given image defined by a maskand
and returns a vertically aligned, bounded image of the object.'''
def align_elliptical_object(img, mask, bg=(255,255,255)):
    img_in = expand_gray2color(img)
    if img_in is None:
        return None
    bb, contour = get_bounding_rect(mask)
    if bb is None:
        return None

    # extract the object
    w,h = (bb[3]-bb[1], bb[2]-bb[0])
    obj = img_in.copy()
    obj[~mask.astype(bool)] = bg

    '''
    # pad the background
    padded_obj = cv2.copyMakeBorder(
            obj[bb[0]:bb[2], bb[1]:bb[3]],
            h, h, w, w,
            cv2.BORDER_CONSTANT,
            value=bg)
    padded_mask = cv2.copyMakeBorder(
            mask[bb[0]:bb[2], bb[1]:bb[3]],
            h, h, w, w,
            cv2.BORDER_CONSTANT,
            value=(0,0,0))
    '''

    # Fit ellipse to obj mask to get rotation angle. To vertically
    # align object need to use the complement of the ellipse major
    # axis angle.
    _, _, angle = cv2.fitEllipse(contour)
    #angle = angle - 90

    # rotate both padded images. need the mask rotated to track
    # the position of the object.
    if mask.dtype == bool:
        mask = 255*mask.astype(np.uint8)
    rot_obj = ndimage.rotate(obj[bb[0]:bb[2], bb[1]:bb[3]], angle, cval=255)
    rot_mask = ndimage.rotate(mask[bb[0]:bb[2], bb[1]:bb[3]], angle, cval=0)
    from PIL import Image
    x = Image.fromarray(rot_obj)
    y = Image.fromarray(rot_mask)
    z = Image.fromarray(mask[bb[0]:bb[2], bb[1]:bb[3]])

    # Rotating results in a padded image. Need to tightly bound the
    # object with another rectangle.
    bb, _ = get_bounding_rect(rot_mask, use_biggest_cnt=True)
    if bb is None:
        return None
    return rot_obj[bb[0]:bb[2], bb[1]:bb[3]]
