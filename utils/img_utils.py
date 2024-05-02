from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import warnings
from operator import attrgetter

import numpy as np
import cv2

from .dtype_conversion import check_dtype
from .dtype_conversion import to_bool


def _rint(value):
    """Round and convert to int"""
    return int(np.round(value))


def _const(*args):
    """
    Return constant depending on OpenCV version.
    Returns first value found for supplied names of constant.
    """
    for const in args:
        try:
            return attrgetter(const)(cv2)
        except AttributeError:
            continue
    raise AttributeError(
        """Installed OpenCV version {:s} has non of the given constants.
         Tested constants: {:s}""".format(cv2.__version__, ', '.join(args))
    )


# Color Conversion Codes (only internally used ones)
_COLOR_CVT_DICT = {
    # gray -> *
    'gray2rgb': _const('cv.CV_GRAY2RGB', 'COLOR_GRAY2RGB'),
    'gray2rgba': _const('cv.CV_GRAY2RGBA', 'COLOR_GRAY2RGBA'),

    # RGB -> *
    'rgb2gray': _const('cv.CV_RGB2GRAY', 'COLOR_RGB2GRAY'),
    'rgb2rgba': _const('cv.CV_RGB2RGBA', 'COLOR_RGB2RGBA'),

    # RGBA -> *
    'rgba2gray': _const('cv.CV_RGB2GRAY', 'COLOR_RGB2GRAY'),
    'rgba2rgb': _const('cv.CV_RGBA2RGB', 'COLOR_RGBA2RGB'),

    # BGR(A) -> RGB(A)
    'bgra2rgba': _const('cv.CV_BGRA2RGBA', 'COLOR_BGRA2RGBA'),
    'bgr2rgb': _const('cv.CV_BGR2RGB', 'COLOR_BGR2RGB'),

    # RGB(A) -> BGR(A)
    'rgba2bgra': _const('cv.CV_RGBA2BGRA', 'COLOR_RGBA2BGRA'),
    'rgb2bgr': _const('cv.CV_RGB2BGR', 'COLOR_RGB2BGR')
}


# interpolation modes (all supported)
_INTERPOLATION_DICT = {
    # bicubic interpolation
    'bicubic': _const('INTER_CUBIC', 'INTER_CUBIC'),
    # nearest-neighbor interpolation
    'nearest': _const('INTER_NEAREST', 'INTER_NEAREST'),
    # bilinear interpolation (4x4 pixel neighborhood)
    'linear': _const('INTER_LINEAR', 'INTER_LINEAR'),
    # resampling using pixel area relation, preferred for shrinking
    'area': _const('INTER_AREA', 'INTER_AREA'),
    # Lanczos interpolation (8x8 pixel neighborhood)
    'lanczos4': _const('INTER_LANCZOS4', 'INTER_LANCZOS4')
}


def to_rgb(img):
    """
    Function to convert an input image to RGB color mode.

    Parameters
    ----------
    img : {numpy.ndarray, list, tuple}
        The image to convert with axes either '01' or '01c' and of dtype
        'uint8', 'uint16' or 'float32'.

    Returns
    -------
    img_rgb : numpy.ndarray
        The converted image with axes '01c' of same dtype.

    """
    # ensure that img is a numpy object
    img = np.asanyarray(img)
    assert check_dtype(img.dtype, (np.uint8, np.uint16, np.float32))

    # convert
    if img.ndim == 2:
        img = cv2.cvtColor(img, _COLOR_CVT_DICT['gray2rgb'])
    elif img.ndim == 3:
        if img.shape[-1] == 4:
            img = cv2.cvtColor(img, _COLOR_CVT_DICT['rgba2rgb'])

    # nothing to do (color mode is already rgb)
    return img


def to_rgba(img):
    """
    Function to convert an input image to RGBA color mode.

    Parameters
    ----------
    img : {numpy.ndarray, list, tuple}
        The image to convert with axes either '01' or '01c' and of dtype
        'uint8', 'uint16' or 'float32'.

    Returns
    -------
    img_rgb : numpy.ndarray
        The converted image with axes '01c'.

    """
    # ensure that img is a numpy object
    img = np.asanyarray(img)
    assert check_dtype(img.dtype, (np.uint8, np.uint16, np.float32))

    # convert
    if img.ndim == 2:
        img = cv2.cvtColor(img, _COLOR_CVT_DICT['gray2rgba'])
    elif img.ndim == 3:
        if img.shape[-1] == 3:
            img = cv2.cvtColor(img, _COLOR_CVT_DICT['rgb2rgba'])

    # nothing to do (color mode is already rgba)
    return img


def to_grayscale(img):
    """
    Function to convert an input image to grayscale.

    Parameters
    ----------
    img : {numpy.ndarray, list, tuple}
        The image to convert with axes either '01' or '01c' and of dtype
        'uint8', 'uint16' or 'float32'.

    Returns
    -------
    img_grayscale : numpy.ndarray
        The converted image with axes '01'.

    """
    # ensure that img is a numpy object
    img = np.asanyarray(img)
    assert check_dtype(img.dtype, (np.uint8, np.uint16, np.float32))

    # Convert
    if img.ndim == 3:
        img = cv2.cvtColor(img, _COLOR_CVT_DICT['rgb2gray'])
    else:
        if img.shape[-1] == 4:
            img = cv2.cvtColor(img, _COLOR_CVT_DICT['rgba2gray'])

    # Nothing to do (color mode is already grayscale)
    return img


to_mask = to_bool


def mask_to_image(mask, color):
    """
    Function to convert a mask to a RGB(A) or grayscale image using a given
    color. The dtype of the resulting image is derived from the given color.

    Parameters
    ----------
    mask: {numpy.ndarray, list, tuple}
        The mask to convert with axes '01'.
    color: {numpy.ndarray}
        The color to use for unmasking. Since `color` determines the output
        dtype, it has to be of type numpy.ndarray.

    Returns
    -------
    img : numpy.ndarray
        The converted mask as RGB image with axes '01c'.

    """
    # ensure that img is a numpy object and of dtype bool
    mask = np.asarray(mask, dtype=np.bool_)

    # check inputs
    assert mask.ndim == 2
    assert isinstance(color, np.ndarray)

    # Create image
    shape = mask.shape if color.shape[0] == 1 else mask.shape+color.shape
    img = np.zeros(shape, dtype=color.dtype)
    img[mask, ...] = color

    return img


def mask_to_rgb(mask, color=(0, 255, 0)):
    warnings.warn("`mask_to_rgb` is deprecated, use `mask_to_image` instead",
                  DeprecationWarning)
    return mask_to_image(mask, color=np.array(color, dtype='uint8'))


def resize(img, shape_or_scale, interpolation='linear'):
    """
    Function to resize a given image.

    Parameters
    ----------
    img : {numpy.ndarray, list, tuple}
        The image to convert with axes either '01' or '01c' and of dtype
        'uint8', 'uint16' or 'float32'.
    shape_or_scale : {float, tuple, list}
        The output image shape as a tuple of ints (height, width), the scale
        factors for both dimensions as a tuple of floats (fy, fx) or a single
        float as scale factor for both dimensions.
    interpolation : str
        Interpolation method to use, one of: 'nearest', 'linear' (default),
        'area', 'bicubic' or 'lanczos4'. For details, see OpenCV documentation.

    Returns
    -------
    img_resized : numpy.ndarray
        The resized input image.

    """
    # ensure that img is a numpy object
    img = np.asanyarray(img)
    assert check_dtype(img.dtype, (np.uint8, np.uint16, np.float32))

    # get current shape
    cur_height, cur_width = img.shape[:2]

    # check shape_or_scale
    if isinstance(shape_or_scale, (tuple, list)) and len(shape_or_scale) == 2:
        if all(isinstance(e, int) for e in shape_or_scale):
            new_height, new_width = shape_or_scale
        elif all(isinstance(e, float) for e in shape_or_scale):
            fy, fx = shape_or_scale
            new_height = _rint(fy*cur_height)
            new_width = _rint(fx*cur_width)
        else:
            raise ValueError("`shape_or_scale` should either be a tuple of "
                             "ints (height, width) or a tuple of floats "
                             "(fy, fx)")
    elif isinstance(shape_or_scale, float):
        new_height = _rint(shape_or_scale * cur_height)
        new_width = _rint(shape_or_scale * cur_width)
    else:
        raise ValueError("`shape_or_scale` should either be a tuple of ints "
                         "(height, width) or a tuple of floats (fy, fx) or a "
                         "single float value")

    # scale image
    if cur_height == new_height and cur_width == new_width:
        return img

    return cv2.resize(img,
                      dsize=(new_width, new_height),
                      interpolation=_INTERPOLATION_DICT[interpolation])


def blend(img1, img2, alpha=0.2):
    """
    Function to alpha composite two images. The output image is calculated
    by img_out = ( 1 - ( alpha*( img2 > 0 ) ) )*img1 + alpha*img2.

    Parameters
    ----------
    img1 : {numpy.ndarray, list, tuple}
        The first image with axes '01' or '01c' and of dtype 'uintX' or
        'floatX'. (background image).
    img2 : {numpy.ndarray, list, tuple}
        The second image with axes '01' or '01c' and of dtype 'uintX' or
        'floatX' (foreground image).
    alpha : {float}
        The alpha value to use: 0.0 <= alpha <= 1.0.

    Returns
    -------
    img_out : numpy.ndarray
        The resulting image.

    """
    # ensure that img is a numpy object
    img1 = np.asanyarray(img1)
    img2 = np.asanyarray(img2)
    assert check_dtype(img1.dtype, (np.unsignedinteger, np.floating))
    assert check_dtype(img2.dtype, (np.unsignedinteger, np.floating))
    assert img1.dtype == img2.dtype
    assert img1.ndim == img2.ndim

    if alpha > 1:
        warnings.warn("`alpha` values > 1 are deprecated, use percentage "
                      "value 0.0 <= `alpha` <= 1.0 instead",
                      DeprecationWarning)
        alpha /= 255.

    # ensure that img is a numpy object
    img1 = np.asanyarray(img1)
    img2 = np.asanyarray(img2)

    # alpha composite images
    if img2.ndim == 3:
        mask = np.any(img2 > 0, axis=2)
    else:
        mask = img2 > 0
    result = img1.copy()
    result[mask, ...] = ((1-alpha)*img1[mask, ...] +
                         alpha*img2[mask, ...]).astype(img1.dtype)
    return result


def save(filepath, img, shape_or_scale=1.0, interpolation='linear', **kwargs):
    """
    Function to write an image to disk.

    Parameters
    ----------
    filepath : str
        Filepath where to store the image.
    img : {numpy.ndarray, list, tuple}
        The image with axes '01' or '01c' and of dtype 'uint8' or 'uint16'.
    shape_or_scale : {float, tuple, list}
        Optional parameter to apply a resizing before writing to disk.
        `shape_or_scale` can be a tuple of ints (height, width) to define the
        image shape, a tuple of floats (fy, fx) to set the scale factors for
        both dimensions or a single float to set the same scale factor for both
        dimensions.
    scale : float
        Optional scale factor to apply before.
    interpolation : str
        Interpolation method to use when `shape_or_scale` is given, one of:
        'nearest', 'linear' (default), 'area', 'bicubic' or 'lanczos4'.
        For details, see OpenCV documentation.

    Raises
    ------
    IOError
        If writing has been failed.

    """
    if 'scale' in kwargs:    # pragma no cover
        warnings.warn('Parameter `scale` is deprecated, use `shape_or_scale`'
                      'instead')
        shape_or_scale = kwargs.pop('scale')

    # ensure that img is a numpy object
    img = np.asanyarray(img)
    assert check_dtype(img.dtype, (np.uint8, np.uint16))

    # Resize and handle dtype
    img = resize(img, shape_or_scale, interpolation)

    if img.ndim == 2:
        cv2.imwrite(filepath, img)
    else:
        if img.shape[-1] == 4:
            color_mode = _COLOR_CVT_DICT['rgba2bgra']
        else:
            color_mode = _COLOR_CVT_DICT['rgb2bgr']
        if not cv2.imwrite(filepath, cv2.cvtColor(img, color_mode)):
            dirname = os.path.dirname(filepath)
            if not os.path.exists(dirname):
                msg = "No such directory: '{}'".format(dirname)
            else:
                msg = "Cannot write image to '{}'".format(filepath)
            raise IOError(msg)


def load(filepath, mode=None):
    """
    Function to load an image from disk. In contrast to OpenCV, this function
    returns images with color mode RGB(A) instead of BGR(A).

    Parameters
    ----------
    filepath : str
        The filepath.
    mode:
        The Read-Mode (see cv::ImreadModes). If `mode` is None,
        `IMREAD_UNCHANGED` is used.

    Returns
    -------
    img_out : numpy.ndarray
        The loaded image with axes '01' or '01c'.

    """
    if not os.path.exists(filepath):
        raise IOError("No such file or directory: '{}'".format(filepath))

    if mode is None:
        mode = cv2.IMREAD_UNCHANGED
    img = cv2.imread(filepath, mode)

    if img.ndim > 2:
        if img.shape[-1] == 4:
            color_mode = _COLOR_CVT_DICT['bgra2rgba']
        else:
            color_mode = _COLOR_CVT_DICT['bgr2rgb']

        img = cv2.cvtColor(img, color_mode)
    return img


def stack(imgs, axis=1, pad=0, pad_value=1):
    """
    Function to stack a list of images.

    Parameters
    ----------
    imgs : {list of {numpy.ndarray, list, tuple}}
        A list of images to stack with axes '01c' or '01' and of dtype 'uintX',
        'floatX' or 'bool'.
    axis : int
        The axis along which the images will be concatenated. 0 means vertical
        stacking, while 1 means horizontal stacking.
    pad : int
        Optional padding between the images.
    pad_value : {int, float, tuple, list}
        Optional value for the padded pixel. The correct format depends on the
        axes of the input images.

    Returns
    -------
    img_stacked : numpy.ndarray
        The stacked image.

    """
    # ensure that img is a numpy object
    imgs = [np.asanyarray(img) for img in imgs]
    assert all(check_dtype(i.dtype, (np.unsignedinteger, np.floating, np.bool))
               for i in imgs)

    # check number of images
    if len(imgs) == 1:
        # only one image -> return
        return imgs[0]

    # check images
    ndim = imgs[0].ndim
    shape = imgs[0].shape
    dtype = imgs[0].dtype
    assert all(img.ndim == ndim for img in imgs[1:])
    assert all(img.shape == shape for img in imgs[1:])
    assert all(img.dtype == dtype for img in imgs[1:])

    # Check axis
    assert axis in [0, 1]

    # cast pad_value
    pad_value = np.asarray(pad_value, dtype=dtype)

    # stack images
    if pad > 0:
        # determine the shape of padding
        pad_shape = list(shape)
        pad_shape[axis] = pad

        # update list of images to stack
        to_stack = []
        for img in imgs[:-1]:
            to_stack.append(img)
            to_stack.append(np.full(pad_shape, pad_value, dtype=dtype))
        to_stack.append(imgs[-1])
    else:
        to_stack = imgs

    # stack images
    return np.concatenate(to_stack, axis=axis)


def clip(img, roi, color=None, warn_when_outside_of_image=True):
    """
    Function to clip a subimage specified by the region of interest (ROI)
    from an image with border handling. Pixels outside the image will be
    filled with defined color.

    Parameters
    ----------
    img : {numpy.ndarray, list, tuple}
        Input image with axes '01' or '01c' and of dtype 'uintX',
        'floatX' or 'bool'.
    roi : tuple
        Region of interest in the form (top, left, bottom, right).
    color : int / float / tuple
        Fill color for pixels that are out of the image, but visible in the
        clipped subimg. If color is None, the pixels are colored black.
        default: None (black filling)
    warn_when_outside_of_image : boolean
        Specifies whether or not to show a warning when the region of interest
        is fully outside of the image.

    Returns
    -------
    subimg : numpy.ndarray
        The clipped subimage with same axes as the input image.

    """
    # ensure that img is a numpy object
    img = np.asanyarray(img)
    assert check_dtype(img.dtype, (np.unsignedinteger, np.floating, np.bool))

    # get shapes
    img_shape = img.shape
    rows, cols = img_shape[:2]
    top, left, bottom, right = roi
    roi_rows = bottom - top
    roi_cols = right - left

    # create (black) subimage
    roi_shape = list(img_shape)
    roi_shape[0] = roi_rows
    roi_shape[1] = roi_cols
    if color is None:
        subimg = np.zeros(roi_shape, dtype=img.dtype)
    else:
        subimg = np.full(roi_shape, fill_value=color, dtype=img.dtype)

    # outside image borders handling
    content_rows = min(rows, bottom) - max(0, top)
    content_cols = min(cols, right) - max(0, left)
    if content_rows <= 0 or content_cols <= 0:
        # subimage is completely outside of the image --> nothing to copy
        if warn_when_outside_of_image:
            warnings.warn('ROI is fully outside of the image.')
        return subimg
    subimg_offset_r = max(-top, 0)
    subimg_offset_c = max(-left, 0)
    img_offset_r = max(0, top)
    img_offset_c = max(0, left)

    # select rows and cols to be copied
    subimg_rows = slice(subimg_offset_r, subimg_offset_r + content_rows)
    subimg_cols = slice(subimg_offset_c, subimg_offset_c + content_cols)
    img_rows = slice(img_offset_r, img_offset_r + content_rows)
    img_cols = slice(img_offset_c, img_offset_c + content_cols)

    # copy content, that is within image
    subimg[subimg_rows, subimg_cols] = img[img_rows, img_cols]

    return subimg
