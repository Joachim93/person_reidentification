from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import operator

import numpy as np


__all__ = ['check_dtype',
           'to_bool',
           'to_uint8', 'to_uint16',
           'to_float16', 'to_float32', 'to_float64']


def check_dtype(dtype, allowed_dtypes):
    """
    Function to check if a given dtype is a subdtype of one of the dtypes in a
    given list.

    Parameters
    ----------
    dtype : {numpy.dtype, type}
        The dtype to check.
    allowed_dtypes : {numpy.dtype, list, tuple}
        Allowed dtype or list of allowed dtypes.

    Returns
    -------
    is_subdtype : bool
        True, if the given dtype is a valid subdtype, otherwise False.

    """
    # ensure that allowed_dtypes is a list of dtypes
    if not isinstance(allowed_dtypes, list):
        if isinstance(allowed_dtypes, tuple):
            allowed_dtypes = list(allowed_dtypes)
        else:
            allowed_dtypes = [allowed_dtypes]

    # fix bool type, since np.issubdtype(any_type, bool) is always true
    # see: https://github.com/numpy/numpy/issues/5711
    try:
        idx = allowed_dtypes.index(np.bool)
        allowed_dtypes[idx] = np.bool_
    except ValueError:
        # np.bool/bool is not in allowed_dtypes
        pass

    # check dtype
    for allowed_dtype in allowed_dtypes:
        if np.issubdtype(dtype, allowed_dtype):
            return True
    return False


def to_bool(img, threshold, relate=operator.gt):
    """
    Function to convert an image from `uintX` or `floatX` to `bool`.

    Parameters
    ----------
    img : {numpy.ndarray, list, tuple}
        The image with axes '01c' or '01'.
    threshold : {int, float}
        Threshold to use for masking (mask = relate(img, threshold).
    relate : builtin_function_or_method
        Operator to use for masking (default: operator.gt -> greater than)

    Returns
    -------
    img : numpy.ndarray
        The input image as mask with bool dtype.

    """
    assert check_dtype(img.dtype, (np.bool, np.floating, np.unsignedinteger))
    if check_dtype(img.dtype, (np.floating, np.unsignedinteger)):
        return relate(np.asanyarray(img), threshold)

    # nothing to do
    return img


def _to_uintx(img, uintx,  uintx_scale):
    """Converts the dtype of an image from `float` or `bool` to `uintX`"""
    # ensure that img is a numpy object
    img = np.asanyarray(img)
    assert check_dtype(img.dtype, (np.bool, np.floating, np.unsignedinteger))

    # handle bool or float
    if check_dtype(img.dtype, (np.bool, np.floating)):
        return (img*uintx_scale).astype(uintx)

    # ensure dtype only (copy only if dtype does not match)
    return np.asarray(img, dtype=uintx)


def to_uint8(img, cvt_scale=2**8-1):
    """
    Function to convert an image from `float` or a mask from `bool` to `uint8`.

    Parameters
    ----------
    img : {numpy.ndarray, list, tuple}
        The image with axes '01c' or '01'.
    cvt_scale : {int, float}
        Scaling factor to apply when converting from bool or float to uint8.
        (default: numpy.iinfo('uint8').max -> 2**8-1 = 255).

    Returns
    -------
    img : numpy.ndarray
        The input image/mask with uint8 dtype.

    """
    return _to_uintx(img, 'uint8', cvt_scale)


def to_uint16(img, cvt_scale=2**16-1):
    """
    Function to convert an image from `float` or a mask from `bool` to `uint16`.

    Parameters
    ----------
    img : {numpy.ndarray, list, tuple}
        The image with axes '01c' or '01'.
    cvt_scale : {int, float}
        Scaling factor to apply when converting from bool or float to uint16
        (default: numpy.iinfo('uint16').max -> 2**16-1 = 65535).

    Returns
    -------
    img : numpy.ndarray
        The input image/mask with uint16 dtype.

    """
    return _to_uintx(img, 'uint16', cvt_scale)


def _to_floatx(img, floatx,  floatx_scale):
    """Converts the dtype of an image from `uintx` or `bool` to `floatx`"""
    # ensure that img is a numpy object
    img = np.asanyarray(img)
    assert check_dtype(img.dtype, (np.bool, np.floating, np.unsignedinteger))

    # handle integer dtypes
    if check_dtype(img.dtype, np.unsignedinteger):
        return img.astype(floatx)*np.asarray(floatx_scale, dtype=floatx)

    # ensure dtype only (copy only if dtype does not match)
    return np.asarray(img, dtype=floatx)


def to_float16(img, cvt_scale=1./(2**8-1)):
    """
    Function to convert an image from `uintx` or a mask from `bool` to
    `float16`.

    Parameters
    ----------
    img : {numpy.ndarray, list, tuple}
        The image with axes '01c' or '01'.
    cvt_scale : {int, float}
        Scaling factor to apply when converting from uintX to float16
        (default: 1./numpy.iinfo('uint8').max -> 1./(2**8-1) = 1./255).

    Returns
    -------
    img : numpy.ndarray
        The input image/mask with float16 dtype.

    """
    return _to_floatx(img, 'float16', cvt_scale)


def to_float32(img, cvt_scale=1./(2**8-1)):
    """
    Function to convert an image from `uintx` or a mask from `bool` to
    `float32`.

    Parameters
    ----------
    img : {numpy.ndarray, list, tuple}
        The image with axes '01c' or '01'.
    cvt_scale : {int, float}
        Scaling factor to apply when converting from uintX to float32
        (default: 1./numpy.iinfo('uint8').max -> 1./(2**8-1) = 1./255).

    Returns
    -------
    img : numpy.ndarray
        The input image/mask with float32 dtype.

    """
    return _to_floatx(img, 'float32', cvt_scale)


def to_float64(img, cvt_scale=1./(2**8-1)):
    """
    Function to convert an image from `uintx` or a mask from `bool` to
    `float64`.

    Parameters
    ----------
    img : {numpy.ndarray, list, tuple}
        The image with axes '01c' or '01'.
    cvt_scale : {int, float}
        Scaling factor to apply when converting from uintX to float64
        (default: 1./numpy.iinfo('uint8').max -> 1./(2**8-1) = 1./255).

    Returns
    -------
    img : numpy.ndarray
        The input image/mask with float64 dtype.

    """
    return _to_floatx(img, 'float64', cvt_scale)
