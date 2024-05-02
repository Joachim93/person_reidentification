"""
This file is for the creation of training and validation data.

Following datasets are being used:
- Market1501
- DukeMTCM-reID
- CUHK03
- MSMT17
"""

import os
import shutil
import functools
import numpy as np
from random import shuffle

from utils.utils import get_files_by_extension
from utils import img_utils
from parameters import parse_datasets


def standardize_cuhk03_np(input_dir, output_dir, class_start_count):
    """
    Standardize CUHK03-NP
    (https://github.com/zhunzhong07/person-re-ranking/tree/master/CUHK03-NP)
    :param input_dir: Input directory with imgs in root
    :param output_dir: Output directory of the later unified (training+val)
        dataset with id folders inside
    :param class_start_count: Starting number of the output id folders
    :return new_class_start_count: Returns the next available ID
    """
    file_list = get_files_by_extension(input_dir, extension='.png',
                                       flat_structure=True)
    i = 0
    last_class = '0001'
    for file_path in file_list:
        file_name = (os.path.split(file_path)[1][:-4]).split('_')
        if last_class != file_name[0]:
            i += 1
            last_class = file_name[0]

        target_folder_name = str().join(
            ['0' for _ in range(4 - len(str(i + class_start_count)))]) + \
            str(i + class_start_count)
        target_dir = os.path.join(output_dir, target_folder_name)

        if not os.path.exists(target_dir):
            os.makedirs(target_dir)

        shutil.copy(file_path, target_dir)
        new_file_path = os.path.join(target_dir, os.path.split(file_path)[1])
        # create new name: xxxx_xx_xx_xxxxxxx_xx = ID_Camera_Scene_Frame_Detect
        new_name = str().join(
            ['0' for _ in range(4 - len(str(i + class_start_count)))]) +\
            str(i + class_start_count) + '_0' + file_name[1][1] + str('_00_') +\
            str().join(['0' for _ in range(7 - len(file_name[2]))]) +\
            file_name[2] + '_00.png'

        os.rename(new_file_path, os.path.join(target_dir, new_name))
    return i + class_start_count + 1


def standardize_dataset(input_dir, output_dir, class_start_count, f_dat,
                        standardize_func=None, extensions=('.jpg', '.png')):
    """
    For the creation of uniform training or validation dataset:
    Every id gets inside its own folder in output_dir, folder numbering starts
    with class_start_count
    :param input_dir: Input directory with imgs in root
    :param output_dir: Output directory of the later unified (training+val)
        dataset with id folders inside
    :param f_dat: Array in format [start_id, end_id, start_cam, end_cam,
        start_scene, end_scene, start_frame, end_frame, start_det, end_det]
    :param class_start_count: Starting number of the output id folders
    :param standardize_func: Function for datasets with variable file names or
        other complications
    :param extensions: File extensions of the dataset
    :return new_class_start_count: Returns the next available ID
    """
    if standardize_func is not None:
        return standardize_func(input_dir, output_dir, class_start_count)

    file_list = get_files_by_extension(input_dir, extension=extensions,
                                       flat_structure=True)

    i = 0
    last_class = file_list[0][f_dat[0]:f_dat[1]]
    for file_path in file_list:
        if last_class != file_path[f_dat[0]:f_dat[1]]:
            i += 1
            last_class = file_path[f_dat[0]:f_dat[1]]

        target_folder_name = str().join(
            ['0' for _ in range(4 - len(str(i + class_start_count)))]) + \
            str(i + class_start_count)
        target_dir = os.path.join(output_dir, target_folder_name)

        if not os.path.exists(target_dir):
            os.makedirs(target_dir)

        shutil.copy2(file_path, target_dir)
        o_name = os.path.split(file_path)[1]
        # create name: xxxx_xx_xx_xxxxxxx_xx = ID_Camera_Scene_Frame_Detection
        # TODO use str format: f.e. print('{num:02d}'.format(num=5))
        new_name = str().join(
            ['0' for _ in range(4 - len(str(i + class_start_count)))]) + \
            str(i + class_start_count) + '_' + \
            \
            str().join(['0' for _ in range(2 - (f_dat[3] - f_dat[2]))]) + \
            str(o_name[f_dat[2]:f_dat[3]]) + '_' + \
            \
            str().join(['0' for _ in range(2 - (f_dat[5] - f_dat[4]))]) + \
            str(o_name[f_dat[4]:f_dat[5]] if f_dat[4] != 256 else '') + '_' + \
            \
            str().join(['0' for _ in range(7 - (f_dat[7] - f_dat[6]))]) + \
            str(o_name[f_dat[6]:f_dat[7]]) + '_' + \
            \
            str().join(['0' for _ in range(2 - (f_dat[9] - f_dat[8]))]) + \
            str(o_name[f_dat[8]:f_dat[9]] if f_dat[8] != 256 else '') + \
            str(o_name[-4:])

        os.rename(os.path.join(target_dir, o_name),
                  os.path.join(target_dir, new_name))
    return i + class_start_count + 1


def standardize_msmt17(input_dir, output_dir, class_start_count):
    """
        Standardize msmt17
        :param input_dir: Input directory with imgs in root
        :param output_dir: Output directory of the later unified (training+val)
            dataset with id folders inside
        :param class_start_count: Starting number of the output id folders
        :return new_class_start_count: Returns the next available ID
    """
    file_list = get_files_by_extension(input_dir, extension='.jpg',
                                       flat_structure=True)
    i = 0
    last_class = "0000"
    for file_path in file_list:
        file_name = file_path.split("/")[-1].split("_",1)
        if last_class != file_name[0]:
            i += 1
            last_class = file_name[0]

        target_folder_name = str().join(
            ['0' for _ in range(4 - len(str(i + class_start_count)))]) + \
                             str(i + class_start_count)
        target_dir = os.path.join(output_dir, target_folder_name)

        if not os.path.exists(target_dir):
            os.makedirs(target_dir)

        shutil.copy(file_path, target_dir)

    return i + class_start_count + 1


def split_training_and_validation(input_dir, output_dir, split_ratio=0.1,
                                  extensions=('.jpg', '.png')):
    """
    Moves a percentage of training image to validation.
    Sorts all folders (classes) by images in class (descending) and starts to
    sort out images to validation.
    For every class considered one image per camera (random pulled) will be
    moved to validation.
    :param input_dir: Input directory with imgs in enumerated class folders
    :param output_dir: Validation output directory
    :param split_ratio: Percentage of train images that will be
        moved to validation
    :param extensions: File extensions of the dataset
    :return acquired_classes: Number of classes in the validation dataset
    """
    file_list = get_files_by_extension(input_dir, extension=extensions,
                                       recursive=True, flat_structure=False)
    file_count = sum([len(v) for v in file_list.values()])
    min_files = round(file_count * split_ratio)
    sorted_folder_list = []
    for path in sorted(file_list, key=lambda path: len(file_list[path]),
                       reverse=True):
        sorted_folder_list.append(path)

    acquired_files = 0
    acquired_classes = 0
    for folder in sorted_folder_list:
        camera = []
        file_names = file_list[folder]
        shuffle(file_names)
        for file_name in file_names:
            if file_name[5:7] not in camera:
                camera.append(file_name[5:7])
                target_dir = os.path.join(output_dir, os.path.split(folder)[1])
                if not os.path.exists(target_dir):
                    os.makedirs(target_dir)
                shutil.move(os.path.join(folder, file_name), target_dir)
                acquired_files += 1

        acquired_classes += 1
        if acquired_files >= min_files:
            break

    return acquired_classes


def img_resize_without_warp(size, img):
    shape = img.shape
    img_resized = img_utils.resize(img,
                                   min(float(size[0])/shape[0],
                                       float(size[1])/shape[1]))
    img_out = np.zeros(size + (3, ), dtype=img.dtype)
    img_out[:img_resized.shape[0], :img_resized.shape[1]] = img_resized
    return img_out


def img_resize_random_h_crop(size, min_h, img):
    """
    Resizes the image with random crop.
    Height of the crop is drawn between max_height and min_h*max_height.
    Width will be chosen in respect to the aspect ratio.

    Parameters
    ----------
    size : tuple
        The output image shape as a tuple of ints (height, width)
    min_h : float
        Minimal height as a scale
    img : {numpy.ndarray, list, tuple}
        The image to convert with axes '01c' and of dtype
        'uint8', 'uint16' or 'float32'.

    Returns
    -------
    Resized image
    """
    shape = img.shape[:2]
    rand = np.random.rand(2)
    # Calculating top and bottom offset in random crop
    height_o = shape[0] * rand[0] * (1 - min_h)
    top_o = int(height_o * rand[1])
    bottom_o = int(shape[0] - height_o * (1 - rand[1]))

    # Calculating left and right offset according to aspect ratio and top crop
    # afterwards resize and zero padding if needed
    target_factor = size[0] / (bottom_o - top_o)
    width_o = int(shape[1] - size[1] / target_factor)
    if width_o >= 0:  # Original is to wide or fits
        left_o = width_o // 2
        right_o = shape[1] - (width_o - left_o)
        return img_utils.resize(img[top_o:bottom_o, left_o:right_o], size)
    else:  # Zero padding is needed and the img will be fitted in the center
        width = int(shape[1] * target_factor)
        width_o = size[1] - width
        left_o = width_o // 2
        right_o = left_o + width
        img_resized = img_utils.resize(img[top_o:bottom_o],
                                       (size[0], width))
        img_out = np.zeros(size + (3,), dtype=img.dtype)
        img_out[:, left_o:right_o] = img_resized
        return img_out


def preprocess_keras_resnet50(arguments, x):
    """
    Transform the input images corresponding to the used ResNet version.

    Parameters
    ----------
    arguments.resnet_version : str
        possible values: ['pytorch', 'keras']
    x : np.array
        input image

    Returns
    -------
    transformed image

    """

    x = x.astype(np.float32, copy=False)

    # convert the images from RGB to BGR, then will zero-center each color
    # channel with respect to the ImageNet dataset without scaling
    if arguments.resnet_version == "keras":
        # 'RGB'->'BGR'
        x = x[..., ::-1]

        # Zero-center by mean pixel
        mean = [103.939, 116.779, 123.68]
        x[..., 0] -= mean[0]
        x[..., 1] -= mean[1]
        x[..., 2] -= mean[2]
    else:
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        x = x / 255.0
        # normalize images
        for i in range(3):
            x[:,:,i] = (x[:,:,i]-mean[i])/std[i]

    return x


def get_preprocess_func(arguments, img_size, f_type, min_h=0.9):
    """
    Get a preprocess function.
    If f_type is not supported a AttributeError will be raised.

    Parameters
    ----------
    img_size : tuple
        Size image will be resized to
    f_type : str
        Dictates which function should be returned. Possible functions are:
            - 'resize_no_warp_keras_resnet50'
            - 'resize_no_warp_h_crop_keras_resnet50'
    min_h : float
        Parameter for 'resize_no_warp_h_crop_keras_resnet50'
        To be more specific for the function img_resize_random_h_crop

    Returns
    -------
    A preprocess function that takes the image path as an argument and
    returns the preprocessed image
    """

    if f_type is 'resize_no_warp_keras_resnet50':
        def preprocess_func(img_path):
            resize_func = functools.partial(img_resize_without_warp, img_size)
            return preprocess_keras_resnet50(arguments, resize_func(img_utils.load(img_path)))
        return preprocess_func

    elif f_type is 'resize_no_warp_h_crop_keras_resnet50':
        def preprocess_func(img_path):
            resize_func = functools.partial(img_resize_random_h_crop,
                                            img_size, min_h)
            return preprocess_keras_resnet50(
                resize_func(img_utils.load(img_path)))
        return preprocess_func

    raise AttributeError('Function type ' + f_type + ' is not supported')


def preprocess_imgs(img_dir, output_dir, preprocess_style, output_size,
                    h_flip=True, extensions=('.jpg', '.png'), warp=False):
    """
    Load img, normalize img, resize img and save img in output_dir as numpy
    array. Img can be saved as a flipped and not flipped version. Img will be
    saved with channels_last.
    Output names: str(N)+original_name and str(F)+original_name
    :param img_dir: Path to img dir with a folder for every class
    :param output_dir: Path to output dir
    :param preprocess_style: Which function should be used to norm the img
    :param output_size: (height, width) Size the img has to be resized to
    :param h_flip: (Bool) If an horizontal flip of the image should be save
    :param extensions: File extensions of the dataset
    :param warp: If the image should be resized with respect to aspect ratio
    :return: _
    """
    file_list = get_files_by_extension(img_dir, extension=extensions,
                                       recursive=True, flat_structure=False)
    for folder in file_list:
        for img_name in file_list[folder]:
            img_path = os.path.join(folder, img_name)
            img = img_utils.load(img_path)

            if preprocess_style == 'Keras_ResNet50':
                if warp:
                    img_resized = img_utils.resize(img, output_size)
                else:
                    img_resized = img_resize_without_warp(output_size, img)
                img_norm = preprocess_keras_resnet50(img_resized)
                img_name = img_name[:-4] + '.npy'
                save = np.save
            elif preprocess_style == 'nothing':
                img_norm = img
                save = img_utils.save
            else:
                print('The chosen preproces style ' + preprocess_style +
                      ' is not supported')
                return

            target_dir = os.path.join(output_dir, os.path.split(folder)[1])
            if not os.path.exists(target_dir):
                os.makedirs(target_dir)
            save(os.path.join(target_dir, ('N' if h_flip else '') +
                              img_name), img_norm)
            if h_flip:
                img_horizontal_flipped = np.flip(img_norm, 1)

                save(os.path.join(target_dir, 'F' + img_name),
                     img_horizontal_flipped)


def main():

    arguments = parse_datasets()
    input_dirs = arguments.input_dirs
    names = arguments.dataset_names
    assert len(input_dirs) == len(names),  "Number of given paths doesn't match number of given dataset names!"

    if os.path.exists(arguments.output_dir):
        shutil.rmtree(arguments.output_dir)
    if arguments.validation_dir and os.path.exists(arguments.validation_dir):
        shutil.rmtree(arguments.validation_dir)

    class_start_count = 0
    if "market" in names:
        idx = names.index("market")
        path = input_dirs[idx]
        splits = [-23, -19, -17, -16, -15, -14, -13, -7, -6, -4]
        class_start_count = standardize_dataset(path,
                                                arguments.output_dir,
                                                class_start_count,
                                                splits)
    if "duke" in names:
        idx = names.index("duke")
        path = input_dirs[idx]
        splits = [-20, -16, -14, -13, 256, 256, -11, -4, 256, 256]
        class_start_count = standardize_dataset(path,
                                                arguments.output_dir,
                                                class_start_count,
                                                splits)
    if "cuhk" in names:
        idx = names.index("cuhk")
        path = input_dirs[idx]
        class_start_count = standardize_cuhk03_np(path,
                                                  arguments.output_dir,
                                                  class_start_count)
    if "msmt" in names:
        idx = names.index("msmt")
        path = input_dirs[idx]
        class_start_count = standardize_msmt17(path,
                                               arguments.output_dir,
                                               class_start_count)

    print('There are ' + str(class_start_count)
          + ' classes in the provided datasets.')

    if arguments.validation_dir:
        val_classes = split_training_and_validation(arguments.output_dir,
                                                    arguments.validation_dir,
                                                    split_ratio=arguments.split_ratio)

        print('The validation dataset consists of ' + str(val_classes) + 'classes.')


if __name__ == "__main__":
    main()
