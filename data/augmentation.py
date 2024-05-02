"""
Function to realize random erasing augmentation
"""

import random
import math


def random_erasing_augmentation(image, probability=0.5, sl=0.02, sh=0.4, r1=0.3, mean=(0.485, 0.456, 0.406)):
    """
    Realizes Random Erasing Augmentation like it is used in Bag of Tricks (https://arxiv.org/abs/1903.07071),

    Parameters
    ----------
    image : np.array
        input image
    probability : float
        probability of applying the augmentation
    sl : float
        minimum proportion of erased area
    sh : float
        maximum proportion of erased area
    r1: float
        maximum aspect ratio of erased area
    mean : tuple
        mean RGB values of the ImageNet dataset, which replace the erased pixels.

    Returns
    -------
    transformed image
    """
    if random.uniform(0, 1) >= probability:
        return image

    for attempt in range(100):
        area = image.shape[0] * image.shape[1]

        target_area = random.uniform(sl, sh) * area
        aspect_ratio = random.uniform(r1, 1 / r1)

        h = int(round(math.sqrt(target_area * aspect_ratio)))
        w = int(round(math.sqrt(target_area / aspect_ratio)))

        if w < image.shape[1] and h < image.shape[0]:
            x1 = random.randint(0, image.shape[0] - h)
            y1 = random.randint(0, image.shape[1] - w)

            new_img = image.numpy()
            new_img[x1:x1 + h, y1:y1 + w, 0] = mean[0]
            new_img[x1:x1 + h, y1:y1 + w, 1] = mean[1]
            new_img[x1:x1 + h, y1:y1 + w, 2] = mean[2]

            return new_img

    return image
