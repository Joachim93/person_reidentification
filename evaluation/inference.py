"""
Functions for evaluation of trained models
"""

import os
import tensorflow as tf
from tensorflow import keras
import numpy as np
from sklearn.metrics import average_precision_score
from tqdm import tqdm

from data import create_mini_batches
from evaluation import market1501_evaluation
from utils.utils import get_files_by_extension

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'


def _tf_euclidean(a, b, square_result=False):
    """
    A Tensorflow graph that calculates all pairwise euclidean distances
    between a and b

    Parameters
    ----------
    a : 2D tensor
        A batch of vectors shaped (B1, F)
    b : 2D tensor
        A batch of vectors shaped (B1, F)
    square_result : bool
        If the euclidean distance should be squared (squared euclidean)

    Returns
    -------
        The matrix of all pairwise distances between all vectors in `a` and in
        `b`, will be of shape (B1, B2).
    """
    diffs = tf.expand_dims(a, axis=1) - tf.expand_dims(b, axis=0)
    if square_result:
        return tf.reduce_sum(tf.square(diffs), axis=-1)
    else:
        return tf.sqrt(tf.reduce_sum(tf.square(diffs), axis=-1) + 1e-12)


def cosine_distances(a, b):
    """
    Computes pairwise cosine distances between two matrices.

    Parameters
    ----------
    a : 2D tensor
        A batch of vectors shaped (B1, F)
    b : 2D tensor
        A batch of vectors shaped (B1, F)

    Returns
    -------
        The matrix of all pairwise distances between all vectors in `a` and in
        `b`, will be of shape (B1, B2).
    """
    normalize_a = tf.nn.l2_normalize(a,1)
    normalize_b = tf.nn.l2_normalize(b,1)
    distance = 1 - tf.matmul(normalize_a, normalize_b, transpose_b=True)
    return distance


def evaluate_like_market1501(model, args, query_dir, gallery_dir, prep_func,
                             batch_size=16, cosine=False, epoch=None):
    """
    Evaluates a model with the Market1501 test data

    Parameters
    ----------
    model : keras.Model
        Model to evaluate
    query_dir : str
        Path to query data from dataset
    gallery_dir : str
        Path to test data from dataset
    prep_func : function
        Preprocess function for the data sequencer which should match the
        preprocess function used to train the model
    batch_size : int
        Batch size for input data. Cannot be to big. Output of the matching
        will be: (feature_size, batch_size, len(gallery_fids)).
        len(gallery_fids) is 19733 and if batch_size is too big the matching
        output will not fit into memory and the evaluation process will crash.
    cosine: bool
        whether to use cosine distance for inference (if false -> euclidean distance)
    epoch: int
        current epoch (used for logging)
    """
    gallery_fids = get_files_by_extension(gallery_dir, extension=('.png', '.jpg'),
                                          flat_structure=True)

    try:
        excluder = market1501_evaluation.ExcluderMarket(gallery_fids)
    except:
        try:
            excluder = market1501_evaluation.ExcluderDuke(gallery_fids)
        except:
            excluder = market1501_evaluation.ExcluderMSMT17(gallery_fids)

    gallery_pids = excluder.gallery_pids

    query_fids = get_files_by_extension(query_dir, extension=('.png', '.jpg'),
                                        flat_structure=True)
    batches_fids = np.split(query_fids,
                            range(batch_size, len(query_fids), batch_size))

    if cosine:
        distance_func = cosine_distances
    else:
        distance_func = _tf_euclidean

    gallery_sequ = create_mini_batches.ValidationDataSequence(gallery_dir,
                                                              batch_size,
                                                              num_class=args.num_class,
                                                              preprocess=prep_func,
                                                              test_mode=True)
    query_sequ = create_mini_batches.ValidationDataSequence(query_dir,
                                                            batch_size,
                                                            num_class=args.num_class,
                                                            preprocess=prep_func,
                                                            test_mode=True)

    if args.architecture in ("mgn", "embedding"):
        feature = -1
    elif bool(args.feature_vector == "after"):
        feature = 2
    else:
        feature = 1

    # if test time augmentation is activated, feature vectors for the original image and
    # the horizontal flipped version are calculated and averaged as final feature vector

    if args.test_time_augmentation:
        gallery_batches_out = []
        for batch in tqdm(gallery_sequ):
            flipped = []
            for img in batch[0]:
                flipped.append(np.flip(img, axis=1))
            batch_flipped = np.stack(flipped), batch[1]
            pred_1 = model(batch, mgn=bool(args.architecture == "mgn"))[feature]
            pred_2 = model(batch_flipped, mgn=bool(args.architecture == "mgn"))[feature]
            pred = (pred_1 + pred_2) /2
            gallery_batches_out.append(pred)

        batches_outs = []
        for batch in tqdm(query_sequ):
            flipped = []
            for img in batch[0]:
                flipped.append(np.flip(img, axis=1))
            batch_flipped = np.stack(flipped), batch[1]
            pred_1 = model(batch, mgn=bool(args.architecture == "mgn"))[feature]
            pred_2 = model(batch_flipped, mgn=bool(args.architecture == "mgn"))[feature]
            pred = (pred_1 + pred_2) /2
            batches_outs.append(pred)
    else:
        gallery_batches_out = [model(gallery_sequ[i], mgn=bool(args.architecture == "mgn"))[feature]
                           for i in tqdm(range(len(gallery_sequ)))]
        batches_outs = [model(query_sequ[i], mgn=bool(args.architecture == "mgn"))[feature]
                    for i in tqdm(range(len(query_sequ)))]

    gallery_outs = np.concatenate(gallery_batches_out)

    # Run over every query batch and calculate distances between query batch
    # and the complete gallery. Exclude every unwanted match and keep track
    # of the results for AP and CMC curve
    aps = []
    cmc = np.zeros(len(gallery_pids), dtype=np.int32)

    # for inference with euclidean distance on some machines GPU RAM is not sufficient und CPU musst be used
    with tf.device('/gpu:0'):
        for batch_outs, batch_fids in tqdm(zip(batches_outs, batches_fids)):

            # Calculate distances between query_batch and all test_data
            distances = distance_func(batch_outs, gallery_outs)

            # Get the pid_matches of the query batch in the gallery
            query_pids = np.array([os.path.splitext(os.path.basename(fid))[0][:4]
                                   for fid in batch_fids])
            pid_matches = np.array(gallery_pids[None] == query_pids[:, None])

            # Get the mask which excludes the unwanted distance pairings from the
            # evaluation (possible reasons: same camera or junk label).
            # Use the mask to exclude the pairings in such a way that they won't
            # affect CMC or mAP
            mask = excluder(batch_fids)
            #distances[mask] = np.inf
            distances = tf.where(tf.logical_not(mask), distances, np.inf)
            pid_matches[mask] = False

            # Keep track of statistics. Invert distances to scores using any
            # arbitrary inversion, as long as it's monotonic and well-behaved,
            # it won't change anything.
            scores = 1 / (1 + distances)
            for i in range(len(distances)):
                ap = average_precision_score(pid_matches[i], scores[i])

                if np.isnan(ap):
                    print()
                    print("WARNING: encountered an AP of NaN!")
                    print("This usually means a person only appears once.")
                    print("In this case, it's because of {}.".format(batch_fids[i]))
                    print("I'm excluding this person from eval and carrying on.")
                    print()
                    continue

                aps.append(ap)
                # Find the first true match and increment the cmc data from there on.
                k = np.where(pid_matches[i, np.argsort(distances[i])])[0][0]
                cmc[k:] += 1

    # Compute the actual cmc and mAP values
    cmc = cmc / len(query_fids)
    mean_ap = np.mean(aps)

    # Print out a short summary.
    print(
        'mAP: {:.2%} | top-1: {:.2%} top-2: {:.2%} | top-5: {:.2%} | top-10: {:.2%} | top-20: {:.2%} | nAUC5: {:.2%}'.format(
            mean_ap, cmc[0], cmc[1], cmc[4], cmc[9], cmc[19], cmc[:5].mean()))

    logs = {}
    if epoch:
        logs["epoch"] = epoch
    logs["val_mAP"] = round(mean_ap.item()*100,2)
    logs["val_CMC1"] = round(cmc[0].item()*100,2)
    logs["val_CMC2"] = round(cmc[1].item()*100,2)
    logs["val_CMC5"] = round(cmc[4].item()*100,2)
    logs["val_CMC10"] = round(cmc[9].item()*100,2)
    logs["val_CMC20"] = round(cmc[19].item()*100,2)
    logs["val_nAUC5"] = round(cmc[:5].mean()*100,2)

    return logs
