"""
Functions for evaluation of trained models on validation data
"""

import numpy as np
from sklearn.metrics import average_precision_score
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.metrics import categorical_accuracy

from evaluation.inference import cosine_distances, _tf_euclidean
from tqdm import tqdm


def evaluate_model(model, arguments, val_seq):
    """
    Evaluates a model with the Market1501 test data

    Parameters
    ----------
    model : keras.Model
        Model to evaluate
    arguments: Namespace
        Contains all arguments, which where defined from command line when running the script.
    val_seq: keras sequence
        sequence with validation data

    Returns
    -------
    features: list of tensors
        extracted features from model
    pids: list of integers
        person id for every feature
    loss: tensor
        loss on validation data
    accuracy: float
        accuracy on validation data
    """
    features = []
    pids = []
    loss = 0
    accs = []

    for i in tqdm(range(len(val_seq))):
        images, labels = val_seq[i]

        if arguments.feature_constraint_loss:
            logits, _, feature, _ = model([images, labels])
        else:
            logits, _, feature = model([images, labels])

        if arguments.classification_loss == "softmax":
            loss_batch = keras.losses.CategoricalCrossentropy()(labels, logits)
        else:
            loss_batch = keras.losses.CategoricalCrossentropy(from_logits=True)(labels, logits)

        out = [feature]
        out += [loss_batch]
        accuracy = categorical_accuracy(labels, logits)
        out += [accuracy]

        pids += list(labels.argmax(1))

        features += list(out[0])
        loss += out[1]
        accs += list(out[2])

    return features, pids, loss / len(val_seq), np.mean(accs)


def evaluate_map_and_cmc(features_dict, m_size, distance_metric):
    """
    Evaluates validation data on market1501 test protocol

    Parameters
    ----------
    features_dict : dictionary
        contains a list of feature vectors for each pid
    m_size: int
        batch size
    distance_metric: str
        metric to use for evaluation ('cosine# or 'euclidean')

    Returns
    -------
    map: float
        mean average precision
    cmc: np.array
        rank-n-accuracy for every possible rank
    """
    query_outs = []
    query_pids = []
    gallery_outs = []
    gallery_pids = []
    for pid, key in enumerate(features_dict.keys()):
        features = features_dict[key]
        query_outs.append(features[0])
        query_pids.append(pid)
        gallery_outs += features[1:]
        gallery_pids += [pid for _ in features[1:]]
    batch_outs = [[query_outs[i * m_size:(i + 1) * m_size],
                   query_pids[i * m_size:(i + 1) * m_size]]
                  for i in range(int(np.ceil(len(query_outs)
                                             / m_size)))]

    with tf.device('/cpu:0'):
        aps = []
        cmc = np.zeros(len(gallery_pids), dtype=np.int32)
        for b_query_outs, b_pids in batch_outs:

            if distance_metric == "cosine":
                distances = cosine_distances(b_query_outs, gallery_outs)
            else:
                distances = _tf_euclidean(b_query_outs, gallery_outs)

            pid_matches = np.array(np.asarray(gallery_pids)[None] ==
                                   np.asarray(b_pids)[:, None])
            scores = 1 / (1 + distances)
            for i in range(len(distances)):
                try:
                    ap = average_precision_score(pid_matches[i], scores[i])
                except ValueError:
                    print()
                    print("Value Error")
                    print()
                    continue

                if np.isnan(ap):
                    print()
                    print("WARNING: encountered an AP of NaN!")
                    print("Reason: In validation a person has only one image")
                    print()
                    continue
                aps.append(ap)
                k = np.where(pid_matches[i, np.argsort(distances[i])])[0][0]
                cmc[k:] += 1

    mean_ap = np.mean(aps)
    cmc = cmc / len(query_pids)
    return mean_ap, cmc


def on_epoch_end(model, arguments, val_seq, batch_size, epoch=0):
    """
    Evaluates a model with the Market1501 test data

    Parameters
    ----------
    model : keras.Model
        Model to evaluate
    arguments: Namespace
        Contains all arguments, which where defined from command line when running the script.
    val_seq: keras sequence
        sequence with validation data
    batch_size: int
        batch size
    epoch: int
        current epoch

    Returns
    -------
    logs: dictionary
        contains all tracked values on validation data
    """

    features, pids, val_loss, val_acc = evaluate_model(model, arguments, val_seq)

    features_dict = {}
    for feature, pid in zip(features, pids):
        if pid in features_dict:
            features_dict[pid] += [feature]
        else:
            features_dict[pid] = [feature]

    val_map, val_cmc = evaluate_map_and_cmc(features_dict, batch_size, arguments.distance_metric)

    logs = {}
    logs["epoch"] = epoch
    logs["val_loss"] = round(val_loss.numpy().item(), 4)
    logs["val_acc"] = round(val_acc.item(), 4)
    logs["val_mAP"] = round(val_map.item()*100, 2)
    logs["val_CMC1"] = round(val_cmc[0].item()*100, 2)
    logs["val_CMC2"] = round(val_cmc[1].item()*100, 2)
    logs["val_CMC5"] = round(val_cmc[4].item()*100, 2)
    logs["val_CMC10"] = round(val_cmc[9].item()*100, 2)
    logs["val_CMC20"] = round(val_cmc[19].item()*100, 2)
    logs["val_nAUC5"] = round(val_cmc[:5].mean()*100, 2)
    print('eval_on_epoch_end_finished')
    print(logs)
    return logs
