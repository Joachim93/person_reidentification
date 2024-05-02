"""
Class for writing training and evaluation loss to csv
"""

import csv
import os


class CSVLogger(object):
    """
    Class for saving results to csv.

    Parameters
    ----------
    path : str
        where logging results should be stored
    """
    def __init__(self, path):
        super(CSVLogger, self).__init__()
        self._path = path

    def write_logs(self, logs):
        if not os.path.exists(self._path):
            with open(self._path, 'w') as f:
                w = csv.DictWriter(f, list(logs.keys()))
                w.writeheader()

        with open(self._path, 'a') as f:
            w = csv.DictWriter(f, list(logs.keys()))
            w.writerow(logs)
