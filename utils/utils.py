from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
from collections import OrderedDict


def get_files_by_extension(path,
                           extension='.png',
                           flat_structure=False,
                           recursive=False,
                           follow_links=True):
    """
    Function to search for files with given extension.

    Parameters
    ----------
    path : str
        Path to the directory to search for files.
    extension :  {str, tuple, None}
        File extension(s). Can be a str, a tuple (str, str, ...) or None to
        search for all files.
    flat_structure : bool
        If `flat_structure` is True, a simple list containing all filepaths is
        returned.
    recursive : bool
        If `recursive` is True, subdirectories will be searched too.
    follow_links : bool
        If `follow_links` is True, links are followed.

    Returns
    -------
    files : {OrderedDict, list}
        If `flat_structure` is False, an OrderedDict with keys equal to the
        directories and values equal to a list of found files (basenames) is
        returned. If `flat_structure` is True, a simple list containing all
        filepaths is returned.
        Empty directories are omitted.
        All entries are sorted.
    """
    # check input args
    if not os.path.exists(path):
        raise IOError("No such file or directory: '{}'".format(path))

    if flat_structure:
        filelist = []
    else:
        filelist = {}

    # path is a file
    if os.path.isfile(path):
        basename = os.path.basename(path)
        if extension is None or basename.lower().endswith(extension):
            if flat_structure:
                filelist.append(path)
            else:
                filelist[os.path.dirname(path)] = [basename]
        return filelist

    # get filelist
    filter_func = lambda f: extension is None or f.lower().endswith(extension)
    for root, _, filenames in os.walk(path, topdown=True,
                                      followlinks=follow_links):
        filenames = list(filter(filter_func, filenames))
        if filenames:
            if flat_structure:
                filelist.extend((os.path.join(root, f) for f in filenames))
            else:
                filelist[root] = sorted(filenames)
        if not recursive:
            break

    # return
    if flat_structure:
        return sorted(filelist)
    else:
        return OrderedDict(sorted(filelist.items()))
