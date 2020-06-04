import numpy as np
from scipy.signal import medfilt, medfilt2d, wiener


def median_filter(size = 3, axis = 2, frames = None):
    """
    Applies a median filter in 1 or 2 dimensions
    :param size: int. kernel size
    :param axis: int. 0 for X, 1 for Y, 2 for X and Y
    :return:
    """

    if size % 2 == 0: size += 1
    sh = frames.shape
    for i in range(len(frames)):
        if axis == 0:
            c = frames[i].transpose().flatten()
            frames[i] = np.reshape(medfilt(c, kernel_size = size), (sh[2],sh[1])).transpose()
        elif axis == 1:
            c = frames[i].flatten()
            frames[i] = np.reshape(medfilt(c, kernel_size = size), sh[1:3])
        else:
            frames[i] = medfilt2d(frames[i], kernel_size = size)
    frames[frames == 0.] = frames.mean()
    return frames


def wiener_filter(size = 3, axis = 2, frames = None):
    """
    Applies a Wiener filter in 1 or 2 dimensions
    :param size: int. kernel size
    :param axis: int. 0 for X, 1 for Y, 2 for X and Y
    :return:
    """

    if size is None: size = 3
    if axis is None: axis = 0
    sh = frames.shape
    for i in range(len(frames)):
        if axis == 0:
            c = frames[i].transpose().flatten()
            frames[i] = np.reshape(wiener(c, mysize = size), (sh[2],sh[1])).transpose()
        elif axis == 1:
            c = frames[i].flatten()
            frames[i] = np.reshape(wiener(c, mysize = size), sh[1:3])
        else:
            frames[i] = wiener(frames[i], mysize = size)
    return frames

try:
    from xicam.plugins.operationplugin import OperationPlugin
    class MedianFilterOperation(OperationPlugin):
        output_names = ('frames' )
        
        _func = median_filter

    class WienerFilterOperation(OperationPlugin):
        output_names = ('frames' )
        
        _func = wiener_filter

except ModuleNotFoundError:
    print('xi-cam not installed, could not import OperationPlugin for FilterOperations')
   
