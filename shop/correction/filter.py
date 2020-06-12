import numpy as np
from scipy.signal import medfilt, medfilt2d, wiener
from skimage.restoration import denoise_tv_chambolle, denoise_wavelet, \
                                denoise_nl_means, estimate_sigma


# equiv to denoise
def denoise(frames, mode='nl', weight=0.05):
    """
    Applies some basic denoise algorithm from skimage.restoration.
    :param frames: list of ndarray
    :param mode: string. 'tv', 'wavelet' or 'nl'
    :param weight: float.  make a good guess.
    :return processed_frames: list of ndarray
    """
    processed_frames = []
    for i in range(len(frames)):
        if mode == 'tv':
            processed_frames.append(denoise_tv_chambolle(frames[i], weight=weight, multichannel=False))
        elif mode == 'wavelet':
            processed_frames.append(denoise_wavelet(frames[i], multichannel=False))
        elif mode == 'nl':
            processed_frames.append(denoise_nl_means(frames[i], h=weight))
        else:
            print("no applicable denoise mode selected")
            pass
    return processed_frames

# equiv despike
def despike(frames):
    """
    Applies despike algorithm
    :param frames:
    :return:
    """
    despiked_frames = frames.copy()
    filtered_frames = median_filter(frames)
    peakIndices = np.where(np.abs(filtered_frames - frames) > 1. * np.abs(filtered_frames - frames).std())
    despiked_frames[peakIndices] = filtered_frames[peakIndices]
    despiked_frames = despiked_frames[:, 1:-1, 1:-1]
    shape = filtered_frames.shape
    return despiked_frames, shape

# equiv to MedianFilter
def median_filter(size = 3, axis = 2, frames = None):
    """
    Applies a median filter from scipy.signal in 1 or 2 dimensions
    :param size: int. kernel size
    :param axis: int. 0 for X, 1 for Y, 2 for X and Y
    :return:
    """
    filtered_frames = frames.copy()
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
    return filtered_frames

# equiv to nlMeansFilter()
def nonlocal_means_filter(frames):
    """
    Applies a non-local means filter in 2 dimensions
    :param frames: 3D numpy array (nEnergiesxMxN)
    :return:
    """
    processed_frames = []
    for i in range(len(frames)):
        sigma_est = np.mean(estimate_sigma(frames[i], multichannel=False))
        processed_frames.append(denoise_nl_means(frames[i], \
                                h=0.6 * sigma_est, sigma=sigma_est,
                                fast_mode=True)
    return processed_frames

# equiv to WienerFilter
def wiener_filter(size = 3, axis = 2, frames = None):
    """
    Applies a Wiener filter from scipy.signal in 1 or 2 dimensions
    :param size: int. kernel size
    :param axis: int. 0 for X, 1 for Y, 2 for X and Y
    :return:
    """
    filtered_frames = frames.copy()
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
    return filtered_frames




""" Import OperationPlugin to use functions as operations in Xi-CAM"""
try:
    from xicam.plugins.operationplugin import OperationPlugin


    class MedianFilterOperation(OperationPlugin):
        output_names = ('frames')

        _func = median_filter


    class WienerFilterOperation(OperationPlugin):
        output_names = ('frames')

        _func = wiener_filter

except ModuleNotFoundError:
    print('xi-cam not installed, could not import OperationPlugin for FilterOperations')