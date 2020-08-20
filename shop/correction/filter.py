import numpy as np
from scipy.signal import medfilt, medfilt2d, wiener
from skimage.restoration import denoise_tv_chambolle, denoise_wavelet, \
                                denoise_nl_means, estimate_sigma

#TODO pool filter in one operation
def filters():
    pass

# equiv to denoise
def denoise(frames: np.ndarray,
            mode: str='nl',
            weight: float=0.05):
    """
    Applies some basic denoise algorithm from skimage.restoration.
    :param frames: list of ndarray
    :param mode: string. 'tv', 'wavelet' or 'nl'
    :param weight: float.  make a good guess.
    :return processed_frames: list of ndarray
    """
    processed_frames = frames.copy

    if mode == 'tv':
        processed_frames = denoise_tv_chambolle(frames, weight=weight, multichannel=False)
    elif mode == 'wavelet':
        processed_frames = denoise_wavelet(frames, multichannel=False)
    elif mode == 'nl':
        for i in range(len(processed_frames)):
            processed_frames[i] = denoise_nl_means(frames[i], h=weight)
    else:
        print("no applicable denoise mode selected")
        pass
    return processed_frames

# equiv despike
def despike(frames: np.ndarray):
    """
    Applies despike algorithm
    :param frames:
    :return:
    """
    #create a copy to get array with same shape
    despiked_frames = frames.copy()
    filtered_frames = median_filter(frames)
    peakIndices = np.where(np.abs(filtered_frames - frames) > 1. * np.abs(filtered_frames - frames).std())
    despiked_frames[peakIndices] = filtered_frames[peakIndices]
    despiked_frames = despiked_frames[:, 1:-1, 1:-1]
    shape = filtered_frames.shape
    # return despiked_frames, shape
    return despiked_frames

# equiv to MedianFilter
def median_filter(frames: np.ndarray,
                  size: int = 3,
                  axis: int = 2):
    """
    Applies a median filter from scipy.signal in 1 or 2 dimensions
    :param size: int. size of Median filter in each dimension
    :param axis: int. 0 for X, 1 for Y, 2 for X and Y
    :return:
    """
    filtered_frames = frames.copy()
    if size % 2 == 0: size += 1
    sh = frames.shape
    for i in range(len(frames)):
        if axis == 0:
            c = frames[i].transpose().flatten()
            filtered_frames = np.reshape(medfilt(c, kernel_size = size), (sh[2], sh[1])).transpose()
        elif axis == 1:
            c = frames[i].flatten()
            filtered_frames = np.reshape(medfilt(c, kernel_size = size), sh[1:3])
        else:
            filtered_frames = medfilt2d(frames[i], kernel_size = size)
    filtered_frames[filtered_frames == 0.] = filtered_frames.mean()
    return filtered_frames

# equiv to WienerFilter
def wiener_filter(frames: np.ndarray,
                  size: int= 3,
                  axis: int = 2,
                  noise: float = None):
    """
    Applies a Wiener filter from scipy.signal in 1 or 2 dimensions
    :param size: int. size of Wiener filter in each dimension
    :param axis: int. 0 for X, 1 for Y, 2 for X and Y
    :param noise: float if None then noise is estimated as the average of the local variance of the input.
    :return: filtered_frames: np.ndarray
    """
    filtered_frames = frames.copy()
    sh = frames.shape
    for i in range(len(frames)):
        if axis == 0:
            c = frames[i].transpose().flatten()
            filtered_frames = np.reshape(wiener(c, mysize = size, noise=noise), (sh[2], sh[1])).transpose()
        elif axis == 1:
            c = frames[i].flatten()
            filtered_frames = np.reshape(wiener(c, mysize = size, noise=noise), sh[1:3])
        else:
            filtered_frames = wiener(frames[i], mysize = size, noise=noise)
    return filtered_frames

# equiv to nlMeansFilter()
def nonlocal_means_filter(frames: np.ndarray):
    """
    Applies a non-local means filter in 2 dimensions
    :param frames: 3D numpy array (nEnergiesxMxN)
    :return:
    """
    filtered_frames = []
    for i in range(len(frames)):
        sigma_est = np.mean(estimate_sigma(frames[i], multichannel=False))
        filtered_frames.append(denoise_nl_means(frames[i], \
                                h=0.6 * sigma_est, sigma=sigma_est,
                                fast_mode=True))
    return filtered_frames




""" Import OperationPlugin to use functions as operations in Xi-CAM"""
try:
    from xicam.plugins.operationplugin import OperationPlugin

    class MedianFilterOperation(OperationPlugin):
        name = 'Median filter'
        output_names = ('filtered_frames')

        _func = median_filter


    class WienerFilterOperation(OperationPlugin):
        name = 'Wiener filter'
        output_names = ('filtered_frames')

        _func = wiener_filter

    class nlMeansFilter(OperationPlugin):
        name = 'Non-local means filter'
        output_names = ('filtered_frames')

        _func = nonlocal_means_filter

    class Despike(OperationPlugin):
        name = 'Despike'
        output_names = ('despiked_frames')

        _func = despike

    class Denoise(OperationPlugin):
        name = 'Denoise'
        output_names = ('processed_frames')

        _func = denoise


except ModuleNotFoundError:
    print('xi-cam not installed, could not import OperationPlugin for FilterOperations')