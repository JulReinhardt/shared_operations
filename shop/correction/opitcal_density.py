import numpy as np

def get_I0_mask(frames: np.ndarray):
    """
    This generates a mask based on a histogram of the data.
    It attempts to find the region outside of the sample to get the I0.
    :param frames: ndarray
    :return I0_mask: ndarray
    """
    avg_frame = frames.mean(axis=0)
    hist = np.histogram(avg_frame, bins=5)
    threshold = hist[1][-2]
    I0_mask = avg_frame > threshold
    return I0_mask

def calc_optical_density(frames: np.ndarray, I0: np.ndarray=None):
    """
    Calculate optical density -log(I/I0) from frames

    :param frames: ndarray
    :param I0: None
    :return optical_density: ndarray
    """
    if I0 is None:
        mask = get_I0_mask(frames)
        I0 = (frames * mask).sum(axis=(1, 2)) / mask.sum()
        I0 = np.reshape(I0, (len(I0), 1, 1))
    frames[frames == 0.] = 1.
    optical_density = -np.log(frames/I0)
    return optical_density



""" Import OperationPlugin to use functions as operations in Xi-CAM"""
try:
    from xicam.plugins.operationplugin import OperationPlugin


    class CalcOpticalDensity(OperationPlugin):
        name = 'Optical Density'
        output_names = ('optical_density')

        _func = calc_optical_density

except ModuleNotFoundError:
    print('xi-cam not installed, could not import OperationPlugin for FilterOperations')