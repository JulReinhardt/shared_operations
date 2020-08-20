import numpy as np
import matplotlib.pyplot as plt
import cv2
from scipy import ndimage, stats
from scipy.ndimage import fourier_shift
from skimage import data
from skimage.filters import sobel, gaussian
from skimage.registration import phase_cross_correlation as register_translation# scikit-image > 0.17 required, otherwise from skimage.feature import register_translation


def register_frames_stack(frames: np.ndarray,
                          mode: str = 'translation', 
                          sobelFilter: bool = True, 
                          autocrop: bool = True):
    """
        Function to register a stack of frames using different registration modes. 
        For use in COSMIC/ALS pystxm package and Xi-CAM.spectral plugin.
        Only translation mode implemented yet.

        Parameters
        ----------
        frames : ndarray
            list of ndarray frames
        mode : str
            Alignment mode
        sobelFilter :  bool
            toggle Sobelfilter on/off
        autocrop : bool
            toggle automatic cropping on/off

        Returns
        -------
        ndarray
            2D aligned image
    """
    aligned_frames = frames.copy()
    shift_list = [] 
    for i in range(1, len(frames)):
        if i == 1 :  # for i = 1 i.e. the second frame of the stack of frames, the first frame (index 0) is used as reference
            aligned_frames[i] = _register_images(aligned_frames[i - 1], frames[i], sobelFilter = sobelFilter, mode=mode)[0]
            shifts = _register_images(aligned_frames[i - 1], frames[i], sobelFilter = sobelFilter, mode=mode)[1]
            # aligned_frames[i], warp_matrix = _register_images(aligned_frames[i - 1], frames[i], sobelFilter = sobelFilter)
        elif i > 1: # for all following frames i>1 the avg of the previously registered frames are used as reference 
            aligned_frames[i] = _register_images(aligned_frames[0:i - 1].mean(axis = 0), frames[i], sobelFilter = sobelFilter, mode=mode)[0]
            shifts = _register_images(aligned_frames[i - 1], frames[i], sobelFilter = sobelFilter, mode=mode)[1]
            # aligned_frames[i], warp_matrix = _register_images(aligned_frames[0:i - 1].mean(axis = 0), frames[i], sobelFilter = sobelFilter)
        # shifts = warp_matrix[0,2],warp_matrix[1,2]
        shift_list.append(shifts)
    shifts = np.array(shift_list)
    maxY, minY = abs(np.int(round(shifts[:,0].min()))), abs(np.int(round(shifts[:,0].max())))
    maxX, minX = abs(np.int(round(shifts[:,1].min()))), abs(np.int(round(shifts[:,1].max())))
    if maxX == 0: maxX += 1
    if maxY == 0: maxY += 1
    if autocrop: aligned_frames = aligned_frames[:,minY:-maxY, minX:-maxX]
    #TODO: David, you really want an additional cropping on top?
    # if autocrop: aligned_frames = aligned_frames[:,2:-2,2:-2]
    return aligned_frames


def _register_images(ref_image, moving_image, mode = 'translation', sobelFilter = True, upsample_factor=100):
    """ Internal registration function wrapping phase_cross_correlation 
        from skimage.registration for linear translation.
        Further alignment algorithms to be implemented.

        Parameters
        ----------
        ref_image : ndarray
            2D image array used as fixed reference for alignment
        moving_image: ndarray
            2D image array to be aligned to reference
        mode : str
            Alignment mode
        sobelFilter :  bool
            toggle Sobelfilter on/off
        upsample_factor : int
            upsampling factor for sub-pixel registration

        Returns
        -------
        ndarray
            2D aligned image
        ndarray
            2D shift vector to align image
    """
    
    if mode == 'translation':
        if sobelFilter == True:
            shifts, error, phase_diff = register_translation(sobel(ref_image), sobel(moving_image), upsample_factor=upsample_factor)
        else:
            shifts, error, phase_diff = register_translation(ref_image, moving_image, upsample_factor=upsample_factor)
        aligned_image = ndimage.interpolation.shift(moving_image, shifts, mode = 'wrap')
        #TODO warp matrix only needed for other alignment than translation mode
        # warp_matrix = np.eye(2, 3, dtype=np.float32)
        # warp_matrix[0,2],warp_matrix[1,2] = shifts
    else:
        pass
        #TODO: implement other registration modes
        # src_image = _ecc_align(ref_image, moving_image, sobel = sobelFilter, mode = mode)
    # return src_image, warp_matrix
    return aligned_image, shifts



""" Import OperationPlugin to use functions as operations in Xi-CAM"""
try:
    from xicam.plugins.operationplugin import OperationPlugin

    class RegisterOperation(OperationPlugin):
        name = 'Register frames stack'
        output_names = ('aligned_frames')

        _func = register_frames_stack

except ModuleNotFoundError:
    print('xi-cam not installed, could not import OperationPlugin for RegisterOperation')

