import numpy as np

#linFitSpectra
def lin_fit_spectra(image_data, target_spectra):
    """
    This method uses Singular Value Decomposition to solve for the maps of chemical weights.
    :param image_data: 3D numpy array (nEnergiesxMxN)
    :param target_spectra: list. Each element is 1D numpy (nEnergies,1)
    :return:
    """
    n_spectra = len(target_spectra)  # list of spectra
    target_spectra = np.vstack(target_spectra)  # convert to matrix
    image_data = np.transpose(image_data, axes=(1, 2, 0))
    nY, nX, n_energies = image_data.shape
    retries = 5
    while retries > 0:
        try:
            u, s, vh = np.linalg.svd(target_spectra, full_matrices=False)
        except np.linalg.LinAlgError:
            retries -= 1
        else:
            break
    mu_inverse = np.dot(np.dot(vh.T, np.linalg.inv(np.diag(s))), u.T)
    target_SVDmaps = np.dot(image_data, mu_inverse)
    target_SVDmaps = np.transpose(np.reshape(target_SVDmaps, (nY, nX, n_spectra),
                                  order='F').real, axes=(2, 0, 1))
    target_SVDmaps *= target_SVDmaps > 0.
    target_inverse_spectra = mu_inverse
    # target_spectra = target_spectra
    return target_SVDmaps, target_inverse_spectra

#lstsqFit
def lstsq_fit(image_data, target_spectra):
    """
    This method uses least squares to solve for the maps of chemical weights.
    :param image_data: 3D numpy array (nEnergiesxMxN)
    :param target_spectra: list. Each element is 1D numpy (nEnergies,1)
    :return:
    """
    n_spectra = len(target_spectra)  # list of spectra
    target_spectra = np.vstack(target_spectra).T  # convert to matrix
    image_data = np.transpose(image_data, axes=(1, 2, 0))
    nY, nX, n_energies = image_data.shape
    image_data = np.reshape(image_data, (nY * nX, n_energies)).T
    s = np.linalg.lstsq(target_spectra, image_data)
    r_map = np.reshape(s[1], (nY, nX))
    target_SVDmaps = np.reshape(s[0], (n_spectra, nY, nX))
    target_SVDmaps *= target_SVDmaps > 0.
    target_spectra = target_spectra.T
    return r_map


#nnls
def nn_lstsq(image_data, target_spectra):
    """
     This method uses non-negative least squares to solve for the maps of chemical weights.
     :param image_data: 3D numpy array (nEnergiesxMxN)
     :param target_spectra: list. Each element is 1D numpy (nEnergies,1)
     :return:
     """
    n_spectra = len(target_spectra)  # list of spectra
    target_spectra = np.vstack(target_spectra).T  # convert to matrix
    image_data = np.transpose(image_data, axes=(1, 2, 0))
    nY, nX, n_energies = image_data.shape
    image_data = np.reshape(image_data, (nY * nX, n_energies)).T
    X = np.zeros((n_spectra, image_data.shape[1]))
    resids_nnls = np.zeros(image_data.shape[1])
    for i in np.arange(image_data.shape[1]):
        X[:, i], resids_nnls[i] = scipy.optimize.nnls(target_spectra, image_data[:, i])
    r_map = np.reshape(resids_nnls, (nY, nX))
    target_SVDmaps = np.reshape(X, (n_spectra, nY, nX))
    target_spectra = target_spectra.T
    return r_map
