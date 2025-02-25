# -*- coding: utf-8 -*-
"""
Created on Wed Jan 30 12:39:18 2019

@author: Fritsch
"""
import logging, traceback
import re
import os
import numpy as np
import matplotlib.pyplot as plt
import hyperspy.api as hs
from skimage.transform import hough_circle, hough_circle_peaks, rescale
from skimage.morphology import disk
from skimage.feature import canny
from skimage.filters import median
from scipy.ndimage import rotate
from skimage.exposure import equalize_adapthist
from scipy.signal import find_peaks_cwt, savgol_filter
from scipy.optimize import curve_fit, leastsq
from scipy.special import wofz
from scipy.ndimage import map_coordinates
import pandas as pd
import datetime
from functools import reduce
from itertools import zip_longest, tee


# %% from pyabel:


def reproject_image_into_polar(data, origin=None, Jacobian=False, dr=1, dt=None):
    """
    Reprojects a 2D numpy array (``data``) into a polar coordinate system.
    "origin" is a tuple of (x0, y0) relative to the bottom-left image corner,
    and defaults to the center of the image.

    Parameters
    ----------
    data : 2D np.array
    origin : tuple
        The coordinate of the image center, relative to bottom-left
    Jacobian : boolean
        Include ``r`` intensity scaling in the coordinate transform.
        This should be included to account for the changing pixel size that
        occurs during the transform.
    dr : float
        Radial coordinate spacing for the grid interpolation
        tests show that there is not much point in going below 0.5
    dt : float
        Angular coordinate spacing (in radians)
        if ``dt=None``, dt will be set such that the number of theta values
        is equal to the maximum value between the height or the width of
        the image.

    Returns
    -------
    output : 2D np.array
        The polar image (r, theta)
    r_grid : 2D np.array
        meshgrid of radial coordinates
    theta_grid : 2D np.array
        meshgrid of theta coordinates

    Notes
    -----
    Adapted from:
    http://stackoverflow.com/questions/3798333/image-information-along-a-polar-coordinate-system

    """
    # bottom-left coordinate system requires numpy image to be np.flipud
    data = np.flipud(data)

    ny, nx = data.shape[:2]
    if origin is None:
        origin = (nx // 2, ny // 2)

    # Determine that the min and max r and theta coords will be...
    x, y = index_coords(data, origin=origin)  # (x,y) coordinates of each pixel
    r, theta = cart2polar(x, y)  # convert (x,y) -> (r,θ), note θ=0 is vertical

    nr = int(np.ceil((r.max() - r.min()) / dr))

    if dt is None:
        nt = max(nx, ny)
    else:
        # dt in radians
        nt = int(np.ceil((theta.max() - theta.min()) / dt))

    # Make a regular (in polar space) grid based on the min and max r & theta
    r_i = np.linspace(r.min(), r.max(), nr, endpoint=False)
    theta_i = np.linspace(theta.min(), theta.max(), nt, endpoint=False)
    theta_grid, r_grid = np.meshgrid(theta_i, r_i)

    # Project the r and theta grid back into pixel coordinates
    X, Y = polar2cart(r_grid, theta_grid)

    X += origin[0]  # We need to shift the origin
    Y += origin[1]  # back to the bottom-left corner...
    xi, yi = X.flatten(), Y.flatten()
    coords = np.vstack((yi, xi))  # (map_coordinates requires a 2xn array)

    zi = map_coordinates(data, coords)
    output = zi.reshape((nr, nt))

    if Jacobian:
        output = output * r_i[:, np.newaxis]

    return output, r_grid, theta_grid


def index_coords(data, origin=None):
    """
    Creates x & y coords for the indicies in a numpy array

    Parameters
    ----------
    data : numpy array
        2D data
    origin : (x,y) tuple
        defaults to the center of the image. Specify origin=(0,0)
        to set the origin to the *bottom-left* corner of the image.

    Returns
    -------
        x, y : arrays
    """
    ny, nx = data.shape[:2]
    if origin is None:
        origin_x, origin_y = nx // 2, ny // 2
    else:
        origin_x, origin_y = origin

    x, y = np.meshgrid(np.arange(float(nx)), np.arange(float(ny)))

    x -= origin_x
    y -= origin_y
    return x, y


def cart2polar(x, y):
    """
    Transform Cartesian coordinates to polar

    Parameters
    ----------
    x, y : floats or arrays
        Cartesian coordinates

    Returns
    -------
    r, theta : floats or arrays
        Polar coordinates

    """
    r = np.sqrt(x**2 + y**2)
    theta = np.arctan2(x, y)  # θ referenced to vertical
    return r, theta


def polar2cart(r, theta):
    """
    Transform polar coordinates to Cartesian

    Parameters
    -------
    r, theta : floats or arrays
        Polar coordinates

    Returns
    ----------
    x, y : floats or arrays
        Cartesian coordinates
    """
    y = r * np.cos(theta)  # θ referenced to vertical
    x = r * np.sin(theta)
    return x, y


# end from Pabel


# %%
class LSqEllipse:
    """Demonstration of least-squares fitting of ellipses

    Taken from: 10.5281/zenodo.2578663

    __author__ = "Ben Hammel, Nick Sullivan-Molina"
    __credits__ = ["Ben Hammel", "Nick Sullivan-Molina"]
    __maintainer__ = "Ben Hammel"
    __email__ = "bdhammel@gmail.com"
    __status__ = "Development"

    Requirements
    ------------
    Python 2.X or 3.X
    np
    matplotlib

    References
    ----------
    (*) Halir, R., Flusser, J.: 'Numerically Stable Direct Least Squares
        Fitting of Ellipses'
    (**) http://mathworld.wolfram.com/Ellipse.html
    (***) White, A. McHale, B. 'Faraday rotation data analysis with least-squares
        elliptical fitting'
    """

    def fit(self, data):
        """Least Squares fitting algorithm

        Theory taken from (*)
        Solving equation Sa=lCa. with a = |a b c d f g> and a1 = |a b c>
            a2 = |d f g>

        Args
        ----
        data (list:list:float): list of two lists containing the x and y data of the
            ellipse. of the form [[x1, x2, ..., xi],[y1, y2, ..., yi]]

        Returns
        ------
        coef (list): list of the coefficients describing an ellipse
           [a,b,c,d,f,g] corresponding to ax**2+2bxy+cy**2+2dx+2fy+g
        """
        x, y = np.asarray(data, dtype=float)

        # Quadratic part of design matrix [eqn. 15] from (*)
        D1 = np.mat(np.vstack([x**2, x * y, y**2])).T
        # Linear part of design matrix [eqn. 16] from (*)
        D2 = np.mat(np.vstack([x, y, np.ones(len(x))])).T

        # forming scatter matrix [eqn. 17] from (*)
        S1 = D1.T * D1
        S2 = D1.T * D2
        S3 = D2.T * D2

        # Constraint matrix [eqn. 18]
        C1 = np.mat("0. 0. 2.; 0. -1. 0.; 2. 0. 0.")

        # Reduced scatter matrix [eqn. 29]
        M = C1.I * (S1 - S2 * S3.I * S2.T)

        # M*|a b c >=l|a b c >. Find eigenvalues and eigenvectors from this equation [eqn. 28]
        eval, evec = np.linalg.eig(M)

        # eigenvector must meet constraint 4ac - b^2 to be valid.
        cond = 4 * np.multiply(evec[0, :], evec[2, :]) - np.power(evec[1, :], 2)
        a1 = evec[:, np.nonzero(cond.A > 0)[1]]

        # |d f g> = -S3^(-1)*S2^(T)*|a b c> [eqn. 24]
        a2 = -S3.I * S2.T * a1

        # eigenvectors |a b c d f g>
        self.coef = np.vstack([a1, a2])
        self._save_parameters()

    def _save_parameters(self):
        """finds the important parameters of the fitted ellipse

        Theory taken form http://mathworld.wolfram

        Args
        -----
        coef (list): list of the coefficients describing an ellipse
           [a,b,c,d,f,g] corresponding to ax**2+2bxy+cy**2+2dx+2fy+g

        Returns
        _______
        center (List): of the form [x0, y0]
        width (float): major axis
        height (float): minor axis
        phi (float): rotation of major axis form the x-axis in radians
        """

        # eigenvectors are the coefficients of an ellipse in general form
        # a*x^2 + 2*b*x*y + c*y^2 + 2*d*x + 2*f*y + g = 0 [eqn. 15) from (**) or (***)
        a = self.coef[0, 0]
        b = self.coef[1, 0] / 2.0
        c = self.coef[2, 0]
        d = self.coef[3, 0] / 2.0
        f = self.coef[4, 0] / 2.0
        g = self.coef[5, 0]

        # finding center of ellipse [eqn.19 and 20] from (**)
        x0 = (c * d - b * f) / (b**2.0 - a * c)
        y0 = (a * f - b * d) / (b**2.0 - a * c)

        # Find the semi-axes lengths [eqn. 21 and 22] from (**)
        numerator = 2 * (a * f * f + c * d * d + g * b * b - 2 * b * d * f - a * c * g)
        denominator1 = (b * b - a * c) * ((c - a) * np.sqrt(1 + 4 * b * b / ((a - c) * (a - c))) - (c + a))
        denominator2 = (b * b - a * c) * ((a - c) * np.sqrt(1 + 4 * b * b / ((a - c) * (a - c))) - (c + a))
        width = np.sqrt(numerator / denominator1)
        height = np.sqrt(numerator / denominator2)

        # angle of counterclockwise rotation of major-axis of ellipse to x-axis [eqn. 23] from (**)
        # or [eqn. 26] from (***).
        phi = 0.5 * np.arctan((2.0 * b) / (a - c))

        self._center = [x0, y0]
        self._width = width
        self._height = height
        self._phi = phi

    @property
    def center(self):
        return self._center

    @property
    def width(self):
        return self._width

    @property
    def height(self):
        return self._height

    @property
    def phi(self):
        """angle of counterclockwise rotation of major-axis of ellipse to x-axis
        [eqn. 23] from (**)
        """
        return self._phi

    def parameters(self):
        return self.center, self.width, self.height, self.phi


# %%
######################################### files handling ################################################


def create_timestamp():

    timestamp = str(datetime.now()).split(".")[0].split(":")
    return reduce(lambda x, y: x + y, timestamp)


def get_emi_metadata(file):
    """
    get metadata from the emi file using HyperSpy
    input:      TIA emi file
    output:     pixelsize (unit: nm-1),
                capture time (time stamp, inteval in second),
                illumation area (unit: um),
                stage z height (unit um),
                C2 lens (% of power),
                C3 lens (% of power)
                What matters here is actually the C2/C3 ratio that defines the
                location of the cross-over before the Obj-pre field
    """
    # try:
    #    emi_file = hs.load(file)
    # except MemoryError:
    emi_file = hs.load(file, lazy=True)

    cal_x = emi_file.original_metadata.ser_header_parameters.CalibrationDeltaX
    time = emi_file.original_metadata.ser_header_parameters.Time
    illuArea = emi_file.original_metadata.ObjectInfo.ExperimentalDescription.Illuminated_Area_Diameter_um
    z = emi_file.original_metadata.ObjectInfo.ExperimentalDescription.Stage_Z_um
    C2 = emi_file.original_metadata.ObjectInfo.ExperimentalDescription.C2_lens_
    try:
        C3 = emi_file.original_metadata.ObjectInfo.ExperimentalDescription.C3_lens_
    except AttributeError:
        C3 = 1

    return cal_x, time, illuArea, z, C2, C3


def get_emd_metadata(file):
    """
    get relevant metadata from the Velox emd file using HyperSpy
    input:      Velox emd file
    output:     pixelsize (unit: nm-1),
                capture time (time stamp, inteval in second),
                indicated beam convergence (unit: mrad),
                stage z height (unit um),
                C2 lens (% of power),
                C3 lens (% of power)
                What matters here is actually the C2/C3 ratio that defines the
                location of the cross-over before the Obj-pre field
    """
    #    try:
    #        s = hs.load(file)
    #    except MemoryError:
    s = hs.load(file, lazy=True)

    cal_x = float(s.original_metadata.BinaryResult.PixelSize.width) * 1e-9
    time = float(s.original_metadata.Acquisition.AcquisitionStartDatetime.DateTime)
    Conv = float(s.original_metadata.Optics.BeamConvergence) * 1e3
    Stage_z = float(s.original_metadata.Stage.Position.z) * 1e6
    Stage_x = float(s.original_metadata.Stage.Position.x) * 1e6
    Stage_y = float(s.original_metadata.Stage.Position.y) * 1e6
    C2 = float(s.original_metadata.Optics.C2LensIntensity)
    try:
        C3 = float(s.original_metadata.Optics.C3LensIntensity)
    except AttributeError:
        C3 = 1
    return cal_x, time, Conv, Stage_x, Stage_y, Stage_z, C2, C3


def ensure_folder(directory):
    """
    Creates a path if it does not exsist already. Else: it does nothing.
    Inspired from https://gist.github.com/keithweaver/562d3caa8650eefe7f84fa074e9ca949

    input: Directory as list or tuple
    """

    directory = os.path.join(*directory)

    try:

        if not os.path.exists(directory):

            os.makedirs(directory)

    except OSError:

        logging.Error("Error: Creating directory. {}".format(directory))


def select_files(file_list, pattern="Ceta \d+.tif"):
    """
    Input: List of strings, pattern = re string

    Return: List of strings matching the given re pattern
    """

    return [file for file in file_list if bool(re.search(pattern, file))]


def preview_files(file_list, start=0, **kwargs):
    """
    Saves an image of every image contained in the emd files given. The files must be in the cwd.

    Input: List of strings containing the filenames of interest.
    """
    for file in file_list:

        logging.debug("preview_files --> load {}".format(file))

        img = hs.load(file, lazy=True)

        shape = img.data.astype("float64").shape

        logging.debug("preview_files --> Shape of {} is {}".format(file, shape))
        filename = reduce(_kit_str, file.split(".")[:-1])

        title = "_preview.png"

        if len(shape) == 2:

            fig, ax = plt.subplots(figsize=(5, 5), **kwargs)

            ax.imshow(img.data.astype("float64") ** (1 / 3), cmap="binary")
            ax.set_axis_off()
            ax.set_title(filename)

            fig.savefig(filename + title)

            fig.clf()
            plt.close()

            # img.close_file()

            logging.debug(file + " done")

        elif len(shape) == 3:

            try:

                for num in range(shape[0]):

                    if num < start:

                        continue
                    # img = hs.load(file, lazy = True)
                    try:
                        subtitle = filename + "_{}".format(num)

                        logging.debug("preview_files --> Start with {}".format(subtitle))

                        fig, ax = plt.subplots(**kwargs)

                        ax.imshow(img.data.astype("float64")[num] ** (1 / 3), cmap="binary")
                        ax.set_axis_off()
                        ax.set_title(subtitle)
                        fig.savefig(subtitle + title)
                        fig.clf()
                        plt.close()

                    except TypeError:

                        logging.error("preview_files --> TypeError for position {} in {}".format(num, file))

                    logging.debug("preview_files --> " + subtitle + " done")

            except TypeError:

                logging.error("preview_files --> Cannot iterate over {}. I skip it".format(file))

                continue

            logging.debug("preview_files --> " + subtitle + " done")

        else:

            logging.error("Unknown matrix format for " + file)


def extract_files(file_list, datatype="tiff", show=True, start=0, **kwargs):
    """
    Saves an image of every image contained in the emd files given. The files must be in the cwd.

    Input: List of strings containing the filenames of interest.
    """
    for file in file_list:

        logging.debug("extract_files --> load {}".format(file))

        img = hs.load(file, lazy=True)

        shape = img.data.astype("float64").shape

        logging.debug("preview_files --> Shape of {} is {}".format(file, shape))
        filename = reduce(_kit_str, file.split(".")[:-1])

        if len(shape) == 2:

            plt.figure()
            plt.imsave(filename + f".{datatype}", img.data.astype("float64"), cmap="binary", **kwargs)
            plt.close("all")
            plt.clf()
            if show:
                fig, ax = plt.subplots(figsize=(5, 5))

                ax.imshow(img.data.astype("float64") ** (1 / 3), cmap="binary")
                ax.set_axis_off()
                ax.set_title(filename)

                plt.show()

                fig.clf()
                plt.close()

            # img.close_file()

            logging.debug(file + " done")

        elif len(shape) == 3:

            try:

                for num in range(shape[0]):

                    if num < start:

                        continue
                    # img = hs.load(file, lazy = True)
                    try:
                        subtitle = filename + f"_{num}"

                        logging.debug("extract_files --> Start with {}".format(subtitle))
                        plt.figure()
                        plt.imsave(subtitle + f".{datatype}", img.data.astype("float64")[num], cmap="binary", **kwargs)
                        plt.close("all")
                        plt.clf()
                        if show:
                            fig, ax = plt.subplots(**kwargs)

                            ax.imshow(img.data.astype("float64")[num] ** (1 / 3), cmap="binary")
                            ax.set_axis_off()
                            ax.set_title(subtitle)
                            plt.show()
                            fig.clf()
                            plt.close()

                    except TypeError:

                        logging.error("extract_files --> TypeError for position {} in {}".format(num, file))

                    logging.debug("extract_files --> " + subtitle + " done")

            except TypeError:

                logging.error("extract_files --> Cannot iterate over {}. I skip it".format(file))

                continue

            logging.debug("extract_files --> " + subtitle + " done")

        else:

            logging.error("extract_files --> Unknown matrix format for " + file)


def get_cwd(directory):
    """
    Changes cwd to directory using os

    Input: Directory as list or tuple
    """
    directory = os.path.join(*directory)

    if os.getcwd() != directory:
        os.chdir(directory)


def load_file(filename, directory=None):
    """
    input: Filename as string; show: bool, if truthy, the picture, its filename and shape are displayed.

    output: file as array
    """

    if directory is not None:
        get_cwd(directory)

    # import image
    img = hs.load(filename, lazy=True)  # .data.astype('float64')

    return img


def prepare_matrix(img):
    """
    Input: img as np.array

    Return: np.array
    """

    # normalize array to [-1:1]:
    if img.max() >= np.abs(img.min()):
        img = img / img.max()

    else:
        img = img / np.abs(img.min())

    return img


def image4hough(img, scale_factor, mask=None, show=True, hough_filter=True, **kwargs):
    """
    use median filter on a CLAHE-prepared img, apply a canny algo and rescale by scale_factor
    kwargs are passed to canny
    """

    if mask is not None:

        img = img * mask

    # don't know if disk(3) is too large...
    if hough_filter:
        img_filtered = median(equalize_adapthist(img, clip_limit=1.0), disk(3))

    else:
        img_filtered = img

    # rescale image to safe memory. Results will be rescaled
    img_filtered = rescale(img_filtered, 1 / scale_factor, anti_aliasing=True, mode="constant")

    # use canny algo to find edges. Sometimes, an additional gaussian filter with sigma = 2 is required.
    if mask is None:

        img_filtered = canny(img_filtered, **kwargs)  # , sigma = 2)

    else:

        img_filtered = canny(img_filtered, **kwargs)

    if show:

        plt.title("Canny Reslut for Hough-Transform")
        plt.imshow(img_filtered ** (1 / 3), cmap="binary")
        plt.show()
        plt.clf()
        plt.close("all")

    return img_filtered


def find_rings(img, m, n, i, mask=None, scale_factor=4, max_peaks=5, filename="test.png", show=True, old_center=None, hough_filter=True, canny_use_quantiles=True, canny_sigma=1):
    """
    finds circles in diffraction image based on hough transformation.

    INPUT:

    img (np.array) = image as grayscale

    m (int) = lower radius boundary

    n (int) = upper radius boundary

    i (int) = number of radii between m and n

    kwargs:

    mask (None,np.array): Array of ones except a region with np.nan. If none (default), mask is ignored

    scale_factor (num) = downsampling factor for images to save memory.
    Furthermore, a scale factor between 4 - 9 seems to yield the best results.
    This might be due to refiltering with a gaussian filter...

    max_peaks (int) = number of circles to be found

    filename = Name of the tif image file to process

    show (bool) = Display results on original image


    RETURN:

    np.array containing fitresults with the columns (x,y,radius)
    """
    # mask_file = 'mask_Ceta_1.tif'
    # mask area if a mask is given
    #    if mask is not None:
    #
    #        img = img * mask
    #        allowed_peaks = max_peaks
    #        max_peaks = np.inf

    # save a copy of img and process it by using Contrast Limited Adaptive
    # Histogram Equalization (CLAHE) followed by denoising
    img_filtered = image4hough(img, scale_factor, mask=mask, show=show, hough_filter=hough_filter, use_quantiles=canny_use_quantiles, sigma=canny_sigma)

    # find circles and scale values to filtered image
    hough_radii = np.arange(m / scale_factor, n / scale_factor, i)
    hough_res = hough_circle(img_filtered, hough_radii)
    # Select the most prominent circles (max_peaks)
    accums, cx, cy, radii = hough_circle_peaks(hough_res, hough_radii, total_num_peaks=max_peaks)
    # rescale
    cx = cx * scale_factor
    cy = cy * scale_factor
    radii = radii * scale_factor

    # store results in numpy array
    out = np.array(list(zip(cx, cy, radii)))

    if show:
        # Draw circles
        fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(10, 4))
        for x, y, r in out:
            # draw circle center
            ax.plot(
                [x],
                [y],
                marker="X",
                markersize=5,
            )
            # draw circle
            circle = plt.Circle((x, y), r, fill=False, color="deeppink")
            ax.add_artist(circle)

        # draw original image with false colors
        ax.imshow(img ** (1 / 3), cmap="binary")

        # save results
        fig.savefig(filename, bbox_inches="tight", dpi=300)

        # output result
        plt.show()
        fig.clf()
        plt.close()

    return out


def get_mean(circs):
    """
    Input: np.array containing circle coordinates

    Return: Tuple of averaged x, y circle center
    """

    xc = circs[..., 0].mean()

    yc = circs[..., 1].mean()

    return (xc, yc)


def img2polar(img, center, show=True, clahe=True, median_filter=True, jacobian=True, dr=1, dt=None, filename="test.png"):
    """
    Wrapper around at.polar.reproject_image_into_polar
    """

    # abel module requires a coordinate system originating from the bottom left corner. Thus, center has to be transformed
    abel_center = center[0], img.shape[0] - center[1]
    # abel_center = center

    # Perform CLAHE and noise filtering, then transform to polar coordinate system
    if clahe:

        img = equalize_adapthist(img)

    if median_filter:

        img = median(img)

    polar, r_grid, theta_grid = reproject_image_into_polar(img, abel_center, Jacobian=jacobian, dr=dr, dt=dt)

    if show:

        fig, ax = plt.subplots()

        center = round(center[0], 3), round(center[1], 3)

        ax.set(title="Center={}, jacobian={}, CLAHE={}, median Filter={}".format(center, jacobian, clahe, median_filter))
        ax.imshow(polar ** (1 / 3), cmap="binary")
        # save results
        fig.savefig(filename, bbox_inches="tight", dpi=300)

        plt.show()
        plt.clf()
        plt.close()

    return polar, r_grid, theta_grid


def find_average_peaks(polar, r_grid, theta_grid, peak_widths=[20], dr=1, min_radius=None, max_radius=None, show=True, filename="test.png"):
    """
    custom variation of abel.tools.vmi.average_radial_intensity plus peak fitting function

    min_radius (int or None): If truthy (int): Only peaks above the given radius are taken into account
    """
    dt = theta_grid[0, 1] - theta_grid[0, 0]
    polar = polar * r_grid * np.abs(np.sin(theta_grid))
    intensity = np.trapz(polar, axis=1, dx=dt)

    # get fitting r:
    r = r_grid[: intensity.shape[0], 0]

    # find peaks
    indexes_real = find_peaks_cwt(intensity, peak_widths)
    indexes = [int(round(idx * dr)) for idx in indexes_real]

    logging.debug("find_average_peaks --> find_peaks_cwt returned indexes: {}".format(indexes))

    # filter peak if min_radius is given
    if min_radius:

        indexes = [i for i in indexes if i >= min_radius]

        logging.debug("find_average_peaks --> Indexes after min_radius ({}) filtering: {}".format(min_radius, indexes))

    # filter peak if max_radius is given
    if max_radius:

        indexes = [i for i in indexes if i <= max_radius]

        logging.debug("find_average_peaks --> Indexes after max_radius ({}) filtering: {}".format(max_radius, indexes))

    if show:

        if min_radius:
            indexes_real = [i for i in indexes_real if i >= min_radius / dr]
        if max_radius:
            indexes_real = [i for i in indexes_real if i <= max_radius / dr]
        logging.debug("find_average_peaks --> plot results")

        fig, ax = plt.subplots()

        ax.plot(r, intensity, label="Data")
        ax.plot(np.asarray(indexes_real) * dr, intensity[indexes_real], "x", markersize=8, label="{} Peaks found".format(len(indexes)))
        ax.set_title("Integrated Intensity with peak detection")
        ax.set(xlabel="Radius / px", ylabel="Angular-integrated intensity / a. u.")

        for i in indexes_real:

            ax.annotate(str(int(round(i * dr))), (int(i * dr), intensity[i]))

        ax.legend(loc=0)

        # save results
        fig.savefig(filename + ".png", bbox_inches="tight", dpi=300)
        plt.show()
        fig.clf()
        plt.close()

    return indexes, r, intensity


def linear_func(xdata, a, b):
    """
    Linear function = a * xdata + b
    """

    return a * xdata + b


def gaussian_FWHM_Func(xdat, y0, xc, A, w):
    """
    FWHM Version of Gaussian:

    y0 = base, xc = center, A = area, w = FWHM


    yc = y0 + A/(w*np.sqrt(np.pi/(4*np.log(2))))
    """

    denominator = A * np.exp((-4 * np.log(2) * (xdat - xc) ** 2) / w**2)

    numerator = w * np.sqrt(np.pi / (4 * np.log(2)))

    return y0 + denominator / numerator


def lorentzian_func(xdat, xc, A, w, y0):
    """
    FWHM Version of Lorentzian:

    y0 = base, xc = center, A = area, w = FWHM

    yc = y0 + A/(w*np.sqrt(np.pi/(4*np.log(2))))

    Height of the Curve (yc - y0); H = 2 * A / (PI * w)
    """

    denominator = 2 * A * w

    enumerator = np.pi * 4 * (xdat - xc) ** 2 + w**2

    return y0 + denominator / enumerator


def get_errors(p_opt, p_cov, x_dat, y_dat, y_fit):
    """
    create Rsqaured and errors out of fitted data.
    Arguments: p_opt, p_cov, x_dat, y_dat, y_fit
    """

    # Get standard errors:
    errors = np.sqrt(np.diag(p_cov))

    r_squared = calc_r2(y_dat, y_fit)
    # R squared:
    # r_squared = 1 - np.sum((y_dat - y_fit)**2) / np.sum((y_dat - np.mean(y_dat))**2)

    # Corrected R squared
    r_squared_corr = 1 - (1 - r_squared) * (x_dat.shape[0] - 1) / (x_dat.shape[0] - len(p_opt))

    return errors, r_squared_corr


def calc_r2(y, f):
    """
    Corrected R squared
    """
    sstot = np.sum((y - y.mean()) ** 2)
    ssres = np.sum((y - f) ** 2)
    if sstot < ssres:
        # print('Hey, deine totale Streuung ist größer als deine unerklärte Streuung.')
        pass
    return 1 - ssres / sstot


def df2excel(df_fitresults, name="test", newfile=True):
    """
    Takes a DataFrame (df_fitresults) and saves it under the given 'name' in the cwd.
    If newfile is truthy (default), then the exported name will have a timestamp so that every run creates a new file.
    """

    # create saving directory:
    curr_dir = os.getcwd()

    output_dir = [".", "output"]
    ensure_folder(output_dir)

    os.chdir(os.path.join(*output_dir))

    if newfile:
        # creates a timestamp if newfile is truthy
        timestamp = str(datetime.datetime.now()).split(".")[0].split(":")
    else:
        timestamp = [""]

    # create file name as folder of cwd and with moment of execution (if newfile is truthy)
    export_name = reduce(lambda x, y: x + y, timestamp) + " " + name + ".xlsx"
    # Create a Writer Excel object
    writer = pd.ExcelWriter(export_name, engine="xlsxwriter")
    # Convert the df to the Writer object
    df_fitresults.to_excel(writer)

    logging.info(f"Saved {export_name}.")

    # restore curr_dir
    os.chdir(curr_dir)


def fit_gaussian2peaks(peaks, r, intensity, val_range=40, show=True, dr=1, filename="test.png", functype="gauss"):
    """
    INPUT:

        peaks: array of guessed peak maxima (indexes from find_average_peaks)

        r, intensity: arrays from find_average_peaks containing the radius/intensity information

    KWARGS:

        val_range (int):
            amount of data points used to estimate the gaussian.
        It is centerred at the respective peak from peaks and should amount to
        twice the value passed to find_peaks_cwt as peak_width

        show (bool): If True, the results are plotted using matplotlib

        functype (str): Possible values:

            'gauss' --> use gaussian function
            'lorentz' --> use lorentzian function

            else --> default to 'gauss'


    RETURN:

        dictionary of the shape {key:{y0:val, xc:val, A:val, w:val, y0err:val, xcerr:val, Aerr:val, werr:val, r2adj:val}}

        y0, xc, A, w: optimized values from curve fit, containing  (y0 = base, xc = center, A = area, w = FWHM)

        *err: Standard Deviation of y0, xc, A, w

        r2adj (float): Adjusted coefficient of determination

        key (int): Position of peak in peaks (0,1,...)

    """
    if show:

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(r, intensity, label="Full data range")
        ax.set(
            ylabel="Angular-integrated intensity / a. u.",
            xlabel="Radius / px",
        )
        # title = f'{filename} radius determination')

    out = {}
    # iterate over all peaks

    if functype.lower() == "gauss":

        func = gaussian_FWHM_Func

    elif functype.lower() == "lorentz":

        func = lorentzian_func

    elif functype.lower() == "voigt":

        func = voigt

    else:

        func = gaussian_FWHM_Func
        logging.warning(f"Cannot regognize function in fit_gaussian2peaks for {filename}. Default to {func}.")

    logging.debug("fit_gaussian2peaks --> enter loop over peak estimations {}".format(peaks))

    val_range = int(round(val_range / dr))
    peaks = sorted(peaks)
    for num, pk in enumerate(peaks):

        i = int(round(pk / dr))

        for tries in range(1, 5):

            interval = val_range * tries

            low, up = int(i) - interval // 2, int(i) + interval // 2
            x = r[low:up]
            y = intensity[low:up]

            logging.debug("fit_gaussian2peaks --> Prepare fitting of peak around {}".format(i))

            ##fit gaussian:
            # find starting values:
            if functype == "voigt":
                xc_start = pk
                w_start = interval / 8
                g_start = interval / 8
                A_start = np.trapz(y)  # interval/2 * y.max()
                p0 = xc_start, A_start, w_start, g_start

                if num == 0:
                    lb_xc = 0
                else:
                    lb_xc = peaks[num - 1]

                try:
                    ub_xc = peaks[num + 1]
                except IndexError:
                    ub_xc = x.max()

                # ub_wg = x.max() - x.min()
                boundaries = [[lb_xc, 0, 0, 0], [ub_xc, np.inf, np.inf, np.inf]]

            else:
                y0 = 0
                xc_start = pk
                w_start = interval / 2
                A_start = w_start * max(y)
                p0 = y0, xc_start, A_start, w_start
                boundaries = 0, np.inf
            # perform fitting
            try:
                popt, pcov = curve_fit(func, x, y, maxfev=10000000, bounds=boundaries, p0=p0)

                yfit = func(r, *popt)

                errors, r2adj = get_errors(popt, pcov, x, y, func(x, *popt))

            except ValueError:
                print(sorted(peaks))
                print(num, p0)
                print(boundaries)
                raise ValueError

            except RuntimeError:

                popt = [np.nan, np.nan, np.nan, np.nan]
                errors = [np.inf, np.inf, np.inf, np.inf]
                r2adj = 0
                yfit = func(r, *p0)

                logging.info("RuntimeError: Peak {} around {} could not be estimated.".format(pk, (low, up)))

            # safe output in out:
            #            y0, xc, A, w = popt
            #            y0err, xcerr, Aerr, werr = errors

            key = num

            if functype == "voigt":
                xc = popt[0]
                xc_err = errors[0]
            else:
                xc = popt[1]
                xc_err = errors[1]

            if (low * dr <= xc - xc_err) and (up * dr >= xc + xc_err):

                logging.debug(
                    "fit_gaussian2peaks --> fitting of peak around {} succesful after {} tries. Optimized values are {}. Errors are {}. Boundaries are {}".format(
                        i, tries, popt, errors, (low, up)
                    )
                )

                break

            else:

                logging.debug(
                    "fit_gaussian2peaks --> fitting of peak around {} not succesful after {} tries. Optimized values are {}. Errors are {}. Boundaries are {}".format(
                        i, tries, popt, errors, (low, up)
                    )
                )

        logging.debug("fit_gaussian2peaks --> Store values to out")

        if functype == "voigt":
            out["xc {}".format(key)] = popt[0]
            out["A {}".format(key)] = popt[1]
            out["sigma {}".format(key)] = popt[2]
            out["gamma {}".format(key)] = popt[3]
            out["xc_error {}".format(key)] = errors[0]
            out["A_error {}".format(key)] = errors[1]
            out["sigma_error {}".format(key)] = errors[2]
            out["gamma_error {}".format(key)] = errors[3]
        else:
            out["y0 {}".format(key)] = popt[0]
            out["xc {}".format(key)] = popt[1]
            out["A {}".format(key)] = popt[2]
            out["FWHM {}".format(key)] = popt[3]
            out["y0_error {}".format(key)] = errors[0]
            out["xc_error {}".format(key)] = errors[1]
            out["A_error {}".format(key)] = errors[2]
            out["FWHM_error {}".format(key)] = errors[3]

        out["R2adj {}".format(key)] = r2adj

        if show:

            logging.debug("fit_gaussian2peaks --> create overview image of plot function")

            ax.plot(x, y, linewidth=1, label=f"Selected data around {int(round(xc))} Iteration {tries}")
            ax.plot(r, yfit, ":", linewidth=3, label=f"{functype.capitalize()} fit around {int(round(xc))}")

            ax.annotate(str(round(xc, 2)), (xc, intensity[i]))

    logging.debug("fit_gaussian2peaks --> exit loop over peaks")

    if show:

        ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
        # save results
        try:
            fig.savefig(filename, bbox_inches="tight", dpi=300)
        except Exception as e:
            logging.error(type(e).__name__)
            logging.error(traceback.format_exc())
            logging.error("Cannot save " + filename)
        plt.show()
        fig.clf()
        plt.close()

    return out


def model_background(dct, radius, intens, background_voigt_num=4, show=True, filename="test.png", min_radius=0, max_radius=1000, dr=1, only_1_peak=False):

    # correct min and max radius
    min_radius = int(round(min_radius / dr))
    max_radius = int(round(max_radius / dr))
    ##substract fitted peak regions

    # retreive peaks
    peaks = [int(peak[3:]) for peak in dct if peak.startswith("xc ")]

    # remove region for every peak using conditional selection in
    fitted_peaks = np.zeros(intens.shape)

    for peak in peaks:

        # get peak characteristics
        xc = dct[f"xc {peak}"]
        area = dct[f"A {peak}"]
        sigma = dct[f"sigma {peak}"]
        gamma = dct[f"gamma {peak}"]

        if only_1_peak:
            area /= 1.25
            sigma /= 1.25
            gamma /= 1.25

        # crop data to region of interest
        if min_radius <= xc <= max_radius:
            fitted_peaks += voigt(radius, xc, area, sigma, gamma)

    # substract peaks
    residuals = intens - fitted_peaks
    # clip data to 0:
    condition = residuals > 0
    r = radius
    radius = radius[condition]
    residuals = residuals[condition]

    ##fit data
    # get parameter guesses and boundaries
    # max. boundaries:
    area_max = np.trapz(residuals, radius)
    max_width = max_radius - min_radius

    #    xc0 = radius.min()
    #    xc3 = radius.max()
    #    xc1 = xc3 - 0.75*xc0
    #    xc2 = xc3 - 0.25*xc0

    # different guesses if only one hkl peak is expected in the chosen range:
    if only_1_peak:
        xcs = [radius.mean() for i in range(background_voigt_num)]
        A = area_max * 10
        sigma = max_width * 10

    else:
        xcs = np.linspace(radius.min(), radius.max() - 0.2 * np.mean(radius), background_voigt_num)
        A = area_max / 2
        sigma = max_width / 2

    gamma = 0

    p0 = []
    # xc0, A/2, sigma, gamma, xc1, A/4, sigma/2, gamma/2, xc2, A/4, sigma, gamma, xc3, A/4, sigma/4, gamma/4
    for xc in xcs:
        p0.extend((xc, A / 3, sigma, gamma))

    if not only_1_peak:
        boundaries = [
            [
                min_radius,
                0,
                0,
                0,
            ]
            * background_voigt_num,
            [
                max_radius,
                area_max,
                max_width,
                max_width,
            ]
            * background_voigt_num,
        ]
    else:
        boundaries = 0, np.inf

    ##curve fitting
    # pass deviation of residuals from original spectrum as uncertainties:
    uncertainties = np.abs(intens[condition] - residuals)

    try:
        popt, pcov = curve_fit(multi_voigt, radius, residuals, p0=p0, bounds=boundaries, sigma=uncertainties, maxfev=int(1e8))
        yfit = multi_voigt(radius, *popt)
        errors, r2adj = get_errors(popt, pcov, radius, residuals, yfit)

    except ValueError:
        popt, pcov = curve_fit(multi_voigt, radius, residuals, p0=p0, bounds=(0, np.inf), sigma=uncertainties, maxfev=int(1e8))
        yfit = multi_voigt(radius, *popt)
        errors, r2adj = get_errors(popt, pcov, radius, residuals, yfit)

    except RuntimeError:
        popt = [np.nan] * len(p0)
        yfit = multi_voigt(radius, *p0)
        errors = [np.inf] * len(p0)
        r2adj = np.nan

    # append results to fit_dict
    strt = "background_voigt"
    dct[f"{strt} R2adj"] = r2adj
    strt_err = strt + "_errors"
    for n, (params, errs) in enumerate(zip(grouper(popt, 4, np.nan), grouper(errors, 4, np.nan))):
        dct[f"{strt} xc {n}"], dct[f"{strt} A {n}"], dct[f"{strt} sigma {n}"], dct[f"{strt} gamma {n}"] = params
        dct[f"{strt_err} xc {n}"], dct[f"{strt_err} A {n}"], dct[f"{strt_err} sigma {n}"], dct[f"{strt_err} gamma {n}"] = errs

    # show results
    if show:

        # use guesses in case of failed fit
        if np.isnan(r2adj):
            popt = p0

        fig, ax = plt.subplots(figsize=(5, 5), dpi=100)
        ax.plot(r, intens, "--", label="Full data range")
        ax.errorbar(radius, residuals, uncertainties, lolims=True, label="selected data range", zorder=0, mec="gray", mfc="gray")
        ax.plot(radius, yfit, label="background fit", zorder=1)
        for n, params in enumerate(grouper(popt, 4, 0)):
            n += 1
            ax.plot(radius, voigt(radius, *params), ":", label=f"Voigt {n}", zorder=1)

        ax.set(
            ylabel="Angular-integrated intensity / a. u.",
            xlabel="Radius / px",
        )
        ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
        plt.show()
        fig.clf()
        plt.close("all")

    return dct


def increment2radians(phi, phimax):
    """
    Calculates the radian value of an angle phi, if the respective rircle is
    incremented phimax times.
    """
    return phi * 2 * np.pi / phimax


def radians2increment(phi, phimax):
    """
    Calculates the increment value of an angle phi in radians, if the respective rircle is
    incremented phimax times.
    """
    return phi * phimax / (2 * np.pi)


def polar2carthesian(r, phi, abel_result=True):
    """
    INPUT: r, phi (in radians)

    RETURN: carthesian coordinates x,y
    """

    if abel_result:

        # shift origin to start lower
        phi += np.pi / 2

        # change rotation direction
        # phi *= -1

    x = r * np.cos(phi)
    y = r * np.sin(phi)

    return x, y


def carthesian2polar(x, y):
    """
    INPUT: carthesian coordinates x,y

    RETURN: r, phi (in radians)
    """

    r = np.hypot(x, y)
    phi = np.arctan2(y / x)

    return r, phi


def _get_intensity_threshold(arr, intensity_threshold):

    if type(intensity_threshold) == str:

        if " " in intensity_threshold:

            # separate string on ' '. Code will fail if more than one space is present
            factor, intensity_threshold = intensity_threshold.split()
            # convert factor to float. Code will fail, if factor is not convertible
            factor = float(factor)
            factor_avg = factor

        else:

            factor = 1
            factor_avg = 0

        if intensity_threshold == "std":

            thres = arr.mean() - factor * arr.std()

        elif intensity_threshold == "2std":

            thres = arr.mean() - factor * 2 * arr.std()

        elif intensity_threshold == "-std":

            thres = arr.mean() + factor * arr.std()

        elif intensity_threshold == "-2std":

            thres = arr.mean() + factor * 2 * arr.std()

        elif intensity_threshold == "mean":

            thres = arr.mean() + factor_avg * arr.mean()

        elif intensity_threshold == "median":

            if type(arr) == np.ndarray:
                thres = np.median(arr) + factor_avg * np.median(arr)
            else:
                thres = arr.median() + factor_avg * arr.median()

    elif type(intensity_threshold) == float:

        assert 0 <= intensity_threshold <= 1, f"intensity threshold is not between 0 and 1. Value is {intensity_threshold}"

        thres = arr.max() * intensity_threshold

    else:

        thres = 0
        logging.warning(f"No intensity filtering could be applied. All values are used for ellipse fitting. intensity_threshold == {intensity_threshold}")

    return thres


def _apply_intensity_threshold(df, intensity_threshold):

    thres = _get_intensity_threshold(df["intensity"], intensity_threshold)

    # apply threshold:
    return df[df["intensity"] >= thres]


def _apply_intensity_threshold_to_list(out, intensity_threshold, colnum=3):

    # ensure numpy array
    out_arr = np.asarray(out)
    # get intensity column
    maxvals = out_arr[..., colnum]
    # calculate threshold value
    threshold = _get_intensity_threshold(maxvals, intensity_threshold)
    # filter array
    out_arr = out_arr[maxvals >= threshold]

    return out_arr


def _select_values_by_intensity(rrange, r, rlist, minr, intensity_threshold="std", filename="test", local_intensity_scope=False, dr=1, show=False, dt=None):

    out = []

    # iterate over every column in the selected image part
    threshold = _get_intensity_threshold(rrange, intensity_threshold)

    for i in range(rrange.shape[1]):

        correct_ring = True
        counter = 0
        skipped_val = 0

        while correct_ring and counter < rrange.shape[1]:

            continue_loop = False
            break_loop = False

            counter += 1

            sliced = rrange[..., i]

            if (sliced < threshold).all():

                break

            sliced_lst = list(sliced)
            sliced_lst.sort(reverse=True)

            # get maximum value
            maxval = sliced_lst[skipped_val]

            # skip if maximum intensity is minimum intensity
            if maxval == min(sliced):

                continue_loop = True
                continue

            # get radius coordinate relative to selected image region
            r_maxval = np.where(sliced == maxval)

            # skip, if the maximum position is ambiguous
            if len(r_maxval) < 1:

                break_loop = True
                break

            # check whether value is closest to selected radius:
            rcheck = {}
            for r_i in rlist:

                # calculate distance to selected radius
                rcheck[np.sqrt((r_i / dr - r) ** 2)] = r_i / dr

            # if not, go to second
            if round(rcheck[min(rcheck)]) != round(r):

                # print(rcheck[min(rcheck)], type(rcheck[min(rcheck)]))
                # print(r, type(r))
                skipped_val += 1

                continue_loop = True

                continue
            else:

                pass

            # break loop
            correct_ring = False

        if break_loop:

            break

        if continue_loop:

            continue
        # get position of maximum relative to selected radius region rrange.
        # IMPORTANT: The 0 in peak_dct key is autogenerated and attributes to the first maximum passed to fit_gaussian2peaks
        local_r = r_maxval[0] + minr

        # store results
        out.append([r, i, local_r[0], maxval, np.nan])

    # keep results only if threshold is exceeded:
    out_arr = _apply_intensity_threshold_to_list(out, intensity_threshold)

    return out_arr


def select_ring_values(
    polar, rlist, tolerance=0.1, show=False, intensity_threshold="std", filename="test", local_intensity_scope=False, dr=1, dt=None, fit_gauss=True, double_peak=False
):
    """
    Input:

    polar: polar-transformed image
    rlist: Radius list as iterable, eg. circ[...,2]

    kwargs:

    tolerance (float): value to define the relative region around the given radius.

    show (bool): If truthy, interim results are plotted. Slows algorithm down.

    intensity_threshold (string or float):

        valid string values:

        'std' --> Only use the brightest values based on a 1 sigma standard deviation
        '2std' --> Only use the brightest values based on a 2 sigma standard deviation
        'mean' --> Only use the brightest values based on the mean intensity
        'median' --> Only use the brightest values based on the median intensity

        if float:

            Value x of interval [0,1].

    local_intensity_scope (bool):

        False (default) --> intensity_threshold is applied to all rings simulatneously
        True  --> intensity_threshold is applied to all rings separately

    Return:

    pandas.DataFrame containing the resulting values
    """

    # assert len(rlist) != 0, 'rlist is empty'

    # define a list to store results

    logging.debug("Selecting ring values. {} given as rlist".format(rlist))

    #    out = _select_values_by_intensity(rlist, polar, tolerance, dr, show = False)
    if type(tolerance) != float:

        rlist = [int(np.mean(tolerance))]
    # iterate over preselected radii
    for r in rlist:

        logging.debug(f"Iterate over ring value {r}")
        # select a region based on a given percentage of r.
        # ensure integer values for indexing

        if type(tolerance) == float:
            r = int(round(r / dr))
            val = r * tolerance
            val = int(round(val))

            logging.debug(f"Iterate over ring value {r}. Calculated value range as {val} based on dr = {dr} and tolerance = {tolerance}")

            # boundary conditions for polar-slicing
            minr, maxr = r - val, r + val

        else:

            minr, maxr = tolerance
            minr = int(round(minr / dr))
            maxr = int(round(maxr / dr))

            logging.debug(f"Peak interval is set manually to {tolerance}.Iterate over ring value {r}.")

        # get region of interest
        rrange = polar[minr:maxr]

        if show:
            fig, ax = plt.subplots()

            ax.set_title("Radius at {} px".format(r))

            ax.imshow(rrange ** (1 / 3), cmap="binary")

            plt.close("all")
            plt.clf()

        if len(rrange) == 0:

            logging.debug("rrange is empty for {}".format(r))

            continue

        if fit_gauss:
            out = _select_values_by_fit(
                rrange.T,
                r,
                rlist,
                minr,
                intensity_threshold=intensity_threshold,
                filename=filename,
                local_intensity_scope=local_intensity_scope,
                dr=dr,
                dt=dt,
                show=show,
                double_peak=double_peak,
            )
        else:
            out = _select_values_by_intensity(
                rrange,
                r,
                rlist,
                minr,
                intensity_threshold=intensity_threshold,
                filename=filename,
                local_intensity_scope=local_intensity_scope,
                dr=dr,
                dt=dt,
            )
        #

        logging.debug("Ring {}, {} values found.".format(r, len(out)))

    # make intensity dataFrame
    df = pd.DataFrame(out, columns=["Hough-Radius", "angle", "radius", "intensity", "radius error"])

    # calculate angles in radians for further analyses.
    df["angle / rad"] = df["angle"].apply(lambda x: increment2radians(x, polar.shape[1]))
    # calculate carthesian coordinates:
    df["x"], df["y"] = polar2carthesian(df["radius"] * dr, df["angle / rad"])

    if show:

        fig, ax = plt.subplots()
        save_str = f"{filename} selected pixels"
        ax.imshow(polar ** (1 / 3), cmap="binary")
        ax.set(ylabel=f"Radius / {dr} px", xlabel="Angular increment / a. u.", title=save_str)
        axright = ax.twinx()
        axright.set(ylim=np.asarray(ax.get_ylim()) / max(polar.shape), ylabel="Relative radius")

        for bound in [minr, maxr]:
            ax.axhline(bound, alpha=0.5, color="C0")

        for i in set(df["Hough-Radius"]):

            vls = df[df["Hough-Radius"] == i]

            ax.plot(vls["angle"], vls["radius"], "x", alpha=0.33, color="C1")

        fig.savefig(f"{save_str}.png", bbox_inches="tight", dpi=300)
        plt.show()
        plt.close("all")

    # rescale radii to fit to carthesian values:
    df["radius"] *= dr

    return df


def voigt(x, xc, A, sigma, gamma):
    """
    Fit a voigt function.

    Params:
    x (np.array): Data
    xc: peak shift
    A: area under curve
    sigma: Gaussian standard deviation
    gamma: Lorentzian HWHM
    """
    denom = sigma * np.sqrt(2)
    w = wofz((x - xc + 1.0j * gamma) / denom)

    return A * np.real(w) / denom


def multi_voigt(x_data, *params):

    y = np.zeros(x_data.shape)

    for xc0, A0, sigma0, gamma0 in grouper(params, 4, 0):
        y += voigt(x_data, xc0, A0, sigma0, gamma0)

    return y


def _gaussian_background(
    xdat,
    a,
    b,
    x1,
    A1,
    w1,
    y0,
):

    yc = poly_func(xdat, a, b, y0)
    yc += gaussian_FWHM_Func(xdat, 0, x1, A1, w1)

    return yc


def _double_gaussian_background(
    xdat,
    a,
    b,
    x1,
    A1,
    w1,
    x2,
    A2,
    w2,
    y0,
):

    yc = _gaussian_background(
        xdat,
        a,
        b,
        x1,
        A1,
        w1,
        y0,
    )
    yc += gaussian_FWHM_Func(xdat, 0, x2, A2, w2)

    return yc


def _triple_gaussian_zero(
    xdat,
    x1,
    A1,
    w1,
    x2,
    A2,
    w2,
    x3,
    A3,
    w3,
):

    yc = gaussian_FWHM_Func(
        xdat,
        0,
        x1,
        A1,
        w1,
    )
    yc += gaussian_FWHM_Func(xdat, 0, x2, A2, w2)
    yc += gaussian_FWHM_Func(xdat, 0, x3, A3, w3)
    return yc


def _lorentzian_background(
    xdat,
    a,
    b,
    x1,
    A1,
    w1,
    y0,
):

    yc = poly_func(xdat, a, b, y0)
    yc += lorentzian_func(xdat, x1, A1, w1, 0)

    return yc


def _double_lorentzian_background(
    xdat,
    a,
    b,
    x1,
    A1,
    w1,
    x2,
    A2,
    w2,
    y0,
):

    yc = _lorentzian_background(
        xdat,
        a,
        b,
        x1,
        A1,
        w1,
        y0,
    )
    yc += lorentzian_func(xdat, x2, A2, w2, 0)

    return yc


def _lorentzian_jac(xdat, x1, A1, w1, y0, single=True):

    binom = 4 * np.pi * (xdat - x1) ** 2
    w1w1 = w1**2
    A1_4 = 4 * A1

    binom_w1w1 = binom + w1w1

    dy0 = xdat * 0
    dA1 = 2 * w1 / binom_w1w1
    dw1 = 2 * A1 / binom - A1_4 * w1w1 / binom_w1w1**2
    dx1 = A1_4 * 2 * np.pi * (xdat - x1) * dA1 / binom_w1w1

    out = np.array((dx1, dA1, dw1, dy0))

    if single:
        out = out.T

    return out


def _double_lorentzian_jac(
    xdat,
    x1,
    A1,
    w1,
    x2,
    A2,
    w2,
    y0,
):

    dx1, dA1, dw1, dy0 = _lorentzian_jac(xdat, x1, A1, w1, y0, single=False)
    dx2, dA2, dw2, __ = _lorentzian_jac(xdat, x2, A2, w2, 0, single=False)

    return np.array((dx1, dA1, dw1, dx2, dA2, dw2, dy0)).T


def _double_lorentzian(
    xdat,
    x1,
    A1,
    w1,
    x2,
    A2,
    w2,
    y0,
):

    yc = lorentzian_func(
        xdat,
        x1,
        A1,
        w1,
        y0 / 2,
    )
    yc += lorentzian_func(xdat, x2, A2, w2, y0 / 2)

    return yc


def _pseudo_voigt(
    xdat,
    x1,
    A1,
    w1,
    eta1,
    y0,
):

    l = lorentzian_func(
        xdat,
        x1,
        A1,
        w1,
        y0 / 2,
    )
    g = gaussian_FWHM_Func(xdat, y0 / 2, x1, A1, w1)

    return eta1 * l + (1 - eta1) * g


def _double_pseudo_voigt(
    xdat,
    x1,
    A1,
    w1,
    eta1,
    x2,
    A2,
    w2,
    eta2,
    y0,
):

    yc = _pseudo_voigt(
        xdat,
        x1,
        A1,
        w1,
        eta1,
        y0 / 2,
    )
    yc += _pseudo_voigt(xdat, x2, A2, w2, eta2, y0 / 2)

    return yc


def _allowed_peak_deviation(p, x, scale_factor=5) -> tuple:

    interval_bound = len(x) / scale_factor
    max_peak_deviation = p + interval_bound
    min_peak_deviation = p - interval_bound
    max_x = x.max()

    if max_peak_deviation > max_x:
        p_upper = max_x
    else:
        p_upper = max_peak_deviation

    if min_peak_deviation < 0:
        p_lower = 0
    else:
        p_lower = min_peak_deviation

    return p_lower, p_upper


def _highest(lst1, lst2):

    dct = {y: x for x, y in zip(lst1, lst2)}

    highest = max(dct)

    return dct[highest]


def _select_values_by_fit(
    rrange, r, rlist, minr, intensity_threshold="std", filename="test", local_intensity_scope=False, dr=1, dt=None, filo_num=4, show=True, double_peak=True, fit=False
):
    """ """

    # assert len(rlist) != 0, 'rlist is empty'

    # define a list to store results
    out = []

    threshold = _get_intensity_threshold(rrange, intensity_threshold)

    if double_peak:
        # integrate sliced region
        double_peak_profile = np.trapz(rrange.T)
        # find peaks
        averaged_peaks = np.asarray(find_peaks_cwt(double_peak_profile, [10], noise_perc=10, min_snr=1))
        # clip to two highest peaks
        p_averaged_1 = _highest(averaged_peaks, double_peak_profile[averaged_peaks])
        averaged_peaks_2 = averaged_peaks[averaged_peaks != p_averaged_1]
        p_averaged_2 = _highest(averaged_peaks_2, double_peak_profile[averaged_peaks_2])
        averaged_peaks = np.array([p_averaged_1, p_averaged_2])

        if show:
            fig, ax = plt.subplots()
            ax.plot(double_peak_profile, label="Integrated rrange")
            ax.plot(averaged_peaks, double_peak_profile[averaged_peaks], "x", label="Peaks")
            ax.legend(loc=0)
            ax.set(title="integrated trial")
            plt.show()
            fig.clf()
            plt.close()

    # iterate over every column in the selected image part
    for i, sliced in enumerate(filo(rrange, filo_num)):

        two_peaks = double_peak

        sliced = mean_arr(sliced)

        # check for threshold value
        if (sliced < threshold).all():
            continue

        # exlude regions where sliced is 0:
        sliced = sliced[sliced > 0][:-1]

        # peak detection
        peaks = find_peaks_cwt(sliced, [10], noise_perc=10, min_snr=1)

        #        reference_radius = r-minr
        x, y = np.arange(len(sliced)), sliced
        #        p1 = closest(peaks, reference_radius)
        p1 = _highest(peaks, sliced[peaks])
        #        y0 = y.mean()

        # skip if no peak was detected
        if len(peaks) == 0:
            continue
        # skip if two peaks are assumed but not detected
        elif double_peak:

            if len(peaks) < 2:
                two_peaks = False

            else:
                # p2 is assumed to be smaller than p1
                p2_peaks = peaks[peaks != p1]
                p2 = _highest(p2_peaks, sliced[p2_peaks])

                if sliced[p2] <= 1.25 * sliced.min():
                    two_peaks = False

                else:

                    both_peaks = np.array((p1, p2))
                    major_peak = closest(both_peaks, p_averaged_1)
                    minor_peak = both_peaks[both_peaks != major_peak][0]

        ##try to fit:
        if fit:

            logging.WARNING("_select_values_by_fit: Fit option is disabled")
 
        # if no fit was used
        else:

            p1_err = np.nan
            try:
                maxval = sliced[major_peak]
            except UnboundLocalError:
                maxval = sliced[p1]

            if two_peaks:

                p2_err = p1_err
                maxval2 = sliced[minor_peak]
                # create negative Hough radius for second peak
                out.append([r, i, major_peak + minr, maxval, p1_err])
                out.append([-r, i, minor_peak + minr, maxval2, p2_err])

            else:
                # sort peaks by radius:
                out.append([r, i, p1 + minr, maxval, p1_err])

        if show:

            show_plot_estimator = rrange.shape[0] // 5
            ndigits = len(str(show_plot_estimator)) - 1

            if not i % round(show_plot_estimator, -ndigits):

                # y_quadrat = poly_func(x_fit, popt[0], popt[1], popt[-1])

                fig, ax = plt.subplots()
                # plot data
                x_plot = x + minr
                ax.plot(x_plot, y, ".", label=f"Ang. Incr. {i}")

                # plot cwt-peak guess result
                if two_peaks:
                    ax.plot([p1 + minr, p2 + minr], sliced[[p1, p2]], "x", label="Peak guess")
                else:
                    ax.plot([p1 + minr], sliced[p1], "x", label="Peak guess")

                ax.set(xlabel="Radius / px", ylabel="Intensity / a. u.")
                ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))

                savename = filename.split(".png")[0] + f"_angular_fit_{i}.png"
                fig.savefig(savename, bbox_inches="tight", dpi=300)
                plt.show()

                fig.clf()
                plt.close("all")

    # convert to
    out = np.array(out)

    # separate radii:
    arrays = []
    for rad in set(out[..., 0]):

        # apply intensity threshold:
        filtered_array = _apply_intensity_threshold_to_list(out[out[..., 0] == rad], intensity_threshold)

       
        arrays.append(filtered_array)

    # merge subarrays again
    out_arr = np.concatenate(arrays)
    return out_arr


def guess_ellipse(df, show=False, filename="test.png", skip_ring=None, img=None, real_center=(0, 0)):
    """
    Creates ellipses for every ring analyzed by select_ring_values.

    Return: Dicitionary containing Ellipse objects. Key is the respective Hough-radius.
    To access the data, call the parameters attribute on the Ellipse.
    It returns (center, width, height, phi).

    kwargs:

    show (bool): If truthy, interim results are plotted. Slows algorithm down.

    skip_ring (list of integers or None (default)): List of ring numbers (starting from 0, referring to the rings in df) to skip.

    RETURN:

    Due to lazyness, the code returns retundant information:
    First: np.array of centers, then the averaged radii, then all parameters of ellipse
    """
    # Create output containers

    logging.debug(
        f"""Enterred guess_ellipse. args are:
                  show = {show},
                  filename = {filename},
                  skip_ring = {skip_ring},
                  img was given ({df is None}),
                  real_center = {real_center}""",
    )
    radii = []
    centers = []
    parameters = []
    radius_deviation = []

    if show:

        filename_split = filename.split(".")
        filename_split.insert(-1, "_Data_Separation")
        data_lst = []

    rings = list(set(df["Hough-Radius"]))
    rings.sort()

    logging.debug("Enter loop over ring values. rings is {}".format(rings))

    for j, circle in enumerate(rings):

        logging.debug("Iterate over Pos. {} in {}".format(j, rings))

        if skip_ring is not None:

            if j in skip_ring:

                logging.debug("skip ring nr. {}".format(j))
                continue

        rad = df[df["Hough-Radius"] == circle]
        #        #define intensity threshold:
        #
        #        thres = rad['intensity'].mean()-rad['intensity'].std()
        #        rad = rad[rad['intensity'] >= thres]

        # create Ellipse Object
        ellipse = LSqEllipse()
        # perform fitting
        try:
            ellipse.fit((rad["x"], rad["y"]))
        except Exception as e:
            logging.error(f"Fit ellipse to ring {circle} in sample {filename} failed due to {type(e).__name__}. Continuing.")

            continue

        logging.debug("Fitted ellipse to ring {}. Parameters are {}".format(circle, ellipse.parameters()))

        # store values. Important!
        radii.append(np.mean([ellipse.width, ellipse.height]))
        centers.append(ellipse.center)

        logging.debug(f"Ellipse center {ellipse.center} was added to centers-Container")

        parameters.append(list(ellipse.parameters()))

        # calculate and store relative radius deviation of raw data:
        rel_rad_std = rad["radius"].std() / np.asarray(radii[-1])
        radius_deviation.append(1 / rel_rad_std)

        # create ellipse data based on fit
        center = ellipse.center
        width = ellipse.width
        height = ellipse.height
        phi = ellipse.phi

        if show:
            # data for plotting
            t = np.linspace(0, 2 * np.pi, 2000)  # len(rad['x']))
            logging.debug(f"Ellipse x and y data are calculated by using the angle {phi} and the center {center}.")
            ellipse_x = center[0] + width * np.cos(t) * np.cos(phi) - height * np.sin(t) * np.sin(phi)
            ellipse_y = center[1] + width * np.cos(t) * np.sin(phi) + height * np.sin(t) * np.cos(phi)

            fig, ax = plt.subplots(3, 1, figsize=(8, 6))

            ax[0].plot(rad["angle"], rad["radius"], "x")
            ax[0].set(xlabel="Angle / Increment", ylabel="radius / px")

            ax[1].plot(rad["angle"], rad["intensity"], "x")
            ax[1].set(xlabel="Angle / Increment", ylabel="Intensity / a. u.")

            title = r"{}: r $\approx$ {} px".format(filename, circle)
            ax[0].set_title(title)

            ax[0].tick_params(direction="in", which="both")
            ax[1].tick_params(direction="in", which="both")

            filename_split = filename.split(".")
            filename_split.insert(-1, f"_Data_Separation_{j}")
            filename_fig = reduce(_kit_str, filename_split)

            # plot ellipse in polar coordinates
            ellipse_polar = np.array([cart2polar(x, y) for x, y in zip(ellipse_x, ellipse_y)])
            ellipse_r = np.flipud(ellipse_polar[..., 0])
            ellipse_phi = np.flipud(radians2increment(ellipse_polar[..., 1], rad["angle"].max()))

            # recalculate data in polar coord
            data = np.array([cart2polar(x, y) for __, x, y in rad[["x", "y"]].itertuples()])
            data_r = np.flipud(data[..., 0])
            data_phi = np.flipud(radians2increment(data[..., 1], rad["angle"].max()))

            #            ellipse_r = sorted(ellipse_r, key = lambda x: sorted(ellipse_phi))
            #            ellipse_phi = sorted(ellipse_phi)
            #            data_r = sorted(data_r,key = lambda x: sorted(data_phi))
            #            data_phi = sorted(data_phi)

            ax[2].plot(data_phi, data_r, "x")
            ax[2].plot(ellipse_phi, ellipse_r, ".", label="Ellipse fit", markersize=0.5)  #
            ax[2].set(ylabel="Radius / px", xlabel="Angle / Transformed Increment")
            ax[2].tick_params(direction="in", which="both")
            ax[2].legend(loc=0)
            plt.tight_layout()

            # save raw data for plotting
            x, y = rad["x"] + real_center[0], rad["y"] + real_center[1]
            center = center[0] + real_center[0], center[1] + real_center[1]
            ell_x, ell_y = ellipse_x + real_center[0], ellipse_y + real_center[1]
            data_lst.append((x, y, rel_rad_std, center, ell_x, ell_y, circle))  # ((rad['x'], rad['y'], rel_rad_std, center, ellipse_x, ellipse_y, circle))

            plt.show()
            fig.savefig(filename_fig, bbox_inches="tight", dpi=300)
            plt.clf()
            plt.close()

    logging.debug("Exit loop over ring values.")

    if show:

        logging.debug("Plot results over img. --> Create figure")

        # create second figure
        fig2, ax2 = plt.subplots(figsize=(7, 7))

        if img is not None:
            ax2.imshow(img ** (1 / 3), cmap="binary")

        # get raw data

        logging.debug("Plot results over img. --> iterate over data_lst. len(data_lst) == {}".format(len(data_lst)))
        for x, y, std, center, ell_x, ell_y, circle in data_lst:

            # plotting
            color = next(plt.gca()._get_lines.prop_cycler)["color"]

            ax2.plot(x, y, "o", label="Rel. Std: {}%".format(round(std * 100, 2)), markersize=2, color=color)
            # plot center
            ax2.plot(center[0], center[1], "x", label="Center {}".format((round(center[0], 2), round(center[1], 2))), color=color)
            # plot ellipse
            ax2.plot(ell_x, ell_y, label=r"Ellipse r $\approx$ {} px".format(circle))

        # layouting
        ax2.set(xlabel="x / px", ylabel="y / px")
        # ax2.axis('equal')
        # ax2.tick_params(direction='out', which = 'both')
        ax2.legend(loc="center left", bbox_to_anchor=(1, 0.5))
        # ax2.grid()

        filename_split.pop(-2)
        filename_split.insert(-1, "_Ellipse_Fit")
        filename_fig2 = reduce(_kit_str, filename_split)

        fig2.savefig(filename_fig2, bbox_inches="tight", dpi=500)
        # plt.tight_layout()
        plt.show()
        fig2.clf()
        plt.close()

    # to format the relative center compatible to plt.imshow, it has to originate from the top left corner:
    logging.debug(f"Invert y coordinates of centers {centers} to originate from top left")
    centers = np.array(centers)
    centers[..., 1] *= -1
    logging.debug(f"Finished inverting the y coordinate of centers {centers} to originate from top left")

    return centers, radii, np.array(parameters, dtype="object"), radius_deviation


def fit_ellipse(
    x,
    y,
    ring="220",
    filename="test.png",
):
    """
    Creates ellipses for every ring analyzed by select_ring_values.

    Return: Dicitionary containing Ellipse objects. Key is the respective Hough-radius.
    To access the data, call the parameters attribute on the Ellipse.
    It returns (center, width, height, phi).

    kwargs:

    show (bool): If truthy, interim results are plotted. Slows algorithm down.

    skip_ring (list of integers or None (default)): List of ring numbers (starting from 0, referring to the rings in df) to skip.

    RETURN:

    p_opt, p_err of the ellipse. Each is a numpy array containing values for width, height, center, phi
    """

    # create Ellipse Object
    ellipse = LSqEllipse()
    # perform fitting
    try:
        ellipse.fit((x, y))
    except IndexError:
        logging.error("Fit ellipse to ring {} in sample {} failed. Continuing.".format(ring, filename))

    logging.debug("Fit ellipse to ring {} using non-iterative approach. Parameters are {}".format(ring, ellipse.parameters()))
    # create ellipse data based on fit
    center = np.asarray(ellipse.center)
    width = ellipse.width
    height = ellipse.height
    phi = ellipse.phi


    # to format the angle, it has to be adjusted:
    phi += np.pi / 2
    logging.debug(f"Add pi/2 to angle, which is now {phi}")

    p_opt = width, height, center, phi

    return p_opt


def _kit_str(a, b):

    return a + "." + b


def residuals(
    center_guess,
    x,
    y,
):

    xguess, yguess = center_guess
    # np.array is used so that pd.Series can be passed as x,y
    rs = np.array(np.hypot(x - xguess, y - yguess))

    return rs - np.mean(rs)


def _fit_center(x, y, center_guess):
    """
    INPUT:

    x, y = Coordinates in carthesian form

    center_guess = initially guessed center (e.g. via Hough-Transform)
    """

    # search for optimized center by minimizing the distance in x and y using a
    # least-square-based algorithm, as suggested by Florian Niekiel

    return leastsq(residuals, center_guess, args=(x, y), ftol=1e-12, xtol=1e-12, maxfev=10000)


def ellipse_polar(a, b, center, phi):
    """
    calculates the ellipse function based on its polar coordinates. The coordinate system
    originates from the center of the ellipse

    INPUT:

    data: r, phi

    a, b = axes
    alpha = angle between major axis and x axis

    """

    t = np.linspace(0, 2 * np.pi, 2000)

    x = center[0] + a * np.cos(t) * np.cos(phi) - b * np.sin(t) * np.sin(phi)
    y = center[1] + a * np.cos(t) * np.sin(phi) + b * np.sin(t) * np.cos(phi)

    return x, y


def hkl2distance_cubic(hkl, a=0.4078):
    """
    Calculates lattice plane distance for cubic crystals (including fcc).


    Input: h,k,l as miller indice tuple (h,k,l); a=0.4078 nm as lattice constant at 25 °C (Dutta1963)

    Returns lattice plane distance (unit depends on a)
    """

    h, k, l = hkl

    return a / np.sqrt(h**2 + k**2 + l**2)


def thermal_expansion_au(l2, t1=25, hkl=(2, 2, 0)):
    """
    Estimation of thermal expansion. The formula of Dutta1963 is used.

    Input: t1,t2 as temperatures in °C, l1,2 corresponds to the lattice distance in nm
    """

    # calculate a thermal expansion coefficient
    alpha = 13.99e-6 + 0.491e-8 * t1

    a1 = hkl2distance_cubic(hkl)

    (
        h,
        k,
        l,
    ) = hkl
    # convert from lattice plane to crystal distance using a = d *sqrt(h²+k²+l²)
    a2 = l2 * np.sqrt(h**2 + k**2 + l**2)

    dT = (a2 - a1) / (alpha * a1)

    return dT


def read_temperature_file(
    filename,
    path=[
        "K:\\",
        "sp_oe520",
        "DFG-Uni",
        "04_GRK_InsituMesstechnik",
        "Dokumentationen",
        "Messtechnik",
        "TEM",
        "181115_Au_thin-film_temperature_measurement",
        "2018-11-14 LC Thermal",
    ],
):
    """
    filename : 'Ceta 1327 25-100C 2CpS.csv'

    path as list. Put an empty list if not necessary.
    """
    internal_path = [] + path
    if path:

        internal_path.append(filename)
        filename = os.path.join(*internal_path)

    df = pd.read_csv(filename)
    df["Time / s"] = df["Time"] / 1000
    df["T / °C"] = df[" Channel A Temperature"]

    return df


def file2df(filename, path, output_folder=True):
    """
    INPUT:
    filename (str): Name of the excel or csv file to import

    path (str): directory of the file of interest

    kwargs:
    output_folder (bool): If truthy (default), a folder named "output" is appended to "path"


    RETURN:

    pandas.DataFrame object
    """
    filepath = [] + path

    if output_folder:
        filepath.append("output")

    filepath.append(filename)

    directory = os.path.join(*filepath)

    if filename.endswith(".csv"):
        df = pd.read_csv(directory)

    elif filename.endswith(".xlsx"):
        df = pd.read_excel(directory)

    else:
        logging.warning(
            "WARNING: file2df cannot create a DataFrame, because {} is not \
              recognized. Only csv and xlsx files are supported.".format(
                filename
            )
        )

        df = None

    return df


def calculate_beta(slope, a_s=3.3 / 2):
    """
    calculate beta in mrad
    """
    return slope * a_s * 1000


def make_label(val, i=2):
    """
    Creates a string out of val with the desired amount i of decimal places, even if they end with 0.
    """
    val = str(round(val, i))

    if len(val) == i + 1:

        val += "0"

    return val


def estimate_incident_angle(df, show=True, peak_criterion="all", image_size=(np.nan, np.nan), tolerance=20, radius_range=(0, np.inf), number_of_peaks=100):
    """
    Estimates the incident angle beta for every ring identified in all images.

    Input: pd.DataFrame

    kwargs: show (bool): If truthy (default), the result is displayed as figure.
            peak_criterion (list of strings str): List of strings to select columns of interest or 'all'. It takes
                'Integration' --> Uses fitted peak of azimuthal integration
                'Ellipse' --> Uses value of non-iterative Ellipse fitting in carthesian coordinates (does not provide an error value)
                'Distortions' --> Uses value of iterative Distortion fitting in polar coordinates
                'all' --> all three methods are used
            tolerance (scalar): Value to which the rounded values have to match to


    Return: betalist, beta_errs as nested lists and the conditions as list
    """

    crit_dct = {"Integration": "^xc \d+$", "Ellipse": "Ellipse \d+ radius / px", "Distortions": "distortion correction \d+ radius / px", "Full fit": "^xc_full_fit \d+$"}

    error_dct = {"Integration": "^xc_error \d+", "Ellipse": None, "Distortions": "distortion correction \d+ radius error / px", "Full fit": "^xc_full_fit_error \d+$"}

    if type(peak_criterion) == str:
        if peak_criterion.lower() == "all":
            peak_criterion = crit_dct

        elif peak_criterion.lower().capitalize() in crit_dct:

            peak_criterion = [peak_criterion]

        else:

            raise KeyError(
                '{} not in crit_dct. Only "Integration", "Ellipse", "Distortions", or a list containing those is recognized. "all" is a shortcut for ["Integration", "Ellipse", "Distortions"]'.format(
                    peak_criterion
                )
            )

    betalist = []
    beta_errs = []
    labels = []
    for crit in peak_criterion:
        # extract peak numbers out of df and iterate over those. Works only for up to 10 peaks:
        crit = crit.lower().capitalize()
        betalist.append([])
        beta_errs.append([])
        labels.append(crit)

        # for peak_num in [int(re.search('\d+', i).group()) for i in df.columns if re.search(peak_criterion, i)]:
        for xcol in [i for i in df.columns if re.search(crit_dct[crit], i)][:number_of_peaks]:

            if "background" in xcol:
                continue

            # xcol = 'xc {}'.format(peak_num)
            peak_num = int(re.search("\d+", xcol).group())

            # exclude NaNs:
            xraw = df[df[xcol].apply(np.isfinite)]["z height"] - df["z height"].median()
            yraw = df[df[xcol].apply(np.isfinite)][xcol]

            # exclude errorously sorted values
            minthres_lst = [*yraw.apply(lambda x: round(x, -1))]
            minthres = max(minthres_lst, key=minthres_lst.count)

            y = yraw[(yraw >= minthres - tolerance) & (yraw <= minthres + tolerance)]
            x = xraw[(yraw >= minthres - tolerance) & (yraw <= minthres + tolerance)]

            if not radius_range[0] <= y.mean() <= radius_range[1]:
                continue

            # normalize y values to median radius
            y2 = y / df[xcol].median()

            # deal with errors:
            if error_dct[crit]:

                errcols = [i for i in df.columns if re.search(error_dct[crit], i)]
                errcol = [i for i in errcols if str(peak_num) in i][0]

                y_err = df[df[xcol].apply(np.isfinite)][(yraw >= minthres - tolerance) & (yraw <= minthres + tolerance)][errcol]

                # add systematic error of center finding
                if crit == "Distortions":
                    y_err = y_err + df[df[xcol].apply(np.isfinite)][(yraw >= minthres - tolerance) & (yraw <= minthres + tolerance)]["Center Displacement / px"]

                y2_err = y_err / y * y2

            else:
                y2_err = None

            # skip, if circle is larger than the image
            if y.mean() > min(image_size) / 2:

                continue

            if show:
                # create figure
                fig, ax = plt.subplots(figsize=(4.5, 4.5), dpi=100)
                try:
                    ax.errorbar(
                        x,
                        y2,
                        yerr=y2_err,
                        label="Ring radius {} px".format(int(round(y.mean()))),
                        marker=".",
                        capsize=3,
                        elinewidth=1,
                        markeredgewidth=1,
                        ls="",
                    )
                    # ax.plot(x, y2,'o', label = 'Ring radius {} px'.format(int(round(y.mean()))))
                except ValueError:
                    pass
                # set axis label
                ax.set(xlabel="Z / µm", ylabel="Relative radius", title=crit)

            # use a least-square algorithm to fit a linear function to the data
            try:

                popt, pcov = curve_fit(linear_func, x, y2, sigma=y2_err, maxfev=10000)

            except (RuntimeError, TypeError):

                logging.error("Error for Peak {}".format(peak_num))

                if show:
                    ax.tick_params(direction="in", which="both")
                    ax.legend(loc=0)
                    ax.grid()
                    plt.show()
                    fig.savefig("Alignment_fit_RingNr_{}_unsuccesful.png".format(peak_num), bbox_inches="tight")
                    fig.clf()
                continue

            # calculate incident angle of electron beam
            yfit = linear_func(x, *popt)
            popt_err, rsquare = get_errors(popt, pcov, x, y2, yfit)

            beta = calculate_beta(popt[0])
            beta_err = abs(calculate_beta(popt_err[0]))

            # store results in container
            betalist[-1].append(beta)
            beta_errs[-1].append(beta_err)

            if show:

                # plot data and errorbars

                # plot fitresult
                digits = 6
                beta_digits = digits - 3
                labeltxt = "Slope: {}$\pm${} \n$\\beta$: ({}$\pm${}) mrad".format(
                    round(popt[0], digits), make_label(popt_err[0], digits), round(beta, beta_digits), make_label(beta_err, beta_digits)
                )
                ax.plot(x, yfit, "-", label=labeltxt)

                # design stuff
                ax.tick_params(direction="in", which="both")
                ax.legend(loc=0)
                ax.grid()

                # output figure
                plt.show()
                fig.savefig("Alignment_fit_Method_{}_RingNr_{}_rel_ex_{}.png".format(crit, peak_num, list(set(df["C2/C3"]))[0]), bbox_inches="tight")
                fig.clf()
                plt.close("all")

    return betalist, beta_errs, labels


def poly_func(xdat, *args):

    y = 0

    for n, i in enumerate(args[::-1]):

        y += float(i) * xdat**n

    return y


def calculate_distortion(arr):
    """
    Input (np.array): 2D array that stores width and height of the ellipse in the 1,2 position
    (as does the parameters output value of guess_ellipse)

    return: Distortion (float)
    """
    eta_lst = []

    try:

        for i in arr[..., 1:3:]:
            rad_ratio = min(i) / max(i)

            eta_lst.append((1 - rad_ratio) / (1 + rad_ratio))

    except TypeError:

        rad_ratio = min(arr[1:3]) / max(arr[1:3])
        eta_lst.append((1 - rad_ratio) / (1 + rad_ratio))

    return np.mean(eta_lst)


def sort_rings(i_vals, rlist, tolerance=0.1, show=True):
    """
    to be filled
    """


    cache = [[], [], [], []]

    for i in i_vals:


        dct = []
        for r in set(rlist):

            # define data corridor
            dr = r * tolerance

            minr, maxr = r - dr, r + dr

            if minr <= i[0] <= maxr:

                dct.append(r)

        # skip if edct is empty
        if not len(dct):

            continue

        elif len(dct) == 1:

            r = dct[0]

        else:

            dct2 = {abs(i[0] - pr): pr for pr in dct}

            r = dct2[min(dct2)]

        # append angles
        cache[0].append(i[2])
        # append radii
        cache[1].append(i[0])
        # append intensity
        cache[2].append(i[1])
        # append r
        cache[3].append(r)

    df = pd.DataFrame(np.array(cache).transpose(), columns="angle radius intensity Hough-Radius".split())

    return df


def check_ring_values(df, max_phi=2048, vals=5):

    # get quadrant
    q1 = (0, max_phi // 4)
    q2 = (q1[1], max_phi // 2)
    q3 = (q2[1], (3 * max_phi) // 4)
    q4 = (q3[1], max_phi)

    # list of data_Frames
    q_dfs = [df[(df["angle"] >= l) & (df["angle"] < m)] for l, m in (q1, q2, q3, q4)]
    # list with lengths of data Frames
    q = [len(i) for i in q_dfs]

    # array with True and False values of lengths
    len_arr = np.array(q) >= vals

    # count amount of quadrants with sufficient values
    positive_qs = [*len_arr].count(True)

    # three or four quadrants are fine
    if positive_qs >= 3:

        out = df

    # one or no quadrant is fine
    elif positive_qs <= 1:

        out = df[[False] * len(df)]

    # two quadrants are fine
    else:

        # now, these quadrants can either be neiqhbouring each other (bad), or sit on opposit sites (good):
        # check if they sit opposite to each other. Then, the first and third argument must be similar
        if len_arr[0] == len_arr[2]:

            out = df

        else:

            out = df[[False] * len(df)]

    return out


def _retreive_center(filename, df) -> tuple:
    """
    reads out a center guess out of a DataFrame having the columns "File" and columns with center information
    """

    # select file data
    data = df[df["File"] == filename]

    # get available center information
    x = data["Center x"].iloc[0]
    y = data["Center y"].iloc[0]

    return x, y


def optimize_center(
    img,
    center_guess,
    file="test.png",
    show_ellipse=True,
    show_all=False,
    max_iter=100,
    mask=None,
    tolerance=0.05,
    median_for_polar=True,
    clahe_for_polar=True,
    int_thres="std",
    max_tries=6,
    value_precision=0.1,
    sigma=2,
    radius_boundaries=(400, 1024),
    dr_polar=1,
    dt_polar=None,
    local_intensity_scope=False,
    vals_per_ring=20,
    jacobian_for_polar=False,
    skip_ring=None,
    fit_gauss=True,
    double_peak=False,
    only_one_polar_transform=False,
    show_select_values=False,
):
    """
    Routine to iteraviely find the true origin of the diffraction pattern.

    Parameters
    ----------
    img : array
        Diffraction pattern of interest.
    center_guess : Tuple
        Coordinates of the initial center guess in px.
    file : string, optional
        Desired name for saving intermediate results as plot. The default is
        'test.png'.
    show_ellipse : bool, optional
        If truthy, the ellipse fits are shown. Overwritten if show_all is True.
        The default is True.
    show_all : bool, optional
        If truthy, all intemediate results are plotted. The default is False.
    max_iter : int, optional
        Maximum number of iterations. The default is 100.
    mask : array or None, optional
        Array to be multiplied with img if it is not None. The default is None.
    tolerance : float, optional
        Convergence criterion for the iteration. It describes the deviation as
        euclidian distance between the center positions in px.
        The default is 0.05.
    median_for_polar : bool, optional
        Applies median filtering prior to polar transform. The default is True.
    clahe_for_polar : TYPE, optional
        Applies CLAHE (https://en.wikipedia.org/wiki/Adaptive_histogram_equalization#Contrast_Limited_AHE)
        prior to polar transform.  The default is True.
    int_thres : str, optional
        String to select a statistical measure as described by . The default is 'std'.
    max_tries : int, optional
        Maximum attempts to shift the center per iteration. The default is 6.
    value_precision : int, optional
        Convergence criterion, describing the euclidian distance between the center iterations in px. The default is 0.1.
    sigma : TYPE, optional
        Experimentally. Do not use. The default is 2.
    radius_boundaries : tuple, optional
        Radius boundaries for ring detection in px or relative units. The default is (400, 1024).
    dr_polar : TYPE, optional
        Experimentally. Do not use. The default is 1.
    dt_polar : TYPE, optional
        Experimentally. Do not use. The default is None.
    local_intensity_scope : TYPE, optional
        DESCRIPTION. The default is False.
    vals_per_ring : int, optional
        Minimum values per ring. The default is 20.
    jacobian_for_polar : bool, optional
        Intensity correction for the polar transform. The default is False.
    skip_ring : TYPE, optional
        Experimentally. Do not use. The default is None.
    fit_gauss : TYPE, optional
        Experimentally. Do not use. The default is True.
    double_peak : TYPE, optional
        Experimentally. Do not use. The default is False.
    only_one_polar_transform bool, optional
        Flag that performs only a single polar transform without center optimization. The default is False.

    Returns
    -------
    fit_dict : dict
        contains fit results
    ring_df : DataFrame
        contains the actual pixel information
    polar_transformed_results : dict
        Dictionary containing the results of the final polar transform as returned
        by at.polar.reproject_image_into_polar:
            {'polar':2D-array, 'r_grid':radius grid (1D), 'theta_grid':angle grid (1D)}
    """

    logging.debug("optimize_center --> File {}: start optimize_center".format(file))

    if show_all:

        show_ellipse = show_img2polar = show_peaks = show_select_values = True

    else:

        show_img2polar = show_peaks = False

    # ensure full algorithm processing
    # six different options for algorithm to get a better result. trys starts from 0
    min_trys = 6

    if max_tries < min_trys:

        logging.info("Maximum number of iterations {} is less than the number of different optimization strategies {}.".format(max_tries, min_trys))

    center_lst = []
    displacement_lst = []

    center_old = "not defined --> first run"

    center = center_guess

    # store initial guess for comparison:
    center_lst.append(center)

    # create dummy variables for initial displacement and radius deviation
    displacement_old = mean_rad_dev_old = fit_dict = None

    i = 0
    total_counter = 0
    max_counter = max_iter * 2
    trys = 0
    distance = np.nan

    min_rad, max_rad = radius_boundaries

    # start iteration loop

    logging.debug("optimize_center --> File {}: Enter while loop for iteration".format(file))

    # set increments to default values for center opt.
    dr, dt = 1, None
    # define variable to store that interpolation has been performed
    if dr == dr_polar and dt == dt_polar and dr != 1 and dt != None:
        interpolated = True
    else:
        interpolated = False

    while all([max_iter >= i, trys <= max_tries, np.nan not in list(center), total_counter < max_counter]):  # and :
        ##Transform to obtain polar coordinates
        # transform img to polar coordinates
        logging.debug("optimize_center --> File {}: Tranform img to polar coordinates --> Enter img2polar. Center is {}".format(file, center))
        polar, r_grid, theta_grid = img2polar(
            img,
            center,
            show=show_img2polar,
            clahe=clahe_for_polar,
            median_filter=median_for_polar,
            jacobian=jacobian_for_polar,
            dr=dr,
            dt=dt,
            filename=file + "_Abel_result_{}.png".format(i),
        )
        logging.debug("optimize_center --> File {}: Tranform img to polar coordinates --> Exit img2polar".format(file))
        ###Find average peaks by integrating over the angle
        # get first guess and 1D structure
        logging.debug("optimize_center --> File {}: Find first peak guesses --> Enter find_average_peaks".format(file))
        peaks, r, intensity = find_average_peaks(
            polar,
            r_grid,
            theta_grid,
            peak_widths=[int(round(20 * max(img.shape) / (2048 * dr)))],
            min_radius=min_rad,
            max_radius=max_rad,
            dr=dr,
            show=show_peaks,
            filename=f"{file}_Peak_guessing_{i}.png",
        )
        logging.debug("optimize_center --> File {}: Find first peak guesses --> Exit find_average_peaks. Peaks found at {}".format(file, peaks))

        fit_dict_new = {}

        rlist = peaks

        logging.debug("optimize_center --> File {}: Find precise peak positions --> rlist is {}".format(file, rlist))

        ###center optimization

        ##use found radii to select ring values for center optimization
        logging.debug("optimize_center --> File {}: Extract ring values --> Enter select_ring_values".format(file))

        ring_df = select_ring_values(
            polar,
            rlist,
            tolerance=value_precision,
            show=show_select_values,
            intensity_threshold=int_thres,
            filename=f"{file}",
            local_intensity_scope=local_intensity_scope,
            dr=dr,
            fit_gauss=fit_gauss,
            double_peak=double_peak,
        )

        logging.debug("optimize_center --> File {}: Extract ring values --> Exit select_ring_values".format(file))
        # ensure that significant values for each ring (10 in at least 2 quadrants that do not touch) are found
        ring_df = pd.concat((check_ring_values(ring_df[ring_df["Hough-Radius"] == i], vals=round(vals_per_ring / 4), max_phi=polar.shape[1]) for i in set(ring_df["Hough-Radius"])))

        try:
            logging.debug("optimize_center --> File {}: Fit Ellispe --> Enter guess_ellipse. Center is {}".format(file, center))

            centers, radii, parameters, rdev = guess_ellipse(ring_df, show=show_ellipse, filename=file + f"_Ellipse_{i}.png", skip_ring=skip_ring, img=img, real_center=center)

            logging.debug("optimize_center --> File {}: Fit Ellispe --> Exit guess_ellipse. Center is {}. Parameters are {}".format(file, center, parameters))

        except:
            displacement = np.average(centers[..., 0], weights=rdev), np.average(centers[..., 1], weights=rdev)
            center_old = center
            center = center[0] + displacement[0], center[1] + displacement[1]
            distance = np.hypot(*displacement)

            logging.error(
                "optimize_center --> File {}: Could not guess ellipse . Abort optimization after {} runs containing {} iteration improvements.\tCenter distance is {}.".format(
                    file, total_counter, i, distance
                )
            )

            break

        ###shift center

        logging.debug("optimize_center --> File {}: Evaluate center --> centers is {}. weights is {}".format(file, centers, rdev))
        displacement = np.average(centers[..., 0], weights=rdev), np.average(centers[..., 1], weights=rdev)
        center_old = center
        logging.debug("optimize_center --> File {}: Evaluate center --> Displacement is {}".format(file, displacement))

        # abort if no center was found:
        if any(np.isnan(displacement)):

            logging.warning("optimize_center --> File {}: Evaluate center --> Displacement is {}. Abort while loop.".format(file, displacement))
            break

        # average radius deviation:
        mean_rad_dev = calculate_distortion(parameters)
        logging.debug("optimize_center --> File {}: Evaluate center --> Distiortion is {}".format(file, mean_rad_dev))

        # make center mutable
        center_old = list(center_old)
        center = list(center)

        # optimize coordinates
        center_old = np.asarray(center_old)
        displacement = np.asarray(displacement)

        logging.debug("optimize_center --> File {}: Correct center. trys == {}. Both coordiantes shifted.".format(file, trys))

        if trys == 0:

            center = center_old[0] + displacement[0], center_old[1] - displacement[1]
            comment = "added Pos. 0, substracted Pos. 1 of displacement from center_old"

        distance = np.hypot(*displacement)

        if only_one_polar_transform:
            break

        # save first displacement and distortion for comparison
        if displacement_old is None:

            logging.debug("optimize_center --> File {}: Define displacement_old and mean_rad_dev_old".format(file))

            displacement_old = displacement
            mean_rad_dev_old = mean_rad_dev

            # add initial displacement to list
            displacement_lst.append(distance)

            logging.debug("continue")
            # skip evaluation
            continue

        distance_old = np.hypot(*displacement_old)

        logging.debug("optimize_center --> File {}: iteration round {}, run {}, total run {}".format(file, i, trys, total_counter))
        logging.debug(
            "optimize_center --> File {}: initial center {}, old center {} with distance {}, new center {} with distance {}".format(
                file, center_guess, center_old, distance_old, center, distance
            )
        )

        ##only take new center if old new one shows a better displacement:

        # if new displacement is smaller without inreasing the mean radius deviation, then fine. If not, try oppisite shift signs.
        tolerance_factor = 2
        max_rad_dev = tolerance_factor * mean_rad_dev_old

        if distance_old <= distance or all((mean_rad_dev > max_rad_dev, distance > tolerance)):

            logging.debug(
                "optimize_center --> File {}: old distance {} <= new distance {} or distortion {} > {} * old distortion {}".format(
                    file, distance_old, distance, mean_rad_dev, tolerance_factor, mean_rad_dev_old
                )
            )

            # try changing half of maximum value
            if trys > 0:

                if trys == 1:

                    q = trys

                elif trys % 2:

                    q += 1

                else:

                    q = 1

                # change only center position of larger displacement
                center = center_old[0] + displacement[0] / q, center_old[1] - displacement[1] / q
                comment = f"added Pos. 0 /{q}, substracted Pos.1/{q} of displacement from center_old"

        # only store best displacement and center
        else:

            displacement_lst.append(distance)
            center_lst.append(center)
            # overwrite peak data
            fit_dict = fit_dict_new

            # store ellipse results
            for k, ell in enumerate(parameters):

                # also possible with ellipse_params and later merging
                fit_dict[f"Ellipse {k} center / px"] = ell[0]
                fit_dict[f"Ellipse {k} width / px"] = ell[1]
                fit_dict[f"Ellipse {k} height / px"] = ell[2]
                fit_dict[f"Ellipse {k} radius / px"] = np.mean((ell[1:3]))
                fit_dict[f"Ellipse {k} phi / rad"] = ell[3]

            logging.debug(f"optimize_center --> File {file}: Added Ellipse parameters {parameters} to fit_dict")

            # leap one step forward
            i += 1
            logging.info(
                "optimize_center --> File {}: Iteration step increased. Operation: {}, Displacement {}, Distance {}, New iteration: {}".format(
                    file, comment, displacement, distance, i
                )
            )
            # reset trys
            trys = 1

            logging.debug("optimize_center --> File {}: Start iteration round: {}".format(file, i))
            logging.debug("optimize_center --> File {}: Run {}, initial center {}".format(file, total_counter, center_guess))
            logging.debug("optimize_center --> File {}: old center: {}; old displacement {}; old distortion {}".format(file, center_old, displacement_old, mean_rad_dev_old))
            logging.debug("optimize_center --> File {}: new center: {}; new displacement {}; new distortion {}".format(file, center, displacement, mean_rad_dev))

            displacement_old = displacement
            mean_rad_dev_old = mean_rad_dev

        # keep track with loop number
        total_counter += 1
        trys += 1

        # break if displacement tolerance is reached:
        if distance <= tolerance:

            logging.info(
                "optimize_center --> File {}: Tolerance value {} reached after {} runs containing {} iteration improvements. Center distance is {}.".format(
                    file, tolerance, total_counter, i, distance
                )
            )

            break

        if i == max_iter:

            logging.info(
                "optimize_center --> File {}: Maximum number of iterations ({}) reached after {} runs. Center is {}, Center distance is {}".format(
                    file, i, total_counter, center_old, distance_old
                )
            )

        if trys == max_tries:

            if interpolated:
                break

            # one last try using interpolation
            else:
                interpolated = True
                # reset trys
                trys = 1

            logging.info(
                "optimize_center --> File {}: Maximum number of tries ({}) reached. {} iteration improvements could be performed. Center is {}, Center distance is {}".format(
                    file, trys, i, center_old, distance_old
                )
            )

    logging.debug("optimize_center --> File {}: while loop exit".format(file))
    # check, whether fit_dict is still None (Means that iteration did not improve anything):
    if fit_dict is None:

        logging.info("optimize_center --> File {}: Center improvement failed. Stick with center guess {} and displacement {}".format(file, center_guess, displacement_old))

        fit_dict = fit_dict_new

    # add center with last imrovment to dictionary
    fit_dict["Center x"], fit_dict["Center y"] = center_lst[-1]
    fit_dict["Center Displacement / px"] = distance
    # add filename to dictionary:
    fit_dict["File"] = file

    center_arr = np.array(center_lst)
    logging.debug(f"optimize_center --> File {file}: Convert center_lst {center_lst} to array {center_arr}")

    if show_ellipse and not only_one_polar_transform:

        logging.debug("optimize_center --> File {}: Create Centers -- img overlay image".format(file))

        fig, ax = plt.subplots(figsize=(5, 5))

        name = file + "_center_optimizaion"

        ax.plot(center_arr.T[0], center_arr.T[1], ":", c="black")
        for num, i in enumerate(center_arr):
            ax.plot(
                i[0],
                i[1],
                "x",
                label=num,
            )  # c=c)
        ax.set(xlabel="x / px", ylabel="y / px", title=f"{file} Center Optimization Coordinates")
        ax.axis("equal")
        ax.legend(loc="best")
        ax.tick_params(direction="in", which="both")
        plt.show()
        fig.savefig(name + ".png", bbox_inches="tight")
        fig.clf()
        plt.close("all")

        # plot distance
        f, ax = plt.subplots(figsize=(5, 5))
        ax.plot(range(len(displacement_lst)), displacement_lst, ":", c="black")
        ax.set_yscale("log")
        for num, i in enumerate(zip(range(len(displacement_lst)), displacement_lst)):
            # c=next(color)
            ax.plot(i[0], i[1], "o", label=num)

        ax.hlines(tolerance, 0, len(displacement_lst), label="Threshold")
        ax.legend(loc=0)
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Displacement / px")
        ax.set_title(f"{file} Center Displacement")
        ax.tick_params(direction="in", which="both")
        f.savefig(f"{file}_displ_development_tolerance_{tolerance}.png", bbox_inches="tight")
        plt.show()

        plt.clf()
        plt.close("all")

    # merge polar transfom results for later processing
    polar_transformed_results = {"polar": polar, "r_grid": r_grid, "theta_grid": theta_grid}

    return fit_dict, ring_df, polar_transformed_results


def correct_distortions(fit_dict, ring_df, skip_ring=None, show=True):
    """
    Fits the sum of fourfold distortions to every ring in ring_df and fit_dict

    returns a dictionary containing the fitparameters per ring merged with fit_dict
    """

    # output dictionary:
    out = {}

    if show:
        # container for data storage:
        rdat_container = []
        phidat_container = []
    # extract number of rings
    logging.debug(f'File {fit_dict["File"]}: Correct_distortions --> search ring numbers in {fit_dict.keys()}')
    # rings = [int(re.search('-?\d+', k).group()) for k in fit_dict if re.search('Ellipse \d+ radius / px', k)]
    rings = list(range(len(set(ring_df["Hough-Radius"]))))
    logging.debug(f'File {fit_dict["File"]}: Correct_distortions --> Found rings: {rings}')
    # loop over rings
    hough_radii = sorted(list(set(ring_df["Hough-Radius"])))

    for ring in rings.copy():

        if skip_ring is not None:

            skip_ring = np.asarray(skip_ring)
            relevant_rings = skip_ring[skip_ring <= ring]
            skipped = len(relevant_rings) + ring

        else:

            skipped = ring

        try:

            rad = ring_df[ring_df["Hough-Radius"] == hough_radii[skipped]]

        except IndexError:

            logging.error(
                f'File {fit_dict["File"]}: Correct_distortions --> Loop over rings rings: Ring {ring}. list index out of range. List: {hough_radii}. Index: {skipped}. Continue.'
            )
            rings.remove(ring)

            continue

        logging.debug(f'File {fit_dict["File"]}: Correct_distortions --> Loop over rings rings: Ring {ring}')

        # get data to fit on
        rdat, phidat, rdat_error = rad["radius"], rad["angle / rad"], 1 / rad["intensity"]

        # get starting parameters p0 based on the non-iterative ellipse fitting
        r = rdat.mean()
        try:
            alpha = fit_dict[f"Ellipse {ring} phi / rad"]
        except KeyError:
            alpha = 0
        alpha1 = alpha2 = alpha3 = alpha
        ab = rdat.min(), rdat.max()  # fit_dict[f'Ellipse {ring} height / px'], fit_dict[f'Ellipse {ring} width / px']
        eta1 = (1 - min(ab) / max(ab)) / (1 + min(ab) / max(ab))
        eta2 = eta1
        eta3 = 0.25 * eta2

        p0 = r, alpha1, alpha2, alpha3, eta1, eta2, eta3

        logging.debug(
            f'File {fit_dict["File"]}: Correct_distortions --> \
                              Created fit guesses for {ring}. p0 is {p0}'
        )

        # print(rdat, phidat)
        try:
            # try to fit rings
            popt, pcov = curve_fit(polar_dist4th_order, phidat, rdat, p0=p0, sigma=rdat_error, bounds=([0, -np.inf, -np.inf, -np.inf, 0, 0, 0], np.inf), maxfev=10000)
            perr, rsquare = get_errors(popt, pcov, phidat, rdat, polar_dist4th_order(phidat, *popt))  # np.sqrt(np.diag(pcov))

            logging.info(f'File {fit_dict["File"]}: Correct_distortions --> Fit succesful. Parameters are {popt}, Errors are {perr}.')

        except Exception as e:

            # log
            message = f'{type(e).__name__} in file {fit_dict["File"]}.'
            logging.error(message)
            logging.error(traceback.format_exc())

            popt = p0
            perr = [np.inf] * len(p0)

            logging.info(f'File {fit_dict["File"]}: Correct_distortions --> Fit failed. Use input parameters. Parameters are {popt}, Errors are {perr}.')

        # save values
        out[f"distortion correction {ring} radius / px"] = popt[0]
        out[f"distortion correction {ring} phi1 / rad"] = popt[1]
        out[f"distortion correction {ring} phi2 / rad"] = popt[2]
        out[f"distortion correction {ring} phi3 / rad"] = popt[3]
        out[f"distortion correction {ring} eta1"] = popt[4]
        out[f"distortion correction {ring} eta2"] = popt[5]
        out[f"distortion correction {ring} eta3"] = popt[6]
        out[f"distortion correction {ring} radius error / px"] = perr[0]
        out[f"distortion correction {ring} phi1 error / rad"] = perr[1]
        out[f"distortion correction {ring} phi2 error / rad"] = perr[2]
        out[f"distortion correction {ring} phi3 error / rad"] = perr[3]
        out[f"distortion correction {ring} eta1 error"] = perr[4]
        out[f"distortion correction {ring} eta2 error"] = perr[5]
        out[f"distortion correction {ring} eta3 error"] = perr[6]
        out[f"distortion correction {ring} R2adj"] = rsquare
        out[f"distortion correction {ring} values"] = len(phidat)
        # store data for plotting
        if show:
            # ydata
            rdat_container.append(rdat)
            # xdata
            phidat_container.append(phidat)
    # show results
    if show:

        fig, ax = plt.subplots(len(rings), 1, figsize=(10, 3 * len(rings)), sharex=True)

        if type(ax) != np.ndarray:
            ax = [ax]

        t = np.linspace(0, 2 * np.pi, 10000)
        # plot in terms of pi

        t_plot = t / np.pi
        for i in rings:

            if out[f"distortion correction {i} radius error / px"] == np.inf:

                ax[i].text(1, 0.9, "FIT FAILED", {"color": "m", "fontsize": 18}, va="top", ha="right")

            r = out[f"distortion correction {i} radius / px"]
            alpha1 = out[f"distortion correction {i} phi1 / rad"]
            alpha2 = out[f"distortion correction {i} phi2 / rad"]
            alpha3 = out[f"distortion correction {i} phi3 / rad"]
            eta1 = out[f"distortion correction {i} eta1"]
            eta2 = out[f"distortion correction {i} eta2"]
            eta3 = out[f"distortion correction {i} eta3"]
            # built functions:
            # sum function
            sum_dist = polar_dist4th_order(t + np.pi / 2, r, alpha1, alpha2, alpha3, eta1, eta2, eta3)
            # each distortion contributes to 1/3 to r. Also, shift t by pi/2 to fit plotted results in [0,2pi]
            dist2 = polar_dist2(t + np.pi / 2, r / 3, eta1, alpha1, n=2) + 2 / 3 * r
            dist3 = polar_dist2(t + np.pi / 2, r / 3, eta2, alpha2, n=3) + 2 / 3 * r
            dist4 = polar_dist2(t + np.pi / 2, r / 3, eta3, alpha3, n=4) + 2 / 3 * r

            # shift phi data to fit plotted results in [0,2pi]
            phidat_container[i] = np.asarray(phidat_container[i])
            # phidat_container[i] -= phidat_container[i].min()
            # plot in terms of pi
            phidat_container[i] -= np.pi / 2
            phidat_container[i] /= np.pi

            # plot
            r2 = round(out[f"distortion correction {i} R2adj"], 4)
            amount = out[f"distortion correction {i} values"]
            ax[i].plot(phidat_container[i], rdat_container[i], "x", label=f"{amount} pixels")
            # ax[i].scatter(phidat_container[i], rdat_container[i], c=, cmap='Blues')
            ax[i].plot(t_plot, sum_dist, label=f"sum of distortions\nR$^2$ = {r2}")
            ax[i].plot(t_plot, dist2, dashes=[6, 2], label="2-fold dist. + 2/3r")
            ax[i].plot(t_plot, dist3, "--", label="3-fold dist. + 2/3r")
            ax[i].plot(t_plot, dist4, ":", label="4-fold dist. + 2/3r")
            ax[i].set_ylabel("Radius / px")
            ax[i].tick_params(direction="in", which="both")
            ax[i].hlines(r, min(t_plot), max(t_plot), label=f"Radius {round(r,4)} px")
            ax[i].legend(loc="center left", bbox_to_anchor=(1, 0.5))
            ax[i].grid()

            ax[i].set_xlabel(r"Polar angle / $\pi$")

        plt.tight_layout()
        plt.show()
        fig.savefig(f'{fit_dict["File"]}_Distortion_fit.png', bbox_inches="tight")
        fig.clf()
        plt.close("all")

    # merge dictionaries
    out = {**fit_dict, **out}

    return out


def reshape_image_by_function(polar, func, norm_factor=1, show=True, filename="test", dr=1, *funcparams):
    """
    uses skimage.transform.rescale on every columns in polar. The scaling factor
    is calculated using func and r.

    args:

        polar (array-like): input 2D array, e.g. a polar-transformed image

        func (callable): Function to calculate scaling factors. Must return real numbers


    kwargs:
        norm_factor (int or float): value to normalize the calculated scaling factors

        *funcparams: Arguments passed to func

    """
    # calculate scaling factors based on func
    phi_arr = increment2radians(np.arange(polar.shape[1]), polar.shape[1])
    #    if abel:
    phi_arr += np.pi / 2
    #
    scale_vector = func(phi_arr, *funcparams) / dr

    if show:
        fig, ax = plt.subplots()
        ax.imshow(polar ** (1 / 3), cmap="gray")
        ax.plot(np.arange(polar.shape[1]), scale_vector)
        ax.set(xlabel="Angular increment / a. u.", ylabel="Radius / px", title="uncorrected")
        plt.show()
        fig.savefig(f"{filename}_Distortion_fitted_for_correction.png", bbox_inches="tight")
        fig.clf()
        plt.close("all")

    # normalize scale_vector to oscillate around norm_factor:
    scale_vector = scale_vector / norm_factor
    # invert
    scale_vector = 1 / scale_vector
    # plot
    if show:
        fig, ax = plt.subplots()
        ax.plot(phi_arr, scale_vector, "--", label="normalized distortions")

    #    #shift scale vector to values greater 1 (interpolation only)
    #    scale_vector = scale_vector/scale_vector.min()
    #
    if show:
        ax.plot(phi_arr, scale_vector, label="scale vector")
        ax.legend(loc=0)
        ax.set(xlabel="Angle / rad", ylabel="Relative Magnitude")
        plt.show()
        fig.savefig(f"{filename}_Scale_vector.png", bbox_inches="tight")
        fig.clf()
        plt.close("all")
    # get smallest array length for reconcatenation
    #    min_scale =

    # length = int(polar.shape[0] * scale_vector.min())

    # output list
    out = []

    for col, scale in zip(polar.T, scale_vector):

        rescaled = rescale(col, scale)  # [:length]

        difference = polar.shape[0] - rescaled.shape[0]

        if difference < 0:
            out.append(rescaled[:difference])
        else:
            out.append(np.append(rescaled, np.zeros(difference)))

    return np.array(out).T


def polar_dist2(phi, r, eta, alpha, n=2):
    """
    Calcluate n-fold distortion in polar coordinates. n=2 equals an ellipse centerred at the origin of polar coordinates,
    as given in Niekiel.2017

    args:

    phi = angular increment in radians

    r = mean radius

    alpha = rotation of distortion relative to x axis

    """

    arg = 1 + eta**2 - 2 * eta * np.cos(n * (phi + alpha))

    return r * (1 - eta**2) / np.sqrt(arg)


def polar_dist4th_order(phi, r, alpha1, alpha2, alpha3, eta1, eta2, eta3):
    """
    Calculates the sum of 2,3, and 4-fold distortions.

    args:

    phi = angular increment in radians
    r = mean radius
    alpha1, alpha2, alpha3 = distortion orientation of 2,3,4th order
    eta1, eta2, eta3 = distortion strength
    """

    out = 0

    for n, eta, alpha in zip((2, 3, 4), (eta1, eta2, eta3), (alpha1, alpha2, alpha3)):

        out += polar_dist2(phi, r / 3, eta, alpha, n)

    return out


def weighted_error_average(vals, val_errs, **kwargs):
    """
    Calculates a weighted average of vals based on their absolute uncertainties val_errs using np.average.

    The transformation to weights is performed by calculating the inverted relative error.
    Normalizing is performed by np.average.

    kwargs are passed to np.average
    """
    # ensure numpy compatibility of values
    vals = np.asanyarray(vals)
    val_errs = np.asanyarray(val_errs)

    ###transform errors to weights
    # calculate relative error
    val_errs /= vals

    # invert errors to weight small errors higher and vice verca
    val_errs = 1 / val_errs

    # bypass 'ZeroDivisionError: Weights sum to zero, can't be normalized'
    try:
        out = np.average(vals, weights=np.abs(val_errs), **kwargs)
    except ZeroDivisionError:

        text = "Encounterred ZeroDivisionError in weighted_error_average. Try without weighting"
        logging.warning(text)
        # inform user during alignment
        print(text)

        out = np.average(vals, **kwargs)

    return out


def weighted_mean_error(vals, val_errs):
    """
    Uses np.average to calculate the weigth of val_errs depending on thei relative precision of vals (vals/valerrs).
    Furthermore, gaussian error distribution is used to divide the result by the square root of the amout of values.
    """
    weights = np.asarray(vals) / np.asarray(val_errs)
    if weights.sum() == 0:
        weights = None
    return np.average(val_errs, weights=weights) / np.sqrt(len(vals))


def grouper(iterable, n, fillvalue=None):
    """
    Collect data into fixed-length chunks or blocks. Fillvalue is returned in case of division rest.

    grouper( 'ABCDEFG', 3, 'x') --> ABC DEF Gxx


    From https://docs.python.org/3.0/library/itertools.html
    """
    args = [iter(iterable)] * n
    return zip_longest(*args, fillvalue=fillvalue)


def filo(iterable, n=2):
    """
    Creates a rolling window (FILO) out of an interable object with the length
    n.

    INPUT: iterable as an iterable object.

    n (integer) = window size. Must not exceed len(iterable).


    RETURN: zip object containing the rolling window tuples

    """

    assert len(iterable) >= n, "n must not exceed len(iterable)"

    tees = tee(iterable, n)

    for i in range(n):

        for _ in range(i):

            next(tees[i], None)

    return zip(*tees)


def closest(vals, ref):
    """
    Finds closest distance between points in a 1D array and a reference point

    INPUT:

        vals (iterable): An iterable containing floats or ints;

        ref (float or int): Reference value to calculate distance to.



    RETURN:

        Value of vals with minimum distance to ref
    """
    # create a dictionary containing the peak distances to pk and the respective intensity
    distances = {abs(ref - k): k for k in vals}
    # store only minimum distance values (radius, intensity)
    return distances[min(distances)]


def mean_arr(iterable):
    """
    Input: iterable containing objects that support addition and division

    Return: element-wise average of all elements in iterable
    """

    return reduce(lambda a, b: a + b, iterable) / len(iterable)


# "better" peak-finding function


def select_pixels(polar, peaks, minr=None, maxr=None, chunksize=4, sigma=2.5, filename="test", show=True):
    """
    Function to select ellipse on a polar-transformed image

    INPUT:

    polar (array): polar-transformed image
    peaks (list): List of averaged ellipse radii



    args:
    minr, maxr (int or None): minumum, maximum radius to regard. Default: None
    chunksize (int): Number of angle increments to average before searching pixels to enhance signal/noise ratio
    sigma (float): Only pixels with an intensity higher than sigma * array.std() are regarded. array is the (processed) polar array
    filename (string): If show is truthy, filename will be used to name the figures
    show (bool): If truthy (default), intermediate results are shown

    OUTPUT:

    i_vals (array): Array containing the found pixel characteristics. Access the radii with i_vals[:,0],
    the intensities with radii = i_vals[:,1], and the angle increments with i_vals[:,2]


    """

    # polar is cropped to save computation time and avoid some unreal peaks
    # also, it is transposed here
    filtered_polar = prepare_matrix(polar[minr:maxr].T)

    # calculate two sigma
    two_sgm = sigma * filtered_polar.std()

    #    #set beam stopper area to 0:
    filtered_polar = np.where(filtered_polar > two_sgm, filtered_polar, 0)

    # create a data cache
    peak_dct = {}

    # define maximum number of peaks based on the averaged ones:
    maxpeaks = len(peaks)

    # iterate over angles
    for n, i in enumerate(grouper(filtered_polar, chunksize, 0)):

        # average spectra in chunk
        i = reduce(lambda x, y: x + y, i) / chunksize
        # get averaged angle from chunksize
        angle = np.arange(n * chunksize, (n + 1) * chunksize).mean()
        # find values
        indexes = find_peaks_cwt(i, [7.5], min_snr=2, noise_perc=30)

        # filter indices by intensity. Only regard peaks if they are in brighter
        # than two_sgm
        # indexes = [p for p in indexes if i[p] >= two_sgm]
        # indexes = indexes[indexes >= two_sgm]
        if not len(indexes):
            continue
        # filter indices by min_dist:

        # change max peak number if it exceeds the amount of found peaks
        if indexes.shape[0] < maxpeaks:

            mp = indexes.shape[0]

        else:

            mp = maxpeaks

        # arbitrary distance of peaks in px
        min_dist = 30
        intensity = i[indexes]
        lst = []

        # iterate over found peaks
        for k, val in enumerate(indexes):

            if k == 0:

                lst.append(val)

                continue

            old = indexes[k - 1]

            diff = val - old

            # select value with maximum intensity
            if diff <= min_dist:

                if old in lst:

                    lst.remove(old)

                vals = [old, val]
                intens = [intensity[k - 1], intensity[k]]

                lst.append(vals[intens.index(max(intens))])

            else:

                lst.append(val)

        indexes, index_old = lst, indexes

        # index:intensity
        index_dct = dict(zip(indexes, i[indexes]))
        # build dictionary that consists only of the four most intense peaks.
        top4 = {a + minr: index_dct[a] for a in sorted(index_dct, key=lambda x: index_dct[x], reverse=True)[:mp]}
        # store angle

        peak_dct[angle] = top4

        # lst = _remove_peaks(indexes, i[indexes],  5)
        if show:

            if n % 50:

                continue

            # plotting
            f, ax = plt.subplots()
            ax.set(ylabel="Intensity / a. u.", xlabel="Radius / px")
            ax.plot(np.arange(minr, len(i) + minr), i, label=f"Angle: {angle}")

            ax.plot(np.array(index_old) + minr, i[index_old], "s", label="Found Peaks")
            # plt.plot(indexes, i[indexes], 'o')
            ax.plot(np.array([*top4]), list(top4.values()), "x", label="Chosen Peaks")
            ax.legend(loc=0)
            plt.show()
            plt.clf()
            plt.close("all")

    # unpack cache:
    i_vals = np.array(
        reduce(lambda x, y: x + y, ((list(zip(dct[0].keys(), dct[0].values(), dct[1]))) for dct in ((peak_dct[key], [key] * len(peak_dct[key])) for key in peak_dct)))
    )

    if show:

        angles = i_vals[:, 2]
        radii = i_vals[:, 0]

        f, ax = plt.subplots(figsize=(10, 6))
        ax.set_title(f"Found pixels in {filename}")
        ax.plot(angles, radii, "x")
        ax.imshow(equalize_adapthist(prepare_matrix(polar)) ** (1 / 3), cmap="binary")

        for end in (".png", ".pdf"):
            f.savefig(f"{filename}_found_pixels{end}", bbox_inches="tight", dpi=300)
        plt.show()
        plt.clf()
        plt.close("all")

    return i_vals


def averaged_ellipse(
    imgstack,
    minr,
    maxr,
    hminr,
    hmaxr,
    scale_factor=5,
    show=True,
    filename="Averaged files",
    average_range=(3, 51),
    hough=True,
    sigma=2.5,
    hough_filter=True,
    hough_rings=3,
    median_for_polar=True,
    clahe_for_polar=True,
    canny_use_quantiles=False,
    canny_sigma=1,
    dr_polar=1,
    dt_polar=None,
):

    try:

        raw_name = filename

        # adjust filename
        filename = f"{raw_name}_Averaged_over_{average_range[1]-average_range[0]}"

        # average files:
        img = 0

        if len(imgstack.data.shape) == 2:  # , 'more than one image is required'

            img = np.asarray(imgstack.data)

        else:

            for mat in imgstack.data[average_range[0] : average_range[1]]:

                img += np.asarray(mat)

        # display file
        if show:

            plt.title(filename)
            plt.imshow(img ** (1 / 3), cmap="binary")
            plt.savefig(f"{filename}.png", dpi=300)
            plt.clf()
            plt.close()

        # convert img to [0,1]
        img = prepare_matrix(img)

        if hough:

            # create a mask to block beamstopper
            mask = np.ones(img.shape)
            s = int(img.shape[0] / 20 * 3)
            lowy, upy = int(img.shape[0] / 2 - s), int(img.shape[0] / 2 + s)
            lowx, upx = int(img.shape[1] / 2 - s), int(img.shape[1] / 2 + s)
            mask[lowx:upx, lowy:upy] = np.nan

            # perform hough transformation
            circ = find_rings(
                img,
                hminr,
                hmaxr,
                10,
                scale_factor=scale_factor,
                max_peaks=hough_rings,
                mask=mask,
                filename=filename + "_Hough_result_avg.png",
                show=show,
                hough_filter=hough_filter,
                canny_use_quantiles=canny_use_quantiles,
                canny_sigma=canny_sigma,
            )

            # get circle center in numpy coordinates, originating from the top left corner
            center = get_mean(circ)[::1]

        else:
            center = np.array(img.shape) / 2

        polar, r_grid, theta_grid = img2polar(img, center, show=show, clahe=clahe_for_polar, median_filter=median_for_polar, jacobian=False, dr=dr_polar, dt=dt_polar)
        # get first estimation of the peak values by integrating around the Hough center
        peaks, __, __ = find_average_peaks(
            polar,
            r_grid,
            theta_grid,
            peak_widths=[
                int(round(2 * max(img.shape) / (2048 * dr_polar))),
            ],
            min_radius=minr,
            max_radius=maxr,
            show=show,
        )

        i_vals = select_pixels(polar, peaks, minr, maxr, filename=filename, show=show, sigma=sigma)

        angles = i_vals[:, 2]  # [cond]
        radii = i_vals[:, 0]  # [cond]

        df = pd.DataFrame(np.array([radii, angles]).transpose(), columns="radius angle".split())
        # calculate angles in radians for further analyses.
        df["angle / rad"] = df["angle"].apply(lambda x: increment2radians(x, polar.shape[1]))
        # calculate carthesian coordinates:
        df["x"], df["y"] = polar2carthesian(df["radius"], df["angle / rad"])

        # fit ellipse to data
        ell_opt = fit_ellipse(df["x"], df["y"], ring=df["radius"].mean(), filename=filename)
        a, b, c0, alpha = ell_opt
        xvals, yvals = ellipse_polar(
            b,
            a,
            center + c0,
            alpha,
        )

        # use fitresults to define a data corridor:

        if show:
            f, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
            ax1.set_title(f"{filename}")
            ax1.plot(angles, radii, "x")
            ax1.imshow(equalize_adapthist(prepare_matrix(polar)) ** (1 / 3), cmap="binary")
            ax2.plot(df["x"] + center[0], df["y"] + center[1], "x")
            ax2.plot(xvals, yvals, "-")
            ax2.imshow(equalize_adapthist(img) ** (1 / 3), cmap="binary")
            plt.tight_layout()
            for end in (".png", ".pdf"):
                f.savefig(f"{filename}{end}", bbox_inches="tight", dpi=300)
            plt.show()
            plt.clf()
            plt.close()

    # log error, save and terminate
    except Exception as e:

        # log
        message = "{} in file {}.".format(type(e).__name__, filename)
        logging.critical(message)
        logging.critical(traceback.format_exc())

        # terminate
        raise e

    return ell_opt


def alignment_check(df, **kwargs):

    beta = []
    beta_err = []
    condition = []
    # iterate over different alignemnt conditions:
    for cond in sorted(set(df["C2/C3"])):

        dfcond = df[df["C2/C3"] == cond]
        # calculate and plot beta out of df:
        # try:
        betalist, betaerrs, label = estimate_incident_angle(dfcond, **kwargs)

        for i in range(len(betalist)):

            # calculate weighted averaged beta:
            beta_mean = weighted_error_average(betalist[i], betaerrs[i])

            # CATION THIS MAY NOT BE VALID FOR GAUSSIAN ERROR PROPAGATION
            beta_err_mean = weighted_mean_error(betalist[i], betaerrs[i])

            betalist[i] = beta_mean
            betaerrs[i] = beta_err_mean

        beta.append(betalist)
        beta_err.append(betaerrs)
        condition.append(cond)

    beta = np.array(beta).transpose()
    beta_err = np.array(beta_err).transpose()

    ##plot alignment
    fig, ax = plt.subplots(figsize=(4.5, 4.5), dpi=100)
    # plot data and errorbars
    for m, (y, yerr) in enumerate(zip(beta, beta_err)):

        # try:
        ax.errorbar(
            condition,
            y,
            yerr=yerr,
            label=label[m],
            marker=".",
            capsize=3,
            elinewidth=1,
            markeredgewidth=1,
        )
    #        except Exception as e:
    #            print(type(e).__name__,e)

    ax.legend(loc=0)
    ax.set(ylabel=r"$\beta$" + " / mrad", xlabel="Relative Excitation C2/C3")
    ax.tick_params(direction="in", which="both")
    ax.grid()
    plt.show()
    fig.savefig(f"Alignment condition {cond}.png", bbox_inches="tight")
    plt.close()

    df = pd.DataFrame([(i, j, k) for i, j, k in zip(beta, beta_err, condition)], columns=["Beta / mrad", "Beta_error / mrad", "C2/C3"])
    df2excel(df, name="Alignment results", newfile=False)


def _get_last_center(num, tolerance, indices):
    """
    enters the indices dictionary to retreive the num-1th file center if the center displacement deceeds tolerance.
    """
    # get old number
    old_num = num - 1

    try:
        old_data = indices[old_num]
    except KeyError:

        logging.warning(f"_get_last_center: {old_num} not in indices")
        # exit function
        return None

    # only get the center if the tolerance value is reached
    if old_data["Center Displacement / px"] <= tolerance:
        x = old_data["Center x"]
        y = old_data["Center y"]
        center = x, y
    else:
        center = None

    return center


def _retreive_fit_parameters(out):
    """
    Returns fit parameters of best fit based on R² value as list
    """
    # get ring numbers
    rings = [int(re.search("-?\d+", k).group()) for k in out if re.search("distortion correction -?\d+ radius / px", k)]
    # create output container
    r2s = {}
    ring_values = []
    # retreive fit parameters for every ring

    for ring in rings:
        r2s[out[f"distortion correction {ring} values"]] = ring

    ring = r2s[max(r2s)]

    ring_values.append(out[f"distortion correction {ring} radius / px"])
    ring_values.append(out[f"distortion correction {ring} phi1 / rad"])
    ring_values.append(out[f"distortion correction {ring} phi2 / rad"])
    ring_values.append(out[f"distortion correction {ring} phi3 / rad"])
    ring_values.append(out[f"distortion correction {ring} eta1"])
    ring_values.append(out[f"distortion correction {ring} eta2"])
    ring_values.append(out[f"distortion correction {ring} eta3"])

    # create numpy array and average ring numbers
    return ring_values  # popt[-1]#


def _multiple_voigt(
    xdat,
    y0,
    *params,
):

    # assert not len(params) % 4, f'Parameter number is not divisible by 4. {len(params)} parameters given.'

    y = 0
    for x, A, w, eta in grouper(params, 4, 0):
        y += _pseudo_voigt(xdat, x, A, w, eta, y0 / (len(params) / 4))

    return y


def _multi_voigt_background(xdat, *params):

    poly_params = params[-2:]
    params = params[:-2]

    return multi_voigt(xdat, *params) + poly_func(xdat, *poly_params)


def fwhm_voigt(sigma, gamma):
    # https://www.sciencedirect.com/science/article/pii/0022407377901613?via%3Dihub
    # https://de.wikipedia.org/wiki/Voigt-Profil#Die_Breite_des_Voigt-Profils

    return 0.5346 * 2 * gamma + np.sqrt(0.2166 * (2 * gamma) ** 2 + np.sqrt(8 * np.log(2)) * sigma)


def fit_full_peak_spectrum(
    rrange,
    intensity,
    bounds,
    filename="test.png",
    show=False,
    dr=1,
    expected_peak_num=4,
    background_voigt_num=3,
    holz=2,
    zolz_guesses=None,
    full_spectrum_fit=True,
    distortion_fit_radius=[],
    only_1_peak=False,
    only_zolz=True,
    double_peak=False,
    use_last_fit=True,
    popt_old=None,
    max_intensity=None,
):

    popt = None

    # unpack boundaries
    low, up = bounds

    # apply boundaries to data
    r = rrange[low:up]
    cropped_intensity = intensity[low:up]

    ##select peaks
    # calculate peak positions based on hkl values
    if isinstance(zolz_guesses, (tuple, list, np.array)):

        peaks = []
        relative_distances = np.array((np.sqrt(1**2 + 1**2 + 1**2), np.sqrt(2**2 + 0**2 + 0**2), np.sqrt(2**2 + 2**2 + 0**2), np.sqrt(3**2 + 1**2 + 1**2)))

        h, k, l = zolz_guesses

        for distortion_fit in distortion_fit_radius:

            assert distortion_fit_radius, "No distortion fit radius is given"

            p_positions = relative_distances / np.sqrt(h**2 + k**2 + l**2) * distortion_fit
            peaks.extend(p_positions)

    # calculate guesses based on a list or tuple of given values. Deprecated?
    elif zolz_guesses is not None:

        peaks = list(zolz_guesses)

    # try to get them from shape of cropped intensity
    else:

        grad1 = np.gradient(
            savgol_filter(
                cropped_intensity,
                35,
                3,
                mode="interp",
            )
        )
        grad2 = savgol_filter(
            np.gradient(grad1),
            55,
            3,
            mode="interp",
        )

        # find radii of index change to detect zeros in second derivation
        inflictions = np.where(np.sign(grad2[:-1]) != np.sign(grad2[1:]))[0] + 1

        intensity_lst = np.split(cropped_intensity, inflictions)
        rrange_lst = np.split(r, inflictions)
        grad2_lst = np.split(grad2, inflictions)

        peaks = []
        peak_intensity = []
        for intens, rad, gra in zip(intensity_lst, rrange_lst, grad2_lst):

            i_max = intens.max()

            if gra[np.where(np.absolute(gra).max())] < 0 and i_max >= 0.25 * intensity.max():

                p = rad[intens == i_max][0]
                peaks.append(p)
                peak_intensity.append(i_max)

        # reduce amount of peaks to expected_peak_num:
        p_arr = np.array((peaks, peak_intensity))
        peaks = p_arr[0][p_arr[1].argsort()][::-1][:expected_peak_num]

    # ensure to take higher order diffraction peaks into account, too:
    peaks = np.asarray(peaks)
    if not only_zolz:
        # ensure to take higher order diffraction peaks into account, too:
        peaks = np.append(peaks, (holz * peaks)[holz * peaks <= up])

    else:
        # cut spectrum half way between last zolz and first holz peak
        last_zolz = peaks[-1]
        first_holz = peaks[0] * 2

        up = int(round(np.mean((last_zolz, first_holz))))

        # apply new boundaries to data
        r = rrange[low:up]
        cropped_intensity = intensity[low:up]

    ##create guesses
    # fit voigt function to peaks separately

    peak_fits = fit_gaussian2peaks(
        peaks,
        rrange,
        intensity,
        filename=filename.split(".")[0] + "_single_peaks.png",
        val_range=10,
        functype="voigt",
        show=show,
        dr=dr,
    )

    if full_spectrum_fit and double_peak:
        peak_fits = fit_separate_double_peak(
            peak_fits,
            rrange,
            intensity,
            val_range=30,
            show=show,
            filename=filename.split(".")[0] + "_double_peaks.png",
        )

    if full_spectrum_fit and not only_1_peak:
        # retreive peak numbers:
        peaks = [int(peak[3:]) for peak in peak_fits if peak.startswith("xc ")]
        # retreive initial guesses for peak fitting and calculate boundaries
        peak_guesses = []
        low_bounds = []
        up_bounds = []

        for n, pk in enumerate(peaks):

            xc = peak_fits[f"xc {pk}"]
            area = peak_fits[f"A {pk}"]
            sigma = peak_fits[f"sigma {pk}"]
            gamma = peak_fits[f"gamma {pk}"]
            fwhm = fwhm_voigt(sigma, gamma)
            max_width = max((sigma, gamma, fwhm))

            if n in (2, 3):
                peak_guesses.extend((xc, area / 100, sigma / 100, gamma / 100))
                low_bounds.extend((xc - fwhm / 2, 0, 0, 0))
                up_bounds.extend((xc + fwhm / 2, area * 0.75, max_width, max_width))

            else:
                peak_guesses.extend((xc, area, sigma / 10, gamma / 10))
                low_bounds.extend((xc - fwhm / 2, 0, 0, 0))
                up_bounds.extend((xc + fwhm / 2, area * 1.5, max_width * 1.5, max_width * 1.5))

        # add boundaries for background
        low_bounds.extend([low, 0, 0, 0] * background_voigt_num)
        up_bounds.extend([up, np.inf, np.inf, np.inf] * background_voigt_num)

        if use_last_fit and popt_old is not None:
            try:
                p0 = popt_old
                fitted = True

            except NameError:
                print("##################### fitted is False #####################")
                fitted = False
        else:
            fitted = False

        if not use_last_fit or not fitted:

            # fit residual spectrum
            peak_fits = model_background(
                peak_fits,
                r,
                cropped_intensity,
                background_voigt_num=background_voigt_num,
                show=show,
                filename=filename.split(".")[0] + "_residual_background.png",
                min_radius=low,
                max_radius=up,
                dr=dr,
                only_1_peak=only_1_peak,
            )

        if not use_last_fit or not fitted:
            ##combine guesses for modelling the spectrum
            # retreive background guesses,
            strt = "background_voigt"

            popt_background_voigts = []

            for i in range(background_voigt_num):
                popt_background_voigts.extend(
                    [
                        peak_fits[f"{strt} xc {i}"],
                        peak_fits[f"{strt} A {i}"],
                        peak_fits[f"{strt} sigma {i}"],
                        peak_fits[f"{strt} gamma {i}"],
                    ]
                )

            p0 = peak_guesses + popt_background_voigts

        # ensure mutable p0_
        p0 = np.asarray(p0)
        # check for feasible p0 conditions:
        for n, (l, g, u) in enumerate(zip(grouper(low_bounds, 4), grouper(p0, 4), grouper(up_bounds, 4))):
            for i in range(4):

                if not l[i] < u[i]:
                    low_bounds[n * 4 + i] = 0
                    up_bounds[n * 4 + i] = np.inf  # max((l[i], u[i], g[i]))
                    print(f"Boundaries {i} in Peak {n} reassigned: {l[i]} --> {low_bounds[n*4+i]}")
                    print(f"Boundaries {i} in Peak {n} reassigned: {u[i]} --> {up_bounds[n*4+i]}")

                    if not low_bounds[n * 4 + i] < g[i] < up_bounds[n * 4 + i]:
                        p0[n * 4 + i] = low_bounds[n * 4 + i]

                    continue

                if not l[i] <= g[i] <= u[i]:
                    print(f"Improper guess for value {i} in Peak {n}: {g[i]}")
                    if not np.isinf(u[i]):
                        p0[n * 4 + i] = np.mean((l[i], u[i]))
                    else:
                        p0[n * 4 + i] = l[i]
                    print(f"Value {i} in Peak {n} reassigned: {g[i]} --> {p0[n*4+i]}")
        # fit
        try:
            popt, pcov = curve_fit(multi_voigt, r, cropped_intensity, p0=p0, bounds=(low_bounds, up_bounds), maxfev=100000)
            errors, r_squared_corr = get_errors(popt, pcov, r, cropped_intensity, multi_voigt(r, *popt))
        except Exception as e:

            print("ValueError occured!")

            for n, (l, g, u) in enumerate(zip(grouper(low_bounds, 4), grouper(p0, 4), grouper(up_bounds, 4))):
                print("Peak Nr", n)
                for i in range(4):
                    print("\t", l[i], g[i], u[i])

            raise e
            popt, pcov = curve_fit(multi_voigt, r, cropped_intensity, p0=p0, bounds=(0, np.inf), maxfev=100000)
            errors, r_squared_corr = get_errors(popt, pcov, r, cropped_intensity, multi_voigt(r, *popt))

        # store output parameters:
        peak_fits["R2adj_full_fit"] = r_squared_corr
        for pk, (popts, errs) in enumerate(zip((grouper(popt, 4, np.nan)), (grouper(errors, 4, np.inf)))):
            xc, area, sigma, gamma = popts
            xc_err, area_err, sigma_err, gamma_err = errs

            if pk >= len(popt) / 4:
                strt = "background_voigt "
            else:
                strt = ""

            peak_fits[f"{strt}xc_full_fit {pk}"] = xc
            peak_fits[f"{strt}A_full_fit {pk}"] = area
            peak_fits[f"{strt}sigma_full_fit {pk}"] = sigma
            peak_fits[f"{strt}gamma_full_fit {pk}"] = gamma
            peak_fits[f"{strt}xc_full_fit_error {pk}"] = xc_err
            peak_fits[f"{strt}A_full_fit_error {pk}"] = area_err
            peak_fits[f"{strt}sigma_full_fit_error {pk}"] = sigma_err
            peak_fits[f"{strt}gamma_full_fit_error {pk}"] = gamma_err

        # plot
        if show:
            fig, ax = plt.subplots(figsize=(5, 5), dpi=100)
            ax.plot(rrange, intensity, label="Data")
            # ax.plot(rrange, model_whole_spectrum(rrange, *p0), '--', label ='Guess')
            ax.plot(r, multi_voigt(r, *popt), "--", label=f"Fit $R^2$={round(r_squared_corr*100,2)}%")

            # plot voigt peaks:
            ls = "-."
            for n, params in enumerate(grouper(popt, 4, 0)):

                if n >= len(popt) / 4 - background_voigt_num:
                    ls = ":"

                ax.plot(r, voigt(r, *params), ls, label=f"Voigt at {int(round(params[0]))} px")

            # ax.axvline(low, label='Start data range')
            # ax.axvline(up, label='End Data Range')
            ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
            ax.set(xlabel="Radius / px", ylabel="Angular-integrated intensity / a. u.", ylim=max_intensity)

            plt.show()
            fig.savefig(
                filename,
                bbox_inches="tight",
            )
            plt.clf()
            plt.close("all")

    if full_spectrum_fit and only_1_peak:
        print("NOT DEFINED")

    return peak_fits, popt


def fit_separate_double_peak(
    dct,
    r,
    intensity,
    val_range=40,
    show=True,
    dr=1,
    filename="test.png",
):

    peaks = [int(peak[3:]) for peak in dct if peak.startswith("xc ")]

    #    if show:
    #            f, ax = plt.subplots(dpi=100,figsize=(5,5))
    #            ax.plot(r, intensity, label='Full data range', c = 'black')

    for n, (p1, p2) in enumerate(grouper(peaks, 2, None)):

        if p2 is None:
            continue

        # retreive params
        xc1 = dct[f"xc {p1}"]
        area1 = dct[f"A {p1}"] / 2
        sigma1 = dct[f"sigma {p1}"] / 2
        gamma1 = dct[f"gamma {p1}"] / 2
        xc2 = dct[f"xc {p2}"]
        area2 = dct[f"A {p2}"] / 2
        sigma2 = dct[f"sigma {p2}"] / 2
        gamma2 = dct[f"gamma {p2}"] / 2

        meanxc = np.mean((xc1, xc2))
        max_area = max((area1, area2))
        fwhm = np.ceil(max((fwhm_voigt(sigma1, gamma1), fwhm_voigt(sigma2, gamma2))))

        if n == 1:
            area1 /= 2
            area2 /= 2
            fwhm /= 2

        p0 = [xc1, area1, sigma1, gamma1, xc2, area2, sigma2, gamma2, 0, intensity.min()]

        # select area of interest
        low = int(xc1) - val_range
        up = int(np.ceil(xc2)) + val_range

        x = r[low:up]
        y = intensity[low:up]
        # restrict boundaries
        lb = [
            low,
            0,
            0,
            0,
            low,
            0,
            0,
            0,
            -np.inf,
            -np.inf,
        ]
        ub = [up, max_area, fwhm * 2, fwhm * 2, up, max_area, fwhm * 2, fwhm * 2, np.inf, np.inf]

        # first, try without parabolic background:
        if True:  # n = 1:

            func = _multi_voigt_background
            pb = None
        try:
            popt, pcov = curve_fit(func, x, y, p0=p0[:pb], bounds=(lb[:pb], ub[:pb]), maxfev=100000)
        except ValueError as e:

            print(e)
            print(lb)
            print(p0)
            print(ub)
            ub2 = [up, np.inf, np.inf, np.inf, up, np.inf, np.inf, np.inf, np.inf, np.inf]
            popt, pcov = curve_fit(func, x, y, p0=np.abs(p0[:pb]), bounds=(lb, ub2), maxfev=100000)
        yfit = func(x, *popt)
        errors, r2adj = get_errors(popt, pcov, x, y, yfit)

        # overwrite results
        dct[f"xc {p1}"] = popt[0]
        dct[f"A {p1}"] = popt[1]
        dct[f"sigma {p1}"] = popt[2]
        dct[f"gamma {p1}"] = popt[3]
        dct[f"xc {p2}"] = popt[4]
        dct[f"A {p2}"] = popt[5]
        dct[f"sigma {p2}"] = popt[6]
        dct[f"gamma {p2}"] = popt[7]

        if show:
            f, ax = plt.subplots(dpi=100, figsize=(5, 5))
            ax.plot(
                x,
                y,
                label="Data ",
            )
            ax.plot(x, func(x, *popt), "--", label=f"Double peak at {int(round(meanxc))} R2: {round(r2adj*100, 2)}")
            ax.plot(r, voigt(r, *popt[:4]), "-.", label=f"Voigt at {int(round(popt[0]))}")
            ax.plot(r, voigt(r, *popt[4:-2]), "-.", label=f"Voigt at {int(round(popt[4]))}")
            ax.plot(x, poly_func(x, *popt[-2:]), ":", label="Background")

            ax.set(xlabel="Angular-integrated intensity / a. u.", ylabel="Radius / px")
            ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
            # save results
            filename = f"{filename}_{int(round(meanxc))}.png"
            f.savefig(filename, bbox_inches="tight")
            plt.show()
            plt.clf()
            plt.close("all")

    return dct


def model_full_spectrum(
    rrange,
    intensity,
    starting_peaks,
    low_r,
    up_r,
    peak_num,
    background_voigt_num,
    zolz_guesses,
    filename,
    full_spectrum_fit,
    double_peak,
    popt_old,
    reuse_last_fit_params=True,
    show=True,
):

    i = 0
    j = 0
    # try max 4 times and then live with the result. filtering via R² is possible anyway...
    while i < 2 and j < 4:
        try:
            i += 1
            j += 1
            peak_fits, popt_old = fit_full_peak_spectrum(
                rrange,
                intensity,
                (low_r, up_r),
                show=show,
                dr=1,
                expected_peak_num=peak_num,
                background_voigt_num=background_voigt_num,
                zolz_guesses=zolz_guesses,
                filename=f"{filename}_Spectrum_fit.png",
                full_spectrum_fit=full_spectrum_fit,
                distortion_fit_radius=starting_peaks,
                double_peak=double_peak,
                popt_old=popt_old,
                use_last_fit=reuse_last_fit_params,
                max_intensity=None,
            )
        except Exception as e:
            print(e)
            raise e

        if peak_fits["R2adj_full_fit"] >= 0.993:
            break
        elif i == 0:
            continue
        else:
            i = 0
            popt_old = None

    return peak_fits, popt_old


def main(
    show=False,
    test=False,
    show_test=False,
    reverse=False,
    filepath=".",
    rescaling=False,
    dfname="test",
    filo_num=1,
    tolerance=0.01,
    skip=[],
    ring_deviation=0.05,
    scale=10,
    re_pattern=".em[di]$",
    radius_range="auto",
    average_range=(0, 2),
    filter_for_hough=True,
    test_interval=10,
    median_for_polar=False,
    clahe_for_polar=False,
    jacobian_for_polar=True,
    int_thres="std",
    local_intensity_scope=True,
    min_vals_per_ring=20,
    fit_func="gauss",
    rescale_sigma=2.5,
    canny_use_quantiles=True,
    hough_rings=1,
    canny_sigma=1,
    dr_polar=1,
    dt_polar=None,
    hough_radius_range=None,
    skip_ring=None,
    center_guess=None,
    use_last_center=True,
    double_peak=False,
    peak_num=5,
    background_voigt_num=4,
    zolz_guesses=None,
    full_spectrum_fit=True,
    cwt_based_pixel_extraction=True,
):
    """
    Main Excecution function.

    Returns a DataFrame with the results


    Kwargs:

    show (bool): If truthy, Images of (intermediate) results are displayed and saved. Slows algorithm down

    test (bool): If truthy, only every nth image in an image stack is processed. n is specified by test_interval. Default: 10.

    show_test (bool): If truthy, the (intermediate) results of every test_interval'th pattern are displayed and saved. Overruled by show = True or test = True.

    test_interval (int): Value to specify in which period image are processed if test is truthy.

    reverse (bool): If truthy, the found images are processed in reverse order.

    filepath (string or container containing strings): Path to the desired working directory.
             Default: '.'. To write a code independent of the operating system, one can pass lists or tuples
             containing the folder names such as ['D:\\', 'Algotest', 'TEM', '190412_Alignment_test', 'all'].

    re_pattern (string): A pattern to search for files of interest in the set working directory using regular
                         expression syntax.

    rescaling (bool): If truthy, the ellipticity of the images is corrected by rescaling the image matrix.
                      Required for strong elliptic distortion.

    dfname (string): Name of the to-be-saved excel sheet, additionally, a timestamp is added to the filename.

    filo_num (integer): Amount of images to average using a first-in-first-out process. 1 equals no averaging

    tolerance (float): Value between the used center for polar transform and the calculated center in px after
                       which the center optimization is considered succesfully.

    skip (List of integers or empty list): Amount of images to skip in an image container.

    ring_deviation (float or tuple of ints): Value to define the relative interval size around the given radius. If too small,
                    the diffraction ring might be displayed incorrectly. If too large, values of a neighbouring ring
                    may be taken into accout. For heavy distortions, use rescaling.

    scale (integer): Downsampling factor for Hough transform (required to find the intial circle center).
                    For low singal/noise ratio or sparse diffraction rings, a high number (e.g. 9) may yield
                    better results.

    radius_range (tuple of two numbers (int or float) or 'auto'): Radius range of interest in px for finding diffraction rings.
                    Can be used to exclude small/large rings. If 'auto', the radii are calculated as a fraction
                    of the image size. If both tuple values (e.g. (0.5, 0.78)) are in [0:1], they are interpreted as
                    percentages of the image dimensions.

    filter_for_hough (bool): If truthy, a combination of median filtering and
                            Contrast Limited Adaptive Histogram Equalization is performed on the image to
                            prepare the Hough-Transform.

    fit_func (str): Function to fit the integrated peaks:
            'gauss' --> gaussian function is used
            'lorentz' --> lorentzian function is used
            else --> default to 'gauss'

    int_thres (string or float (0,1]): Intensity threshold for ring value selection:

        valid string values:

        'std' --> Only exclude the darkest values based on a 1 sigma standard deviation
        '2std' --> Only exclude the darkest values based on a 2 sigma standard deviation
        '-std' --> Only use the brightest values based on a 1 sigma standard deviation
        '-2std' --> Only use the brightest values based on a 2 sigma standard deviation
        'mean' --> Only use the brightest values based on the mean intensity
        'median' --> Only use the brightest values based on the median intensity

        if float:

            Value must be in [0,1]. Thus, The allowed intensity must be above this percentage
            --> 1.0 means 0 data,

    min_vals_per_ring (int): Minimum Number of values required to count for a ring.
                Also, these values have to be in at least two orthogonal quadrants.
                If set to 0, it will disable the this option.

    median_for_polar (bool): if truthy, a median filter is used on the polar-transformed image

    clahe_for_polar (bool): if truthy, a Contrast Limited Adaptive Histogram Equalization is
                            used on the polar-transformed image

    jacobian_for_polar (bool): if truthy, a jacobian matrix-based algorithm rescales the intensity of the polar-transformed image.

    sigma (float): If rescaling is truthy, only pixels with an intensity higher
                  than sigma * array.std() are regarded in select_pixels.
                  array is the (processed) polar array. Default for noisy images: 2.5

    canny_use_quantiles (bool): If truthy, thresholding in canny algorithm is done differently.

    hough_rings (int): Number of rings to choose in Hough-Transform.

    canny_sigma (float): Sigma value for gauss filter in canny algorithm

    hough_radius_range (tuple of two numbers (int or float) or None): adius range of interest in px for finding diffraction rings during Hough Transform.
                    Can be used to exclude small/large rings. If None (default), radius_range is used.

    center_guess(string or None): path to Excel file containing center information for the requested data set.

    use_last_center(bool): retreives previous center if the latter was passing the tolerance value. Overruled by center_guess.

    fit_gauss(bool): If truthy, every data point is modelled by a gauissan peak with a parabolic background

    double_peak(bool): If truthy, the peak detection routine is looking for two peaks in the sliced polar data range.
                        Only works, if fit_gauss is truthy. Default: False
    """

    logging.info("Program started")
    logging.critical(f"Operating Parameters are:\n{locals()}")
    # create indices dict
    indices = {}

    # list to store distortion-corrected spectra
    corrected_spectra = []

    # go to files
    if type(filepath) == str:

        pass

    elif type(filepath) == tuple or type(filepath) == list:

        filepath = os.path.join(*filepath)

    if reverse:

        dfname += "_reverse"

    # change current working directory
    os.chdir(filepath)

    # keep in mind that "pattern" accepts regex syntax
    file_selection = select_files(os.listdir(), pattern=re_pattern)

    logging.info('{} files selected, based on "{}".'.format(len(file_selection), re_pattern))

    # iterate over files to fit center
    num = 0

    # create a variable to store initial show value
    init_show = show

    logging.info("Enter loop over files. Reverse sorting = {}".format(reverse))

    if center_guess:
        old_df = pd.read_excel(center_guess)

    popt_old = None
    # sort files by name
    for file in sorted(file_selection, reverse=reverse):

        logging.info("starting with " + file)
        print("\tstarting with " + file)

        # load file
        imgstack = load_file(file)

        # check for multiple images in imgstack
        shape = imgstack.data.shape

        if len(shape) == 3:

            # store amount of images
            imgs_in_stack = shape[0]

            if radius_range == "auto":

                minr = min(shape[1:]) // 5
                maxr = max(shape[1:]) // 2

            # calculate minr, maxr in percentage of image size
            elif all((0 <= radius_range[0] <= 1, 0 <= radius_range[1] <= 1)):

                minr = int(min(shape[1:]) * radius_range[0])
                maxr = int(max(shape[1:]) * radius_range[1])

            else:

                minr, maxr = (int(i) for i in radius_range)

            ##deal with Hough radius boundaries:
            if hough_radius_range is None:

                hminr, hmaxr = minr, maxr

            elif all((0 <= hough_radius_range[0] <= 1, 0 <= hough_radius_range[1] <= 1)):

                hminr = int(min(shape[1:]) * hough_radius_range[0])
                hmaxr = int(max(shape[1:]) * hough_radius_range[1])

            else:

                hminr, hmaxr = (int(i) for i in hough_radius_range)

            # average files and store results for ellipse correction
            if rescaling:

                ell_opt = averaged_ellipse(
                    imgstack,
                    min(shape[1:]) // 6,
                    max(shape[1:]) // 2,
                    hminr=hminr,
                    hmaxr=hmaxr,
                    scale_factor=scale,
                    show=show,
                    average_range=average_range,
                    median_for_polar=median_for_polar,
                    clahe_for_polar=clahe_for_polar,
                    filename=f"{file}_averaged",
                    hough_filter=filter_for_hough,
                    sigma=rescale_sigma,
                    canny_use_quantiles=canny_use_quantiles,
                    hough_rings=hough_rings,
                )

        elif len(shape) == 2:

            # set amout of image
            imgs_in_stack = 1

            if radius_range == "auto":

                minr = min(shape) // 5
                maxr = max(shape) // 2

                # calculate minr, maxr in percentage of image size

            elif all((0 <= radius_range[0] <= 1, 0 < radius_range[1] <= 1)):

                minr = int(min(shape) * radius_range[0])
                maxr = int(max(shape) * radius_range[1])

            else:

                minr, maxr = (int(i) for i in radius_range)

            ##deal with Hough radius boundaries:
            if hough_radius_range is None:

                hminr, hmaxr = minr, maxr

            elif all((0 <= hough_radius_range[0] <= 1, 0 <= hough_radius_range[1] <= 1)):

                hminr = int(min(shape) * hough_radius_range[0])
                hmaxr = int(max(shape) * hough_radius_range[1])

            else:

                hminr, hmaxr = (int(i) for i in hough_radius_range)

            if rescaling:
                # average files and store results for ellipse correction
                if init_show or show_test:
                    show_avg = True

                else:
                    show_avg = False

                    ell_opt = averaged_ellipse(
                        imgstack,
                        min(shape[1:]) // 6,
                        max(shape[1:]) // 2,
                        hminr=hminr,
                        hmaxr=hmaxr,
                        scale_factor=scale,
                        show=show,
                        average_range=average_range,
                        median_for_polar=median_for_polar,
                        clahe_for_polar=clahe_for_polar,
                        filename=f"{file}_averaged",
                        hough_filter=filter_for_hough,
                        sigma=rescale_sigma,
                        canny_use_quantiles=canny_use_quantiles,
                        hough_rings=hough_rings,
                    )

        else:

            logging.warning("Unknown Image format for {}. Continuing...".format(file))

            continue

        ###for Ming:
        ring_deviation = minr, maxr

        if rescaling:

            logging.debug(f"ell_opt for ellipticity correction is {ell_opt}")

            tup = ell_opt[:2] / np.mean(ell_opt[:2])
        if imgs_in_stack == 1:

            imgs = [imgstack.data.astype("float64")]

        else:

            imgs = imgstack.data.astype("float64")

        # iterate over all images in stack
        for n, images in enumerate(filo(imgs, filo_num)):
            # for i in range(894,1148):

            logging.info(f"Averaging over {filo_num} file(s)")
            img = mean_arr(images)

            # redefine filename
            filename = reduce(_kit_str, file.split(".")[:-1])

            # skip part without information
            if n in skip:

                logging.info(f"Skip {filename}, because n = {n} is in {skip}")
                continue

            # apply test only if n not in skip
            elif test and n % test_interval:

                logging.info(f"Skip {filename}, because test is {test}. test_interval is {test_interval}")
                continue

            # set show accordingly to show_test if init_show is False
            logging.debug(f"Test for skipping images: init_show is {init_show}")
            if all((not init_show, not test, show_test)):

                if n % test_interval:
                    show = False
                    logging.debug(f"Test for skipping images: n ({n}) % {test_interval} = {n % test_interval} --> show = {show}")

                else:
                    show = True
                    logging.debug(f"Test for skipping images: n ({n}) % {test_interval} = {n % test_interval} --> show = {show}")

            # make show_test behave differently if test is true and show is false
            elif all((not init_show, test, show_test)):

                if n % test_interval**2:
                    show = False
                    logging.debug(f"Test for skipping images with show_test={show_test} and test={test}: n ({n}) % {test_interval}**2 = {n % test_interval**2} --> show = {show}")

                else:
                    show = True
                    logging.debug(
                        f"Test for skipping images with show_test True and Test True with show_test True and Test True: n ({n}) % {test_interval}**2 = {n % test_interval**2} --> show = {show}"
                    )

            try:

                if show:

                    fig, ax = plt.subplots()

                    ax.imshow(img ** (1 / 3), cmap="binary")
                    ax.set(xlabel="px", ylabel="px")

                    for end in (".png", ".pdf"):
                        fig.savefig(f"{filename}_before_ell_corr{end}", dpi=300)
                    plt.show()
                    plt.clf()
                    plt.close()

                ##correct ellipticity roughly to enable a circular shape to work with:
                # rotate so that ellipse semi-axes match axes of coordinate system
                if rescaling:
                    logging.info("Correcting ellipse for " + filename)
                    # rescale to make pattern circular
                    logging.debug(f"img is {type(img)}")
                    #
                    #                    #change angle to match with rescaling factors.
                    # The problem is that the ellipse fitting does not return a signed angle.
                    phi = np.rad2deg(ell_opt[3] - np.pi / 2)
                    #                    #add 90° if major axis is vertical
                    #
                    # phi = np.rad2deg(ell_opt[3])
                    logging.debug(f"rotate img by {phi}°")
                    img = rotate(img, phi, reshape=False)

                    #                    #rescale
                    logging.debug(f"Shape before rescale: {img.shape}")
                    img = rescale(img, tup, order=3)
                    logging.debug(f"Shape after rescale: {img.shape}")

                    if show:

                        fig, ax = plt.subplots()

                        ax.imshow(img ** (1 / 3), cmap="binary")
                        ax.set(xlabel=f"rescaled by {tup[1]}", ylabel=f"rescaled by {tup[0]}")

                        for end in (".png", ".pdf"):
                            fig.savefig(f"{filename}_ell_corr{end}", dpi=300)
                        plt.show()
                        plt.clf()
                        plt.close()

                ##append filename by i with a proper amount of 0 in front of it for file sorting
                # calculate amount of 0's:
                maxdigits = len(str(imgs_in_stack))
                i_digits = len(str(n))
                zeros = "0" * (maxdigits - i_digits)

                # adjust filename
                filename += "_" + zeros + str(n)

                logging.info("Start center finding for " + filename)
                # print('\tProcessing ' + filename)
                # create mask array:
                mask = np.ones(img.shape)
                s = int(img.shape[0] / 20 * 3)
                lowy, upy = int(img.shape[0] / 2 - s), int(img.shape[0] / 2 + s)
                lowx, upx = int(img.shape[1] / 2 - s), int(img.shape[1] / 2 + s)
                mask[lowx:upx, lowy:upy] = np.nan

                # convert array to the required form
                img = prepare_matrix(img)
                # img = img * mask2

                center = None
                if center_guess:

                    try:
                        center = _retreive_center(filename, old_df)
                    except IndexError:
                        pass

                # use last center
                elif use_last_center:

                    center = _get_last_center(num, tolerance, indices)

                # only perform hough transform if no previous center guess is useful
                if center is None:
                    #                #perform hough transformation
                    circ = find_rings(
                        img,
                        hminr,
                        hmaxr,
                        10,
                        scale_factor=scale,
                        max_peaks=hough_rings,
                        mask=mask,
                        filename=filename + "_Hough_result.png",
                        show=show,
                        hough_filter=filter_for_hough,
                        canny_use_quantiles=canny_use_quantiles,
                        canny_sigma=canny_sigma,
                    )

                    # get circle center in numpy coordinates, originating from the top left corner
                    center = get_mean(circ)

                # center finding
                fit_dict, ring_df, polar_transformed = optimize_center(
                    img,
                    center,
                    max_iter=10,
                    max_tries=2,
                    tolerance=tolerance,
                    file=filename,
                    show_all=False,
                    mask=mask,
                    value_precision=ring_deviation,
                    int_thres=int_thres,
                    median_for_polar=median_for_polar,
                    clahe_for_polar=clahe_for_polar,
                    jacobian_for_polar=jacobian_for_polar,
                    dr_polar=dr_polar,
                    dt_polar=dt_polar,
                    radius_boundaries=(minr, maxr),
                    show_ellipse=show,
                    show_select_values=show,
                    local_intensity_scope=True,
                    vals_per_ring=min_vals_per_ring,
                    skip_ring=skip_ring,
                    fit_gauss=cwt_based_pixel_extraction,
                    only_one_polar_transform=False,
                )

                # correct distortions for final ring data
                fit_dict = correct_distortions(fit_dict, ring_df, show=show, skip_ring=skip_ring)

                ##use fitted distortions on img to correct all rings at once:
                # retreive parameters:
                funcparams = _retreive_fit_parameters(fit_dict)
                norm_factor = funcparams[0]  # radius

                # rescaling
                polar_transformed["polar"] = reshape_image_by_function(
                    polar_transformed["polar"], polar_dist4th_order, norm_factor / dr_polar, show, filename, dr_polar, *funcparams
                )

                # overwrite up based on center
                # low, up = int(min(img.shape)/8), int(min(img.shape)/2)
                low = int(min(img.shape) / 9)
                up = int(min([fit_dict["Center x"], fit_dict["Center y"], img.shape[1] - fit_dict["Center x"], img.shape[0] - fit_dict["Center y"]]))

                # display distortion corrected result
                if show:

                    fig, ax = plt.subplots(dpi=150)
                    ax.imshow(polar_transformed["polar"] ** (1 / 3), cmap="gray")
                    ax.set(xlabel="Angular increment / a. u.", ylabel=f"Radius / {dr_polar} px", title="corrected")
                    # ax.hlines(norm_factor, 0, polar_transformed['polar'].shape[1], colors = 'red')
                    plt.show()
                    fig.savefig(f"{filename}_Distortion_corrected_polar.png", bbox_inches="tight")
                    fig.clf()
                    plt.close("all")
                # limit data range

                # fit_gaussian2peaks
                __, rrange, intensity = find_average_peaks(
                    polar_transformed["polar"],
                    polar_transformed["r_grid"],
                    polar_transformed["theta_grid"],
                    peak_widths=[3],
                    dr=dr_polar,
                    show=show,
                    filename=f"{filename}_after_correction",
                    min_radius=low,
                    max_radius=up,
                )

                # store corrected spectrum
                corrected_spectra.append(intensity)

                distortion_fit_radii = [fit_dict[key] for key in fit_dict if re.search("distortion correction \d+ radius / px", key)]

                peak_fits, popt_old = model_full_spectrum(
                    rrange,
                    intensity,
                    distortion_fit_radii,
                    low,
                    up,
                    peak_num,
                    background_voigt_num,
                    zolz_guesses,
                    filename,
                    full_spectrum_fit,
                    double_peak,
                    popt_old,
                    reuse_last_fit_params=True,
                    show=True,
                )

                # add peaks to indices
                indices[num] = {**fit_dict, **peak_fits}

                ##get metadata:
                if file.endswith(".emi"):

                    cal_x, time, illuArea, z, c2, c3 = get_emi_metadata(file)

                    # create dummy variables for code compatibility
                    conv = x = y = None

                elif file.endswith(".emd"):

                    cal_x, time, conv, x, y, z, c2, c3 = get_emd_metadata(file)

                    # create dummy variables for code compatibility
                    illuArea = None

                else:
                    # create dummy variables for code compatibility
                    cal_x = time = conv = z = y = x = c2 = c3 = illuArea = None

                # save metadata
                indices[num]["cal_x"] = cal_x
                indices[num]["time / s"] = time
                indices[num]["conv"] = conv
                indices[num]["z height"] = z
                indices[num]["x position"] = x
                indices[num]["y position"] = y
                indices[num]["C2"] = c2
                indices[num]["C3"] = c3
                indices[num]["illuArea"] = illuArea
                try:
                    indices[num]["C2/C3"] = c2 / c3

                except TypeError:

                    indices[num]["C2/C3"] = None
                    logging.warning(f'TypeError for {filename} during "C2/C3" assignment.')

            # log error, save and terminate
            except Exception as e:

                # log
                message = "{} in file {}.".format(type(e).__name__, filename)
                logging.critical(message)
                logging.critical(traceback.format_exc())

                #
                #                savename = f'{dfname}_aborted_during_{filename}'
                #                df = pd.DataFrame.from_dict(indices, orient='index')
                #
                #                #save data
                #                df2excel(df, dfname)
                #
                #                ##create summary
                #                #get all distortion radii and errors
                #                summary = [col for col in df.columns if re.search('distortion correction \d radius', col)]
                #                summary.append('Center Displacement / px')
                #                summary.append('File')
                #                df2excel(df[summary], savename + '_summary')
                #
                # terminate
                #               raise e
                # print('!!!!!!!!!!!', message, '!!!!!!!!!!!')
                num += 1

                raise e
            #     continue

            num += 1

    # create a DataFrame
    logging.info("Writing DataFrame")
    df = pd.DataFrame.from_dict(indices, orient="index")

    # print('Execution succesful')

    # ensure that no unnecessary file ending is created:
    dfname = dfname.split(".xls")[0]

    df2excel(df, dfname)
    # write spectra to npz file:
    np.savez_compressed(dfname, *corrected_spectra)

    logging.info("Finished")
    ##create summary
    # get all distortion radii and errors
    try:
        summary = [col for col in df.columns if re.search("distortion correction \d radius", col)]
        summary.append("Center Displacement / px")
        summary.append("File")
        df2excel(df[summary], dfname + "_summary")
    except Exception as e:

        logging.info(f"An error occured while creating the summary file. The summary columns are {summary}.")
        message = "{} in file.".format(type(e).__name__)
        logging.critical(message)
        logging.critical(traceback.format_exc())
        raise e

    return df


# %%

filename = "Alignment Test"  # Name of the related Excel sheet.
#
###
for handler in logging.root.handlers[:]:

    logging.root.removeHandler(handler)

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
    filename="{} {}.log".format(reduce(lambda x, y: x + y, str(datetime.datetime.now()).split(".")[0].split(":")), filename),
)


radius_range = 700 / 2048, 850 / 2048  # Relative radius range for 220
peak_num = 4
df = main(
    show=False,  # True: Processing steps are shown, slows algorithm down
    show_test=False,
    test=False,
    test_interval=100,
    tolerance=0.05,  # Precision of Center finding in px. The smaller, the longer it takes. I remain in [0.01, 0.1]
    radius_range=radius_range,  # see above
    int_thres="0.05 median",  # For calculating the intensity threshold during pixel detection. If not enough pixels are found, try "mean". If that doesn't work, try numbers between 0 and 1.
    dfname=filename,  # see above
    hough_radius_range=(0.2, 0.5),  # Relative Radius range for Hough transform. It tells, which radii to consider for automated center finding
    min_vals_per_ring=20,  # Minimum spots per Ring. I hope you don't need to reduce it...
    hough_rings=1,  # Number of Hough rings to find. If first center guess is bad, increase to 2 or 3.
    peak_num=peak_num,  # Number of Voigt Peaks
    background_voigt_num=3,  # Number of Background Voigt Peaks. 3 for vacuum worked fine. 5 for Liquid. In gas... we will see.
    scale=10,
    zolz_guesses=(2, 2, 0),
    full_spectrum_fit=True,
    use_last_center=False,
    reverse=True,
)

# %%
alignment_check(df, peak_criterion=["Distortions", "Full Fit"], number_of_peaks=peak_num, show=True)
