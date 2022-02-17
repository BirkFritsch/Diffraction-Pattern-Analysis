# -*- coding: utf-8 -*-
"""
'This is without a doupt the worst code I've ever run
-- But it runs' -- https://i.redd.it/6b7und8rs1v21.png

This code is designed to run on diffraction patterns comprising continouos
diffraction rings in fcc geometry with {222} as largest, fully captured
diffraction signal. It has been developed for patterns with a resolution of
1024x1024 px², 2048x2048 px², and 4096x4096 px², respectively, but may work on
different pattern geometries, as well. A description of the general workflow
can be found in the publication referenced below.

If you find this script helpful, please cite our work:
    
Birk Fritsch, Mingjian Wu, Andreas Hutzler, Dan Zhou, Ronald Spruit, Lilian Vogl,
Johannes Will, H. Hugo Pérez Garza, Martin März, Michael P.M. Jank, Erdmann Spiecker:
"Sub-Kelvin thermometry for evaluating the local temperature stability within in situ TEM gas cells",
Ultramicroscopy, 2022, 113494, https://doi.org/10.1016/j.ultramic.2022.113494
 

@Dependencies:
    python 3.7.10
    numpy 1.20.3
    matplotlib 3.4.2
    pyabel 0.8.3
    hyperspy 1.6.3
    scikit-image 0.18.1
    scipy 1.6.3
    pandas 1.2.4
    xlsxwriter 1.4.3
    openpyxl 3.0.7

@author: Birk Fritsch
"""

import logging
import traceback
import re
import os
import numpy as np
import matplotlib.pyplot as plt
import abel.tools as at
import hyperspy.api as hs
from skimage.transform import hough_circle, hough_circle_peaks, rescale
from skimage.morphology import disk
from skimage.feature import canny
from skimage.filters import median
from skimage.exposure import equalize_adapthist
from scipy.signal import find_peaks_cwt
from scipy.optimize import curve_fit
from scipy.special import wofz
import pandas as pd
import datetime
from functools import reduce
from itertools import zip_longest, tee

#%%
class LSqEllipse:
    """Demonstration of least-squares fitting of ellipses
    
    From Ben Hammel and Nick Sullivan-Molina:
        https://zenodo.org/record/2578663 
        doi:10.5281/zenodo.2578663
    
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

        #Quadratic part of design matrix [eqn. 15] from (*)
        D1 = np.mat(np.vstack([x**2, x*y, y**2])).T
        #Linear part of design matrix [eqn. 16] from (*)
        D2 = np.mat(np.vstack([x, y, np.ones(len(x))])).T
        
        #forming scatter matrix [eqn. 17] from (*)
        S1 = D1.T*D1
        S2 = D1.T*D2
        S3 = D2.T*D2  
        
        #Constraint matrix [eqn. 18]
        C1 = np.mat('0. 0. 2.; 0. -1. 0.; 2. 0. 0.')

        #Reduced scatter matrix [eqn. 29]
        M=C1.I*(S1-S2*S3.I*S2.T)

        #M*|a b c >=l|a b c >. Find eigenvalues and eigenvectors from this equation [eqn. 28]
        eval, evec = np.linalg.eig(M) 

        # eigenvector must meet constraint 4ac - b^2 to be valid.
        cond = 4*np.multiply(evec[0, :], evec[2, :]) - np.power(evec[1, :], 2)
        a1 = evec[:, np.nonzero(cond.A > 0)[1]]
        
        #|d f g> = -S3^(-1)*S2^(T)*|a b c> [eqn. 24]
        a2 = -S3.I*S2.T*a1
        
        # eigenvectors |a b c d f g> 
        self.coef = np.vstack([a1, a2])
        self._save_parameters()
            
        
    def _save_parameters(self):
        """finds the important parameters of the fitted ellipse
        
        Theory taken form http://mathworld.wolfram

        Argso
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

        #eigenvectors are the coefficients of an ellipse in general form
        #a*x^2 + 2*b*x*y + c*y^2 + 2*d*x + 2*f*y + g = 0 [eqn. 15) from (**) or (***)
        a = self.coef[0,0]
        b = self.coef[1,0]/2.
        c = self.coef[2,0]
        d = self.coef[3,0]/2.
        f = self.coef[4,0]/2.
        g = self.coef[5,0]
        
        #finding center of ellipse [eqn.19 and 20] from (**)
        x0 = (c*d-b*f)/(b**2.-a*c)
        y0 = (a*f-b*d)/(b**2.-a*c)
        
        #Find the semi-axes lengths [eqn. 21 and 22] from (**)
        numerator = 2*(a*f*f+c*d*d+g*b*b-2*b*d*f-a*c*g)
        denominator1 = (b*b-a*c)*( (c-a)*np.sqrt(1+4*b*b/((a-c)*(a-c)))-(c+a))
        denominator2 = (b*b-a*c)*( (a-c)*np.sqrt(1+4*b*b/((a-c)*(a-c)))-(c+a))
        width = np.sqrt(numerator/denominator1)
        height = np.sqrt(numerator/denominator2)

        # angle of counterclockwise rotation of major-axis of ellipse to x-axis [eqn. 23] from (**)
        # or [eqn. 26] from (***).
        phi = .5*np.arctan((2.*b)/(a-c))

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
    
    

def get_emi_metadata(file):  
    ''' 
    get metadata from the emi file using HyperSpy
    input:      TIA emi file
    output:     pixelsize (unit: nm-1), 
                capture time (time stamp, inteval in second),
                illumation area (unit: um), 
                stage z height (unit um), 
                C2 lens (% of power), 
                C3 lens (% of power)
    '''
    #try:
    #    emi_file = hs.load(file)
    #except MemoryError:
    emi_file = hs.load(file, lazy = True)
        
    cal_x    = emi_file.original_metadata.ser_header_parameters.CalibrationDeltaX
    time     = emi_file.original_metadata.ser_header_parameters.Time
    illuArea = emi_file.original_metadata.ObjectInfo.ExperimentalDescription.Illuminated_Area_Diameter_um
    z        = emi_file.original_metadata.ObjectInfo.ExperimentalDescription.Stage_Z_um
    C2       = emi_file.original_metadata.ObjectInfo.ExperimentalDescription.C2_lens_
    C3       = emi_file.original_metadata.ObjectInfo.ExperimentalDescription.C3_lens_
    
    
    return cal_x, time, illuArea, z, C2, C3



def get_emd_metadata(file):  
    ''' 
    get relevant metadata from the Velox emd file using HyperSpy
    input:      Velox emd file
    output:     pixelsize (unit: nm-1), 
                capture time (time stamp, inteval in second),
                indicated beam convergence (unit: mrad), 
                stage z height (unit um), 
                C2 lens (% of power), 
                C3 lens (% of power)
    '''

    s = hs.load(file, lazy = True)
        
    cal_x    = float(s.original_metadata.BinaryResult.PixelSize.width) * 1e-9
    time     = float(s.original_metadata.Acquisition.AcquisitionStartDatetime.DateTime)
    Conv     = float(s.original_metadata.Optics.BeamConvergence) * 1e3
    Stage_z  = float(s.original_metadata.Stage.Position.z) * 1e6
    Stage_x  = float(s.original_metadata.Stage.Position.x) * 1e6
    Stage_y  = float(s.original_metadata.Stage.Position.y) * 1e6
    C2       = float(s.original_metadata.Optics.C2LensIntensity)
    C3       = float(s.original_metadata.Optics.C3LensIntensity)
    return cal_x, time, Conv, Stage_x, Stage_y, Stage_z, C2, C3



def get_dm_metadata(file):
    ''' 
    get metadata from the emi file using HyperSpy
    input:      dm dm4 file
    output:     pixelsize (Dummy value: 0), 
                capture time (Dummy value: 0),
                illumation area Dummy value: 0), 
                stage z height (unit um), 
                C2 lens (Dummy value: 1), 
                C3 lens (Dummy value: 1Dummy value: 1)
    '''

    dm_file = hs.load(file, lazy = True)
        
    cal_x    = 0 #dm_file.original_metadata.ser_header_parameters.CalibrationDeltaX
    time     = 0 # dm_file.original_metadata.ser_header_parameters.Time
    illuArea = 0 # dm_file.original_metadata.ObjectInfo.ExperimentalDescription.Illuminated_Area_Diameter_um
    z        = dm_file.metadata.Acquisition_instrument.TEM.Stage.z *1000 # unit was mm in metadata
    C2       = 1 # dm_file.original_metadata.ObjectInfo.ExperimentalDescription.C2_lens_
    C3       = 1 #dm_file.original_metadata.ObjectInfo.ExperimentalDescription.C3_lens_
    
    
    return cal_x, time, illuArea, z, C2, C3



def ensure_folder(directory):
    '''
    Creates a path if it does not exsist already. Else: it does nothing.
    Inspired from https://gist.github.com/keithweaver/562d3caa8650eefe7f84fa074e9ca949
    
    input: Directory as list or tuple
    '''
    
    directory = os.path.join(*directory)
    
    try:
        
        if not os.path.exists(directory):
            
            os.makedirs(directory)
            
    except OSError:
        
        logging.Error('Error: Creating directory. {}'.format(directory))



def select_files(file_list, pattern = 'Ceta \d+.tif'):
    '''
    Input: List of strings, pattern = re string
    
    Return: List of strings matching the given re pattern 
    '''
    
    return [file for file in file_list if bool(re.search(pattern,file))]

            
            
def get_cwd(directory):
    """
    Changes cwd to directory using os
    
    Input: Directory as list or tuple
    """
    directory = os.path.join(*directory)
    
    if os.getcwd() != directory:
        os.chdir(directory)
           
            
        
def load_file(filename, directory = None):
    '''
    input: Filename as string; show: bool, if truthy, the picture, its filename and shape are displayed.
    
    output: file as hyperspy signal
    '''
    
    if directory is not None:
        get_cwd(directory)
    
    #import image
    img = hs.load(filename, lazy = True)
        
    return img



def create_timestamp():
    """
    Creates a timestamp as string
    """    
    
    timestamp = str(datetime.now()).split('.')[0].split(':')

    return reduce(lambda x,y: x+y, timestamp)



def prepare_matrix(img):
    """
    Input: img as np.array
    
    Return: np.array
    """
    
     #normalize array to [-1:1]:
    if img.max() >= np.abs(img.min()):
        img = img / img.max()
    
    else:
        img = img / np.abs(img.min())
    
    
    return img



def image4hough(img, scale_factor, mask = None, show = True, hough_filter = True,
                **kwargs):
    '''
    use median filter on a CLAHE-prepared img, apply a canny algo and rescale by scale_factor
    kwargs are passed to canny
    '''
    
    if mask is not None:
        
        img = img * mask
        

    if hough_filter:
        img_filtered = median(equalize_adapthist(img, clip_limit =1.0), disk(3))
        
    else:
        img_filtered = img
    
    #rescale image to safe memory. Results will be rescaled
    img_filtered = rescale(img_filtered, 1/scale_factor, anti_aliasing=True,
                           multichannel=False, mode='constant')
    
    #use canny algorithm to find edges.
    img_filtered = canny(img_filtered, **kwargs)
    
    if show:
        f, ax = plt.subplots(figsize=(5,5))
        
        ax.set(xlabel=f'x / {scale_factor} px', ylabel=f'y / {scale_factor} px')
        ax.imshow(img_filtered, cmap = 'binary')
        ax.tick_params(direction='in', which='both')
        plt.show()
        f.savefig('Canny Reslut for Hough-Transform.png', bbox_inches='tight', dpi=300)
        f.clf()
        plt.close('all') 
          
    return img_filtered

    

def find_rings(img,
               m,
               n,
               i,
               mask=None,
               scale_factor=4,
               max_peaks=5,
               filename = 'test.tif',
               show= True, old_center = None, hough_filter = True,
               canny_use_quantiles = True, canny_sigma = 1):
    '''
    finds circles in diffraction pattern based on hough transformation.
      
    INPUT: 
        
    img (np.array) = pattern as 2D array     
    
    m (int) = lower radius boundary
    
    n (int) = upper radius boundary
    
    i (int) = number of radii between m and n
    
    kwargs:
    
    mask (None,np.array): Array of ones except a region with np.nan. If none
    (default), mask is ignored
        
    scale_factor (num) = downsampling factor for images to save memory.
    Furthermore, a scale factor between 4 - 9 seems to yield the best results.
    This might be due to refiltering with a gaussian filter...
    
    max_peaks (int) = number of circles to be found
    
    filename = Name of the tif image file to process
    
    show (bool) = Display results on original image
    
    
    RETURN:
        
    np.array containing fitresults with the columns (x,y,radius)
    '''
    
    #save a copy of img and process it for Hough Transform
    img_filtered = image4hough(img, scale_factor, mask= mask, show = show,
                               hough_filter = hough_filter,
                               use_quantiles =canny_use_quantiles,
                               sigma = canny_sigma)    

    
    #find circles and scale values to filtered image   
    hough_radii = np.arange(m/scale_factor,n/scale_factor,i)
    hough_res = hough_circle(img_filtered, hough_radii)
    # Select the most prominent circles (max_peaks)
    accums, cx, cy, radii = hough_circle_peaks(hough_res, hough_radii,
                                               total_num_peaks=max_peaks) 
    #rescale
    cx = (cx * scale_factor)
    cy = (cy * scale_factor)
    radii = (radii * scale_factor)

    #store results in numpy array
    out = np.array(list(zip(cx, cy, radii)))
    
    if show:
        # Draw circles
        fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(5, 5))
        
        for x, y, r in out:
            #draw circle center
            ax.plot([x],[y], marker='X', markersize = 5, )
            #draw circle
            circle = plt.Circle((x,y), r, fill=False, color = 'deeppink')
            ax.add_artist(circle)
            
        #draw original image with false colors
        ax.imshow(img,  cmap = 'binary')
        ax.tick_params(direction='in', which = 'both')
        ax.set(xlabel='x / px', ylabel='y / px')
        #save results
        fig.savefig(filename, bbox_inches = 'tight', dpi = 300)
        
        #output result
        plt.show()
        fig.clf()
        plt.close()
    
    return out



def get_mean(circs):
    '''
    Input: np.array containing circle coordinates
    
    Return: Tuple of averaged x, y circle center
    '''
    
    xc = circs[...,0].mean()
    
    yc = circs[...,1].mean()
    
    return (xc,yc)



def img2polar(img, center,
              show = True,
              clahe = True,
              median_filter = True,
              jacobian = True,
              dr = 1,
              dt = None,
              filename = 'test.png'):
    '''
    Wrapper around at.polar.reproject_image_into_polar

    Input:
        img, image as 2D array
        center, tuple of coordinates around which the polar transform is performed.
    
    Return:
        polar, 2D array, polar-transformed of img
        r_grid, theta_grid: 2D arrays mapping the coordinates of polar with radial and theta increments
    
    Optional:
        show (bool): If truthy (default), the result is displayed and saved as "filename".
        clahe (bool): If truthy (default), CLAHE filtering is performed to img.
        median_filter (bool): If truthy (default), a median filter is performed to img.
        jacobian (bool): passed to at.polar.reproject_image_into_polar. Default is True.
        dr (float): passed to at.polar.reproject_image_into_polar. Default is 1.
        dt (float or None): passed to at.polar.reproject_image_into_polar. Default is None.
        filename (str): Storage name. Ignored if show is falsy. Default is "test.png".    
    '''

    #pyabel assumes a coordinate system starting from the bottom left corner.
    #Thus, center has to be transformed
    abel_center = center[0], img.shape[0] - center[1]
    
    #Perform CLAHE and noise filtering, then transform to polar coordinates
    if clahe:
        
        img = equalize_adapthist(img)
        
    if median_filter:
        
        img = median(img)
    
    polar, r_grid, theta_grid = at.polar.reproject_image_into_polar(img,
                                                                    abel_center,
                                                                    Jacobian=jacobian,
                                                                    dr=dr,
                                                                    dt = dt)
    
    if show:
        
        fig, ax = plt.subplots()
        center = round(center[0], 3), round(center[1], 3)
        ax.set(title='Center={}, jac={}, CLAHE={}, median={}'.format(center,
                                                                                 jacobian,
                                                                                 clahe,
                                                                                 median_filter))
        ax.imshow(polar, cmap = 'binary')
        ax.tick_params(direction='in', which = 'both')
        fig.savefig(filename, bbox_inches = 'tight', dpi = 300)
        
        plt.show()
        plt.clf()
        plt.close()
        
        
    return polar, r_grid, theta_grid



def find_average_peaks(polar,
                       r_grid,
                       theta_grid,
                       peak_widths=[20],
                       dr=1,
                       min_radius=None,
                       max_radius=None,
                       show=True,
                       filename='test.png'):
    '''
    custom variation of abel.tools.vmi.average_radial_intensity plus peak fitting function
    
    min_radius (int or None): If truthy (int): Only peaks above the given radius are taken into account
    '''
    dt = theta_grid[0,1] - theta_grid[0,0]
    polar = polar * r_grid * np.abs(np.sin(theta_grid))
    intensity = np.trapz(polar, axis=1, dx=dt)
    
    #get fitting r:
    r = r_grid[:intensity.shape[0],0]

    #find peaks
    indexes_real = find_peaks_cwt(intensity,peak_widths)
    indexes = [int(round(idx*dr)) for idx in indexes_real]
    
    logging.debug('find_average_peaks --> find_peaks_cwt returned indexes: {}'.format(indexes))
    
    #filter peak if min_radius is given
    if min_radius:
        
        
        indexes = [i for i in indexes if i >= min_radius]
        
        logging.debug('find_average_peaks --> Indexes after min_radius ({}) filtering: {}'.format(min_radius, indexes))
    
    #filter peak if max_radius is given
    if max_radius:
        
        
        indexes = [i for i in indexes if i <= max_radius]
    
        logging.debug('find_average_peaks --> Indexes after max_radius ({}) filtering: {}'.format(max_radius, indexes))
    
    
    if show:
        
        if min_radius:
            indexes_real = [i for i in indexes_real if i >= min_radius/dr]
        if max_radius:
            indexes_real = [i for i in indexes_real if i <= max_radius/dr]        
        logging.debug('find_average_peaks --> plot results')
        
        fig, ax = plt.subplots(figsize=(5,5))
        
        ax.plot(r, intensity, label = 'Data')
        ax.plot(np.asarray(indexes_real)*dr,
                intensity[indexes_real],
                'x',
                markersize=8,
                label = '{} Peaks found'.format(len(indexes))
                )
        ax.set_title('Integrated Intensity with peak detection')
        ax.tick_params(direction='in', which = 'both')
        ax.set(xlabel = 'Radius / px', ylabel = 'Angular-integrated intensity / a. u.')
        
        for i in indexes_real:
            
            ax.annotate(str(int(round(i*dr))), (int(i*dr),intensity[i]))
        
        ax.legend(loc = 0)
        
        #save results
        fig.savefig(filename, bbox_inches = 'tight', dpi = 300)
        plt.show()
        fig.clf() 
        plt.close()
        
        
    return indexes, r, intensity



def gaussian_FWHM_Func(xdat, y0, xc, A, w):
    '''
    FWHM Version of Gaussian:
    
    y0 = base, xc = center, A = area, w = FWHM 
        
        
    yc = y0 + A/(w*np.sqrt(np.pi/(4*np.log(2))))
    '''
    
    denominator = A*np.exp( (-4*np.log(2)*(xdat-xc)**2) / w**2 )
    
    numerator = w*np.sqrt(np.pi/(4*np.log(2)))
    
    return y0 + denominator / numerator



def lorentzian_func(xdat, xc, A, w, y0):
    '''
    FWHM Version of Lorentzian:
    
    y0 = base, xc = center, A = area, w = FWHM       
        
    yc = y0 + A/(w*np.sqrt(np.pi/(4*np.log(2))))
    
    Height of the Curve (yc - y0); H = 2 * A / (PI * w) 
    '''
    
    denominator = 2*A*w
    
    enumerator = np.pi * 4 * (xdat - xc)**2 + w**2
    
    return y0 + denominator / enumerator



def get_errors(p_opt, p_cov, x_dat, y_dat, y_fit):
    
    """
    create Rsqaured and errors out of fitted data.
    Arguments: p_opt, p_cov, x_dat, y_dat, y_fit
    """
    
    # Get standard errors:
    errors = np.sqrt(np.diag(p_cov))
    
    r_squared = calc_r2(y_dat, y_fit)
    
    # Corrected R squared
    r_squared_corr = 1 - (1 - r_squared) * (x_dat.shape[0]-1) / (x_dat.shape[0]-len(p_opt))
        
    return errors, r_squared_corr



def calc_r2(y, f):
    """
    Corrected R squared
    """
    sstot = np.sum( (y - y.mean() )**2 )
    ssres = np.sum( (y - f)**2 )

    return 1 - ssres/sstot



def df2excel(df_fitresults,  name = 'test', newfile = True):
    '''
    Takes a DataFrame (df_fitresults) and saves it under the given 'name' in
    the cwd.
    If newfile is truthy (default), then the exported name will have a timestamp
    so that every run creates a new file.
    '''
    
    #create saving directory:
    curr_dir = os.getcwd()

    output_dir = ['.','output']
    ensure_folder(output_dir)

    os.chdir(os.path.join(*output_dir))

    if newfile:
        #creates a timestamp if newfile is truthy
        timestamp = str(datetime.datetime.now()).split('.')[0].split(':')
    else:
        timestamp = ['']
        
    #create file name as folder of cwd and with moment of execution (if newfile is truthy)
    export_name = reduce(lambda x,y: x+y, timestamp) + ' ' + name + '.xlsx'
    #Create a Writer Excel object
    writer = pd.ExcelWriter(export_name, engine='xlsxwriter')
    #Convert the df to the Writer object
    df_fitresults.to_excel(writer)
    # Close the Pandas Excel writer and output the Excel file.
    writer.save()
    
    logging.info(f'Saved {export_name}.')
    
    #restore curr_dir
    os.chdir(curr_dir)
    
    

def fit_gaussian2peaks(peaks, r, intensity, val_range = 40, show = True, dr = 1,
                       filename = 'test.png', functype = 'gauss'):
    '''
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
            
    '''
    if show:
        
        fig, ax = plt.subplots(figsize=(5,5))
        ax.plot(r, intensity, label = 'Full data range')
        ax.set(ylabel = 'Angular-integrated intensity / a. u.',
               xlabel = 'Radius / px',)
        ax.tick_params(direction='in', which = 'both')
    
    out = {}
    #iterate over all peaks
    
    if functype.lower() == 'gauss':
        
        func = gaussian_FWHM_Func
    
    elif functype.lower() == 'lorentz':
        
        func = lorentzian_func
        
    elif functype.lower() == 'voigt':
        
        func = voigt
        
    else:
        
        func = gaussian_FWHM_Func
        logging.warning(f'Cannot regognize function in fit_gaussian2peaks for {filename}. Default to {func}.')
    
    logging.debug('fit_gaussian2peaks --> enter loop over peak estimations {}'.format(peaks))
    
    val_range = int(round(val_range/dr))
    peaks = sorted(peaks)
    for num, pk in enumerate(peaks):
        
        i = int(round(pk/dr))
        
        if pk > r.shape[0]:
            print(f'WARNING: {pk} not in radial range')
            continue
        
        for tries in range(1,5):
        
            interval = val_range * tries   
        
            low, up = int(i) - interval//2, int(i) + interval//2
            x = r[low:up]
            y = intensity[low:up]
           
            logging.debug('fit_gaussian2peaks --> Prepare fitting of peak around {}'.format(i))
    
            ##fit gaussian:
            #find starting values:
            if functype == 'voigt':
                xc_start = pk
                w_start = interval / 8
                g_start = interval / 8
                A_start = np.trapz(y)#interval/2 * y.max()
                p0 = xc_start, A_start, w_start, g_start
                
                if num == 0:
                    lb_xc = 0
                else:
                    lb_xc = peaks[num-1]
                
                try:
                    ub_xc = peaks[num+1]
                except IndexError:
                    ub_xc = x.max()
                    
                #ub_wg = x.max() - x.min()
                boundaries = [[lb_xc,0,0,0],
                          [ub_xc,np.inf,np.inf,np.inf]]
            
            else:
                y0 = 0
                xc_start = pk
                w_start = interval / 2
                A_start = w_start * max(y)
                p0 = y0, xc_start, A_start, w_start 
                boundaries = 0, np.inf
            #perform fitting
            try:
                popt, pcov = curve_fit(func, x, y, maxfev = 10000000,
                                       bounds=boundaries,
                                       p0 = p0)
                
                yfit = func(r,*popt)
                
                errors, r2adj = get_errors(popt, pcov, x, y, func(x, *popt))
                
            except ValueError as e:
                print(sorted(peaks))
                print(num,p0)
                print(boundaries)
                raise e
            
            except RuntimeError: 
                
                popt = [np.nan, np.nan, np.nan, np.nan]
                errors = [np.inf, np.inf, np.inf, np.inf]
                r2adj = 0
                yfit = func(r,*p0)

                logging.info('RuntimeError: Peak {} around {} could not be estimated.'.format(pk, (low, up)))
                
            
            key = num
            
            if functype == 'voigt':
                xc = popt[0]
                xc_err = errors[0]
            else:
                xc = popt[1]
                xc_err = errors[1]
            
            if (low*dr <= xc - xc_err) and (up*dr >= xc + xc_err):
                
                logging.debug('fit_gaussian2peaks --> fitting of peak around {} succesful after {} tries. Optimized values are {}. Errors are {}. Boundaries are {}'.format(i, tries, popt, errors, (low,up)))
                
                break
            
            else:
                
                logging.debug('fit_gaussian2peaks --> fitting of peak around {} not succesful after {} tries. Optimized values are {}. Errors are {}. Boundaries are {}'.format(i, tries, popt, errors, (low,up)))

        
        logging.debug('fit_gaussian2peaks --> Store values to out')
        
        if functype == 'voigt':
            out['xc {}'.format(key)] = popt[0]
            out['A {}'.format(key)] = popt[1]
            out['sigma {}'.format(key)] = popt[2]
            out['gamma {}'.format(key)] = popt[3]
            out['xc_error {}'.format(key)] = errors[0]
            out['A_error {}'.format(key)] = errors[1]
            out['sigma_error {}'.format(key)] = errors[2]
            out['gamma_error {}'.format(key)] = errors[3]
        else:
            out['y0 {}'.format(key)] = popt[0]
            out['xc {}'.format(key)] = popt[1]
            out['A {}'.format(key)] = popt[2]
            out['FWHM {}'.format(key)] = popt[3]
            out['y0_error {}'.format(key)] = errors[0]
            out['xc_error {}'.format(key)] = errors[1]
            out['A_error {}'.format(key)] = errors[2]
            out['FWHM_error {}'.format(key)] = errors[3]
            
        out['R2adj {}'.format(key)] = r2adj
        
        if show:
            
            logging.debug('fit_gaussian2peaks --> create overview image of plot function')
            
            ax.plot(x,y, linewidth=1, label = f'Selected data around {int(round(xc))} Iteration {tries}')
            ax.plot(r, yfit, ':', linewidth = 3, label = f'{functype.capitalize()} fit around {int(round(xc))}')
            
            ax.annotate(str(round(xc,2)),(xc,intensity[i]))
    
    logging.debug('fit_gaussian2peaks --> exit loop over peaks')
    
    if show:    
        
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        #save results
        try:
            fig.savefig(filename, bbox_inches = 'tight', dpi = 300)
        except Exception as e:
            logging.error(type(e).__name__)
            logging.error(traceback.format_exc())
            logging.error('Cannot save ' + filename)
        plt.show()
        fig.clf()
        plt.close()
        
        
    return out



def model_background(dct,
                     radius,
                     intens,
                     background_voigt_num=3,
                     show = True,
                     filename = 'test.png',
                     min_radius = 0,
                     max_radius=1000,
                     dr=1,
                     ):
    """
    

    Parameters
    ----------
    dct : TYPE
        DESCRIPTION.
    radius : TYPE
        DESCRIPTION.
    intens : TYPE
        DESCRIPTION.
    background_voigt_num : TYPE, optional
        DESCRIPTION. The default is 4.
    show : TYPE, optional
        DESCRIPTION. The default is True.
    filename : TYPE, optional
        DESCRIPTION. The default is 'test.png'.
    min_radius : TYPE, optional
        DESCRIPTION. The default is 0.
    max_radius : TYPE, optional
        DESCRIPTION. The default is 1000.
    dr : TYPE, optional
        DESCRIPTION. The default is 1.

    Returns
    -------
    None.

    """
    #correct min and max radius
    min_radius = int(round(min_radius/dr))
    max_radius = int(round(max_radius/dr))    
    ##substract fitted peak regions
    
    #retreive peaks
    peaks = [int(peak[3:]) for peak in dct if peak.startswith('xc ')]
    
    #remove region for every peak using conditional selection in 
    fitted_peaks = np.zeros(intens.shape)
    
    for peak in peaks:
        
        #get peak characteristics
        xc = dct[f'xc {peak}']
        area = dct[f'A {peak}']
        sigma = dct[f'sigma {peak}']
        gamma = dct[f'gamma {peak}']
        
        #crop data to region of interest
        if min_radius <= xc <= max_radius:
            fitted_peaks += voigt(radius, xc, area, sigma, gamma)
        
    
    #substract peaks
    residuals = intens - fitted_peaks
    #clip data to 0:
    condition = residuals > 0
    r = radius
    radius = radius[condition]
    residuals = residuals[condition]
    
    
    ##fit data
    #get parameter guesses and boundaries
    #max. boundaries:
    area_max = np.trapz(residuals, radius)
    max_width = (max_radius - min_radius)
    xcs = np.linspace(radius.min(), radius.max() - 0.2*np.mean(radius), background_voigt_num)
    A = area_max / 2
    sigma = max_width / 2
    
    gamma = 0
    
    p0 = []
    for xc in xcs:
        p0.extend((xc, A/3,sigma, gamma))
    
    boundaries = [[
             min_radius,
             0,
             0,
             0,
             ]*background_voigt_num,
             [
             max_radius,
             area_max,
             max_width,
             max_width,
             ]*background_voigt_num]

    
    ##curve fitting
    #pass deviation of residuals from original spectrum as uncertainties:
    uncertainties = np.abs(intens[condition] - residuals)
    
    try:
        popt, pcov = curve_fit(multi_voigt, radius, residuals,
                           p0=p0,
                           bounds=boundaries,
                           sigma=uncertainties,
                           maxfev=int(1e8))
        yfit = multi_voigt(radius, *popt)
        errors, r2adj = get_errors(popt, pcov, radius, residuals, yfit)
    
    except ValueError:
        popt, pcov = curve_fit(multi_voigt, radius, residuals,
                           p0=p0,
                           bounds=(0, np.inf),
                           sigma=uncertainties,
                           maxfev=int(1e8))
        yfit = multi_voigt(radius, *popt)
        errors, r2adj = get_errors(popt, pcov, radius, residuals, yfit)
    
    
    except RuntimeError:
        popt = [np.nan]*len(p0)
        yfit = multi_voigt(radius, *p0)
        errors = [np.inf]*len(p0)
        r2adj = np.nan
    
    #append results to fit_dict
    strt = 'background_voigt'
    dct[f'{strt} R2adj'] = r2adj
    strt_err = strt+'_errors'
    for n, (params,errs) in enumerate(zip(grouper(popt,4,np.nan),grouper(errors,4,np.nan))):
        dct[f'{strt} xc {n}'], dct[f'{strt} A {n}'], dct[f'{strt} sigma {n}'], dct[f'{strt} gamma {n}'] = params
        dct[f'{strt_err} xc {n}'], dct[f'{strt_err} A {n}'], dct[f'{strt_err} sigma {n}'], dct[f'{strt_err} gamma {n}'] = errs
   
    

    #show results    
    if show:
        
        #use guesses in case of failed fit
        if np.isnan(r2adj):
            popt = p0
            
        fig, ax = plt.subplots(figsize=(5,5),dpi=100)
        ax.plot(r, intens, '--', label = 'Full data range')
        ax.errorbar(radius, residuals, uncertainties, lolims=True,
                    label = 'selected data range',zorder=0, mec='gray', mfc='gray')
        ax.plot(radius, yfit, label = 'background fit',zorder=1)
        for n,params in enumerate(grouper(popt, 4, 0)):
            n += 1
            ax.plot(radius, voigt(radius, *params), ':',label = f'Voigt {n}',zorder=1)
            
            
        
        ax.set(ylabel = 'Angular-integrated intensity / a. u.',
               xlabel = 'Radius / px',)
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        ax.tick_params(direction='in', which = 'both')
        plt.show()
        fig.clf()
        plt.close('all')
        
    return dct



def increment2radians(phi, phimax):
    '''
    Calculates the radian value of an angle phi, if the respective rircle is
    incremented phimax times.
    '''
    return phi * 2 * np.pi / phimax    
    


def radians2increment(phi, phimax):
    '''
    Calculates the increment value of an angle phi in radians, if the respective rircle is
    incremented phimax times.
    '''
    return phi * phimax / (2 * np.pi)    
    


def polar2carthesian(r,phi, abel_result = True):
        '''
        INPUT: r, phi (in radians)
        
        RETURN: carthesian coordinates x,y 
        '''
        
        if abel_result:
            
            #shift origin to start lower
            phi += np.pi/2
            
            #change rotation direction

        x=r*np.cos(phi)
        y=r*np.sin(phi)
        
        return x, y



def carthesian2polar(x,y):
        '''
        INPUT: carthesian coordinates x,y
        
        RETURN: r, phi (in radians)
        '''
        
        r = np.hypot(x,y)
        phi = np.arctan2(y/x)
        
        return r, phi
    
    
    
def _get_intensity_threshold(arr, intensity_threshold):
    
    
    if type(intensity_threshold) == str:
        
        if ' ' in intensity_threshold:
        
            #separate string on ' '. Code will fail if more than one space is present
            factor, intensity_threshold = intensity_threshold.split()
            #convert factor to float. Code will fail, if factor is not convertible
            factor = float(factor)
            factor_avg = factor
        
        else:
            
            factor = 1
            factor_avg = 0
        
        if intensity_threshold == 'std':      
            
            thres = arr.mean()-factor*arr.std()
            
        
        elif intensity_threshold == '2std':
            
            thres = arr.mean() - factor*2*arr.std()
            
        elif intensity_threshold =='-std':
            
            thres = arr.mean()+ factor*arr.std()
            
        elif intensity_threshold == '-2std':
            
            thres = arr.mean() + factor*2*arr.std()
            
        elif intensity_threshold == 'mean':
            
            thres =  arr.mean() + factor_avg*arr.mean()
            
        elif intensity_threshold == 'median':
            
            if type(arr) == np.ndarray:
                thres =  np.median(arr) + factor_avg*np.median(arr)
            else:
                thres =  arr.median() + factor_avg*arr.median()
    
    
    elif type(intensity_threshold) == float:
        
        assert 0 <= intensity_threshold <= 1, f'intensity threshold is not between 0 and 1. Value is {intensity_threshold}'
        
        thres = arr.max() * intensity_threshold
    
    
    else:
        
        thres = 0
        logging.warning(f'No intensity filtering could be applied. All values are used for ellipse fitting. intensity_threshold == {intensity_threshold}')
        
    return thres



def _apply_intensity_threshold(df, intensity_threshold):
    
    
    thres = _get_intensity_threshold(df['intensity'], intensity_threshold)
    
    #apply threshold:
    return df[df['intensity'] >= thres]
        


def _apply_intensity_threshold_to_list(out, intensity_threshold, colnum = 3):    
        
        #ensure numpy array
        out_arr = np.asarray(out)
        #get intensity column
        maxvals = out_arr[..., colnum]
        #calculate threshold value
        threshold = _get_intensity_threshold(maxvals, intensity_threshold)
        #filter array
        out_arr = out_arr[maxvals >= threshold]
        
        return out_arr
    
    
    
def _select_values_by_intensity(rrange, r, rlist, minr, intensity_threshold = 'std',
                           filename = 'test', local_intensity_scope = False, dr = 1, show=False,
                           dt = None):
    
    out = []
    
    #iterate over every column in the selected image part
    threshold = _get_intensity_threshold(rrange, intensity_threshold)

    for i in range(rrange.shape[1]):
        
        
        correct_ring = True
        counter = 0
        skipped_val = 0
        
        while correct_ring and counter < rrange.shape[1]:
            
            continue_loop = False
            break_loop = False
            

            counter += 1
            
            sliced = rrange[...,i]
            
            if (sliced < threshold).all():
            
                break
            
            
            sliced_lst = list(sliced)
            sliced_lst.sort(reverse = True)
        

            #get maximum value
            maxval = sliced_lst[skipped_val]
            
            #skip if maximum intensity is minimum intensity
            if maxval == min(sliced):
                
                continue_loop = True
                continue
            
            #get radius coordinate relative to selected image region
            r_maxval = np.where(sliced == maxval)
            
            #skip, if the maximum position is ambiguous
            if len(r_maxval) < 1:
                
                break_loop = True
                break
            
            #check whether value is closest to selected radius:
            rcheck = {}
            for r_i in rlist:
                
                #calculate distance to selected radius
                rcheck[np.sqrt((r_i/dr - r)**2)] = r_i/dr
            
            #if not, go to second 
            if round(rcheck[min(rcheck)]) != round(r):
                
                #print(rcheck[min(rcheck)], type(rcheck[min(rcheck)]))
                #print(r, type(r))
                skipped_val += 1
                
                continue_loop = True
                
                continue
            else:
                
                pass
            
            #break loop
            correct_ring = False
            
                       
        if break_loop:
            
            break
        
        if continue_loop:
            
            continue
        #get position of maximum relative to selected radius region rrange.
        #IMPORTANT: The 0 in peak_dct key is autogenerated and attributes to the first maximum passed to fit_gaussian2peaks
        local_r = r_maxval[0] + minr
        
        #store results   
        out.append([r, i, local_r[0], maxval, np.nan])
         
    #keep results only if threshold is exceeded:
    out_arr = _apply_intensity_threshold_to_list(out, intensity_threshold)
    
    return out_arr




def select_ring_values(polar, rlist, tolerance = 0.1, show = False, intensity_threshold = 'std',
                       filename = 'test', local_intensity_scope = False, dr = 1, dt = None,
                       fit_gauss=True, double_peak=False):
    """
    Input:
    
    polar: polar-transformed image
    rlist: Radius list as iterable, eg. circ[...,2]
    
    kwargs:
    
    tolerance (float): value to define the relative region around the given radius.
    
    show (bool): If truthy, interim results are plotted. Slows algorithm down.
    
    intensity_threshold (string or float):
        
        some valid string values:
        
        'std' --> Only use the brightest values based on a 1 sigma standard deviation
        '2std' --> Only use the brightest values based on a 2 sigma standard deviation               
        'mean' --> Only use the brightest values based on the mean intensity
        'median' --> Only use the brightest values based on the median intensity
        
        Alternatively, a floating point number can be placed prior to the above
        mentioned string, separated by a space. Examples on an array (arr) of values:
            '0.1 mean' --> arr.mean() + 0.1*arr.mean()
        
        Please refer
        if float:
            
            Value x of interval [0,1].
            
    local_intensity_scope (bool):
        
        False (default) --> intensity_threshold is applied to all rings simulatneously
        True  --> intensity_threshold is applied to all rings separately
        
    Return:
        
    pandas.DataFrame containing the resulting values
    """  
    
    logging.debug('Selecting ring values. {} given as rlist'.format(rlist))
    
    
    if type(tolerance) != float:
        
        rlist = [int(np.mean(tolerance))]
    #iterate over preselected radii    
    for r in rlist:
        
        logging.debug(f'Iterate over ring value {r}')
        #select a region based on a given percentage of r.
        #ensure integer values for indexing
        
        if type(tolerance) == float: 
            r = int(round(r/dr))
            val = r * tolerance
            val = int(round(val))
            
            logging.debug(f'Iterate over ring value {r}. Calculated value range as {val} based on dr = {dr} and tolerance = {tolerance}')
            
            #boundary conditions for polar-slicing
            minr, maxr = r-val, r+val
            
        else:
          
            minr, maxr = tolerance
            minr = int(round(minr/dr))
            maxr = int(round(maxr/dr))
            
            logging.debug(f'Peak interval is set manually to {tolerance}.Iterate over ring value {r}.')

        #get region of interest
        rrange = polar[minr:maxr]
        
        if show:
            fig, ax = plt.subplots()
            
            ax.set_title('Radius at {} px'.format(r))
            
            ax.imshow(rrange, cmap = 'binary')
            
            plt.close('all')
            plt.clf()
            
        if len(rrange) == 0:

            logging.debug('rrange is empty for {}'.format(r))
            
            continue
        
        if fit_gauss:
            out = _select_values_by_fit(rrange.T,
                                              r,
                                              rlist,
                                              minr,
                                              intensity_threshold=intensity_threshold,
                                              filename = filename,
                                              local_intensity_scope=local_intensity_scope,
                                              dr = dr,
                                              dt = dt,
                                              show=show,
                                              double_peak=double_peak
                                              )        
        else:
            out =_select_values_by_intensity(rrange,
                                              r,
                                              rlist,
                                              minr,
                                              intensity_threshold=intensity_threshold,
                                              filename = filename,
                                              local_intensity_scope=local_intensity_scope,
                                              dr = dr,
                                              dt = dt,
                                              )
        
        logging.debug('Ring {}, {} values found.'.format(r, len(out)))
        
    #make intensity dataFrame   
    df = pd.DataFrame(out, columns = ['Hough-Radius', 'angle', 'radius', 'intensity', 'radius error'])
    
    #calculate angles in radians for further analyses.
    df['angle / rad'] = df['angle'].apply(lambda x: increment2radians(x, polar.shape[1]))
    #calculate carthesian coordinates:
    df['x'], df['y'] = polar2carthesian(df['radius']*dr,df['angle / rad'])
    
    
    if show:
        
        fig, ax = plt.subplots()
        
        ax.imshow(polar, cmap = 'binary')
        ax.set(ylabel = f'Radius / {dr} px', xlabel = 'Angular increment / a. u.')
        
        for i in set(df['Hough-Radius']):
            
            vls = df[df['Hough-Radius'] == i]
            
            ax.plot(vls['angle'], vls['radius'], '.')
        
        fig.savefig(filename, bbox_inches = 'tight', dpi = 300)
        plt.show()
        plt.clf()
        plt.close()
    
    #rescale radii to fit to carthesian values:
    df['radius'] *= dr 

    
    return df



def voigt( x, xc, A, sigma, gamma):
    """
    Fit a voigt function.
    
    Params:
    x (np.array): Data
    xc: peak shift
    A: area under curve
    sigma: Gaussian standard deviation
    gamma: Lorentzian HWHM
    """
    denom = sigma*np.sqrt(2)
    w = wofz((x-xc + 1.0j*gamma)/denom)
    
    return A*np.real(w)/denom



def multi_voigt(x_data, *params):
    """
    Calculates the sum of a flexible number of voigt functions, depending on *params.
    
    INPUT:
        x_data (array): values for which to calculate the voigt functions.
        *params (iterable): number of parameters for to be passed to voigt().
        The iterable is chunked into blocks of 4 parameters. A repetitive sequence of
        xc0, A0, sigma0, gamma0, xc1, A1, sigma1, gamma1, ... is expected. If 
        len(params) % 4 != 0, the missing parameters are interpreted as 0.
    
    RETURN:
        y (array): Array of the same shape as x_data containing the calculated
    result.
    """
    y = np.zeros(x_data.shape)
    
    for xc0, A0, sigma0, gamma0 in grouper(params, 4, 0):
        y += voigt(x_data, xc0, A0, sigma0, gamma0)
    
    return y
    



def _highest(lst1, lst2):
    
    dct = {y:x for x,y in zip(lst1,lst2)}
    
    highest = max(dct)

    return dct[highest]



def _select_values_by_fit(rrange, r, rlist, minr, intensity_threshold = 'std',
                       filename = 'test', local_intensity_scope = False, dr = 1, dt = None,
                       filo_num = 4, show=True, double_peak=True, fit=False):
    """
    
    """
    
    
    #define a list to store results
    out = []
    
    threshold = _get_intensity_threshold(rrange, intensity_threshold)
    
    if double_peak:
        #integrate sliced region
        double_peak_profile = np.trapz(rrange.T)
        #find peaks
        averaged_peaks = np.asarray(find_peaks_cwt(double_peak_profile, [10], noise_perc=10, min_snr = 1))
        #clip to two highest peaks
        p_averaged_1 = _highest(averaged_peaks, double_peak_profile[averaged_peaks])
        averaged_peaks_2 = averaged_peaks[averaged_peaks != p_averaged_1]
        p_averaged_2 = _highest(averaged_peaks_2, double_peak_profile[averaged_peaks_2])
        averaged_peaks = np.array([p_averaged_1, p_averaged_2])
        
        
        if show:
            fig, ax = plt.subplots()
            ax.plot(double_peak_profile, label='Integrated rrange')
            ax.plot(averaged_peaks, double_peak_profile[averaged_peaks], 'x', label='Peaks')
            ax.legend(loc=0)
            ax.set(title='integrated trial')
            plt.show()
            fig.clf()
            plt.close()

    
    #iterate over every column in the selected image part
    for i, sliced in enumerate(filo(rrange, filo_num)):
        
        two_peaks = double_peak
        
        sliced = mean_arr(sliced)
        
        #check for threshold value
        if (sliced < threshold).all():
            continue
        
        #exlude regions where sliced is 0:
        sliced = sliced[sliced > 0][:-1]
        
        #peak detection
        try:
            peaks = find_peaks_cwt(sliced, [10], noise_perc=10, min_snr = 1)
        except ValueError:
            peaks = []

        x,y =  np.arange(len(sliced)), sliced
        try:
            p1 = _highest(peaks, sliced[peaks])
        except Exception as e:
            message = '{} in file {}.'.format(type(e).__name__, filename)
            logging.critical(message)
            logging.critical(traceback.format_exc())
            
            p1 = []
        
        #skip if no peak was detected
        if len(peaks) == 0:
            continue
        #skip if two peaks are assumed but not detected
        elif double_peak:
        
            if len(peaks) < 2:
                two_peaks = False
            
            else:
                #p2 is assumed to be smaller than p1
                p2_peaks = peaks[peaks != p1]
                p2 = _highest(p2_peaks, sliced[p2_peaks])
                
                if sliced[p2] <= 1.25 * sliced.min():
                    two_peaks = False
                
                else:
                
                    both_peaks = np.array((p1, p2))
                    major_peak = closest(both_peaks, p_averaged_1)
                    minor_peak = both_peaks[both_peaks != major_peak][0]
                
            
        p1_err = np.nan
        try:
            maxval = sliced[major_peak]
        except UnboundLocalError:
            maxval = sliced[p1]
            
        if two_peaks:
    
            p2_err = p1_err
            maxval2 = sliced[minor_peak]
            #create negative Hough radius for second peak
            out.append([r, i, major_peak + minr,maxval, p1_err])
            out.append([-r, i, minor_peak + minr,maxval2, p2_err])

            
        else:
        #sort peaks by radius:
            out.append([r, i, p1 + minr, maxval, p1_err])
    
    if show:
        
        show_plot_estimator = rrange.shape[0] // 5
        ndigits = len(str(show_plot_estimator)) - 1
        
        if not i % round(show_plot_estimator, -ndigits):
                                          
        
            fig, ax = plt.subplots(figsize=(5,5))
            #plot data
            x_plot = x + minr
            ax.plot(x_plot,y, '.', label = f'Ang. Incr. {i}')
            
            #plot cwt-peak guess result
            if two_peaks:
                ax.plot([p1 + minr, p2 + minr], sliced[[p1, p2]], 'x',
                label='Peak guess')
            else:
                ax.plot([p1 + minr], sliced[p1], 'x', label='Peak guess')
                       
            ax.set(xlabel='Radius / px', ylabel='Intensity / a. u.')
            ax.tick_params(direction='in', which = 'both')
            ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
            
            savename = filename.split('.png')[0] + f'_angular_fit_{i}.png'
            fig.savefig(savename, bbox_inches = 'tight', dpi = 300)
            plt.show()
            
            fig.clf()
            plt.close('all')        
                
    #convert to 
    out = np.array(out)
    
    #separate radii:
    arrays = []
    for rad in set(out[...,0]):
        
        #apply intensity threshold:
        filtered_array = _apply_intensity_threshold_to_list(out[out[...,0]==rad],
                                                         intensity_threshold)            
        arrays.append(filtered_array)
        
    #merge subarrays again
    out_arr = np.concatenate(arrays)
    return out_arr



def guess_ellipse(df, show = False, filename = 'test.png', skip_ring = None,
                  img = None, real_center = (0,0)):
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
    #Create output containers
    
    logging.debug(f'''Enterred guess_ellipse. args are:
                  show = {show},
                  filename = {filename},
                  skip_ring = {skip_ring},
                  img was given ({df is None}),
                  real_center = {real_center}''',  )
    radii = []
    centers = []
    parameters = []
    radius_deviation = []    
    if show:

        filename_split = filename.split('.')
        filename_split.insert(-1,'_Data_Separation')
        data_lst = []

    
    rings = list(set(df['Hough-Radius']))
    rings.sort()
    
    logging.debug('Enter loop over ring values. rings is {}'.format(rings))
    
    for j, circle in enumerate(rings):
        
        logging.debug('Iterate over Pos. {} in {}'.format(j, rings))
        
        if skip_ring is not None:
            
            if j in skip_ring:
                
                logging.debug('skip ring nr. {}'.format(j))
                continue
        
        rad = df[df['Hough-Radius'] == circle]
        
        #create Ellipse Object
        ellipse = LSqEllipse()
        #perform fitting
        try:
            ellipse.fit((rad['x'],rad['y']))
        except Exception as e:
            logging.error(f'Fit ellipse to ring {circle} in sample {filename} failed due to {type(e).__name__}. Continuing.')
            
            continue
        
        logging.debug('Fitted ellipse to ring {}. Parameters are {}'.format(circle, ellipse.parameters()))
        
        #store values. Important!
        radii.append(np.mean([ellipse.width, ellipse.height]))
        centers.append(ellipse.center)
        
        logging.debug(f'Ellipse center {ellipse.center} was added to centers-Container')
        
        parameters.append(list(ellipse.parameters()))
        
        #calculate and store relative radius deviation of raw data:
        rel_rad_std = rad['radius'].std() / np.asarray(radii[-1])
        radius_deviation.append(1/rel_rad_std)
        
        #create ellipse data based on fit
        center = ellipse.center
        width = ellipse.width
        height = ellipse.height
        phi = ellipse.phi
        
        
        if show:
            #data for plotting
            t = np.linspace(0, 2*np.pi, 2000)#len(rad['x']))
            logging.debug(f'Ellipse x and y data are calculated by using the angle {phi} and the center {center}.')
            ellipse_x = center[0] + width*np.cos(t)*np.cos(phi)-height*np.sin(t)*np.sin(phi)
            ellipse_y = center[1] + width*np.cos(t)*np.sin(phi)+height*np.sin(t)*np.cos(phi)
     
            fig, ax = plt.subplots(3,1, figsize = (8,6))

            ax[0].plot(rad['angle'],rad['radius'], 'x')
            ax[0].set(xlabel='Angle / Increment', ylabel = 'radius / px')
            
            ax[1].plot(rad['angle'],rad['intensity'], 'x')
            ax[1].set(xlabel='Angle / Increment', ylabel = 'Intensity / a. u.')
            
            title = r'{}: r $\approx$ {} px'.format(filename, circle)
            ax[0].set_title(title)
            
            ax[0].tick_params(direction='in', which = 'both')
            ax[1].tick_params(direction='in', which = 'both')
            
            filename_split = filename.split('.')
            filename_split.insert(-1,f'_Data_Separation_{j}')
            filename_fig = reduce(_kit_str, filename_split)
            
            #plot ellipse in polar coordinates
            ellipse_polar = np.array([at.polar.cart2polar(x,y) for x,y
                                      in zip(ellipse_x, ellipse_y)])
            ellipse_r = np.flipud(ellipse_polar[...,0])
            ellipse_phi = np.flipud(radians2increment(ellipse_polar[...,1],
                                                      rad['angle'].max()))
            
            #recalculate data in polar coord
            data = np.array([at.polar.cart2polar(x,y) for __, x, y
                             in rad[['x', 'y']].itertuples()])
            data_r = np.flipud(data[...,0])
            data_phi = np.flipud(radians2increment(data[...,1],
                                                   rad['angle'].max()))
            
            ax[2].plot(data_phi, data_r, 'x')
            ax[2].plot(ellipse_phi,ellipse_r, '.', label = 'Ellipse fit',
                       markersize = 0.5)#
            ax[2].set(ylabel = 'Radius / px',
                      xlabel='Angle / Transformed Increment')
            ax[2].tick_params(direction='in', which = 'both')
            ax[2].legend(loc=0)
            plt.tight_layout()            
           
            #save raw data for plotting
            x, y = rad['x'] + real_center[0], rad['y'] + real_center[1]
            center = center[0] + real_center[0], center[1] + real_center[1]
            ell_x, ell_y = ellipse_x + real_center[0], ellipse_y + real_center[1]
            data_lst.append((x, y, rel_rad_std, center, ell_x, ell_y, circle)) 
            
            plt.show()
            fig.savefig(filename_fig, bbox_inches = 'tight', dpi = 300)
            plt.clf()
            plt.close()
            
    logging.debug('Exit loop over ring values.')
        
    if show:
        
        logging.debug('Plot results over img. --> Create figure')
        
        #create second figure
        fig2, ax2 = plt.subplots(figsize = (7,7))
        
        if img is not None:
            ax2.imshow(equalize_adapthist(img), cmap = 'binary')
            ax2.tick_params(direction='in', which = 'both')
        
        #get raw data
        
        logging.debug('Plot results over img. --> iterate over data_lst. len(data_lst) == {}'.format(len(data_lst)))
        for x, y, std, center, ell_x, ell_y, circle in data_lst:
            
            
            #plotting
            color = next(plt.gca()._get_lines.prop_cycler)['color']
        
            ax2.plot(x, y, 'o',
                     label = 'Rel. Std: {}%'.format(round(std*100,2)),
                     markersize = 2, color = color)
            #plot center
            ax2.plot(center[0],center[1], 'x',
                     label = 'Center {}'.format((round(center[0], 2),
                     round(center[1], 2))), color = color)
            #plot ellipse
            ax2.plot(ell_x, ell_y,
                     label = r'Ellipse r $\approx$ {} px'.format(circle))

        #layouting
        ax2.set(xlabel = 'x / px', ylabel = 'y / px')     
        ax2.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        
        
        filename_split.pop(-2)
        filename_split.insert(-1,'_Ellipse_Fit')
        filename_fig2 = reduce(_kit_str, filename_split)
        
        fig2.savefig(filename_fig2, bbox_inches = 'tight', dpi = 500)
        plt.show()
        fig2.clf()
        plt.close()
    
    #to format the relative center compatible to plt.imshow, it has to originate from the top left corner:
    logging.debug(f'Invert y coordinates of centers {centers} to originate from top left')
    centers = np.array(centers)
    centers[...,1] *= -1
    logging.debug(f'Finished inverting the y coordinate of centers {centers} to originate from top left')
    
    return centers, radii, np.array(parameters, dtype='object'), radius_deviation



def fit_ellipse(x, y, ring = '220', filename = 'test.png',):
    """
    Creates ellipses for every ring analyzed by select_ring_values.
    
    Return: Dicitionary containing Ellipse objects. Key is the respective Hough-radius.
    To access the data, call the parameters attribute on the Ellipse.
    It returns (center, width, height, phi).
    
    kwargs:
      
    show (bool): If truthy, interim results are plotted. Slows algorithm down.
    
    skip_ring (list of integers or None (default)): List of ring numbers
    (starting from 0, referring to the rings in df) to skip.
    
    RETURN:
        
    p_opt, p_err of the ellipse.
    
    Each is a numpy array containing values for width, height, center, phi
    """
    
    
    #create Ellipse Object
    ellipse = LSqEllipse()
    #perform fitting
    try:
        ellipse.fit((x,y))
    except IndexError:
        logging.error('Fit ellipse to ring {} in sample {} failed. Continuing.'.format(ring, filename))
        
    
    logging.debug('Fit ellipse to ring {} using non-iterative approach. Parameters are {}'.format(ring, ellipse.parameters()))
    #create ellipse data based on fit
    center = np.asarray(ellipse.center)
    width = ellipse.width
    height = ellipse.height
    phi = ellipse.phi
    
    #to format the angle, it has to be adjusted:
    phi += np.pi/2
    logging.debug(f'Add pi/2 to angle, which is now {phi}')
    
    p_opt = width, height, center, phi 
    
    return p_opt



def _kit_str(a,b):
    
    return a + '.' + b 
 


def ellipse_polar(a, b, center, phi):
    '''
    calculates the ellipse function based on its polar coordinates. The coordinate system
    originates from the center of the ellipse
    
    INPUT:
    
    data: r, phi
    
    a, b = axes
    alpha = angle between major axis and x axis
    
    '''
    
    t = np.linspace(0, 2*np.pi, 2000)

        
    x = center[0] + a*np.cos(t)*np.cos(phi)-b*np.sin(t)*np.sin(phi)
    y = center[1] + a*np.cos(t)*np.sin(phi)+b*np.sin(t)*np.cos(phi)
    
    return x, y



def hkl2distance_cubic(hkl , a = 0.4078):
    """
    Calculates lattice plane distance for cubic crystals (including fcc).
    
    
    Input: h,k,l as miller indice tuple (h,k,l); a=0.4078 nm as lattice constant at 25 °C (Dutta1963)
    
    Returns lattice plane distance (unit depends on a)
    """
    
    h, k, l = hkl
    
    return a / np.sqrt(h**2 + k**2 + l**2)




def file2df(filename, path, output_folder = True):
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
        filepath.append('output')
    
    filepath.append(filename)
    
    directory = os.path.join(*filepath)
    
    if filename.endswith('.csv'):
        df = pd.read_csv(directory)
        
    elif filename.endswith('.xlsx'):
        df = pd.read_excel(directory)
    
    else:
        logging.warning('WARNING: file2df cannot create a DataFrame, because {} is not \
              recognized. Only csv and xlsx files are supported.'.format(filename))
        
        df = None
        
    return df



def poly_func(xdat, *args):
    """
    Calculates a polynomial function of (len(args)-1)th order.
    
    INPUT:
        xdat (array): values for which to calculate the polynomial.
        *params (iterable): coefficients for the polynomial in descending order.
    
    RETURN:
        y (array): Array of the same shape as x_data containing the calculated
    result.
    """    
    y = 0
    for n,i in enumerate(args[::-1]):
        y += float(i) * xdat**n
    
    return y



def calculate_distortion(arr):
    '''
    Input (np.array): 2D array that stores width and height of the ellipse in the 1,2 position
    (as does the parameters output value of guess_ellipse)
    
    return: Distortion (float)
    '''
    eta_lst = []
    
    try:
        
        for i in arr[...,1:3:]:
            rad_ratio = min(i) / max(i)
             
            eta_lst.append((1- rad_ratio)/(1+ rad_ratio))
            
    except TypeError:
        
        rad_ratio = min(arr[1:3]) / max(arr[1:3])
        eta_lst.append((1- rad_ratio)/(1+ rad_ratio))
        
    return np.mean(eta_lst)



def check_ring_values(df, max_phi = 2048, vals = 5):
    """
    Checks whether sufficient pixels are selected.

    Parameters
    ----------
    df : DataFrame as returned by select_ring_values
    max_phi : Int, optional
        Number of polar increments. The default is 2048.
    vals : Int, optional
        Amount of values per quadrant. The default is 5.

    Returns
    -------
    out : DataFrame
    """
    #get quadrant
    q1 = (0, max_phi//4)
    q2 = (q1[1], max_phi//2)
    q3 = (q2[1], (3*max_phi)//4)
    q4 = (q3[1], max_phi)
    
    #list of data_Frames
    q_dfs = [df[(df['angle'] >= l) & (df['angle'] < m)] for l,m in (q1,q2,q3,q4)]
    #list with lengths of data Frames
    q = [len(i) for i in q_dfs]
    
    #array with True and False values of lengths
    len_arr = np.array(q) >= vals

        
    #count amount of quadrants with sufficient values
    positive_qs =[*len_arr].count(True)
    
    #three or four quadrants are fine
    if  positive_qs >= 3:
        
        out = df
    
    #one or no quadrant is fine
    elif positive_qs <= 1:
        
        out = df[[False]*len(df)]
    
    #two quadrants are fine
    else:
        
        #now, these quadrants can either be neiqhbouring each other (bad), or sit on opposit sites (good):
        #check if they sit opposite to each other. Then, the first and third argument must be similar
        if len_arr[0] == len_arr[2]:
            
            out = df
            
        else:
            
            out = df[[False]*len(df)] 
            
    return out



def _retreive_center(filename, df) -> tuple:
    """
    reads out a center guess out of a DataFrame having the columns "File" and
    columns with center information
    
    """

    #select file data
    data = df[df["File"]==filename]
    
    #get available center information
    x = data["Center x"].iloc[0]
    y = data["Center y"].iloc[0]
    
    return x,y



def optimize_center(img,
                    center_guess,
                    file = 'test.png',
                    show_ellipse = True,
                    show_all = False,
                    max_iter =100,
                    mask = None,
                    tolerance = 0.05,
                    median_for_polar = True,
                    clahe_for_polar = True,
                    int_thres = 'std',
                    max_tries = 6,
                    value_precision = 0.1,
                    sigma = 2,
                    radius_boundaries = (400, 1024),
                    dr_polar = 1,
                    dt_polar = None,
                    local_intensity_scope=False,
                    vals_per_ring = 20,
                    jacobian_for_polar =False,
                    skip_ring = None,
                    fit_gauss=True,
                    double_peak=False):
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
        If truthy, the ellipse fits are shwon. Overwritten if show_all is True.
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
    
    
    logging.debug('optimize_center --> File {}: start optimize_center'.format(file))
    
    if show_all:
        
        show_ellipse = show_img2polar = show_peaks = show_select_values = True
        
    else:
        
        show_img2polar = show_peaks = show_select_values = False
    
    #ensure full algorithm processing
    #six different options for algorithm to get a better result. trys starts from 0
    min_trys = 6
    
    if max_tries < min_trys:
        
        logging.info('Maximum number of iterations {} is less than the number of different optimization strategies {}.'.format(max_tries, min_trys))
    
    center_lst = []
    displacement_lst = []
    
    center_old = 'not defined --> first run'
    
    center = center_guess
    
    #store initial guess for comparison:
    center_lst.append(center)
    
    #create dummy variables for initial displacement and radius deviation
    displacement_old = mean_rad_dev_old = fit_dict = None
    
    i = 0
    total_counter = 0
    max_counter = max_iter * 2 
    trys = 0
    distance = np.nan
    
    min_rad, max_rad = radius_boundaries
    
    #start iteration loop
    
    logging.debug('optimize_center --> File {}: Enter while loop for iteration'.format(file))
    
    #set increments to default values for center opt.
    dr, dt = 1, None
    #define variable to store that interpolation has been performed
    if dr == dr_polar and dt == dt_polar and dr != 1 and dt != None:
        interpolated = True
    else:
        interpolated = False
       
    while all([max_iter >= i, trys <= max_tries, np.nan not in list(center), total_counter < max_counter]): #and :
        ##Abel Transform to obtain polar coordinates
        #transform img to polar coordinates
        logging.debug('optimize_center --> File {}: Tranform img to polar coordinates --> Enter img2polar. Center is {}'.format(file, center))
        polar, r_grid, theta_grid = img2polar(img, center,
                                              show = show_img2polar,
                                              clahe = clahe_for_polar,
                                              median_filter = median_for_polar,
                                              jacobian = jacobian_for_polar, 
                                              dr=dr,
                                              dt=dt,
                                              filename= file + '_Abel_result_{}.png'.format(i))
        logging.debug('optimize_center --> File {}: Tranform img to polar coordinates --> Exit img2polar'.format(file))
        ###Find average peaks by integrating over the angle
        #get first guess and 1D structure
        logging.debug('optimize_center --> File {}: Find first peak guesses --> Enter find_average_peaks'.format(file))
        peaks, r, intensity = find_average_peaks(polar, r_grid, theta_grid,
                                                 peak_widths = [int(round(20*max(img.shape)/(2048*dr)))],
                                                 min_radius=min_rad,
                                                 max_radius = max_rad,
                                                 dr = dr,
                                                 show = show_peaks, filename = f'{file}_Peak_guessing_{i}.png')
        logging.debug('optimize_center --> File {}: Find first peak guesses --> Exit find_average_peaks. Peaks found at {}'.format(file,peaks))
      
        fit_dict_new = {}       

        rlist = peaks
        
        logging.debug('optimize_center --> File {}: Find precise peak positions --> rlist is {}'.format(file, rlist))
    
        ###center optimization
        
        ##use found radii to select ring values for center optimization
        logging.debug('optimize_center --> File {}: Extract ring values --> Enter select_ring_values'.format(file))

        
        ring_df = select_ring_values(polar,
                                     rlist,
                                     tolerance=value_precision,
                                     show=show_select_values,
                                     intensity_threshold=int_thres,
                                     filename=f'{file}_selected_vals_{i}.png',
                                     local_intensity_scope=local_intensity_scope,
                                     dr=dr,
                                     fit_gauss=fit_gauss,
                                     double_peak=double_peak
                                     )
        
        logging.debug('optimize_center --> File {}: Extract ring values --> Exit select_ring_values'.format(file))
        #ensure that significant values for each ring (10 in at least 2 quadrants that do not touch) are found
        ring_df = pd.concat((check_ring_values(ring_df[ring_df['Hough-Radius']==i],
                                               vals=round(vals_per_ring/4),
                                               max_phi = polar.shape[1])
                                        for i in set(ring_df['Hough-Radius'])))
        
        try:
            logging.debug('optimize_center --> File {}: Fit Ellispe --> Enter guess_ellipse. Center is {}'.format(file, center))
        
            centers, radii, parameters, rdev = guess_ellipse(ring_df, show = show_ellipse, 
                                                 filename= file + f'_Ellipse_{i}.png',
                                                 skip_ring = skip_ring, img = img,
                                                 real_center = center)
            
            logging.debug('optimize_center --> File {}: Fit Ellispe --> Exit guess_ellipse. Center is {}. Parameters are {}'.format(file, center, parameters))
        
        except:
            displacement = np.average(centers[...,0], weights = rdev), np.average(centers[...,1], weights = rdev)
            center_old = center
            center = center[0]+displacement[0], center[1]+displacement[1]
            distance = np.hypot(*displacement)

            logging.error('optimize_center --> File {}: Could not guess ellipse . Abort optimization after {} runs containing {} iteration improvements.\tCenter distance is {}.'.format(
                  file, total_counter, i, distance))
            
            break
        

        ###shift center
        
        logging.debug('optimize_center --> File {}: Evaluate center --> centers is {}. weights is {}'.format(file, centers, rdev))
        displacement = np.average(centers[...,0], weights = rdev), np.average(centers[...,1], weights = rdev)
        center_old = center
        logging.debug('optimize_center --> File {}: Evaluate center --> Displacement is {}'.format(file, displacement))
        
        #abort if no center was found:
        if any(np.isnan(displacement)):
            
            logging.warning('optimize_center --> File {}: Evaluate center --> Displacement is {}. Abort while loop.'.format(file, displacement))
            break
        
        #average radius deviation:
        mean_rad_dev = calculate_distortion(parameters)
        logging.debug('optimize_center --> File {}: Evaluate center --> Distiortion is {}'.format(file, mean_rad_dev))
        
        
        #make center mutable
        center_old = list(center_old)
        center = list(center)
        
        #optimize coordinates            
        center_old = np.asarray(center_old)
        displacement = np.asarray(displacement)
        
        logging.debug('optimize_center --> File {}: Correct center. trys == {}. Both coordiantes shifted.'.format(file, trys))

        
        if trys == 0:
           
            center = center_old[0] + displacement[0], center_old[1] - displacement[1]
            comment = 'added Pos. 0, substracted Pos. 1 of displacement from center_old'
        
        distance = np.hypot(*displacement)
        
        #save first displacement and distortion for comparison
        if displacement_old is None:
            
            logging.debug('optimize_center --> File {}: Define displacement_old and mean_rad_dev_old'.format(file))
            
            displacement_old = displacement
            mean_rad_dev_old = mean_rad_dev
            
            #add initial displacement to list
            displacement_lst.append(distance)
            
            logging.debug('continue')
            #skip evaluation
            continue
        
        
        distance_old = np.hypot(*displacement_old)

        logging.debug('optimize_center --> File {}: iteration round {}, run {}, total run {}'.format( file, i, trys, total_counter))
        logging.debug('optimize_center --> File {}: initial center {}, old center {} with distance {}, new center {} with distance {}'.format(
                file, center_guess,center_old, distance_old, center, distance))

        ##only take new center if old new one shows a better displacement:
        
        #if new displacement is smaller without inreasing the mean radius deviation, then fine. If not, try oppisite shift signs.
        tolerance_factor = 2
        max_rad_dev = tolerance_factor * mean_rad_dev_old
        
        if distance_old <= distance or all((mean_rad_dev > max_rad_dev,
                                            distance > tolerance)):
            
            logging.debug('optimize_center --> File {}: old distance {} <= new distance {} or distortion {} > {} * old distortion {}'.format(
                    file, distance_old, distance, mean_rad_dev, tolerance_factor, mean_rad_dev_old))

          #try changing half of maximum value    
            if trys > 0:
                
                if trys == 1:
                    
                    q =trys
                
                elif trys % 2:
                    
                    q += 1
                    
                else:
                    
                    q = 1
                
                    
                #change only center position of larger displacement
                center =  center_old[0] + displacement[0]/q, center_old[1] - displacement[1]/q
                comment = f'added Pos. 0 /{q}, substracted Pos.1/{q} of displacement from center_old'
                        
        #only store best displacement and center    
        else:
            
            displacement_lst.append(distance)
            center_lst.append(center)
            #overwrite peak data
            fit_dict = fit_dict_new
            
            #store ellipse results
            for k, ell in enumerate(parameters):
                
                #also possible with ellipse_params and later merging
                fit_dict[f'Ellipse {k} center / px'] = ell[0]
                fit_dict[f'Ellipse {k} width / px'] = ell[1]
                fit_dict[f'Ellipse {k} height / px'] = ell[2]
                fit_dict[f'Ellipse {k} radius / px'] = np.mean((ell[1:3]))
                fit_dict[f'Ellipse {k} phi / rad'] = ell[3]
                
            logging.debug(f'optimize_center --> File {file}: Added Ellipse parameters {parameters} to fit_dict')
                
            #leap one step forward
            i += 1
            logging.info('optimize_center --> File {}: Iteration step increased. Operation: {}, Displacement {}, Distance {}, New iteration: {}'.format(file, comment, displacement, distance , i))
            #reset trys
            trys = 1
            
            logging.debug('optimize_center --> File {}: Start iteration round: {}'.format(file, i))
            logging.debug('optimize_center --> File {}: Run {}, initial center {}'.format(file, total_counter,center_guess))
            logging.debug('optimize_center --> File {}: old center: {}; old displacement {}; old distortion {}'.format(file, center_old, displacement_old, mean_rad_dev_old))
            logging.debug('optimize_center --> File {}: new center: {}; new displacement {}; new distortion {}'.format(file, center, displacement, mean_rad_dev))
            
            displacement_old = displacement
            mean_rad_dev_old = mean_rad_dev
            
        #keep track with loop number
        total_counter += 1
        trys += 1
        
        #break if displacement tolerance is reached:
        if distance <= tolerance:

            logging.info('optimize_center --> File {}: Tolerance value {} reached after {} runs containing {} iteration improvements. Center distance is {}.'.format(
                    file, tolerance, total_counter, i, distance))
            
            break
            
        if i == max_iter:
            
            logging.info('optimize_center --> File {}: Maximum number of iterations ({}) reached after {} runs. Center is {}, Center distance is {}'.format(
                    file, i,total_counter,center_old, distance_old))
        
        if trys == max_tries:
            
            if interpolated:
                break
            
            #one last try using interpolation
            else:
                interpolated = True
                #reset trys
                trys = 1
            
            logging.info('optimize_center --> File {}: Maximum number of tries ({}) reached. {} iteration improvements could be performed. Center is {}, Center distance is {}'.format(
                    file, trys,i,center_old, distance_old))
            
    
    logging.debug('optimize_center --> File {}: while loop exit'.format(file))
    #check, whether fit_dict is still None (Means that iteration did not improve anything):
    if fit_dict is None:
        
        logging.info('optimize_center --> File {}: Center improvement failed. Stick with center guess {} and displacement {}'.format(file, center_guess, displacement_old))
        
        fit_dict = fit_dict_new
        
    #add center with last imrovment to dictionary
    fit_dict['Center x'], fit_dict['Center y'] = center_lst[-1]
    fit_dict['Center Displacement / px'] = distance
    #add filename to dictionary:
    fit_dict['File'] = file
    
    center_arr = np.array(center_lst)
    logging.debug(f'optimize_center --> File {file}: Convert center_lst {center_lst} to array {center_arr}')
    
    if show_ellipse:
        
        logging.debug('optimize_center --> File {}: Create Centers -- img overlay image'.format(file))
        
        
        
        fig, ax = plt.subplots(figsize=(5,5))
        
        name = file + '_center_optimizaion'
        
        ax.plot(center_arr.T[0],center_arr.T[1], ':', c = 'black')
        for num, i in enumerate(center_arr):
            ax.plot(i[0], i[1], 'x', label = num,)# c=c)
        ax.set(title = file, xlabel = 'x / px', ylabel = 'y / px')
        ax.legend(loc='best')
        ax.tick_params(direction='in', which = 'both')
        plt.show()
        fig.savefig(name+'.png', bbox_inches='tight')
        fig.clf()
        plt.close('all')
        
        #plot distance
        f, ax = plt.subplots(figsize= (5,5))
        ax.plot(range(len(displacement_lst)), displacement_lst, ':', c = 'black')
        ax.set_yscale('log')
        for num, i in enumerate(zip(range(len(displacement_lst)),displacement_lst)):
            #c=next(color)
            ax.plot(i[0], i[1], 'o', label = num)
        
        ax.hlines(tolerance, 0, len(displacement_lst), label ='Threshold')
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Displacement / px')
        ax.tick_params(direction='in', which = 'both')
        f.savefig(f'{file}_displ_development_tolerance_{tolerance}.png', bbox_inches='tight')
        plt.show()
        plt.clf()
        plt.close('all')
        
    
    #merge polar transfom results for later processing
    polar_transformed_results = {'polar':polar, 'r_grid':r_grid, 'theta_grid':theta_grid}
    
    return fit_dict, ring_df, polar_transformed_results



def _truncate_colormap(cmap, minval=1/3, maxval=1.0, n=100):
    """
    from https://stackoverflow.com/questions/18926031/how-to-extract-a-subset-of-a-colormap-as-a-new-colormap-in-matplotlib 
    """
    import matplotlib.colors as colors
    
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap



def _minmax_scale(arr):
    """
    scales array to [0,1]
    """
    
    arr = np.asarray(arr)
    
    return (arr - arr.min() ) / (arr.max() - arr.min())
    
    
    
    
def correct_distortions(fit_dict, ring_df, skip_ring = None, show = True):
    """
    Fits the sum of fourfold distortions to every ring in ring_df and fit_dict
    
    returns a dictionary containing the fitparameters per ring merged with fit_dict
    """
    
    #output dictionary:
    out = {}
    
    if show:
        #container for data storage:
        rdat_container = []
        phidat_container = []
    #extract number of rings
    logging.debug(f'File {fit_dict["File"]}: Correct_distortions --> search ring numbers in {fit_dict.keys()}')
    #rings = [int(re.search('-?\d+', k).group()) for k in fit_dict if re.search('Ellipse \d+ radius / px', k)]
    rings = list(range(len(set(ring_df['Hough-Radius']))))
    logging.debug(f'File {fit_dict["File"]}: Correct_distortions --> Found rings: {rings}')
    #loop over rings
    hough_radii = sorted(list(set(ring_df['Hough-Radius'])))
    
    for ring in rings.copy():
        
        if skip_ring is not None:
            
            skip_ring = np.asarray(skip_ring)
            relevant_rings = skip_ring[skip_ring <= ring]
            skipped = len(relevant_rings) + ring
         
        else:
            
            skipped = ring
            
        try:
            
            rad = ring_df[ring_df['Hough-Radius'] == hough_radii[skipped]]
        
        except IndexError:
            
            logging.error(f'File {fit_dict["File"]}: Correct_distortions --> Loop over rings rings: Ring {ring}. list index out of range. List: {hough_radii}. Index: {skipped}. Continue.')
            rings.remove(ring)
            
            continue
        
        logging.debug(f'File {fit_dict["File"]}: Correct_distortions --> Loop over rings rings: Ring {ring}')
        
        #get data to fit on
        rdat, phidat, rdat_error = rad['radius'], rad['angle / rad'], 1/rad['intensity']
        
        #get starting parameters p0 based on the non-iterative ellipse fitting
        r = rdat.mean()
        try:
            alpha = fit_dict[f'Ellipse {ring} phi / rad']
        except KeyError:
            alpha = 0
        alpha1 = alpha2 = alpha3 = alpha
        ab = rdat.min(),rdat.max() #fit_dict[f'Ellipse {ring} height / px'], fit_dict[f'Ellipse {ring} width / px']
        eta1 = (1-min(ab)/max(ab)) / (1+min(ab)/max(ab))
        eta2 = eta1
        eta3 = 0.25 * eta2
        
        p0 = r, alpha1, alpha2, alpha3, eta1, eta2, eta3
        
        logging.debug(f'File {fit_dict["File"]}: Correct_distortions --> \
                              Created fit guesses for {ring}. p0 is {p0}')


        try:
            #try to fit rings
            popt, pcov = curve_fit(polar_dist4th_order,
                                   phidat,
                                   rdat,
                                   p0=p0,
                                   sigma=rdat_error,
                                   bounds=([0,-np.inf,-np.inf,-np.inf,0,0,0],
                                           np.inf),
                                   maxfev=10000
                                   )
            perr, rsquare = get_errors(popt, pcov, phidat, rdat, polar_dist4th_order(phidat, *popt))# np.sqrt(np.diag(pcov))
            
            logging.info(f'File {fit_dict["File"]}: Correct_distortions --> Fit succesful. Parameters are {popt}, Errors are {perr}.')
            
        except Exception as e:
            
            #log
            message = f'{type(e).__name__} in file {fit_dict["File"]}.'
            logging.error(message)
            logging.error(traceback.format_exc())
            
            popt = p0
            perr = [np.inf]*len(p0)
            
            logging.info(f'File {fit_dict["File"]}: Correct_distortions --> Fit failed. Use input parameters. Parameters are {popt}, Errors are {perr}.')
            

        #save values
        out[f'distortion correction {ring} radius / px']= popt[0]
        out[f'distortion correction {ring} phi1 / rad']= popt[1]
        out[f'distortion correction {ring} phi2 / rad']= popt[2]
        out[f'distortion correction {ring} phi3 / rad']= popt[3]
        out[f'distortion correction {ring} eta1']= popt[4]
        out[f'distortion correction {ring} eta2']= popt[5]
        out[f'distortion correction {ring} eta3']= popt[6]
        out[f'distortion correction {ring} radius error / px']= perr[0]
        out[f'distortion correction {ring} phi1 error / rad']= perr[1]
        out[f'distortion correction {ring} phi2 error / rad']= perr[2]
        out[f'distortion correction {ring} phi3 error / rad']= perr[3]
        out[f'distortion correction {ring} eta1 error']= perr[4]
        out[f'distortion correction {ring} eta2 error']= perr[5]
        out[f'distortion correction {ring} eta3 error']= perr[6]
        out[f'distortion correction {ring} R2adj'] = rsquare
        out[f'distortion correction {ring} values'] = len(phidat)
        #store data for plotting
        if show:
            #ydata
            rdat_container.append(rdat)
            #xdata
            phidat_container.append(phidat)
    #show results
    if show:
        
        fig, ax = plt.subplots(len(rings), 1, figsize = (6,
                               5*len(rings)+1*(len(rings)-1)),
                               sharex=True)
        
        if type(ax) != np.ndarray:
                ax = [ax]
                
        t = np.linspace(0, 2*np.pi, 10000)
        #plot in terms of pi

        t_plot = t / np.pi
        for i in rings:
            
            if out[f'distortion correction {i} radius error / px'] == np.inf:
            
                ax[i].text(1, 0.9, 'FIT FAILED', {'color': 'm', 'fontsize': 18}, va="top", ha="right")
            
            r = out[f'distortion correction {i} radius / px']
            alpha1 = out[f'distortion correction {i} phi1 / rad']
            alpha2 = out[f'distortion correction {i} phi2 / rad']
            alpha3 = out[f'distortion correction {i} phi3 / rad']
            eta1 = out[f'distortion correction {i} eta1']
            eta2 = out[f'distortion correction {i} eta2']
            eta3 = out[f'distortion correction {i} eta3']
            #built functions:
            #sum function
            sum_dist = polar_dist4th_order(t+np.pi/2,r, alpha1, alpha2, alpha3, eta1, eta2, eta3)
            #each distortion contributes to 1/3 to r. Also, shift t by pi/2 to fit plotted results in [0,2pi]
            dist2 = polar_dist2(t+np.pi/2, r/3, eta1, alpha1, n= 2) + 2/3 * r
            dist3 = polar_dist2(t+np.pi/2, r/3, eta2, alpha2, n= 3) + 2/3 * r
            dist4 = polar_dist2(t+np.pi/2, r/3, eta3, alpha3, n= 4) + 2/3 * r
            
            #shift phi data to fit plotted results in [0,2pi]
            phidat_container[i] = np.asarray(phidat_container[i]) 
            #plot in terms of pi
            phidat_container[i] -= np.pi/2
            phidat_container[i] /= np.pi
            
            
            #plot
#            r2 = round(out[f'distortion correction {i} R2adj'],4)
#            amount = out[f'distortion correction {i} values']
#            ax[i].plot(phidat_container[i], rdat_container[i], 'x', label = f'{amount} pixels')
#            ax[i].plot(t_plot, sum_dist, label = f'sum of distortions\nR$^2$ = {r2}')
#            ax[i].plot(t_plot, dist2, dashes=[6, 2], label = '2-fold dist. + 2/3r')
#            ax[i].plot(t_plot, dist3, '--', label = '3-fold dist. + 2/3r')
#            ax[i].plot(t_plot, dist4, ':', label = '4-fold dist. + 2/3r')
#            ax[i].set_ylabel('Radius / px')
#            ax[i].tick_params(direction='in', which = 'both')
#            ax[i].hlines(r, min(t_plot), max(t_plot), label = f'Radius {round(r,4)} px')
#            ax[i].legend(loc='center left', bbox_to_anchor=(1, 0.5))
#            ax[i].tick_params(direction='in', which = 'both')
#            
#            ax[i].set_xlabel(r'Polar angle / $\pi$')
            
            cmap = _truncate_colormap(plt.get_cmap('Blues'), minval=1/2, n=rad['intensity'].shape[0])
            
#            r2 = round(out[f'distortion correction {i} R2adj'],4)
#            amount = out[f'distortion correction {i} values']
            c=ax[i].scatter(phidat_container[i], rdat_container[i], c = _minmax_scale(rad['intensity']),
                          marker='x', label = 'Data', cmap=cmap, )
            cbar = fig.colorbar(c,label='Relative Intensity')
            cbar.ax.tick_params(direction='in', which = 'both')
            next(plt.gca()._get_lines.prop_cycler)
            ax[i].plot(t_plot, sum_dist, label = '$R(\\varphi$)')
            ax[i].plot(t_plot, dist2, dashes=[6, 2], label = 'k=2')
            ax[i].plot(t_plot, dist3, '--', label = 'k=3')
            ax[i].plot(t_plot, dist4, ':', label = 'k=4')
            ax[i].set_ylabel('Radius / px')
            ax[i].tick_params(direction='in', which = 'both')
            ax[i].hlines(r, min(t_plot), max(t_plot), label = 'Radius')
            #ax[i].set_ylim((1540,None))
            
            ax[i].legend(loc=0, ncol=3) #loc='center left', bbox_to_anchor=(1, 0.5))
            #ax[i].grid()
            
            ax[i].set_xlabel(r'Polar angle / $\pi$')
        
        plt.tight_layout()
        plt.show()
        fig.savefig(f'{fit_dict["File"]}_Distortion_fit.png', bbox_inches = 'tight')
        fig.clf()
        plt.close('all')
        
        
    #merge dictionaries
    out = {**fit_dict, **out}

    return out



def reshape_image_by_function(polar, func, norm_factor=1, show=True, filename='test', dr=1, *funcparams):
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
    #calculate scaling factors based on func
    phi_arr = increment2radians(np.arange(polar.shape[1]) , polar.shape[1]) 
    phi_arr += np.pi/2        
    scale_vector = func(phi_arr,*funcparams) / dr
    
    if show:
        fig, ax = plt.subplots()
        ax.imshow(polar, cmap='gray')
        ax.plot(np.arange(polar.shape[1]), scale_vector)
        ax.set(xlabel = 'Angular increment / a. u.',
               ylabel = 'Radius / px',
               title = 'uncorrected')
        ax.tick_params(direction='in', which='both')

        plt.show()
        fig.savefig(f'{filename}_Distortion_fitted_for_correction.png', bbox_inches = 'tight')
        fig.clf()
        plt.close('all')
        
    #normalize scale_vector to oscillate around norm_factor:
    scale_vector = scale_vector / norm_factor
    #invert
    scale_vector = 1 / scale_vector
    #plot
    if show:
        fig, ax = plt.subplots(figsize=(5,5))
        ax.plot(phi_arr, scale_vector,  label ='scale vector')
        ax.legend(loc=0)
        ax.set(xlabel='Angle / rad', ylabel = 'Relative Magnitude')
        ax.tick_params(direction='in', which='both')
        plt.show()
        fig.savefig(f'{filename}_Scale_vector.png', bbox_inches = 'tight')
        fig.clf()
        plt.close('all')
    #get smallest array length for reconcatenation
        
    #output list
    out = []
    
    for col, scale in zip(polar.T, scale_vector):
               
        rescaled = rescale(col, scale, multichannel=False)#[:length]
        
        difference = polar.shape[0] - rescaled.shape[0]
        
        if difference < 0:
            out.append(rescaled[:difference])
        else:
            out.append(np.append(rescaled, np.zeros(difference)))
        
    return np.array(out).T



def polar_dist2(phi, r, eta, alpha,  n = 2):
    """
    Calcluate n-fold distortion in polar coordinates. n=2 equals an ellipse centerred at the origin of polar coordinates,
    as given in Niekiel.2017
    
    args:
    
    phi = angular increment in radians
    
    r = mean radius
    
    alpha = rotation of distortion relative to x axis
    
    """
    
    
    arg = 1 + eta**2 - 2*eta*np.cos(n*(phi+alpha))
    
    return r * (1-eta**2)/np.sqrt(arg)



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
    
    for n, eta, alpha in zip((2,3,4), (eta1, eta2, eta3), (alpha1, alpha2, alpha3)):
        
        out += polar_dist2(phi, r/3, eta, alpha, n)
 
    return out


def weighted_error_average(vals, val_errs, **kwargs):
    """
    Calculates a weighted average of vals based on their absolute uncertainties val_errs using np.average.
    
    The transformation to weights is performed by calculating the inverted relative error.
    Normalizing is performed by np.average.
    
    kwargs are passed to np.average
    """
    #ensure numpy compatibility of values
    vals = np.asanyarray(vals)
    val_errs = np.asanyarray(val_errs)
    
    ###transform errors to weights
    #calculate relative error
    val_errs /= vals
    
    #invert errors to weight small errors higher and vice verca
    val_errs = 1/val_errs
    
    #bypass 'ZeroDivisionError: Weights sum to zero, can't be normalized'
    try:
        out = np.average(vals, weights = np.abs(val_errs), **kwargs)
    except ZeroDivisionError:
        
        text = 'Encounterred ZeroDivisionError in weighted_error_average. Try without weighting'
        logging.warning(text)
        #inform user during alignment
        print(text)
        
        out = np.average(vals, **kwargs)
        
    return out



def weighted_mean_error(vals, val_errs):
    '''
    Uses np.average to calculate the weigth of val_errs depending on thei relative precision of vals (vals/valerrs).
    Furthermore, gaussian error distribution is used to divide the result by the square root of the amout of values.
    '''
    
    return np.average(val_errs, weights=(np.asarray(vals)/np.asarray(val_errs)))/np.sqrt(len(vals))



def grouper(iterable, n, fillvalue=None):
    """
    Collect data into fixed-length chunks or blocks. Fillvalue is returned in case of division rest.
    
    grouper( 'ABCDEFG', 3, 'x') --> ABC DEF Gxx
    
    
    From https://docs.python.org/3.0/library/itertools.html
    """
    args = [iter(iterable)] * n
    return zip_longest(*args, fillvalue=fillvalue)



def filo(iterable, n = 2):
    """
    Creates a rolling window (FILO) out of an interable object with the length
    n.
    
    INPUT: iterable as an iterable object.
    
    n (integer) = window size. Must not exceed len(iterable).
    
    
    RETURN: zip object containing the rolling window tuples
    
    """
    
    assert len(iterable) >= n, 'n must not exceed len(iterable)'
    
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
    #create a dictionary containing the peak distances to pk and the respective intensity
    distances = {abs(ref-k):k for k in vals}
    #store only minimum distance values (radius, intensity)
    return distances[min(distances)]



def mean_arr(iterable):
    """
    Input: iterable containing objects that support addition and division
    
    Return: element-wise average of all elements in iterable
    """    
    
    return reduce(lambda a,b: a + b, iterable) / len(iterable)




def select_pixels(polar, peaks, minr = None, maxr = None, chunksize = 4,
                  sigma = 2.5, filename = 'test', show = True):
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
    #polar is cropped to save computation time and avoid some unreal peaks
    #also, it is transposed here
    filtered_polar = prepare_matrix(polar[minr:maxr].T)
    
    #calculate two sigma
    two_sgm = sigma * filtered_polar.std()
    
    
#    #set beam stopper area to 0:
    filtered_polar = np.where(filtered_polar > two_sgm, filtered_polar, 0)

    #create a data cache
    peak_dct = {}
       
    #define maximum number of peaks based on the averaged ones:
    maxpeaks = len(peaks)
    
    #iterate over angles
    for n, i in enumerate(grouper(filtered_polar, chunksize, 0)):
            
            
        #average spectra in chunk
        i = reduce(lambda x,y: x+y, i) / chunksize
        #get averaged angle from chunksize
        angle = np.arange(n*chunksize, (n+1)*chunksize).mean()
        #find values
        indexes = find_peaks_cwt(i, [7.5], min_snr=2, noise_perc=30)

        if not len(indexes):
            continue
        #change max peak number if it exceeds the amount of found peaks
        if indexes.shape[0] < maxpeaks:
            
            mp = indexes.shape[0]
        
        else:
            
            mp = maxpeaks

        #arbitrary distance of peaks in px
        min_dist = 30
        intensity = i[indexes]
        lst = []
        
        #iterate over found peaks
        for k, val in enumerate(indexes):
    
            if k == 0:
                
                lst.append(val)
                
                continue
        
            old = indexes[k-1]
            
            diff = val - old
            
            #select value with maximum intensity
            if diff <= min_dist:
                
                if old in lst:
                    
                    lst.remove(old)
                
                vals = [old, val]
                intens= [intensity[k-1], intensity[k]]
                
                lst.append(vals[intens.index(max(intens))])
                
            else:
                
                lst.append(val)
            
            
        indexes, index_old = lst, indexes    
       
                    #index:intensity
        index_dct = dict(zip(indexes, i[indexes]))
        #build dictionary that consists only of the four most intense peaks. 
        top4 = {a+minr:index_dct[a] for a in sorted(index_dct,
                key= lambda x: index_dct[x], reverse=True)[:mp]}
        #store angle
        
        peak_dct[angle] = top4

        #lst = _remove_peaks(indexes, i[indexes],  5)    
        if show:
            
            if n % 50:
            
                continue

            #plotting
            f, ax = plt.subplots()
            ax.set(ylabel='Intensity / a. u.', xlabel='Radius / px')
            ax.plot(np.arange(minr, len(i)+minr) ,i , label = f'Angle: {angle}')
            
            ax.plot(np.array(index_old) + minr, i[index_old], 's', label = 'Found Peaks')
            ax.plot(np.array([*top4]) ,list(top4.values()),'x', label = 'Chosen Peaks')
            ax.legend(loc=0)
            ax.tick_params(direction='in', which = 'both')
            plt.show()
            plt.clf()
            plt.close('all')
        
    #unpack cache:
    i_vals = np.array(reduce(lambda x,y: x+y, ((list(zip(dct[0].keys(), dct[0].values(), dct[1])))
                                    for dct in ((peak_dct[key], [key]*len(peak_dct[key])) for key in peak_dct))))

    if show:
        
        angles = i_vals[:,2]
        radii = i_vals[:,0]
        
        f, ax = plt.subplots(figsize = (10,6))
        ax.set_title(f'Found pixels in {filename}')
        ax.plot(angles, radii, 'x')
        ax.imshow(equalize_adapthist(prepare_matrix(polar)), cmap = 'binary')
        ax.tick_params(direction='in', which = 'both')
        for end in ('.png', '.pdf'):
            f.savefig(f'{filename}_found_pixels{end}', bbox_inches = 'tight', dpi = 300)
        plt.show()
        plt.clf()
        plt.close('all')
    
    return i_vals
        
    
    
def _get_last_center(num, tolerance, indices):
    """
    enters the indices dictionary to retreive the num-1th file center if the center displacement deceeds tolerance.
    """
    #get old number
    old_num = num - 1
    
    try:
        old_data = indices[old_num]
    except KeyError:
        
        logging.warning(f'_get_last_center: {old_num} not in indices')
        #exit function
        return None

    #only get the center if the tolerance value is reached
    if old_data['Center Displacement / px'] <= tolerance:
            x = old_data['Center x']
            y = old_data['Center y']
            center = x, y
    else:
        center = None
        
    return center



def _retreive_fit_parameters(out):
    """
    Returns fit parameters of best fit based on R² value as list
    """
    #get ring numbers
    rings = [int(re.search('-?\d+', k).group()) for k in out
             if re.search('distortion correction -?\d+ radius / px', k)]
    #create output container
    r2s = {}
    ring_values=[]
    #retreive fit parameters for every ring
    
    for ring in rings:
        r2s[out[f'distortion correction {ring} values']] = ring
    
    ring = r2s[max(r2s)]
        
    ring_values.append(out[f'distortion correction {ring} radius / px'])
    ring_values.append(out[f'distortion correction {ring} phi1 / rad'])
    ring_values.append(out[f'distortion correction {ring} phi2 / rad'])
    ring_values.append(out[f'distortion correction {ring} phi3 / rad'])
    ring_values.append(out[f'distortion correction {ring} eta1'])
    ring_values.append(out[f'distortion correction {ring} eta2'])
    ring_values.append(out[f'distortion correction {ring} eta3'])
       
    return ring_values
                


def _multi_voigt_background(xdat, *params):
    """
    

    Parameters
    ----------
    xdat : TYPE
        DESCRIPTION.
    *params : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """    
    poly_params = params[-2:]
    params = params[:-2]    

    return multi_voigt(xdat, *params) + poly_func(xdat, *poly_params)



def fwhm_voigt(sigma, gamma):
    """
    Returns the full-with at half maximum of a voigt function

    Parameters
    ----------
    sigma : float
        Full width at half-maximum of the Gaussian-contribution
    gamma : float
        Full width at half-maximum of the Lorentzian-contribution

    Returns
    -------
    float

    """

    return 0.5346 *2 *gamma + np.sqrt(0.2166*(2*gamma)**2+np.sqrt(8*np.log(2))*sigma)



def _fit_full_peak_spectrum(rrange,
                           intensity,
                           bounds,
                           hkl_guess,
                           filename='test.png',
                           show=False,
                           dr=1,
                           expected_peak_num=4,
                           background_voigt_num=3,
                           holz=2,
                           full_spectrum_fit = True,
                           distortion_fit_radius = [],
                           only_zolz=True,
                           double_peak=False,
                           use_last_fit=True,
                           popt_old = None,
                           max_intensity = None):
  
    popt = None
    
    #unpack boundaries                    
    low, up = bounds
    
    #apply boundaries to data
    r = rrange[low:up]
    cropped_intensity = intensity[low:up]
    
    ##select peaks
    #calculate peak positions based on hkl values
        
    peaks = []
    relative_distances = np.array((np.sqrt(1**2+1**2+1**2),np.sqrt(2**2+0**2+0**2),np.sqrt(2**2+2**2+0**2),np.sqrt(3**2+1**2+1**2)))
    
    h,k,l = hkl_guess
    
    for distortion_fit in distortion_fit_radius:
        
        assert distortion_fit_radius, 'No distortion fit radius is given'
        
        p_positions = relative_distances / np.sqrt(h**2+k**2+l**2) * distortion_fit
        peaks.extend(p_positions)
    
    #ensure to take higher order diffraction peaks into account, too:
    peaks = np.asarray(peaks)
    if not only_zolz:
        #ensure to take higher order diffraction peaks into account, too:
        peaks = np.append(peaks, (holz * peaks)[holz*peaks <= up])
    
    else:
        #cut spectrum half way between last zolz and first holz peak
        last_zolz = peaks[-1]
        first_holz = peaks[0] * 2
        
        up = int(round(np.mean((last_zolz, first_holz))))

        #apply new boundaries to data
        r = rrange[low:up]
        cropped_intensity = intensity[low:up]
    
        
    ##create guesses
    #fit voigt function to peaks separately
    peak_fits = fit_gaussian2peaks(peaks, rrange, intensity,
                                               filename= filename.split('.')[0] + '_single_peaks.png',
                                               val_range=10,
                                               functype='voigt',
                                               show=show,
                                               dr=dr,
                                               )
    
    if full_spectrum_fit:
        #retreive peak numbers:
        peaks = [int(peak[3:]) for peak in peak_fits if peak.startswith('xc ')]
                    #retreive initial guesses for peak fitting and calculate boundaries
        peak_guesses = []
        low_bounds = []
        up_bounds = []
        
        for n,pk in enumerate(peaks):
            
            xc = peak_fits[f'xc {pk}']
            area = peak_fits[f'A {pk}']
            sigma = peak_fits[f'sigma {pk}']
            gamma= peak_fits[f'gamma {pk}']
            fwhm= fwhm_voigt(sigma, gamma)
            max_width = max((sigma, gamma, fwhm))
            
            if n in (2,3):
                peak_guesses.extend((xc,area/100,sigma/100,gamma/100))
                low_bounds.extend((xc-fwhm/2,0,0,0))
                up_bounds.extend((xc+fwhm/2,area*0.75,max_width,max_width))

            else:
                peak_guesses.extend((xc,area,sigma/10,gamma/10))
                low_bounds.extend((xc-fwhm/2,0,0,0))
                up_bounds.extend((xc+fwhm/2,area*1.5,
                                  max_width*1.5,max_width*1.5))
                         
        #add boundaries for background 
        low_bounds.extend([low,0,0,0]*background_voigt_num)    
        up_bounds.extend([up,np.inf,np.inf,np.inf]*background_voigt_num)
            
        if use_last_fit and popt_old is not None:
            try:
                p0 = popt_old
                fitted = True

            except NameError:
                logging.error('##################### fitted is False #####################')
                fitted = False
        else:
            fitted = False
                
        if not use_last_fit or not fitted:
        
            #fit residual spectrum
            peak_fits = model_background(
                    peak_fits,
                    r,
                    cropped_intensity,
                    background_voigt_num=background_voigt_num,
                    show=show,
                    filename=filename.split('.')[0] + '_residual_background.png',
                    min_radius=low,
                    max_radius=up,
                    dr=dr,
                    )

        if not use_last_fit or not fitted:
            ##combine guesses for modelling the spectrum
            #retreive background guesses,
            strt = 'background_voigt'
            
            popt_background_voigts = []
            
            for i in range(background_voigt_num):
                popt_background_voigts.extend([peak_fits[f'{strt} xc {i}'],
                                      peak_fits[f'{strt} A {i}'],
                                      peak_fits[f'{strt} sigma {i}'],
                                      peak_fits[f'{strt} gamma {i}'], ])
                                                          
            p0 = peak_guesses + popt_background_voigts
        
        
        #ensure mutable p0_
        p0 = np.asarray(p0)
        #check for feasible p0 conditions:
        for n, (l, g, u) in enumerate(zip(grouper(low_bounds,4),
                                          grouper(p0,4), grouper(up_bounds,4))):
            for i in range(4):

                
                if not l[i] < u[i]:
                    low_bounds[n*4+i] = 0 
                    up_bounds[n*4+i] = np.inf#max((l[i], u[i], g[i]))
                    logging.debug(f'Boundaries {i} in Peak {n} reassigned: {l[i]} --> {low_bounds[n*4+i]}')
                    logging.debug(f'Boundaries {i} in Peak {n} reassigned: {u[i]} --> {up_bounds[n*4+i]}')
                    
                    if not low_bounds[n*4+i] < g[i] < up_bounds[n*4+i]:
                        p0[n*4+i] = low_bounds[n*4+i]
                        
                    continue
                
                if not l[i] <= g[i] <= u[i]:
                    logging.debug(f'Improper guess for value {i} in Peak {n}: {g[i]}')
                    if not np.isinf(u[i]):
                        p0[n*4+i] = np.mean((l[i], u[i]))
                    else:
                        p0[n*4+i] = l[i]
                    logging.debug(f'Value {i} in Peak {n} reassigned: {g[i]} --> {p0[n*4+i]}')
        #fit
        try:
            popt, pcov = curve_fit(multi_voigt, r, cropped_intensity, p0=p0,
                                   bounds = (low_bounds, up_bounds),
                                   maxfev=100000)
            errors, r_squared_corr = get_errors(popt,
                                                pcov,
                                                r,
                                                cropped_intensity,
                                                multi_voigt(r, *popt))
        except Exception as e:
            
            message = '{} in file {}.'.format(type(e).__name__, filename)
            logging.error(message)
            logging.error(traceback.format_exc())
            
            for n, (l, g, u) in enumerate(zip(grouper(low_bounds,4),
                                          grouper(p0,4), grouper(up_bounds,4))):
                logging.debug('Peak Nr',n)
                for i in range(4):
                    logging.debug('\t',l[i], g[i], u[i])
            
            raise e
            popt, pcov = curve_fit(multi_voigt, r, cropped_intensity, p0=p0,
                                   bounds = (0, np.inf),
                                   maxfev=100000)
            errors, r_squared_corr = get_errors(popt, pcov, r, cropped_intensity, multi_voigt(r, *popt))
            
            
        #store output parameters:
        peak_fits['R2adj_full_fit'] = r_squared_corr
        for pk, (popts, errs) in enumerate(zip((grouper(popt, 4, np.nan)),
                                               (grouper(errors, 4, np.inf)))):
            xc, area, sigma, gamma = popts
            xc_err, area_err, sigma_err, gamma_err = errs
            
            if pk >= len(popt) / 4:
                strt = 'background_voigt '
            else:
                strt = ""
                
            peak_fits[f'{strt}xc_full_fit {pk}'] = xc
            peak_fits[f'{strt}A_full_fit {pk}'] = area
            peak_fits[f'{strt}sigma_full_fit {pk}'] = sigma
            peak_fits[f'{strt}gamma_full_fit {pk}'] = gamma
            peak_fits[f'{strt}xc_full_fit_error {pk}'] = xc_err
            peak_fits[f'{strt}A_full_fit_error {pk}'] = area_err
            peak_fits[f'{strt}sigma_full_fit_error {pk}'] = sigma_err
            peak_fits[f'{strt}gamma_full_fit_error {pk}'] = gamma_err
               
        #plot
        if show:
            fig, ax = plt.subplots(figsize=(5,5),dpi=100)
            ax.plot(rrange, intensity, label ='Data')
            ax.plot(r, multi_voigt(r, *popt), '--', label =f'Fit $R^2$={round(r_squared_corr*100,2)}%')
            #plot voigt peaks:
            ls = '-.'
            for n, params in enumerate(grouper(popt, 4, 0)):
                
                if n >= len(popt)/4 - background_voigt_num:
                    ls = ':'
                
                ax.plot(r, voigt(r, *params), ls,
                        label=f'Voigt at {int(round(params[0]))} px')
            
            ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
            ax.set(xlabel='Radius / px', ylabel='Angular-integrated intensity / a. u.',
                   ylim = max_intensity)
            ax.tick_params(direction='in', which = 'both')
            
            plt.show()
            fig.savefig(filename, bbox_inches='tight',)
            plt.clf()
            plt.close('all')
    
    
    return peak_fits, popt



def _model_full_spectrum(rrange, intensity, starting_peaks, low_r, up_r, peak_num,
                        background_voigt_num, hkl_guess, filename,
                        full_spectrum_fit, popt_old,
                        reuse_last_fit_params=True, show=True, ):
    
    i = 0
    j = 0
    #try max 4 times and then live with the result. filtering via R² is possible anyway...
    while i < 2 and j < 4:
        i += 1
        j += 1
        peak_fits, popt_old = _fit_full_peak_spectrum(rrange,
                                   intensity,
                                   (low_r, up_r),
                                   show=show,
                                   dr=1,
                                   expected_peak_num=peak_num,
                                   background_voigt_num=background_voigt_num,
                                   hkl_guess=hkl_guess,
                                   filename=f'{filename}_Spectrum_fit.png',
                                   full_spectrum_fit=full_spectrum_fit,
                                   distortion_fit_radius=starting_peaks,
                                   double_peak = False,
                                   popt_old=popt_old,
                                   use_last_fit=reuse_last_fit_params,
                                   max_intensity=None)

        if full_spectrum_fit:
            if peak_fits['R2adj_full_fit'] >= 0.995:
                break
            elif i == 0:
                continue
            else:
                i = 0
                popt_old = None
        else:
            break
        
    return peak_fits, popt_old



def main(show = False,
         test = False,
         show_test = False,
         reverse = False,
         filepath = '.',
         dfname = 'test',
         filo_num = 1,
         tolerance = 0.01,
         skip = [],
         ring_deviation=None,
         scale = 10,
         re_pattern = '.em[di]$',
         radius_range = (1450/4096, 1750/4096),
         average_range = (0,2),
         filter_for_hough = True,
         test_interval = 10,
         median_for_polar = False,
         clahe_for_polar = False,
         jacobian_for_polar=False,
         int_thres = 'std',
         min_vals_per_ring = 20,
         canny_use_quantiles = True,
         hough_rings = 1,
         canny_sigma = 1,
         dr_polar = 1,
         dt_polar = None,
         hough_radius_range = None,
         skip_ring = None,
         center_guess = None,
         use_last_center=True,
         peak_num=5,
         background_voigt_num=3,
         hkl_guess=(2,2,0),
         full_spectrum_fit = True,
         reuse_last_fit_params = True,
         export_polar = False,
         store = True):
    
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
             containing the folder names.
    
    re_pattern (string): A pattern to search for files of interest in the set working directory using regular
                         expression syntax.
                  
    dfname (string): Name of the to-be-saved excel sheet, additionally, a timestamp is added to the filename.
    
    filo_num (integer): Amount of images to average using a first-in-last-out process. 1 equals no averaging
    
    tolerance (float): Value between the used center for polar transform and the calculated center in px after
                       which the center optimization is considered succesfully.
    
    skip (List of integers or empty list): Amount of images to skip in an image container.
        
    ring_deviation (float or tuple of ints, or None): Value to define the relative interval size around the given radius. If too small,
                    the diffraction ring might be displayed incorrectly. If too large, values of a neighbouring ring
                    may be taken into accout. If None, radius_range is used
        
    scale (integer): Downsampling factor for Hough transform (required to find the intial circle center).
                    For low singal/noise ratio or sparse diffraction rings, a high number (e.g. 9) may yield
                    better results.
    
    radius_range (tuple of two numbers (int or float) ): Radius range in px for finding the diffraction ring of interest.
                    If both tuple values (e.g. (0.5, 0.78)) are in [0:1], they are interpreted as
                    percentages of the image dimensions.
        
    filter_for_hough (bool): If truthy, a combination of median filtering and
                            Contrast Limited Adaptive Histogram Equalization is performed on the image to
                            prepare the Hough-Transform.
                            
                            
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
                  
    canny_use_quantiles (bool): If truthy, thresholding in canny algorithm is done differently. 
    
    hough_rings (int): Number of rings to choose in Hough-Transform.
    
    canny_sigma (float): Sigma value for gauss filter in canny algorithm
    
    hough_radius_range (tuple of two numbers (int or float) or None): adius range of interest in px for finding diffraction rings during Hough Transform.
                    Can be used to exclude small/large rings. If None (default), radius_range is used.
                    
    center_guess(string or None): path to Excel file containing center information for the requested data set.
    
    use_last_center( bool): retreives previous center if the latter was passing the tolerance value. Overruled by center_guess.
    
    Returns
    -------
    pd.DataFrame 
    
    which is also stored as .xlsx in an "output" subfolder of filepath.
    
    It summarizes acomprehensive set of data along the analysis pathway:
    -First, the results of the center finding process are displayed, including
     the parameters of the fitted ellipse, the center position and the displacement
    -Second, the Filename is shown
    -Third, the parameters of the distortion correction function are listed,
    -followed by lots of columns describing the initially guessed voigt peaks
     used as initial guesses for the simultaneous fit of the whole spectrum
    -Next, the parameters of the actual simultaneous spectrum fit are displayed.
     They are easily detectable by the '_full_fit' label in the column. Here,
         'R2adj_full_fit' - describes the adjusted R² value of the function
         'xc_full_fit i' - with i as integer, describes the peak position
         'A_full_fit i' - with i as integer, describes the area under the peak
         'sigma_full_fit i' - with i as integer, characterizes the gaussian, and
         'gamma_full_fit i' - with i as integer, the lorentzian contribution to
                              the voigt function.
        Feeding 'sigma_full_fit i' and 'gamma_full_fit i' to fwhm_voigt(sigma, gamma)
        returns the full-with at half-maximum of the respective voigt peak.
        To plot the whole full spectrum fit, one needs to pass all '_full_fit '
        in order of appearance as *params to multi_voigt(xdat, *params).
    -Subsequently, the uncertainties of the respective parameters are listed,
     as well. This is labelled by the '_full_fit_error' tag.
    -Finally, metadata information are extracted from the input data, which is
     of great importance for the parallel-beam alingnment protocol    
    """
    
    logging.info('Program started')
    logging.critical(f'Operating Parameters are:\n{locals()}')
    #create indices dict
    indices = {}
    
    if store:
        #list to store distortion-corrected spectra
        corrected_spectra = []
    

    #go to files
    if type(filepath) == tuple or type(filepath) == list:
        
        filepath = os.path.join(*filepath)
    
    if reverse:
        
        dfname += '_reverse'
    
    #change current working directory
    os.chdir(filepath)
    
    #keep in mind that "pattern" accepts regex syntax
    file_selection = select_files(os.listdir(), pattern = re_pattern)
    
    logging.info('{} files selected, based on "{}".'.format(len(file_selection), re_pattern))

    #iterate over files to fit center
    num = 0
    
    #create a variable to store initial show value
    init_show = show
    
    logging.info('Enter loop over files. Reverse sorting = {}'.format(reverse))
    
    if center_guess:
        
        old_df = pd.read_excel(center_guess)
    
    popt_old = None
    #sort files by name 
    for file in sorted(file_selection, reverse = reverse):

        
        logging.info('starting with ' + file)
        print('\tstarting with ' + file)
       
        #load file
        imgstack = load_file(file)
        
        #check for multiple images in imgstack
        shape = imgstack.data.shape
        
        if len(shape) == 3:
            
                        
            #store amount of images
            imgs_in_stack = shape[0]
            
            if radius_range == 'auto':
                
                minr = min(shape[1:]) // 5
                maxr = max(shape[1:]) // 2
                
            
            #calculate minr, maxr in percentage of image size
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

        elif len(shape) == 2:
            
            #set amout of image
            imgs_in_stack = 1
            
            if radius_range == 'auto':
                
                minr = min(shape) // 5
                maxr = max(shape) // 2
            
                        #calculate minr, maxr in percentage of image size

            elif all((0 <= radius_range[0] <= 1, 0 < radius_range[1] <= 1)):
                
                minr = int(min(shape) * radius_range[0])
                maxr = int(max(shape) * radius_range[1])
                
                
            else:
                
                minr, maxr = [int(i) for i in radius_range]
            
            
            ##deal with Hough radius boundaries:
            if hough_radius_range is None:
                
                hminr, hmaxr = minr, maxr
            
            elif all((0 <= hough_radius_range[0] <= 1, 0 <= hough_radius_range[1] <= 1)):
                
                hminr = int(min(shape) * hough_radius_range[0])
                hmaxr = int(max(shape) * hough_radius_range[1])
            
            else:
                
                hminr, hmaxr = (int(i) for i in hough_radius_range)
        
        else:
            
            logging.warning('Unknown Image format for {}. Continuing...'.format(file))
            
            continue
       
        ##for simplified usage:
        if ring_deviation is None:
            ring_deviation = minr, maxr
        
        if imgs_in_stack == 1:
            
            imgs = [imgstack.data.astype('float64')]
            
        else:
            
            imgs = imgstack.data.astype('float64')
        
        if reverse:
            
            imgs = imgs[::-1]
            amount = len(imgs)-1
            
        #iterate over all images in stack
        for n, images in enumerate(filo(imgs, filo_num)):
        #for i in range(894,1148):
            if reverse:
                n = amount - n
        
            logging.info(f'Averaging over {filo_num} file(s)')
            img = mean_arr(images)

            #redefine filename
            filename = reduce(_kit_str, file.split('.')[:-1])
    
            #skip part without information
            if n in skip:
                
                logging.info(f'Skip {filename}, because n = {n} is in {skip}')
                continue  
            
            #apply test only if n not in skip
            elif test and n % test_interval:
                
                logging.info(f'Skip {filename}, because test is {test}. test_interval is {test_interval}')
                continue
            
            #set show accordingly to show_test if init_show is False
            logging.debug(f'Test for skipping images: init_show is {init_show}')
            if all((not init_show, not test, show_test)):
                
                if n % test_interval:
                    show = False
                    logging.debug(f'Test for skipping images: n ({n}) % {test_interval} = {n % test_interval} --> show = {show}')
  
                else:
                    show = True
                    logging.debug(f'Test for skipping images: n ({n}) % {test_interval} = {n % test_interval} --> show = {show}')
            
            #make show_test behave differently if test is true and show is false
            elif all((not init_show, test, show_test)):
                
                if n % test_interval**2:
                    show = False
                    logging.debug(f'Test for skipping images with show_test={show_test} and test={test}: n ({n}) % {test_interval}**2 = {n % test_interval**2} --> show = {show}')
  
                else:
                    show = True
                    logging.debug(f'Test for skipping images with show_test True and Test True with show_test True and Test True: n ({n}) % {test_interval}**2 = {n % test_interval**2} --> show = {show}')
            
                
            try:
                
                if show:
                    
                    fig, ax = plt.subplots()
                    
                    ax.imshow(img, cmap = 'binary')
                    ax.set(xlabel = 'px', ylabel = 'px')
                    ax.tick_params(direction='in', which = 'both')
                    for end in ('.png', '.pdf'):
                        fig.savefig(f'{filename}_before_ell_corr{end}', dpi = 300)
                    plt.show()
                    plt.clf()
                    plt.close()
                        
                ##append filename by i with a proper amount of 0 in front of it for file sorting
                #calculate amount of 0's:
                maxdigits = len(str(imgs_in_stack))
                i_digits = len(str(n))
                zeros = '0' * (maxdigits - i_digits)
                
                #adjust filename
                filename += '_' + zeros + str(n)


                logging.info('Start center finding for ' + filename)
                #create mask array:
                mask = np.ones(img.shape)
                s = int(img.shape[0]/20 *3)
                lowy, upy = int(img.shape[0]/2 -s), int(img.shape[0]/2 + s)
                lowx, upx = int(img.shape[1]/2 -s), int(img.shape[1]/2 + s)
                mask[lowx:upx,lowy:upy] = np.nan   
                
                #convert array to the required form    
                img = prepare_matrix(img)
                
                center = None
                if center_guess:
                    
                    try:
                        center = _retreive_center(filename, old_df)
                    except IndexError:
                        pass
                
                #use last center
                elif use_last_center:
                    
                    center = _get_last_center(num, tolerance, indices)

                #only perform hough transform if no previous center guess is useful
                if center is None:
                    #perform hough transformation
                    circ = find_rings(img, hminr, hmaxr, 10, scale_factor=scale,
                                      max_peaks=hough_rings,
                                      mask = mask, filename= filename + '_Hough_result.png',
                                      show = show,
                                      hough_filter = filter_for_hough,
                                      canny_use_quantiles = canny_use_quantiles,
                                      canny_sigma = canny_sigma)
                    
                    #get circle center in numpy coordinates, originating from the top left corner
                    center = get_mean(circ)
                
                #center finding                
                fit_dict, ring_df, polar_transformed = optimize_center(img,
                                                                       center,
                                           max_iter = 200,
                                           tolerance = tolerance,
                                           file = filename,
                                           show_all = show,
                                           mask = mask,
                                           value_precision= ring_deviation,
                                           int_thres = int_thres,
                                           median_for_polar = median_for_polar,
                                           clahe_for_polar = clahe_for_polar,
                                           jacobian_for_polar=jacobian_for_polar,
                                           dr_polar = dr_polar,
                                           dt_polar = dt_polar,
                                           radius_boundaries = (minr, maxr),
                                           show_ellipse=False,
                                           local_intensity_scope=True,
                                           vals_per_ring = min_vals_per_ring,
                                           skip_ring = skip_ring,
                                           fit_gauss=True,
                                           )
                
                #correct distortions for final ring data
                fit_dict = correct_distortions(fit_dict, ring_df, show=show,
                                               skip_ring=skip_ring)
                
                ##use fitted distortions on img to correct all rings at once:
                #retreive parameters:
                funcparams = _retreive_fit_parameters(fit_dict)
                norm_factor = funcparams[0] #radius

                #rescaling
                polar_transformed['polar'] = reshape_image_by_function(
                                                polar_transformed['polar'],
                                                polar_dist4th_order,
                                                norm_factor/dr_polar,
                                                show,
                                                filename,
                                                dr_polar,
                                                *funcparams)
                
                #overwrite up based on center
                #low, up = int(min(img.shape)/8), int(min(img.shape)/2)
                low = int(min(img.shape)/12)
                up = int(min([fit_dict['Center x'],
                          fit_dict['Center y'],
                          img.shape[1]-fit_dict['Center x'],
                          img.shape[0]-fit_dict['Center y']]))
                
                
                #display distortion corrected result
                if show:
                    
                    fig, ax = plt.subplots()
                    ax.imshow(polar_transformed['polar'], cmap='binary')
                    ax.set(xlabel = 'Angular increment / a. u.',
                           ylabel = f'Radius / {dr_polar} px',
                           title = 'corrected')
                    ax.tick_params(direction='in', which = 'both')
                    plt.show()
                    fig.savefig(f'{filename}_Distortion_corrected_polar.png', bbox_inches = 'tight')
                    fig.clf()
                    plt.close('all')
                    
                    
                #store distortion-corrected image
                if export_polar:
                    np.save(f'{filename}_Distortion_corrected_polar', polar_transformed['polar'])

                #fit_gaussian2peaks
                __, rrange, intensity = find_average_peaks(
                                   polar_transformed['polar'],
                                   polar_transformed['r_grid'],
                                   polar_transformed['theta_grid'],
                                   peak_widths = [3],
                                   dr = dr_polar,
                                   show = False,
                                   filename = f'{filename}_after_correction',
                                   min_radius =  low,
                                   max_radius = up)
                
                if store:
                    #store corrected spectrum
                    corrected_spectra.append(intensity)
                
                
                distortion_fit_radii = [fit_dict[key] for key in fit_dict
                                        if re.search('distortion correction \d+ radius / px',
                                                     key)]
     
                
                peak_fits, popt_old = _model_full_spectrum(rrange,
                                                          intensity,
                                                          distortion_fit_radii,
                                                    low,
                                                    up,
                                                    peak_num,
                                                    background_voigt_num,
                                                    hkl_guess,
                                                    filename,
                                                    full_spectrum_fit,
                                                    popt_old,
                                                    reuse_last_fit_params=reuse_last_fit_params,
                                                    show=show, )
                

                #add peaks to indices
                indices[num] = {**fit_dict, **peak_fits}
                
                ##get metadata:
                if file.endswith('.emi'):
                    
                    cal_x, time, illuArea, z, c2, c3  = get_emi_metadata(file)
                    
                    #create dummy variables for code compatibility
                    conv = x = y = None
                
                elif file.endswith('.emd'):
                    
                    cal_x, time, conv, x, y, z, c2, c3 = get_emd_metadata(file)
                    
                    #create dummy variables for code compatibility
                    illuArea = None
                
                elif file.endswith('.dm4'):
                    
                    cal_x, time, illuArea, z, c2, c3  = get_dm_metadata(file)
                    
                    #create dummy variables for code compatibility
                    conv = x = y = None
                
                else:
                    #create dummy variables for code compatibility
                    cal_x = time = conv = z = y = x = c2 = c3 = illuArea = None
            
                #save metadata
                indices[num]['cal_x'] = cal_x
                indices[num]['time / s'] = time
                indices[num]['conv'] = conv
                indices[num]['z height'] = z
                indices[num]['x position'] = x
                indices[num]['y position'] = y
                indices[num]['C2'] = c2
                indices[num]['C3'] = c3
                indices[num]['illuArea'] = illuArea
                try:
                    indices[num]['C2/C3'] = c2/c3  
                    
                except TypeError:
                    
                    indices[num]['C2/C3'] = None
                    logging.warning(f'TypeError for {filename} during "C2/C3" assignment.')
            
            #log error and continue with next pattern
            except Exception as e:
                
                #log
                message = '{} in file {}.'.format(type(e).__name__, filename)
                logging.critical(message)
                logging.critical(traceback.format_exc())

                num += 1
                
                raise e
                continue
            
            num += 1


    #create a DataFrame 
    logging.info('Writing DataFrame')
    df = pd.DataFrame.from_dict(indices, orient='index')
        
    #ensure that no unnecessary file ending is created:
    if store:
        dfname = dfname.split('.xls')[0]
        df2excel(df, dfname)
        #write spectra to npz file:
        np.savez_compressed(os.path.join('.', 'output', dfname),
                            *corrected_spectra)
    
    print('''
          If you find this script helpful, please cite our work:\n
          B. Fritsch et al., "Sub-Kelvin thermometry for evaluating the local temperature stability within in situ TEM gas cells",
          Ultramicroscopy, 2022, 113494,
          https://doi.org/10.1016/j.ultramic.2022.113494''')
          
    logging.info('Finished')
    
    return df

#%%

filename = 'Github Test' #Name of the related Excel sheet.
#
####    
for handler in logging.root.handlers[:]:
    
    logging.root.removeHandler(handler)
    
logging.basicConfig(level=logging.WARNING,
                    format= '%(asctime)s - %(levelname)s - %(message)s',
                    filename='{} {}.log'.format(
                            reduce(lambda x,y: x+y, str(datetime.datetime.now()).split('.')[0].split(':')),
                            filename))


radius_range = 1450/4096, 1750/4096 #Radial range of the diffraction ring that should be used for distortion correction. Relative numbers are preferred, although absolute numbers in px units work, too.

df = main(show = True,                    #True: Processing steps are shown, slows algorithm down
          show_test = False,              #True: Shows the processing step only every "test intervals"th time in a 3D data stack  
          test = False,                   #True: Processes only every "test intervals"th pattern in a 3D data stack
          test_interval = 100,            #See above. Note that count starts at 0
          tolerance=0.05,                 #Precision of Center finding in px. The smaller, the longer it takes. I remain in [0.01, 0.1] 
          radius_range=radius_range,      #see above
          int_thres= '0.1 median',        #For calculating the intensity threshold during pixel detection. If not enough pixels are found, try "mean". If that doesn't work, try numbers between 0 and 1.
          hough_radius_range = (0.2,0.5), #Relative Radius range for Hough transform. It tells, which radii to consider for automated center finding
          min_vals_per_ring = 20,         #Minimum spots per Ring.
          hough_rings = 1,                #Number of Hough rings to find. If first center guess is bad, increase to 2 or 3. 
          background_voigt_num=3,         #Number of Voigt peaks used to fit the amorphous background.
          scale = 10,                     #Downscaling factor for Hough-transform
          hkl_guess = (2,2,0),            #hkl indices of the diffraction ring within radius_range as tuple
          use_last_center=True,           #If multple patterns are given: Reuses last center guess to skip Hough transform if truthy
          re_pattern='tif+$',             #A regular expression pattern can be provided to select different file types or prestructured files by their name. Loading is performed via hyperspy. Particularly, it is tested for the following formats: emd, tiff, hspy
          full_spectrum_fit = True,       #if truthy: the radially integrated spectrum is fitted in a single least square optimization. 
          reuse_last_fit_params = True,   #If multple patterns are given: Reuses last optimized fit parameters as starting conditions.
          export_polar=True,              #A regular expression pattern can be provided to select different file types or prestructured files by their name. Loading is performed lazily via hyperspy. Particularly, it is tested for the following formats: emd, tiff, hspy
          store=True,                     #If truthy: Store results on the hard drive
          dfname = filename,              #see above
          )
