# Defines all the intermediate functions that read in data from APF data files
# and calculates radial velocities

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from astropy.io import fits
import astropy.io.ascii
from scipy.io import readsav
from scipy import interpolate
from scipy.fftpack import fft, ifft
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy import special
from scipy import stats
from scipy.optimize import curve_fit
import math
import os
from astropy.table import Table
from astropy.table import Table, join
from astropy.table import Table, vstack
from astropy.table import Table, unique
from astropy.table import Column
import numpy.core.defchararray as np_f
import pandas as pd

# dictionary of star names from common name to file name
# key = names from APF proposal, with spaces and - changed to _
star_dict = dict([('245185', 'HD 245185'), ('36112', 'HD 36112'), ('142666', 'HD 142666'), 
('HIP26295', 'HIP 26295'), ('294268', 'HD 294268'),
('HIP20777', 'DF Tau'), ('W96-899', 'W96 899'), ('HIP20387', 'RY Tau'), 
('LKHA330', 'LkHa 330'), ('283447','HD 283447'), ('DG_TAU','DG Tau'),
('DH_TAU','DH Tau'), ('GK_TAU','GK Tau'), ('HIP22910','AB Aur'),
('TX_ORI','TX ORi'), ('DS_TAU','DS Tau'), ('143006','HD 143006'), 
('EPIC7221','2M J1609-2217'), ('T67938191','CD-22 11432'), ('EPIC2099','VV Sco'), 
('RXJ1615','RX J1615.3-3255'), ('2M1626-23','2M J1626-2356'), ('2M1639-24','2M J1639-2402'), 
('HIP82323','AS 209'), ('163296','HD 163296'), ('2M2023P42','V1515 Cyg'), 
('EPIC9328','AS 205'), ('HIP34042', 'Z CMa') ]) 

C = 299792.458 #in km/s


def rfftfreq(n, d=1.0):
    """
    Return the Discrete Fourier Transform sample frequencies
    (for usage with rfft, irfft).
    The returned float array `f` contains the frequency bin centers in cycles
    per unit of the sample spacing (with zero at the start). For instance, if
    the sample spacing is in seconds, then the frequency unit is cycles/second.
    Given a window length `n` and a sample spacing `d`::
    f = [0, 1, ..., n/2-1, n/2] / (d*n) if n is even
    f = [0, 1, ..., (n-1)/2-1, (n-1)/2] / (d*n) if n is odd
    Unlike `fftfreq` (but like `scipy.fftpack.rfftfreq`)
    the Nyquist frequency component is considered to be positive.
    :param n : Window length
    :type n: int
    :param d: Sample spacing (inverse of the sampling rate). Defaults to 1.
    ;type d: scalar, optional
    :returns: f, Array of length ``n//2 + 1`` containing the sample frequencies.
    :rtype: ndarray
    """
    if not isinstance(n, np.int):
        raise ValueError("n should be an integer")
    val = 1.0 / (n * d)
    N = n // 2 + 1
    results = np.arange(0, N, dtype=np.int)
    return results * val


def syn_spectrum(syn_wl_file_name, syn_fl_file_name, vsini, syn_wl_min, syn_wl_max):
    '''given a PHOENIX file name, return the corresponding wavelength 
    and flux arrays'''

    #read wavelength
    fname = '/Users/zoeko/desktop/research/phoenix_files' + '/' + syn_wl_file_name
    hdul = fits.open(fname)
    hdu = hdul[0]
    hdr = hdu.header
    wl_native = hdu.data
    
    #read flux
    fname = '/Users/zoeko/desktop/research/phoenix_files' + '/' + syn_fl_file_name
    hdul = fits.open(fname)
    hdu = hdul[0]
    hdr = hdu.header
    fl_native = hdu.data    
    
    # Set up the Fourier coordinate
    ss = rfftfreq(len(wl_native))
    
    # Set up the instrumental taper
    FWHM = 3
    sigma =  FWHM / 2.35  # in km/s
   
    # Instrumentally broaden the spectrum by multiplying with a Gaussian in Fourier space
    # this is the fourier transform of the instrumental kernel
    taper = np.exp(-2 * (np.pi ** 2) * (sigma ** 2) * (ss ** 2))
    
    # Do the FFT
    FF_fl = np.fft.rfft(fl_native)
    
    # the calculation of the broadening kernel requires a division by s
    # and since the ss array contains 0 at the first index, 
    # we're going to a divide by zero. So, we first just 
    # set this to a junk value so we don't get a divide by zero error
    ss[0] = 0.01  
   
    # Calculate the stellar broadening kernel (from Gray 2005, Eqn 18.14)
    ub = 2. * np.pi * vsini * ss
    sb = special.j1(ub) / ub - 3 * np.cos(ub) / (2 * ub ** 2) + 3. * np.sin(ub) / (2 * ub ** 3)

    # we know what sb(ss=0) should be, so just set this directly to this value
    # since we messed up the ss coordinate for this component earlier
    # set zeroth frequency to 1 separately (DC term)
    sb[0] = 1.

    # institute vsini and instrumental taper
    FF_tap = FF_fl * sb * taper

    # do IFFT
    fl_tapered = np.fft.irfft(FF_tap)

    # truncate to be from syn_wl_min to syn_wl_max

    min_differences_array = abs(wl_native - syn_wl_min)
    ind_min = int(np.where(min_differences_array == np.amin(min_differences_array))[0])

    max_differences_array = abs(wl_native - syn_wl_max)
    ind_max = int((np.where(max_differences_array == np.amin(max_differences_array)))[0])

    wl_syn = wl_native[ind_min:ind_max]
    fl_syn = fl_tapered[ind_min:ind_max]

    #noramlize synthetic spectrum
    slope, intercept, r_value, p_value, std_err = stats.linregress(wl_syn, fl_syn)
    fl_syn_fit = slope * wl_syn + intercept
    fl_syn_norm = fl_syn / fl_syn_fit


    # truncate ORIGINAl flux array to be from syn_wl_min to syn_wl_max

    fl_syn_orig = fl_native[ind_min:ind_max] / np.max(fl_native[ind_min:ind_max])

    # convert from vaccuum to air
    # wl_syn = wl_syn / (1.0 + (2.735182 * 10 ** -4) + 131.4182 / wl_syn ** 2 + 2.76249E8 / wl_syn ** 4)

    return wl_syn, fl_syn_norm, fl_syn_orig


def APF_spectrum(APF_wl_file_name, APF_fl_file_name, order, APF_wl_min, APF_wl_max, truncate_after_blaze = False, truncate_min = 0, truncate_max = 10000):
    '''given an APF file name, return the corresponding wavelength 
    and flux arrays'''

    # read data
    w_idl = readsav('/Users/zoeko/desktop/research/newest_data/files/' + APF_wl_file_name)
    wa = w_idl["w"]  # wavelengths in AA
    fl_idl = readsav('/Users/zoeko/desktop/research/newest_data/files/' + APF_fl_file_name)
    fl = fl_idl["sp"]  # fluxes in arbitrary units

    # select echelle order
    wl_APF_orig = wa[order]
    fl_APF_orig = fl[order]

    # truncate wavelength and corresponding flux to be between wl_min and wl_max
    wl_below_max = (wl_APF_orig < APF_wl_max)
    wl_above_min = (wl_APF_orig > APF_wl_min)
    wl_APF = wl_APF_orig[wl_below_max * wl_above_min]
    fl_APF = fl_APF_orig[wl_below_max * wl_above_min]

    #normalize APF data by fitting the blaze function

    # find the wl value in the middle of the array
    ind = int(len(wl_APF_orig) / 2)
    center = wl_APF_orig[ind]

    #cheb fit the masked sections of data
    coef = np.polynomial.chebyshev.chebfit(wl_APF-center, fl_APF, deg = 5)

    #chebval to original wavelength grid
    fl_APF_fit = np.polynomial.chebyshev.chebval(wl_APF-center,coef)

    #divide original flux by fit
    fl_APF_norm = fl_APF / fl_APF_fit
    
    if truncate_after_blaze == True:
        wl_below_max = (wl_APF < truncate_max)
        wl_above_min = (wl_APF > truncate_min)
        wl_APF = wl_APF[wl_below_max * wl_above_min]
        fl_APF = fl_APF[wl_below_max * wl_above_min]
        fl_APF_fit = fl_APF_fit[wl_below_max * wl_above_min]
        fl_APF_norm = fl_APF_norm[wl_below_max * wl_above_min]
    
    return wl_APF, fl_APF, fl_APF_fit, fl_APF_norm


def model(velocity, wl_syn, fl_syn_norm, wl_APF):
    '''given a velocity shift, return corresponding
    shifted wavelength array, interpolate shifted wl_syn onto orig fl_syn
    interpolate onto wl_APF to get corresponding APF flux values'''
    shifted_wl_syn = wl_syn[500:(len(wl_syn)-500)] - (wl_syn[500:(len(wl_syn)-500)] * velocity / C)
    dopshift_func = interpolate.interp1d(shifted_wl_syn, fl_syn_norm[500:(len(wl_syn)-500)], kind = 'cubic')
    return dopshift_func(wl_APF)


def chi(model, data, sigma):
    '''given two arrays of the same length,
    calculate chi squared'''
    return np.sum((data - model) ** 2 / sigma ** 2)


def calc_rad_vel(star, APF_wl_file_name, APF_fl_file_name, JD, plot_intermediates, 
syn_wl_file_name, syn_fl_file_name, vsini, syn_wl_min, syn_wl_max, order, APF_wl_min, 
APF_wl_max, vel_min, vel_max, vel_spacing):

    '''Calculates radial velocity for an order number of a given star'''

    syn = syn_spectrum(syn_wl_file_name, syn_fl_file_name, vsini, syn_wl_min, syn_wl_max)
    wl_syn = syn[0]
    fl_syn_norm = syn[1]

    APF = APF_spectrum(APF_wl_file_name, APF_fl_file_name, order, APF_wl_min, APF_wl_max)
    wl_APF = APF[0]
    fl_APF_orig = APF[1]
    fl_APF_fit = APF[2]
    fl_APF_norm = APF[3]

    if plot_intermediates == True:

        if os.path.isdir('/Users/zoeko/desktop/APF_stars/' + star + '/' + str(order)) == False:
                os.makedirs('/Users/zoeko/desktop/APF_stars/' + star + '/' + str(order))

        os.makedirs('/Users/zoeko/desktop/APF_stars/' + star + '/' + str(order) + '/' + str(JD))

        fig, ax = plt.subplots()
        ax.plot(wl_APF, fl_APF_orig)
        ax.set_xlabel('r"$\lambda$ [$\AA$]"')
        ax.set_ylabel('flux')
        ax.set_title('APF Spectrum for Order ' + str(order))
        # plt.show()
        fig.savefig('/Users/zoeko/desktop/APF_stars/' + star + '/' + str(order) + '/' + str(JD) + '/APF_spectrum.png')

        fig, ax = plt.subplots()
        ax.plot(wl_syn, fl_syn_norm, label = 'Synthetic Spectrum') # plot vel shifted spectrum from PHOENIX
        ax.plot(wl_APF, fl_APF_norm, label = 'APF Spectrum') # plot spectrum from APF
        ax.set_title('Synthetic vs. APF')
        ax.set_xlabel('r"$\lambda$ [$\AA$]"')
        ax.set_ylabel('flux')
        ax.legend()
        fig.savefig('/Users/zoeko/desktop/APF_stars/' + star + '/' + str(order) + '/' + str(JD) + '/Synthetic_APF.png')

    # find sigma array
    noise = np.sqrt(fl_APF_orig)
    normalized_noise = noise / fl_APF_fit

    # plot up chi squared values for different velocities
    vel_values = np.arange(vel_min, vel_max, vel_spacing)
    chi_squared_values = []

    for vel in vel_values:
        chi_squared = chi(model(vel, wl_syn, fl_syn_norm, wl_APF), fl_APF_norm, normalized_noise)
        chi_squared_values = np.append(chi_squared_values, chi_squared)

    # find the velocity shift corresponding to the miminum chi squared value
    min_chi = np.amin(chi_squared_values)
    vel_shift = vel_values[chi_squared_values.tolist().index(min_chi)]

    if plot_intermediates == True:

        fig, ax = plt.subplots()
        ax.plot(vel_values, chi_squared_values)
        ax.set_xlabel('NEGATIVE Velocity km/s')
        ax.set_ylabel('Chi Squared')
        ax.set_title('Chi Squared Values')
        fig.savefig('/Users/zoeko/desktop/APF_stars/' + star + '/' + str(order) + '/' + str(JD) + '/chi_squared_vals.png')

        fig, ax = plt.subplots()
        ax.plot(wl_APF, fl_APF_norm, label = 'Data') #plot spectrum from APF
        ax.plot(wl_APF, model(vel_shift, wl_syn, fl_syn_norm, wl_APF), label = 'Shifted Synthetic Spectrum') #plot vel shifted spectrum from PHOENIX
        ax.set_title('Velocity Shifted Synthetic Spectrum vs. Data')
        ax.set_xlabel('$\lambda$ [$\AA$]')
        ax.set_ylabel('Flux')
        ax.legend()
        fig.savefig('/Users/zoeko/desktop/APF_stars/' + star + '/' + str(order)+ '/' + str(JD) + '/vel_shifted_syn_vs_data.pdf')

        plt.close('all')

    return vel_shift

# def read_files(*file_dates):
#     '''Given a date of the file (ex: june19 or dec19) 
#     return the compiled data in all files in form of astropy table'''
#     table = Table()
#     for file in file_dates:
#         fname = '/Users/zoeko/desktop/research/' + file + '_spectra_files/czekala_master.txt'
#         data = astropy.io.ascii.read(
#             fname,
#             names=[
#                 "fl_file",
#                 "name",
#                 "BARY",  # Barycentric correction in m/s. Add this to the RV to bring into BARY frame.
#                 "JD",  # Julian date.
#                 "temp1",
#                 "type",
#                 "temp2",
#                 "temp3",
#                 "temp4",
#                 "temp5",
#             ],
#         )
#         table = vstack([table, data])
#         table = unique(table)
#     return table.group_by('name')

def star_table(data, star, wl_file_name = 'w_rl23.2142', plot_intermediates = False, plot_radvels = False,
                syn_wl_file_name = 'WAVE_PHOENIX-ACES-AGSS-COND-2011.fits', 
                syn_fl_file_name = 'lte07600-4.50-0.0.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits', 
                vsini = 49, syn_wl_min = 5300, syn_wl_max = 5800, 
                order = 39, APF_wl_min = 5444, APF_wl_max = 5516,
                vel_min = -60, vel_max = 60, vel_spacing = 0.5):

    '''given the czekala_master.txt file in the form of an astropy table, 
    the star name, and the wavelength file name,
    return a table for that star with its radial 
    velocity measurements for each epoch and spectrum'''

    mask = data.groups.keys['name'] == star
    star_data = data.groups[mask]

    # find rad vel values for each spectrum
    fl_file_names = np.array(star_data['fl_file'])
    rad_vels = np.array([])
    for fl_file in fl_file_names:
        # find JD corresponding to fl_file
        fl_grouped_star_data = star_data.group_by('fl_file')
        mask = fl_grouped_star_data.groups.keys['fl_file'] == fl_file
        fl_grouped_star_data = fl_grouped_star_data.groups[mask]
        JD = np.array(list(fl_grouped_star_data[0])).item(3)
        vel = calc_rad_vel(star, wl_file_name, fl_file, JD, plot_intermediates, 
        syn_wl_file_name, syn_fl_file_name, vsini, syn_wl_min, syn_wl_max, 
        order, APF_wl_min, APF_wl_max, vel_min, vel_max, vel_spacing)
        rad_vels = np.append(rad_vels, vel)
    
    # add barycentric correction to radial velocities
    bary_corr = np.array(star_data['BARY'])/1000  # convert m/s to km/s
    vels_with_bary_corr = - rad_vels + bary_corr # MAKE VELS NEGATIVE TO ACCOUNT FOR ERROR IN CALCULATING OFFSET, change to + barycentric correction??
    rad_vel_column = Column(name='order ' + str(order) + ' radial velocity (km/s)', data=vels_with_bary_corr)
    star_data.add_column(rad_vel_column)
    star_data.remove_columns(['fl_file', 'name', 'type', 'temp1', 'temp2', 'temp3', 'temp4', 'temp5', 'BARY'])

    if plot_radvels == True:
        jd = star_data['JD']      
        vel = star_data['order ' + str(order) + ' radial velocity (km/s)']
        fig, ax = plt.subplots()
        ax.scatter(jd, vel)
        ax.set_title('Radial Velocity Measurements for Order ' + str(order))
        ax.set_xlabel('JD')
        ax.set_ylabel('Radial Velocity (km/s)')
        # ax.legend()
        fig.savefig('/Users/zoeko/desktop/APF_stars/' + star + '/' + str(order) + '/rad_vels.png')
        plt.close('all')

    return star_data

def combine_orders(star, table1, table2, table3, plot_combined = False):
    com = join(table1, table2)
    com = join(com, table3)
    if plot_combined == True:
        fig, ax = plt.subplots()
        ax.scatter(table1['JD'], table1.columns[1], label = ((table1.colnames)[1]).replace(' radial velocity (km/s)', ' '))
        ax.scatter(table2['JD'], table2.columns[1], label = ((table2.colnames)[1]).replace(' radial velocity (km/s)', ' '))
        ax.scatter(table3['JD'], table3.columns[1], label = ((table3.colnames)[1]).replace(' radial velocity (km/s)', ' '))
        ax.set_xlabel('JD', fontsize=16)
        ax.set_ylabel('radial velocity (km/s)', fontsize=16)
        ax.set_title('Radial Velocities for ' + star_dict[str(star)])
        # ax.set_title('Radial Velocities for Star Y', fontsize=20)
        plt.xticks(fontsize=13)
        plt.yticks(fontsize=13)
        ax.legend(prop={'size': 15})
        fig.savefig('/Users/zoeko/desktop/APF_stars/' + star + '/' + str(star) + '.pdf')
    return com

# def combine_orders(star, firstorder, *otherorders):
#     ''' use MEAN '''
#     star = str(star)
#     JDs = firstorder.columns[0]
#     average_rvs = firstorder.columns[1]
#     total_orders = 1
#     for order in otherorders:
#         average_rvs += order.columns[1]
#         total_orders += 1
#     average_rvs = average_rvs / total_orders
#     combined = Table()
#     combined.add_column(JDs, name = 'JDs')
#     combined.add_column(average_rvs, name = 'Radial Velocity')
#     combined.add_column([star] * len(combined), index = 0, name = 'Star')
#     return combined




# x = np.arange(10)
# y = np.arange(10)
# z = np.arange(10)

# total = [x.tolist(), y.tolist(), z.tolist()]

# # print(total)

# df = pd.DataFrame()
# for i in np.arange(len(total)):
#     df['order ' + str(i)] = total[i]

# for index, row in df.iterrows():
#     print(row.tolist())


def calc_vsini(APF_wl_file_name, APF_fl_file_name,
syn_wl_file_name, syn_fl_file_name, vsini_estimate,
syn_wl_min, syn_wl_max,
order, APF_wl_min, APF_wl_max,
vel_min = -60, vel_max = 60, vel_spacing = 0.5,
vsini_min = 10, vsini_max = 160, vsini_spacing = 0.5,
plot = True):

    '''given a data file and a vsini estimate, calculate the rad vel shift
    and then calculate the vsini value that would minimize chi squared
    for the velocity shifted synthetic spectrum and data'''

    # find rad vel measurement givin a vsini estimate
    syn = syn_spectrum(syn_wl_file_name, syn_fl_file_name, vsini_estimate, syn_wl_min, syn_wl_max)
    wl_syn = syn[0]
    fl_syn_norm = syn[1]
    fl_syn_orig = syn[2]

    APF = APF_spectrum(APF_wl_file_name, APF_fl_file_name, order, APF_wl_min, APF_wl_max)
    wl_APF = APF[0]
    fl_APF_orig = APF[1]
    fl_APF_fit = APF[2]
    fl_APF_norm = APF[3]

    # find sigma array
    noise = np.sqrt(fl_APF_orig)
    normalized_noise = noise / fl_APF_fit

    # plot up chi squared values for different velocities
    vel_values = np.arange(vel_min, vel_max, vel_spacing)
    chi_squared_values = []

    for vel in vel_values:
        chi_squared = chi(model(vel, wl_syn, fl_syn_norm, wl_APF), fl_APF_norm, normalized_noise)
        chi_squared_values = np.append(chi_squared_values, chi_squared)

    # find the velocity shift corresponding to the miminum chi squared value
    min_chi = np.amin(chi_squared_values)
    vel_shift = vel_values[chi_squared_values.tolist().index(min_chi)]



    # shift the unprocessed synthetic spectrum by vel_shift and truncate it
    # so it matches APF data
    shifted_wl_syn = wl_syn[500:(len(wl_syn)-500)] - (wl_syn[500:(len(wl_syn)-500)] * vel_shift / C)
    dopshift_func = interpolate.interp1d(shifted_wl_syn, fl_syn_orig[500:(len(wl_syn)-500)], kind = 'cubic')  # CHANGED FL_SYN_NORM TO FL_SYN_ORIG
    shifted_fl_syn = dopshift_func(wl_APF)

    # process synthetic spectrum

    wl_native = wl_APF
    fl_native = shifted_fl_syn
        
    # Set up the Fourier coordinate
    ss = rfftfreq(len(wl_native))
        
    # Set up the instrumental taper
    FWHM = 3
    sigma =  FWHM / 2.35  # in km/s
    
    # Instrumentally broaden the spectrum by multiplying with a Gaussian in Fourier space
    # this is the fourier transform of the instrumental kernel
    taper = np.exp(-2 * (np.pi ** 2) * (sigma ** 2) * (ss ** 2))
        
    # Do the FFT
    FF_fl = np.fft.rfft(fl_native)
        
    # the calculation of the broadening kernel requires a division by s
    # and since the ss array contains 0 at the first index, 
    # we're going to a divide by zero. So, we first just 
    # set this to a junk value so we don't get a divide by zero error
    ss[0] = 0.01  

    # calculate chi squared value for each vsini value

    chi_squared_values = np.array([])

    for vsini in np.arange(vsini_min, vsini_max, vsini_spacing):

        # Calculate the stellar broadening kernel (from Gray 2005, Eqn 18.14)
        ub = 2. * np.pi * vsini * ss
        sb = special.j1(ub) / ub - 3 * np.cos(ub) / (2 * ub ** 2) + 3. * np.sin(ub) / (2 * ub ** 3)

        # we know what sb(ss=0) should be, so just set this directly to this value
        # since we messed up the ss coordinate for this component earlier
        # set zeroth frequency to 1 separately (DC term)
        sb[0] = 1.

        # institute vsini and instrumental taper
        FF_tap = FF_fl * sb * taper

        # do IFFT
        fl_tapered = np.fft.irfft(FF_tap)

        wl_syn = wl_native
        fl_syn = fl_tapered

        #noramlize synthetic spectrum
        slope, intercept, r_value, p_value, std_err = stats.linregress(wl_syn, fl_syn)
        fl_syn_fit = slope * wl_syn + intercept
        fl_syn_norm = fl_syn / fl_syn_fit

        # fig, ax = plt.subplots()
        # ax.plot(wl_APF, fl_syn_norm)
        # ax.plot(wl_APF, fl_APF_norm)
        # plt.show()

        # NOISE
        noise = np.sqrt(fl_APF_orig)
        normalized_noise = noise / fl_APF_fit

        chi_val = chi(fl_syn_norm, fl_APF_norm, 1)
        chi_squared_values = np.append(chi_squared_values, chi_val)

    vsini_values = np.arange(vsini_min, vsini_max, vsini_spacing)

    # find the vsini value corresponding to the miminum chi squared value
    min_chi = np.amin(chi_squared_values)
    vsini_value = vsini_values[chi_squared_values.tolist().index(min_chi)]



    # process spectrum according to correct vsini value
    # Calculate the stellar broadening kernel (from Gray 2005, Eqn 18.14)
    ub = 2. * np.pi * vsini_value * ss
    sb = special.j1(ub) / ub - 3 * np.cos(ub) / (2 * ub ** 2) + 3. * np.sin(ub) / (2 * ub ** 3)

    # we know what sb(ss=0) should be, so just set this directly to this value
    # since we messed up the ss coordinate for this component earlier
    # set zeroth frequency to 1 separately (DC term)
    sb[0] = 1.

    # institute vsini and instrumental taper
    FF_tap = FF_fl * sb * taper

    # do IFFT
    fl_tapered = np.fft.irfft(FF_tap)

    wl_syn = wl_native
    fl_syn = fl_tapered

    #noramlize synthetic spectrum
    slope, intercept, r_value, p_value, std_err = stats.linregress(wl_syn, fl_syn)
    fl_syn_fit = slope * wl_syn + intercept
    fl_syn_norm = fl_syn / fl_syn_fit


    if plot == True:
        fig, ax = plt.subplots()
        ax.plot(wl_APF, fl_APF_norm - fl_syn_norm)
        ax.set_xlabel('r"$\lambda$ [$\AA$]"')
        ax.set_ylabel('flux')
        ax.set_title('Residuals: Data minus Model')
        print('Average of Residuals: ', np.average(fl_APF_norm - fl_syn_norm))

        fig, ax = plt.subplots()
        ax.plot(wl_APF, fl_APF_norm, label = 'APF Spectrum')
        ax.plot(wl_APF, fl_syn_norm, label = 'Shifted Synthetic Spectrum')
        ax.set_xlabel('r"$\lambda$ [$\AA$]"')
        ax.set_ylabel('flux')
        ax.legend()
        ax.set_title('Vel Shifted Syn Spectrum with vsini against Data')

        fig, ax = plt.subplots()
        ax.plot(vsini_values, chi_squared_values)
        ax.set_xlabel('vsini')
        ax.set_ylabel('chi squared')
        ax.set_title('vsini against chi squared')

        
        
    # process spectrum according to vsini ESTIMATE
    # Calculate the stellar broadening kernel (from Gray 2005, Eqn 18.14)
    ub = 2. * np.pi * vsini_estimate * ss
    sb = special.j1(ub) / ub - 3 * np.cos(ub) / (2 * ub ** 2) + 3. * np.sin(ub) / (2 * ub ** 3)

    # we know what sb(ss=0) should be, so just set this directly to this value
    # since we messed up the ss coordinate for this component earlier
    # set zeroth frequency to 1 separately (DC term)
    sb[0] = 1.

    # institute vsini and instrumental taper
    FF_tap = FF_fl * sb * taper

    # do IFFT
    fl_tapered = np.fft.irfft(FF_tap)

    wl_syn = wl_native
    fl_syn = fl_tapered

    #noramlize synthetic spectrum
    slope, intercept, r_value, p_value, std_err = stats.linregress(wl_syn, fl_syn)
    fl_syn_fit = slope * wl_syn + intercept
    fl_syn_norm = fl_syn / fl_syn_fit
        
    if plot == True:
        fig, ax = plt.subplots()
        ax.plot(wl_APF, fl_APF_norm, label = 'APF Spectrum')
        ax.plot(wl_APF, fl_syn_norm, label = 'Shifted Synthetic Spectrum')
        ax.set_xlabel('r"$\lambda$ [$\AA$]"')
        ax.set_ylabel('flux')
        ax.set_title('Vel Shifted Syn Spectrum with vsini Estimate against Data')
        plt.show()

    return vsini_value

def blaze_normalization(order, APF_fl_file_name, APF_wl_min, APF_wl_max):
    '''find the chebyshev polynomial that fits the blaze function
    for star HD245185 for the given order and given flux file'''

    # read data
    w_idl = readsav('/Users/zoeko/desktop/research/newest_data/files/w_rl23.2142')
    wa = w_idl["w"]  # wavelengths in AA
    fl_idl = readsav('/Users/zoeko/desktop/research/newest_data/files/' + APF_fl_file_name)
    fl = fl_idl["sp"]  # fluxes in arbitrary units

    # select echelle order
    wl_APF_orig = wa[order]
    fl_APF_orig = fl[order]

    # truncate wavelength and corresponding flux to be between wl_min and wl_max
    wl_below_max = (wl_APF_orig < APF_wl_max)
    wl_above_min = (wl_APF_orig > APF_wl_min)
    wl_APF = wl_APF_orig[wl_below_max * wl_above_min]
    fl_APF = fl_APF_orig[wl_below_max * wl_above_min]

    #fit blaze function

    # find the wl value in the middle of the array
    ind = int(len(wl_APF_orig) / 2)
    center = wl_APF_orig[ind]

    #cheb fit data
    coef = np.polynomial.chebyshev.chebfit(wl_APF-center, fl_APF, deg = 5)

    #chebval to original wavelength grid
    fl_APF_fit = np.polynomial.chebyshev.chebval(wl_APF-center,coef)

    #divide original flux by fit
    fl_APF_norm = fl_APF / fl_APF_fit

    # fig, ax = plt.subplots()
    # ax.plot(wl_APF, fl_APF)
    # ax.plot(wl_APF, fl_APF_fit)
    # plt.show()

    return wl_APF, fl_APF_fit

# for using HD245185 as blaze function fit

order38_blaze_norm = blaze_normalization(order = 38, APF_fl_file_name = 'rl27.7459', APF_wl_min = 5380, APF_wl_max = 5460)
# interpolate onto wavelength grid
blaze_fit_38 = interpolate.interp1d(order38_blaze_norm[0], order38_blaze_norm[1], kind = 'cubic')

order39_blaze_norm = blaze_normalization(order = 39, APF_fl_file_name = 'rl27.2625', APF_wl_min = 5450, APF_wl_max = 5520)
blaze_fit_39 = interpolate.interp1d(order39_blaze_norm[0], order39_blaze_norm[1], kind = 'cubic')

order40_blaze_norm = blaze_normalization(order = 40, APF_fl_file_name = 'rl25.9179', APF_wl_min = 5510, APF_wl_max = 5585)
blaze_fit_40 = interpolate.interp1d(order40_blaze_norm[0], order40_blaze_norm[1], kind = 'cubic')

# def APF_spectrum(APF_wl_file_name, APF_fl_file_name, order, APF_wl_min, APF_wl_max, data_date):
#     '''given an APF file name, return the corresponding wavelength 
#     and flux arrays'''

#     #normalize APF data by fitting the blaze function
#     if order == 38:
#         APF_wl_min = 5380
#         APF_wl_max = 5460
#     if order == 39:
#         APF_wl_min = 5450
#         APF_wl_max = 5520
#     if order == 40:
#         APF_wl_min = 5510
#         APF_wl_max = 5585

#     # read data
#     w_idl = readsav('/Users/zoeko/desktop/research/june19_spectra_files/files/' + APF_wl_file_name)
#     wa = w_idl["w"]  # wavelengths in AA
#     fl_idl = readsav('/Users/zoeko/desktop/research/' + data_date + '_spectra_files/files/' + APF_fl_file_name)
#     fl = fl_idl["sp"]  # fluxes in arbitrary units

#     # select echelle order
#     wl_APF_orig = wa[order]
#     fl_APF_orig = fl[order]

#     # truncate wavelength and corresponding flux to be between wl_min and wl_max
#     wl_below_max = (wl_APF_orig < APF_wl_max)
#     wl_above_min = (wl_APF_orig > APF_wl_min)
#     wl_APF = wl_APF_orig[wl_below_max * wl_above_min]
#     fl_APF = fl_APF_orig[wl_below_max * wl_above_min]

#     #chi squared for blaze function
#     sigma = (fl_APF) ** 0.5 
#     blaze_chi = np.array([])
#     a_values = np.arange(0, 10, 0.1)
#     if order == 38:
#         blaze_fit = blaze_fit_38(wl_APF)
#     if order ==39:
#         blaze_fit = blaze_fit_39(wl_APF)
#     if order ==40:
#         blaze_fit = blaze_fit_40(wl_APF)

#     for a in a_values:
#         one_blaze_chi = np.sum((fl_APF - (a * blaze_fit)) ** 2 / sigma ** 2)  # replace order39_blaze_norm[1] with blaze_fit_39(wl_APF)
#         blaze_chi = np.append(blaze_chi, one_blaze_chi)

#     min_chi = np.amin(blaze_chi)
#     a_value = a_values[blaze_chi.tolist().index(min_chi)]

#     blaze_function_fit = a_value * blaze_fit  # replace order39_blaze_norm[1] with blaze_fit_39(wl_APF)
#     fl_APF_norm = fl_APF / blaze_function_fit

#     fl_APF_fit = blaze_fit
    
#     return wl_APF, fl_APF, fl_APF_fit, fl_APF_norm


def duration(data, star):
    '''given the czekala_master.txt file in the form of an astropy table, 
    and the star name
    return the duration of its measurements'''

    mask = data.groups.keys['name'] == star
    star_data = data.groups[mask]

    # print(star_data)
    dates = np.array(star_data['JD'])
    # print(len(dates))
    if len(dates) == 0:
        return 0
    duration = max(dates) - min(dates)
    return duration
