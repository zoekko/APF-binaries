# Save RV values for given target into a csv file under research/RVdata
# Copy code from calc_rad_vel, change all plots to False, DELETE PLOT_COMBINED
# Add an ascii write command

# SAVING ALL RVs:
# 1. Save individual RVs using this file.
# 2. Stack all RV measurements together using "saving_all_rvs.py".

import pandas as pd
import matplotlib.pyplot as plt
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
import astropy
from astropy.table import Table
from astropy.table import Table, join
from astropy.table import Table, vstack
from astropy.table import Table, unique
from astropy.table import Column
import numpy.core.defchararray as np_f
from astropy.table import Table, vstack
from astropy.table import Table, unique
from functions import rfftfreq
from functions import syn_spectrum
from functions import APF_spectrum
from functions import model
from functions import chi
from functions import calc_rad_vel
from functions import star_table
from functions import combine_orders
from functions import blaze_normalization
from astropy.io import ascii

C = 299792.458 #in km/s

# read key

fname = "/Users/zoeko/desktop/research/newest_data/czekala_master.txt"

data = astropy.io.ascii.read(
    fname,
    names=[
        "fl_file",
        "name",
        "BARY",  # Barycentric correction in m/s. Add this to the RV to bring into BARY frame.
        "JD",  # Julian date.
        "temp1",
        "type",
        "temp2",
        "temp3",
        "temp4",
        "temp5",
    ],
)

data = data.group_by('name')

directory = '/Users/zoeko/desktop/research/separated/'

'''for HD142666'''

table_142666_order38 = star_table(data, '142666', 'w_rl23.2142',
plot_intermediates=False, plot_radvels=False,
vsini=63,
order = 38, APF_wl_min = 5380, APF_wl_max = 5460,
vel_min = -60, vel_max = 60, vel_spacing = 0.5)

table_142666_order39 = star_table(data, '142666', 'w_rl23.2142',
plot_intermediates=False, plot_radvels=False,
vsini=63,
order = 39, APF_wl_min = 5450, APF_wl_max = 5520,
vel_min = -60, vel_max = 60, vel_spacing = 0.5)

table_142666_order40 = star_table(data, '142666', 'w_rl23.2142',
plot_intermediates=False, plot_radvels=False,
vsini=63,
order = 40, APF_wl_min = 5510, APF_wl_max = 5575,
vel_min = -60, vel_max = 60, vel_spacing = 0.5)

rad_vels_across_orders = combine_orders('142666', table_142666_order38, table_142666_order39, table_142666_order40, 
)

ascii.write(rad_vels_across_orders, directory + 'HD142666.csv', format='csv')



# '''for HD245185'''

# table_HD245185_order38 = star_table(data, '245185', 'w_rl23.2142',
# plot_intermediates = False, plot_radvels = False,
# order = 5, vsini = 230, APF_wl_min = 3900, APF_wl_max = 3945,
# syn_wl_min = 3400, syn_wl_max = 4200,
# syn_fl_file_name = 'lte09600-4.50-0.0.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits',
# vel_min = -120, vel_max = 60, vel_spacing = 0.5)

# table_HD245185_order39 = star_table(data, '245185', 'w_rl23.2142',
# plot_intermediates = False, plot_radvels = False,
# order = 20, vsini = 230, APF_wl_min = 4450, APF_wl_max = 4510,
# syn_wl_min = 4000, syn_wl_max = 4900,
# syn_fl_file_name = 'lte09600-4.50-0.0.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits',
# vel_min = -120, vel_max = 60, vel_spacing = 0.5)

# table_HD245185_order40 = star_table(data, '245185', 'w_rl23.2142',
# plot_intermediates = False, plot_radvels = False,
# order = 4, vsini = 230, APF_wl_min = 0, APF_wl_max = 3910,
# syn_wl_min = 3500, syn_wl_max = 4200,
# syn_fl_file_name = 'lte09600-4.50-0.0.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits',
# vel_min = -120, vel_max = 60, vel_spacing = 0.5)

# rad_vels_across_orders = combine_orders('245185', table_HD245185_order40, table_HD245185_order38, table_HD245185_order39,
# )

# ascii.write(rad_vels_across_orders, directory + 'HD245185.csv', format='csv')



'''for HD36112'''

table_36112_order38 = star_table(data, '36112', plot_intermediates = False, plot_radvels = False,
syn_fl_file_name = 'lte07600-4.50-0.0.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits', 
vsini = 49.5, syn_wl_min = 5300, syn_wl_max = 5650, 
order = 38, APF_wl_min = 0, APF_wl_max = 5460,
vel_min = -120, vel_max = 60, vel_spacing = 0.5)

table_36112_order39 = star_table(data, '36112', plot_intermediates = False, plot_radvels = False,
syn_fl_file_name = 'lte07600-4.50-0.0.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits', 
vsini = 49.5, syn_wl_min = 5350, syn_wl_max = 5650, 
order = 39, APF_wl_min = 5444, APF_wl_max = 5524,
vel_min = -120, vel_max = 60, vel_spacing = 0.5)

table_36112_order40 = star_table(data, '36112', plot_intermediates = False, plot_radvels = False,
syn_fl_file_name = 'lte07600-4.50-0.0.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits', 
vsini = 49.5, syn_wl_min = 5350, syn_wl_max = 5800, 
order = 40, APF_wl_min = 5525, APF_wl_max = 5575,
vel_min = -120, vel_max = 60, vel_spacing = 0.5)

rad_vels_across_orders = combine_orders('HD36112', table_36112_order38, table_36112_order39, table_36112_order40, 
)

ascii.write(rad_vels_across_orders, directory + 'HD36112.csv', format='csv')




'''for CQ Tau'''

table_HIP26295_order38 = star_table(data, 'HIP26295', plot_intermediates = False, plot_radvels = False,
syn_fl_file_name = 'lte06400-4.50-0.0.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits', 
vsini = 86, syn_wl_min = 5300, syn_wl_max = 5550, 
order = 38, APF_wl_min = 0, APF_wl_max = 5450,
vel_min = -120, vel_max = 60, vel_spacing = 0.5)

table_HIP26295_order39 = star_table(data, 'HIP26295', plot_intermediates = False, plot_radvels = False,
syn_fl_file_name = 'lte06400-4.50-0.0.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits', 
vsini = 86, syn_wl_min = 5350, syn_wl_max = 5600, 
order = 39, APF_wl_min = 5450, APF_wl_max = 5520,
vel_min = -120, vel_max = 60, vel_spacing = 0.5)

table_HIP26295_order40 = star_table(data, 'HIP26295', plot_intermediates = False, plot_radvels = False,
syn_fl_file_name = 'lte06400-4.50-0.0.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits', 
vsini = 86, syn_wl_min = 5350, syn_wl_max = 5750, 
order = 40, APF_wl_min = 5444, APF_wl_max = 5570,
vel_min = -120, vel_max = 60, vel_spacing = 0.5)

rad_vels_across_orders = combine_orders('CQTAU', table_HIP26295_order38, table_HIP26295_order39, table_HIP26295_order40,
)

ascii.write(rad_vels_across_orders, directory + 'CQTAU.csv', format='csv')


'''for HD294268'''
if not os.path.isdir('/Users/zoeko/desktop/APF_stars/294268'):
    os.makedirs('/Users/zoeko/desktop/APF_stars/294268')

table_294268_order38 = star_table(data, '294268', 
plot_intermediates = False, plot_radvels = False,
syn_fl_file_name = 'lte06400-4.50-0.0.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits', 
vsini = 29, syn_wl_min = 5300, syn_wl_max = 5800, 
order = 38, APF_wl_min = 0, APF_wl_max = 5462,
vel_min = -120, vel_max = 60, vel_spacing = 0.5)

table_294268_order39 = star_table(data, '294268', 
plot_intermediates = False, plot_radvels = False,
syn_fl_file_name = 'lte06400-4.50-0.0.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits', 
vsini = 29, syn_wl_min = 5350, syn_wl_max = 5650, 
order = 39, APF_wl_min = 5444, APF_wl_max = 5524,
vel_min = -120, vel_max = 60, vel_spacing = 0.5)

table_294268_order40 = star_table(data, '294268', 
plot_intermediates = False, plot_radvels = False,
syn_fl_file_name = 'lte06400-4.50-0.0.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits', 
vsini = 29, syn_wl_min = 5350, syn_wl_max = 5800, 
order = 40, APF_wl_min = 5444, APF_wl_max = 5592,
vel_min = -120, vel_max = 60, vel_spacing = 0.5)

rad_vels_across_orders = combine_orders('294268', table_294268_order38, table_294268_order39, table_294268_order40,
)

ascii.write(rad_vels_across_orders, directory + 'HD294268.csv', format='csv')


'''for HIP20777 (DF Tau)'''
if not os.path.isdir('/Users/zoeko/desktop/APF_stars/HIP20777'):
    os.makedirs('/Users/zoeko/desktop/APF_stars/HIP20777')

table_HIP20777_order38 = star_table(data, 'HIP20777', 
plot_intermediates = False, plot_radvels = False,
syn_fl_file_name = 'lte04100-4.50-0.0.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits', 
vsini = 100, syn_wl_min = 5800, syn_wl_max = 6400, 
order = 49, APF_wl_min = 0, APF_wl_max = 10000,
vel_min = -120, vel_max = 60, vel_spacing = 0.5)

table_HIP20777_order39 = star_table(data, 'HIP20777', 
plot_intermediates = False, plot_radvels = False,
syn_fl_file_name = 'lte04100-4.50-0.0.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits', 
vsini = 100, syn_wl_min = 5800, syn_wl_max = 6400, 
order = 47, APF_wl_min = 0, APF_wl_max = 10000,
vel_min = -120, vel_max = 60, vel_spacing = 0.5)

table_HIP20777_order40 = star_table(data, 'HIP20777', 
plot_intermediates = False, plot_radvels = False,
syn_fl_file_name = 'lte04100-4.50-0.0.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits', 
vsini = 100, syn_wl_min = 5800, syn_wl_max = 6400, 
order = 48, APF_wl_min = 0, APF_wl_max = 10000,
vel_min = -120, vel_max = 60, vel_spacing = 0.5)

rad_vels_across_orders = combine_orders('DF Tau', table_HIP20777_order38, table_HIP20777_order39, table_HIP20777_order40,
)

ascii.write(rad_vels_across_orders, directory + 'DFTAU.csv', format='csv')


'''for star W96-899'''
if os.path.isdir('/Users/zoeko/desktop/APF_stars/W96-899') == False:
    os.makedirs('/Users/zoeko/desktop/APF_stars/W96-899')

table_W96_899_order38 = star_table(data, 'W96-899', 
plot_intermediates = False, plot_radvels = False,
syn_fl_file_name = 'lte04000-4.50-0.0.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits', 
vsini = 100, syn_wl_min = 5300, syn_wl_max = 5800, 
order = 38, APF_wl_min = 0, APF_wl_max = 5462,
vel_min = -120, vel_max = 60, vel_spacing = 0.5)

table_W96_899_order39 = star_table(data, 'W96-899', 
plot_intermediates = False, plot_radvels = False,
syn_fl_file_name = 'lte04000-4.50-0.0.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits', 
vsini = 100, syn_wl_min = 5350, syn_wl_max = 5650, 
order = 39, APF_wl_min = 5447, APF_wl_max = 5524,
vel_min = -120, vel_max = 60, vel_spacing = 0.5)

table_W96_899_order40 = star_table(data, 'W96-899', 
plot_intermediates = False, plot_radvels = False,
syn_fl_file_name = 'lte04000-4.50-0.0.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits', 
vsini = 100, syn_wl_min = 5350, syn_wl_max = 5800, 
order = 40, APF_wl_min = 5444, APF_wl_max = 5592,
vel_min = -120, vel_max = 60, vel_spacing = 0.5)

rad_vels_across_orders = combine_orders('W96-899', table_W96_899_order38, table_W96_899_order39, table_W96_899_order40,
)

ascii.write(rad_vels_across_orders, directory + 'W96899.csv', format='csv')




'''for star HIP20387'''
# go to FUNCTIONS and comment out first definition of APF_spectrum, use second one
# to use HD245185 as blaze function fit <-- NOT NECESSARY

# order38_blaze_norm = blaze_normalization(order = 38, APF_fl_file_name = 'rl27.7459', APF_wl_min = 5380, APF_wl_max = 5460)
# # interpolate onto wavelength grid
# blaze_fit_38 = interpolate.interp1d(order38_blaze_norm[0], order38_blaze_norm[1], kind = 'cubic')

# order39_blaze_norm = blaze_normalization(order = 39, APF_fl_file_name = 'rl27.2625', APF_wl_min = 5450, APF_wl_max = 5520)
# blaze_fit_39 = interpolate.interp1d(order39_blaze_norm[0], order39_blaze_norm[1], kind = 'cubic')

# order40_blaze_norm = blaze_normalization(order = 40, APF_fl_file_name = 'rl25.9179', APF_wl_min = 5510, APF_wl_max = 5585)
# blaze_fit_40 = interpolate.interp1d(order40_blaze_norm[0], order40_blaze_norm[1], kind = 'cubic')


table_HIP20387_order38 = star_table(data, 'HIP20387', 
plot_intermediates = False, plot_radvels = False,
syn_fl_file_name = 'lte04500-4.50-0.0.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits', 
vsini = 170, syn_wl_min = 5300, syn_wl_max = 5800, 
order = 38, APF_wl_min = 0, APF_wl_max = 5462,
vel_min = -120, vel_max = 60, vel_spacing = 0.5)

table_HIP20387_order39 = star_table(data, 'HIP20387', 
plot_intermediates = False, plot_radvels = False,
syn_fl_file_name = 'lte04500-4.50-0.0.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits', 
vsini = 170, syn_wl_min = 5350, syn_wl_max = 5650, 
order = 39, APF_wl_min = 5444, APF_wl_max = 5524,
vel_min = -120, vel_max = 60, vel_spacing = 0.5)

table_HIP20387_order40 = star_table(data, 'HIP20387', 
plot_intermediates = False, plot_radvels = False,
syn_fl_file_name = 'lte04500-4.50-0.0.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits', 
vsini = 170, syn_wl_min = 5350, syn_wl_max = 5800, 
order = 40, APF_wl_min = 5444, APF_wl_max = 5592,
vel_min = -120, vel_max = 60, vel_spacing = 0.5)

rad_vels_across_orders = combine_orders('HIP20387', table_HIP20387_order38, table_HIP20387_order39, table_HIP20387_order40,
)

ascii.write(rad_vels_across_orders, directory + 'RYTAU.csv', format='csv')


'''for star LkHa330'''

table_LKHA330_order38 = star_table(data, 'LKHA330', 
plot_intermediates = False, plot_radvels = False,
syn_fl_file_name = 'lte06300-4.50-0.0.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits', 
vsini = 50, syn_wl_min = 5100, syn_wl_max = 5800, 
order = 34, APF_wl_min = 0, APF_wl_max = 5215,
vel_min = -120, vel_max = 60, vel_spacing = 0.5)

table_LKHA330_order39 = star_table(data, 'LKHA330', 
plot_intermediates = False, plot_radvels = False,
syn_fl_file_name = 'lte06300-4.50-0.0.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits', 
vsini = 50, syn_wl_min = 5100, syn_wl_max = 5800, 
order = 35, APF_wl_min = 0, APF_wl_max = 5270,
vel_min = -120, vel_max = 60, vel_spacing = 0.5)

table_LKHA330_order40 = star_table(data, 'LKHA330', 
plot_intermediates = False, plot_radvels = False,
syn_fl_file_name = 'lte06300-4.50-0.0.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits', 
vsini = 50, syn_wl_min = 5100, syn_wl_max = 5800, 
order = 41, APF_wl_min = 0, APF_wl_max = 10000,
vel_min = -120, vel_max = 60, vel_spacing = 0.5)

rad_vels_across_orders = combine_orders('LKHA330', table_LKHA330_order38, table_LKHA330_order39, table_LKHA330_order40,
)

ascii.write(rad_vels_across_orders, directory + 'LKHA330.csv', format='csv')





'''for star HD283447'''
if os.path.isdir('/Users/zoeko/desktop/APF_stars/283447') == False:
    os.makedirs('/Users/zoeko/desktop/APF_stars/283447')

table_283447_order38 = star_table(data, '283447', 
plot_intermediates = False, plot_radvels = False,
syn_fl_file_name = 'lte04700-4.50-0.0.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits', 
vsini = 58.5, syn_wl_min = 5100, syn_wl_max = 5800, 
order = 39, APF_wl_min = 0, APF_wl_max = 5520,
vel_min = -120, vel_max = 60, vel_spacing = 0.5)

table_283447_order39 = star_table(data, '283447', 
plot_intermediates = False, plot_radvels = False,
syn_fl_file_name = 'lte04700-4.50-0.0.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits', 
vsini = 58.5, syn_wl_min = 5100, syn_wl_max = 5800, 
order = 41, APF_wl_min = 0, APF_wl_max = 5650,
vel_min = -120, vel_max = 60, vel_spacing = 0.5)

table_283447_order40 = star_table(data, '283447', 
plot_intermediates = False, plot_radvels = False,
syn_fl_file_name = 'lte04700-4.50-0.0.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits', 
vsini = 58.5, syn_wl_min = 5100, syn_wl_max = 5800, 
order = 42, APF_wl_min = 0, APF_wl_max = 5720,
vel_min = -120, vel_max = 60, vel_spacing = 0.5)

rad_vels_across_orders = combine_orders('283447', table_283447_order38, table_283447_order39, table_283447_order40,
)

ascii.write(rad_vels_across_orders, directory + 'HD283447.csv', format='csv')


'''for star DG_Tau'''
if os.path.isdir('/Users/zoeko/desktop/APF_stars/DG_TAU') == False:
    os.makedirs('/Users/zoeko/desktop/APF_stars/DG_TAU')

table_DG_TAU_order38 = star_table(data, 'DG_TAU', 
plot_intermediates = False, plot_radvels = False,
syn_fl_file_name = 'lte04200-4.50-0.0.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits', 
vsini = 120, syn_wl_min = 5500, syn_wl_max = 6200, 
order = 46, APF_wl_min = 0, APF_wl_max = 10000,
vel_min = -120, vel_max = 60, vel_spacing = 0.5)

table_DG_TAU_order39 = star_table(data, 'DG_TAU', 
plot_intermediates = False, plot_radvels = False,
syn_fl_file_name = 'lte04200-4.50-0.0.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits', 
vsini = 120, syn_wl_min = 5500, syn_wl_max = 6300, 
order = 44, APF_wl_min = 0, APF_wl_max = 10000,
vel_min = -120, vel_max = 60, vel_spacing = 0.5)

table_DG_TAU_order40 = star_table(data, 'DG_TAU', 
plot_intermediates = False, plot_radvels = False,
syn_fl_file_name = 'lte04200-4.50-0.0.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits', 
vsini = 120, syn_wl_min = 5700, syn_wl_max = 6300, 
order = 48, APF_wl_min = 0, APF_wl_max = 10000,
vel_min = -120, vel_max = 60, vel_spacing = 0.5)

rad_vels_across_orders = combine_orders('DG_TAU', table_DG_TAU_order38, table_DG_TAU_order39, table_DG_TAU_order40,
)

ascii.write(rad_vels_across_orders, directory + 'DGTAU.csv', format='csv')


'''for star DH_Tau'''
if os.path.isdir('/Users/zoeko/desktop/APF_stars/DH_TAU') == False:
    os.makedirs('/Users/zoeko/desktop/APF_stars/DH_TAU')

table_DH_TAU_order38 = star_table(data, 'DH_TAU', 
plot_intermediates = False, plot_radvels = False,
syn_fl_file_name = 'lte04700-4.50-0.0.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits', 
vsini = 58.5, syn_wl_min = 5300, syn_wl_max = 5800, 
order = 38, APF_wl_min = 0, APF_wl_max = 10000,
vel_min = -120, vel_max = 60, vel_spacing = 0.5)

table_DH_TAU_order39 = star_table(data, 'DH_TAU', 
plot_intermediates = False, plot_radvels = False,
syn_fl_file_name = 'lte04700-4.50-0.0.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits', 
vsini = 58.5, syn_wl_min = 5300, syn_wl_max = 5800, 
order = 39, APF_wl_min = 0, APF_wl_max = 10000,
vel_min = -120, vel_max = 60, vel_spacing = 0.5)

table_DH_TAU_order40 = star_table(data, 'DH_TAU', 
plot_intermediates = False, plot_radvels = False,
syn_fl_file_name = 'lte04700-4.50-0.0.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits', 
vsini = 58.5, syn_wl_min = 5300, syn_wl_max = 5800, 
order = 40, APF_wl_min = 0, APF_wl_max = 10000,
vel_min = -120, vel_max = 60, vel_spacing = 0.5)

rad_vels_across_orders = combine_orders('DH_TAU', table_DH_TAU_order38, table_DH_TAU_order39, table_DH_TAU_order40,
)

ascii.write(rad_vels_across_orders, directory + 'DHTAU.csv', format='csv')


'''for GK Tau'''
if os.path.isdir('/Users/zoeko/desktop/APF_stars/GK_TAU') == False:
    os.makedirs('/Users/zoeko/desktop/APF_stars/GK_TAU')

table_GK_TAU_order38 = star_table(data, 'GK_TAU', 
plot_intermediates = False, plot_radvels = False,
syn_fl_file_name = 'lte04100-4.50-0.0.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits', 
vsini = 100, syn_wl_min = 5300, syn_wl_max = 5800, 
order = 38, APF_wl_min = 0, APF_wl_max = 10000,
vel_min = -120, vel_max = 60, vel_spacing = 0.5)

table_GK_TAU_order39 = star_table(data, 'GK_TAU', 
plot_intermediates = False, plot_radvels = False,
syn_fl_file_name = 'lte04100-4.50-0.0.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits', 
vsini = 100, syn_wl_min = 5300, syn_wl_max = 5800, 
order = 39, APF_wl_min = 0, APF_wl_max = 10000,
vel_min = -120, vel_max = 60, vel_spacing = 0.5)

table_GK_TAU_order40 = star_table(data, 'GK_TAU', 
plot_intermediates = False, plot_radvels = False,
syn_fl_file_name = 'lte04100-4.50-0.0.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits', 
vsini = 100, syn_wl_min = 5300, syn_wl_max = 5800, 
order = 40, APF_wl_min = 0, APF_wl_max = 10000,
vel_min = -120, vel_max = 60, vel_spacing = 0.5)

rad_vels_across_orders = combine_orders('GK_TAU', table_GK_TAU_order38, table_GK_TAU_order39, table_GK_TAU_order40,
)

ascii.write(rad_vels_across_orders, directory + 'GKTAU.csv', format='csv')


'''for star AB Aur (HIP 22910)'''
if os.path.isdir('/Users/zoeko/desktop/APF_stars/HIP22910') == False:
    os.makedirs('/Users/zoeko/desktop/APF_stars/HIP22910')

table_HIP22910_order38 = star_table(data, 'HIP22910', 
plot_intermediates = False, plot_radvels = False,
syn_fl_file_name = 'lte09600-4.50-0.0.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits', 
vsini = 200, syn_wl_min = 3500, syn_wl_max = 4300, 
order = 4, APF_wl_min = 0, APF_wl_max = 3910,
vel_min = -120, vel_max = 60, vel_spacing = 0.5)

table_HIP22910_order40 = star_table(data, 'HIP22910', 
plot_intermediates = False, plot_radvels = False,
syn_fl_file_name = 'lte09600-4.50-0.0.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits', 
vsini = 200, syn_wl_min = 3800, syn_wl_max = 4400,
order = 10, APF_wl_min = 0, APF_wl_max = 10000,
vel_min = -120, vel_max = 60, vel_spacing = 0.5)

table_HIP22910_order39 = star_table(data, 'HIP22910', 
plot_intermediates = False, plot_radvels = False,
syn_fl_file_name = 'lte09600-4.50-0.0.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits', 
vsini = 200, syn_wl_min = 4200, syn_wl_max = 4800, 
order = 20, APF_wl_min = 4455, APF_wl_max = 4510,
vel_min = -120, vel_max = 60, vel_spacing = 0.5)

rad_vels_across_orders = combine_orders('HIP22910', table_HIP22910_order38, table_HIP22910_order39, table_HIP22910_order40,
)

ascii.write(rad_vels_across_orders, directory + 'ABAUR.csv', format='csv')


'''for star TX Ori'''
if os.path.isdir('/Users/zoeko/desktop/APF_stars/TX_ORI') == False:
    os.makedirs('/Users/zoeko/desktop/APF_stars/TX_ORI')

table_TX_ORI_order38 = star_table(data, 'TX_ORI', 
plot_intermediates = False, plot_radvels = False,
syn_fl_file_name = 'lte04400-4.50-0.0.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits', 
vsini = 160, syn_wl_min = 5300, syn_wl_max = 5800, 
order = 38, APF_wl_min = 0, APF_wl_max = 10000,
vel_min = -120, vel_max = 60, vel_spacing = 0.5)

table_TX_ORI_order39 = star_table(data, 'TX_ORI', 
plot_intermediates = False, plot_radvels = False,
syn_fl_file_name = 'lte04400-4.50-0.0.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits', 
vsini = 160, syn_wl_min = 5300, syn_wl_max = 5800, 
order = 39, APF_wl_min = 0, APF_wl_max = 10000,
vel_min = -120, vel_max = 60, vel_spacing = 0.5)

table_TX_ORI_order40 = star_table(data, 'TX_ORI', 
plot_intermediates = False, plot_radvels = False,
syn_fl_file_name = 'lte04400-4.50-0.0.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits', 
vsini = 160, syn_wl_min = 5300, syn_wl_max = 5800, 
order = 40, APF_wl_min = 0, APF_wl_max = 10000,
vel_min = -120, vel_max = 60, vel_spacing = 0.5)

rad_vels_across_orders = combine_orders('TX_ORI', table_TX_ORI_order38, table_TX_ORI_order39, table_TX_ORI_order40,
)

ascii.write(rad_vels_across_orders, directory + 'TXORI.csv', format='csv')


'''for star HD 143006'''
if os.path.isdir('/Users/zoeko/desktop/APF_stars/143006') == False:
    os.makedirs('/Users/zoeko/desktop/APF_stars/143006')

table_143006_order38 = star_table(data, '143006', 
plot_intermediates = False, plot_radvels = False,
syn_fl_file_name = 'lte05800-4.50-0.0.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits', 
vsini = 13, syn_wl_min = 5300, syn_wl_max = 5800, 
order = 38, APF_wl_min = 0, APF_wl_max = 10000,
vel_min = -120, vel_max = 60, vel_spacing = 0.5)

table_143006_order39 = star_table(data, '143006', 
plot_intermediates = False, plot_radvels = False,
syn_fl_file_name = 'lte05800-4.50-0.0.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits', 
vsini = 13, syn_wl_min = 5300, syn_wl_max = 5800, 
order = 39, APF_wl_min = 0, APF_wl_max = 10000,
vel_min = -120, vel_max = 60, vel_spacing = 0.5)

table_143006_order40 = star_table(data, '143006', 
plot_intermediates = False, plot_radvels = False,
syn_fl_file_name = 'lte05800-4.50-0.0.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits', 
vsini = 13, syn_wl_min = 5300, syn_wl_max = 5800, 
order = 40, APF_wl_min = 0, APF_wl_max = 10000,
vel_min = -120, vel_max = 60, vel_spacing = 0.5)

rad_vels_across_orders = combine_orders('143006', table_143006_order38, table_143006_order39, table_143006_order40,
)

ascii.write(rad_vels_across_orders, directory + 'HD143006.csv', format='csv')


'''for 2M_J1609-2217 (EPIC7221)'''
if os.path.isdir('/Users/zoeko/desktop/APF_stars/EPIC7221') == False:
    os.makedirs('/Users/zoeko/desktop/APF_stars/EPIC7221')

table_EPIC7221_order38 = star_table(data, 'EPIC7221', 
plot_intermediates = False, plot_radvels = False,
syn_fl_file_name = 'lte03900-4.50-0.0.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits', 
vsini = 60, syn_wl_min = 5300, syn_wl_max = 5800, 
order = 38, APF_wl_min = 0, APF_wl_max = 10000,
vel_min = -120, vel_max = 60, vel_spacing = 0.5)

table_EPIC7221_order39 = star_table(data, 'EPIC7221', 
plot_intermediates = False, plot_radvels = False,
syn_fl_file_name = 'lte03900-4.50-0.0.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits', 
vsini = 60, syn_wl_min = 5300, syn_wl_max = 5800, 
order = 39, APF_wl_min = 0, APF_wl_max = 10000,
vel_min = -120, vel_max = 60, vel_spacing = 0.5)

table_EPIC7221_order40 = star_table(data, 'EPIC7221', 
plot_intermediates = False, plot_radvels = False,
syn_fl_file_name = 'lte03900-4.50-0.0.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits', 
vsini = 60, syn_wl_min = 5300, syn_wl_max = 5800, 
order = 40, APF_wl_min = 0, APF_wl_max = 10000,
vel_min = -120, vel_max = 60, vel_spacing = 0.5)

rad_vels_across_orders = combine_orders('EPIC7221', table_EPIC7221_order38, table_EPIC7221_order39, table_EPIC7221_order40,
)

ascii.write(rad_vels_across_orders, directory + '2MJ16092217.csv', format='csv')


'''for CD-22 11432 (T67938191)'''
if os.path.isdir('/Users/zoeko/desktop/APF_stars/T67938191') == False:
    os.makedirs('/Users/zoeko/desktop/APF_stars/T67938191')

table_T67938191_order38 = star_table(data, 'T67938191', 
plot_intermediates = False, plot_radvels = False,
syn_fl_file_name = 'lte04900-4.50-0.0.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits', 
vsini = 41, syn_wl_min = 5300, syn_wl_max = 5800, 
order = 38, APF_wl_min = 0, APF_wl_max = 10000,
vel_min = -120, vel_max = 60, vel_spacing = 0.5)

table_T67938191_order39 = star_table(data, 'T67938191', 
plot_intermediates = False, plot_radvels = False,
syn_fl_file_name = 'lte04900-4.50-0.0.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits', 
vsini = 41, syn_wl_min = 5300, syn_wl_max = 5800, 
order = 39, APF_wl_min = 0, APF_wl_max = 10000,
vel_min = -120, vel_max = 60, vel_spacing = 0.5)

table_T67938191_order40 = star_table(data, 'T67938191', 
plot_intermediates = False, plot_radvels = False,
syn_fl_file_name = 'lte04900-4.50-0.0.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits', 
vsini = 41, syn_wl_min = 5300, syn_wl_max = 5800, 
order = 40, APF_wl_min = 0, APF_wl_max = 10000,
vel_min = -120, vel_max = 60, vel_spacing = 0.5)

rad_vels_across_orders = combine_orders('T67938191', table_T67938191_order38, table_T67938191_order39, table_T67938191_order40,
)

ascii.write(rad_vels_across_orders, directory + 'CD2211432.csv', format='csv')


'''for VV Sco (EPIC2099)'''
if os.path.isdir('/Users/zoeko/desktop/APF_stars/EPIC2099') == False:
    os.makedirs('/Users/zoeko/desktop/APF_stars/EPIC2099')

table_EPIC2099_order38 = star_table(data, 'EPIC2099', 
plot_intermediates = False, plot_radvels = False,
syn_fl_file_name = 'lte04400-4.50-0.0.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits', 
vsini = 100, syn_wl_min = 5300, syn_wl_max = 5800, 
order = 38, APF_wl_min = 0, APF_wl_max = 10000,
vel_min = -120, vel_max = 60, vel_spacing = 0.5)

table_EPIC2099_order39 = star_table(data, 'EPIC2099', 
plot_intermediates = False, plot_radvels = False,
syn_fl_file_name = 'lte04400-4.50-0.0.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits', 
vsini = 100, syn_wl_min = 5300, syn_wl_max = 5800, 
order = 39, APF_wl_min = 0, APF_wl_max = 10000,
vel_min = -120, vel_max = 60, vel_spacing = 0.5)

table_EPIC2099_order40 = star_table(data, 'EPIC2099', 
plot_intermediates = False, plot_radvels = False,
syn_fl_file_name = 'lte04400-4.50-0.0.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits', 
vsini = 100, syn_wl_min = 5300, syn_wl_max = 5800, 
order = 40, APF_wl_min = 0, APF_wl_max = 10000,
vel_min = -120, vel_max = 60, vel_spacing = 0.5)

rad_vels_across_orders = combine_orders('EPIC2099', table_EPIC2099_order38, table_EPIC2099_order39, table_EPIC2099_order40,
)

ascii.write(rad_vels_across_orders, directory + 'VVSCO.csv', format='csv')


'''for 2M J1626-2356 (2M1626-23)'''
if os.path.isdir('/Users/zoeko/desktop/APF_stars/2M1626-23') == False:
    os.makedirs('/Users/zoeko/desktop/APF_stars/2M1626-23')

table_2M1626_23_order38 = star_table(data, '2M1626-23', 
plot_intermediates = False, plot_radvels = False,
syn_fl_file_name = 'lte03900-4.50-0.0.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits', 
vsini = 60, syn_wl_min = 5300, syn_wl_max = 5800, 
order = 38, APF_wl_min = 0, APF_wl_max = 10000,
vel_min = -120, vel_max = 60, vel_spacing = 0.5)

table_2M1626_23_order39 = star_table(data, '2M1626-23', 
plot_intermediates = False, plot_radvels = False,
syn_fl_file_name = 'lte03900-4.50-0.0.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits', 
vsini = 60, syn_wl_min = 5300, syn_wl_max = 5800, 
order = 39, APF_wl_min = 0, APF_wl_max = 10000,
vel_min = -120, vel_max = 60, vel_spacing = 0.5)

table_2M1626_23_order40 = star_table(data, '2M1626-23', 
plot_intermediates = False, plot_radvels = False,
syn_fl_file_name = 'lte03900-4.50-0.0.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits', 
vsini = 60, syn_wl_min = 5300, syn_wl_max = 5800, 
order = 40, APF_wl_min = 0, APF_wl_max = 10000,
vel_min = -120, vel_max = 60, vel_spacing = 0.5)

rad_vels_across_orders = combine_orders('2M1626-23', table_2M1626_23_order38, table_2M1626_23_order39, table_2M1626_23_order40,
)

ascii.write(rad_vels_across_orders, directory + '2MJ16262356.csv', format='csv')


'''for 2M J1639-2402 (2M1639-24)'''
if os.path.isdir('/Users/zoeko/desktop/APF_stars/2M1639-24') == False:
    os.makedirs('/Users/zoeko/desktop/APF_stars/2M1639-24')

table_2M1639_24_order38 = star_table(data, '2M1639-24', 
plot_intermediates = False, plot_radvels = False,
syn_fl_file_name = 'lte03900-4.50-0.0.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits', 
vsini = 60, syn_wl_min = 5300, syn_wl_max = 5800, 
order = 38, APF_wl_min = 0, APF_wl_max = 10000,
vel_min = -120, vel_max = 60, vel_spacing = 0.5)

table_2M1639_24_order39 = star_table(data, '2M1639-24', 
plot_intermediates = False, plot_radvels = False,
syn_fl_file_name = 'lte03900-4.50-0.0.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits', 
vsini = 60, syn_wl_min = 5300, syn_wl_max = 5800, 
order = 39, APF_wl_min = 0, APF_wl_max = 10000,
vel_min = -120, vel_max = 60, vel_spacing = 0.5)

table_2M1639_24_order40 = star_table(data, '2M1639-24', 
plot_intermediates = False, plot_radvels = False,
syn_fl_file_name = 'lte03900-4.50-0.0.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits', 
vsini = 60, syn_wl_min = 5300, syn_wl_max = 5800, 
order = 40, APF_wl_min = 0, APF_wl_max = 10000,
vel_min = -120, vel_max = 60, vel_spacing = 0.5)

rad_vels_across_orders = combine_orders('2M1639-24', table_2M1639_24_order38, table_2M1639_24_order39, table_2M1639_24_order40,
)

ascii.write(rad_vels_across_orders, directory + '2MJ16392402.csv', format='csv')


'''for AS 209 (HIP82323)'''
if os.path.isdir('/Users/zoeko/desktop/APF_stars/HIP82323') == False:
    os.makedirs('/Users/zoeko/desktop/APF_stars/HIP82323')

table_HIP82323_order38 = star_table(data, 'HIP82323', 
plot_intermediates = False, plot_radvels = False,
syn_fl_file_name = 'lte04600-4.50-0.0.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits', 
vsini = 40, syn_wl_min = 5300, syn_wl_max = 5800, 
order = 38, APF_wl_min = 0, APF_wl_max = 10000,
vel_min = -120, vel_max = 60, vel_spacing = 0.5)

table_HIP82323_order39 = star_table(data, 'HIP82323', 
plot_intermediates = False, plot_radvels = False,
syn_fl_file_name = 'lte04600-4.50-0.0.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits', 
vsini = 40, syn_wl_min = 5300, syn_wl_max = 5800, 
order = 39, APF_wl_min = 0, APF_wl_max = 10000,
vel_min = -120, vel_max = 60, vel_spacing = 0.5)

table_HIP82323_order40 = star_table(data, 'HIP82323', 
plot_intermediates = False, plot_radvels = False,
syn_fl_file_name = 'lte04600-4.50-0.0.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits', 
vsini = 40, syn_wl_min = 5300, syn_wl_max = 5800, 
order = 40, APF_wl_min = 0, APF_wl_max = 10000,
vel_min = -120, vel_max = 60, vel_spacing = 0.5)

rad_vels_across_orders = combine_orders('AS209', table_HIP82323_order38, table_HIP82323_order39, table_HIP82323_order40,
)

ascii.write(rad_vels_across_orders, directory + 'AS209.csv', format='csv')


'''for HD 163296'''
if os.path.isdir('/Users/zoeko/desktop/APF_stars/163296') == False:
    os.makedirs('/Users/zoeko/desktop/APF_stars/163296')

table_163296_order38 = star_table(data, '163296', 
plot_intermediates = False, plot_radvels = False,
syn_fl_file_name = 'lte09200-4.50-0.0.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits', 
vsini = 90, syn_wl_min = 3500, syn_wl_max = 4000, 
order = 0, APF_wl_min = 3746, APF_wl_max = 3778,
vel_min = -120, vel_max = 60, vel_spacing = 0.5)

table_163296_order39 = star_table(data, '163296', 
plot_intermediates = False, plot_radvels = False,
syn_fl_file_name = 'lte09200-4.50-0.0.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits', 
vsini = 90, syn_wl_min = 3400, syn_wl_max = 4200, 
order = 1, APF_wl_min = 3778, APF_wl_max = 3790,
vel_min = -120, vel_max = 60, vel_spacing = 0.5)

table_163296_order40 = star_table(data, '163296', 
plot_intermediates = False, plot_radvels = False,
syn_fl_file_name = 'lte09200-4.50-0.0.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits', 
vsini = 90, syn_wl_min = 3400, syn_wl_max = 4200, 
order = 4, APF_wl_min = 3860, APF_wl_max = 3910,
vel_min = -120, vel_max = 60, vel_spacing = 0.5)

rad_vels_across_orders = combine_orders('163296', table_163296_order38, table_163296_order39, table_163296_order40,
)

ascii.write(rad_vels_across_orders, directory + 'HD163296.csv', format='csv')


'''for V1515 Cyg (2M2023P42)'''
if os.path.isdir('/Users/zoeko/desktop/APF_stars/2M2023P42') == False:
    os.makedirs('/Users/zoeko/desktop/APF_stars/2M2023P42')

table_2M2023p42_order38 = star_table(data, '2M2023P42', 
plot_intermediates = False, plot_radvels = False,
syn_fl_file_name = 'lte06000-4.50-0.0.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits', 
vsini = 40, syn_wl_min = 5300, syn_wl_max = 5800, 
order = 38, APF_wl_min = 0, APF_wl_max = 10000,
vel_min = -120, vel_max = 60, vel_spacing = 0.5)

table_2M2023p42_order39 = star_table(data, '2M2023P42', 
plot_intermediates = False, plot_radvels = False,
syn_fl_file_name = 'lte06000-4.50-0.0.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits', 
vsini = 40, syn_wl_min = 5300, syn_wl_max = 5800, 
order = 39, APF_wl_min = 0, APF_wl_max = 10000,
vel_min = -120, vel_max = 60, vel_spacing = 0.5)

table_2M2023p42_order40 = star_table(data, '2M2023P42', 
plot_intermediates = False, plot_radvels = False,
syn_fl_file_name = 'lte06000-4.50-0.0.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits', 
vsini = 40, syn_wl_min = 5300, syn_wl_max = 5800, 
order = 40, APF_wl_min = 0, APF_wl_max = 10000,
vel_min = -120, vel_max = 60, vel_spacing = 0.5)

rad_vels_across_orders = combine_orders('2M2023P42', table_2M2023p42_order38, table_2M2023p42_order39, table_2M2023p42_order40,
)

ascii.write(rad_vels_across_orders, directory + 'V1515CYG.csv', format='csv')


'''for AS 205 (EPIC9328)'''
if os.path.isdir('/Users/zoeko/desktop/APF_stars/EPIC9328') == False:
    os.makedirs('/Users/zoeko/desktop/APF_stars/EPIC9328')

table_EPIC9328_order38 = star_table(data, 'EPIC9328', 
plot_intermediates = False, plot_radvels = False,
syn_fl_file_name = 'lte05300-4.50-0.0.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits', 
vsini = 100, syn_wl_min = 5800, syn_wl_max = 6300, 
order = 48, APF_wl_min = 0, APF_wl_max = 10000,
vel_min = -120, vel_max = 60, vel_spacing = 0.5)

table_EPIC9328_order39 = star_table(data, 'EPIC9328', 
plot_intermediates = False, plot_radvels = False,
syn_fl_file_name = 'lte05300-4.50-0.0.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits', 
vsini = 100, syn_wl_min = 5800, syn_wl_max = 6400, 
order = 49, APF_wl_min = 0, APF_wl_max = 10000,
vel_min = -120, vel_max = 60, vel_spacing = 0.5)

table_EPIC9328_order40 = star_table(data, 'EPIC9328', 
plot_intermediates = False, plot_radvels = False,
syn_fl_file_name = 'lte05300-4.50-0.0.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits', 
vsini = 100, syn_wl_min = 6200, syn_wl_max = 6700, 
order = 52, APF_wl_min = 0, APF_wl_max = 10000,
vel_min = -120, vel_max = 60, vel_spacing = 0.5)

rad_vels_across_orders = combine_orders('AS205', table_EPIC9328_order38, table_EPIC9328_order39, table_EPIC9328_order40,
)

ascii.write(rad_vels_across_orders, directory + 'AS205.csv', format='csv')





'''for star HIP34042'''

# os.makedirs('/Users/zoeko/desktop/HIP34042')

table_HIP34042_order38 = star_table(data, 'HIP34042', 
plot_intermediates = False, plot_radvels = False,
syn_fl_file_name = 'lte06400-4.50-0.0.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits', 
vsini = 50, syn_wl_min = 5100, syn_wl_max = 5800, 
order = 34, APF_wl_min = 0, APF_wl_max = 5215,
vel_min = -120, vel_max = 60, vel_spacing = 0.5)

table_HIP34042_order39 = star_table(data, 'HIP34042', 
plot_intermediates = False, plot_radvels = False,
syn_fl_file_name = 'lte06400-4.50-0.0.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits', 
vsini = 50, syn_wl_min = 5100, syn_wl_max = 5800, 
order = 35, APF_wl_min = 0, APF_wl_max = 5270,
vel_min = -120, vel_max = 60, vel_spacing = 0.5)

table_HIP34042_order40 = star_table(data, 'HIP34042', 
plot_intermediates = False, plot_radvels = False,
syn_fl_file_name = 'lte06400-4.50-0.0.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits', 
vsini = 50, syn_wl_min = 5100, syn_wl_max = 5800, 
order = 41, APF_wl_min = 0, APF_wl_max = 10000,
vel_min = -120, vel_max = 60, vel_spacing = 0.5)

rad_vels_across_orders = combine_orders('HIP34042', table_HIP34042_order38, table_HIP34042_order39, table_HIP34042_order40,
)

ascii.write(rad_vels_across_orders, directory + 'HIP34042.csv', format='csv')