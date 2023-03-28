# Report the number of observations for each target

from astropy.io import ascii
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
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
from functions import rfftfreq
from functions import syn_spectrum
from functions import APF_spectrum
from functions import model
from functions import chi
from functions import calc_rad_vel
from functions import combine_orders
from functions import blaze_normalization

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
print(data)

def num_obs(star):
    '''given a star name in the file, 
    return the number of observations recorded'''
    mask = data.groups.keys['name'] == star
    star_data = data.groups[mask]
    return len(star_data)

# stars = np.array(['36112',
# '245185'])

stars = np.array([
'283447',
'LKHA330',
'294268',
'2M2023P42',
'T67938191',
'EPIC7221',
'143006',
'HIP26295',
'DG_TAU',
'163296',
'245185',
'HIP22910',
'36112',
'142666',
'EPIC2099',
'HIP20387',
'HIP82323',
'2M1626-23',
'2M1639-24',
'GK_TAU',
'TX_ORI',
'W96-899',
'DH_TAU',
'EPIC9328',
'HIP20777',
'HIP34042',
'RXJ1615',
'DS_TAU'
])

nobs = np.array([])
for star in stars:
    obs = num_obs(star)
    nobs = np.append(nobs, obs)

stars_and_nobs = Table([stars, nobs], 
names=('star', 'nobs'))
print(stars_and_nobs)

# ascii.write(stars_and_nobs, '/Users/zoeko/desktop/nobs.csv', format='csv')
