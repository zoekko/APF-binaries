# Create a table containing the duration of the observations for each target

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
from astropy.table import Table
from astropy.table import Table, join
from astropy.table import Table, vstack
from astropy.table import Table, unique
from astropy.table import Column
import numpy.core.defchararray as np_f
import astropy
from astropy.table import Table, vstack
from astropy.table import Table, unique
from astropy.io import ascii
from astropy.table import QTable
from functions import duration

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

stars = [
    'LKHA330',     
    '283447',
    'HIP20387',      
    'HIP20777',      
    'DG_TAU', 
    'DH_TAU', 
    'GK_TAU', 
    'HIP22910',      
    '36112',      
    '245185', 
    'HIP26295',      
    'TX_ORI', 
    'W96-899', 
    '294268',      
    'HIP34042', 
    '142666',      
    '143006', 
    'EPIC7221', 
    'EPIC9328',      
    'T67938191', 
    'EPIC2099',      
    '2M1626-23', 
    '2M1639-24', 
    'HIP82323',      
    '163296', 
    '2M2023P42', 
    ]


table = Table(dtype=[('star', 'S2'), ('duration', 'i4')])

for i in np.arange(26):
    table.add_row((stars[i], duration(data, stars[i])))

durations = np.array(table['duration'])
max_duration = max(durations)
table.pprint_all()

print('max duration is: ', max_duration)
print('average duration is: ', np.average(durations))


plt.hist(durations, bins = 10)
plt.show()