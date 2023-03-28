# read in csv files containing calculated RVs for targets and plot estimated orbitals
# using estimated semi-major amplitudes and periods
# assuming e = 0 (use sine curve)

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
from astropy.table import Table, QTable, join, vstack, unique, Column
import numpy.core.defchararray as np_f
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
from sympy import symbols, Eq, solve

C = 299792.458 #in km/s
G = 6.67408 * 10 ** -11 #m3 kg-1 s-2

def solar_mass_to_kg(m):
    '''convert solar masses to kg'''
    return m * 1.989 * 10 ** 30

def kg_to_solar_mass(m):
    '''convert solar masses to kg'''
    return m / (1.989 * 10 ** 30)

def km_to_m(k):
    '''convert semimajor amplitude from km to m'''
    return k * 1000

def day_to_sec(t):
    '''convert period from days to seconds'''
    return t * 24 * 60 * 60

def mass_ratio(m1, sinI, K, T):
    '''calculate mass ratio and mass of companion star'''
    x = symbols('x')
    constant = (((2 * np.pi * G) / T) ** (1/3)) * sinI
    expr = ((x / ((m1 + x) ** (2/3))) * constant) - K
    m2 = kg_to_solar_mass(solve(expr)[0]) # in solar masses
    mass_ratio = m2 / (kg_to_solar_mass(m1))
    return mass_ratio, m2





'''HD 36112'''
mass = solar_mass_to_kg(1.5)

fname = "/Users/zoeko/Desktop/research/RVdata/HD36112.csv"


data = astropy.io.ascii.read(
    fname,
    names=[
        "Star",
         "JD",
        "RVs"
    ]
)

# load RV dataset
t1 = np.array(data['JD']) # times of observations
rv1_data = np.array(data['RVs']) # radial velocities (in km/s)
rv1_err = np.array([3] * 26) # uncertainty on rv measurement (in km/s) 

# plot data and error bars
fig, ax = plt.subplots(figsize=(7.1, 4.7))
plt.errorbar(t1, rv1_data, yerr=rv1_err, fmt=".b")

# plot orbital estimates

k1 = 2 # semi major amplitude [km/s]
T1 = 90 # period [days]
time = np.arange(t1[0], t1[-1] + 1)
fit1 = -(k1 * np.sin(2 * np.pi * (time + 10) / T1) - 20)
plt.plot(time, fit1, label = 'K=2 km/s, T=90 days')


k2 = 3 # semi major amplitude [km/s]
T2 = 110 # period [days]
time = np.arange(t1[0], t1[-1] + 1)
fit2 = -(k2 * -np.sin(2 * np.pi * (time + 40) / T2) - 20)
plt.plot(time, fit2, label = 'K=3 km/s, T=110 days')


k3 = 5 # semi major amplitude [km/s]
T3 = 230 # period [days]
time = np.arange(t1[0], t1[-1] + 1)
fit3 = -(k3 * np.sin(2 * np.pi * (time + 10) / T3) - 20)
plt.plot(time, fit3, label = 'K=5 km/s, T=230 days')


plt.xlabel("time [days]", fontsize=10)
plt.ylabel("radial velocity [km/s]", fontsize=10)
plt.title("HD 36112", fontsize=10)
plt.legend()
fig.savefig('/Users/zoeko/desktop/36112_orb_est.pdf')
# plt.show()

# 36112 table

# k and t unit conversion
k1_m, k2_m, k3_m = list(map(km_to_m, [k1, k2, k3]))
T1_s, T2_s, T3_s = list(map(day_to_sec, [T1, T2, T3]))
first_45 = mass_ratio(mass, (2 ** 1/2) / 2, k1_m, T1_s)
first_90 = mass_ratio(mass, 1, k1_m, T1_s)
sec_45 = mass_ratio(mass, (2 ** 1/2) / 2, k2_m, T2_s)
sec_90 = mass_ratio(mass, 1, k2_m, T2_s)
third_45 = mass_ratio(mass, (2 ** 1/2) / 2, k3_m, T3_s)
third_90 = mass_ratio(mass, 1, k3_m, T3_s)

star = ['HD 36112'] * 6
K = np.array([k1, k1, k2, k2, k3, k3])
T = np.array([T1, T1, T2, T2, T3, T3])
I = np.array([45, 90, 45, 90, 45, 90])
mass_ratios = np.array([first_45[0], first_90[0], sec_45[0], sec_90[0], third_45[0], third_90[0]])
masses = np.array([first_45[1], first_90[1], sec_45[1], sec_90[1], third_45[1], third_90[1]])

mr_36112 = QTable([star, K, T, I, mass_ratios, masses],
            names=('star', 'semi major amplitude [km/s]', 'period [days]', 'angle of inclination [deg]', 'mass ratio', 'mass of secondary star [solar masses]'))

ascii.write(mr_36112, '/Users/zoeko/desktop/HD36112_mass_ratios.csv', format='csv')





'''HD 294268'''
mass = solar_mass_to_kg(1.6)

fname = "/Users/zoeko/desktop/research/RVdata/HD294268.csv"

data = astropy.io.ascii.read(
    fname,
    names=[
         "Star",
        "JD",
        "RVs"
    ],
)

# load RV dataset
t1 = np.array(data['JD']) # times of observations
rv1_data = np.array(data['RVs']) # radial velocities (in km/s)
rv1_err = np.array([0.5] * 6) # uncertainty on rv measurement (in km/s) 

# plot data and error bars
fig, ax = plt.subplots(figsize=(7.1, 4.7))
plt.errorbar(t1, rv1_data, yerr=rv1_err, fmt=".b")

# plot orbital estimates

k1 = 1.5 # semi major amplitude [km/s]
T1 = 130 # period [days]
time = np.arange(t1[0], t1[-1] + 1)
fit1 = -(k1 * np.sin(2 * np.pi * (time - 12) / T1) - 25)
plt.plot(time, fit1, label = 'K=1.5 km/s, T=130 days')


k2 = 2 # semi major amplitude [km/s]
T2 = 170 # period [days]
time = np.arange(t1[0], t1[-1] + 1)
fit2 = -(k2 * -np.sin(2 * np.pi * (time + 55) / T2) - 25.5)
plt.plot(time, fit2, label = 'K=2 km/s, T=170 days')


k3 = 2 # semi major amplitude [km/s]
T3 = 360 # period [days]
time = np.arange(t1[0], t1[-1] + 1)
fit3 = -(k3 * np.sin(2 * np.pi * (time ) / T3) - 25.5)
plt.plot(time, fit3, label = 'K=2 km/s, T=360 days')


plt.xlabel("time [days]")
plt.ylabel("radial velocity [km/s]")
plt.title("HD 294268")
plt.legend()
# plt.show()
fig.savefig('/Users/zoeko/desktop/HD294268_orb_est.pdf')


# HD 294268 table

# k and t unit conversion
k1_m, k2_m, k3_m = km_to_m(k1), km_to_m(k2), km_to_m(k3)
T1_s, T2_s, T3_s = day_to_sec(T1), day_to_sec(T2), day_to_sec(T3)
first_45 = mass_ratio(mass, (2 ** 1/2) / 2, k1_m, T1_s)
first_90 = mass_ratio(mass, 1, k1_m, T1_s)
sec_45 = mass_ratio(mass, (2 ** 1/2) / 2, k2_m, T2_s)
sec_90 = mass_ratio(mass, 1, k2_m, T2_s)
third_45 = mass_ratio(mass, (2 ** 1/2) / 2, k3_m, T3_s)
third_90 = mass_ratio(mass, 1, k3_m, T3_s)

star = ['HD 294268'] * 6
K = np.array([k1, k1, k2, k2, k3, k3])
T = np.array([T1, T1, T2, T2, T3, T3])
I = np.array([45, 90, 45, 90, 45, 90])
mass_ratios = np.array([first_45[0], first_90[0], sec_45[0], sec_90[0], third_45[0], third_90[0]])
masses = np.array([first_45[1], first_90[1], sec_45[1], sec_90[1], third_45[1], third_90[1]])

mr_294268 = QTable([star, K, T, I, mass_ratios, masses],
            names=('star', 'semi major amplitude [km/s]', 'period [days]', 'angle of inclination [deg]', 'mass ratio', 'mass of secondary star [solar masses]'))

ascii.write(mr_294268, '/Users/zoeko/desktop/294268_mass_ratios.csv', format='csv')




'''HD 142666'''
mass = solar_mass_to_kg(1.64)

fname = "/Users/zoeko/desktop/research/RVdata/HD142666.csv"

data = astropy.io.ascii.read(
    fname,
    names=[
         "Star",
        "JD",
        "RVs"
    ],
)

# load RV dataset
t1 = np.array(data['JD']) # times of observations
start_time = min(t1)
rv1_data_38 = np.array(data['RVs']) # radial velocities (in km/s)
# rv1_data_39 = np.array(data['Order 39 RVs']) # radial velocities (in km/s)
# rv1_data_40 = np.array(data['Order 40 RVs']) # radial velocities (in km/s)
rv1_err = np.array([3] * 24) # uncertainty on rv measurement (in km/s) 

# plot data and error bars
fig, ax = plt.subplots(figsize=(7.1, 4.7))
plt.errorbar(t1 - start_time, rv1_data_38, yerr=rv1_err, fmt=".b")
# plt.errorbar(t1, rv1_data_39, yerr=rv1_err, fmt=".g")
# plt.errorbar(t1, rv1_data_40, yerr=rv1_err, fmt=".r")

# plot orbital estimates
k3 = 5 # semi major amplitude [km/s]
T3 = 70 # period [days]
time = np.arange(t1[0], t1[-1] + 1)
fit3 = k3 * np.sin(2 * np.pi * ((time+30) + 10) / T3)
# plt.plot(time - start_time, fit3, label = 'K=5 km/s, T=70 days')

k2 = 5 # semi major amplitude [km/s]
T2 = 100 # period [days]
time = np.arange(t1[0], t1[-1] + 1)
fit2 = k2 * np.sin(2 * np.pi * ((time+35) + 10) / T2)
# plt.plot(time - start_time, fit2, label = 'K=5 km/s, T=100 days')

k1 = 7 # semi major amplitude [km/s]
T1 = 120 # period [days]
time = np.arange(t1[0], t1[-1] + 1)
fit1 = k1 * np.sin(2 * np.pi * time / T1)
plt.plot(time - start_time, fit1, label = 'K=7 km/s, T=120 days')

plt.xlabel("Time - " +str(int(start_time)) + " MJDs [days]")
plt.ylabel("Radial Velocity [km/s]")
plt.title("HD 142666 Possible RV Curve")
plt.legend()
# plt.show()
fig.savefig('/Users/zoeko/desktop/HD142666_orb_est.pdf')

# 142666 table

# k and t unit conversion
k1_m, k2_m, k3_m = km_to_m(k1), km_to_m(k2), km_to_m(k3)
T1_s, T2_s, T3_s = day_to_sec(T1), day_to_sec(T2), day_to_sec(T3)
first_45 = mass_ratio(mass, (0.88294), k1_m, T1_s)
sec_45 = mass_ratio(mass, (0.88294), k2_m, T2_s)
third_45 = mass_ratio(mass, (0.88294), k3_m, T3_s)

star = ['HD 142666'] * 3
K = np.array([k1, k2, k3])
T = np.array([T1, T2, T3])
I = np.array([62, 62, 62])
mass_ratios = np.array([first_45[0], sec_45[0], third_45[0]])
masses = np.array([first_45[1], sec_45[1], third_45[1]])

mr_142666 = QTable([star, K, T, I, mass_ratios, masses],
            names=('star', 'semi major amplitude [km/s]', 'period [days]', 'angle of inclination [deg]', 'mass ratio', 'mass of secondary star [solar masses]'))

ascii.write(mr_142666, '/Users/zoeko/desktop/HD142666_mass_ratios.csv', format='csv')


'''VVSco (EPIC2099)'''
mass = solar_mass_to_kg(0.56)

fname = "/Users/zoeko/desktop/research/RVdata/VVSCO.csv"

data = astropy.io.ascii.read(
    fname,
    names=[
         "Star",
        "JD",
        "RVs"
    ],
)

# load RV dataset
t1 = np.array(data['JD']) # times of observations
rv1_data_38 = np.array(data['RVs']) # radial velocities (in km/s)
# rv1_data_39 = np.array(data['Order 39 RVs']) # radial velocities (in km/s)
# rv1_data_40 = np.array(data['Order 40 RVs']) # radial velocities (in km/s)
rv1_err = np.array([3] * 14) # uncertainty on rv measurement (in km/s) 

# plot data and error bars
fig, ax = plt.subplots(figsize=(7.1, 4.7))
plt.errorbar(t1, rv1_data_38, yerr=rv1_err, fmt=".b")
# plt.errorbar(t1, rv1_data_39, yerr=rv1_err, fmt=".g")
# plt.errorbar(t1, rv1_data_40, yerr=rv1_err, fmt=".r")

# plot orbital estimates
k1 = 10 # semi major amplitude [km/s]
T1 = 20 # period [days]
time = np.arange(t1[0], t1[-1] + 1)
fit1 = -(k1 * np.sin(2 * np.pi * (time+2) / T1) + 13) + 17
plt.plot(time, fit1, label = 'K=10, T=20')

k2 = 4 # semi major amplitude [km/s]
T2 = 20 # period [days]
time = np.arange(t1[0], t1[-1] + 1)
fit2 = -(k2 * np.sin(2 * np.pi * (time+2) / T2) + 7) + 5
plt.plot(time, fit2, label = 'K=4, T=20')

k3 = 3 # semi major amplitude [km/s]
T3 = 13 # period [days]
time = np.arange(t1[0], t1[-1] + 1)
fit3 = -(k3 * np.sin(2 * np.pi * (time) / T3) + 5)
plt.plot(time, fit3, label = 'K=3, T=13')

plt.xlabel("time [days]")
plt.ylabel("radial velocity [km/s]")
plt.title("VVSco")
plt.legend()
fig.savefig('/Users/zoeko/desktop/VVSco_orb_est.pdf')

# VVSco table

# k and t unit conversion
k1_m, k2_m, k3_m = km_to_m(k1), km_to_m(k2), km_to_m(k3)
T1_s, T2_s, T3_s = day_to_sec(T1), day_to_sec(T2), day_to_sec(T3)
first_45 = mass_ratio(mass, (2 ** 1/2) / 2, k1_m, T1_s)
first_90 = mass_ratio(mass, 1, k1_m, T1_s)
sec_45 = mass_ratio(mass, (2 ** 1/2) / 2, k2_m, T2_s)
sec_90 = mass_ratio(mass, 1, k2_m, T2_s)
third_45 = mass_ratio(mass, (2 ** 1/2) / 2, k3_m, T3_s)
third_90 = mass_ratio(mass, 1, k3_m, T3_s)

star = ['VV Sco'] * 6
K = np.array([k1, k1, k2, k2, k3, k3])
T = np.array([T1, T1, T2, T2, T3, T3])
I = np.array([45, 90, 45, 90, 45, 90])
mass_ratios = np.array([first_45[0], first_90[0], sec_45[0], sec_90[0], third_45[0], third_90[0]])
masses = np.array([first_45[1], first_90[1], sec_45[1], sec_90[1], third_45[1], third_90[1]])

mr_VVSco = QTable([star, K, T, I, mass_ratios, masses],
            names=('star', 'semi major amplitude [km/s]', 'period [days]', 'angle of inclination [deg]', 'mass ratio', 'mass of secondary star [solar masses]'))

ascii.write(mr_VVSco, '/Users/zoeko/desktop/VVSco_mass_ratios.csv', format='csv')

'''DF Tau'''
mass = solar_mass_to_kg(0.32)

fname = "/Users/zoeko/desktop/research/RVdata/DFTAU.csv"

data = astropy.io.ascii.read(
    fname,
    names=[
        "Star",
        "JD",
        "RVs"
    ],
)

# load RV dataset
t1 = np.array(data['JD']) # times of observations
rv1_data_49 = np.array(data['RVs']) # radial velocities (in km/s)
# rv1_data_47 = np.array(data['Order 47 RVs']) # radial velocities (in km/s)
# rv1_data_48 = np.array(data['Order 48 RVs']) # radial velocities (in km/s)
rv1_err = np.array([3] * 5) # uncertainty on rv measurement (in km/s) 

# plot data and error bars
fig, ax = plt.subplots(figsize=(7.1, 4.7))
plt.errorbar(t1, rv1_data_49, yerr=rv1_err, fmt=".b")
# plt.errorbar(t1, rv1_data_47, yerr=rv1_err, fmt=".g")
# plt.errorbar(t1, rv1_data_48, yerr=rv1_err, fmt=".r")

# plot orbital estimates

k1 = 5 # semi major amplitude [km/s]
T1 = 90 # period [days]
time = np.arange(t1[0], t1[-1] + 1)
fit1 = -(k1 * np.sin(2 * np.pi * (time + 10) / T1) - 20)
plt.plot(time, fit1, label = 'K=5, T=90')


k2 = 5 # semi major amplitude [km/s]
T2 = 110 # period [days]
time = np.arange(t1[0], t1[-1] + 1)
fit2 = -(k2 * -np.sin(2 * np.pi * (time + 40) / T2) - 20)
plt.plot(time, fit2, label = 'K=5, T=110')


k3 = 5 # semi major amplitude [km/s]
T3 = 230 # period [days]
time = np.arange(t1[0], t1[-1] + 1)
fit3 = -(k3 * np.sin(2 * np.pi * (time + 10) / T3) - 21)
plt.plot(time, fit3, label = 'K=5, T=230')


plt.xlabel("time [days]", fontsize=10)
plt.ylabel("radial velocity [km/s]", fontsize=10)
plt.title("DF Tau")
plt.legend()
fig.savefig('/Users/zoeko/desktop/DF_Tau_orb_est.pdf')
# plt.show()

# DF Tau table

# k and t unit conversion
k1_m, k2_m, k3_m = list(map(km_to_m, [k1, k2, k3]))
T1_s, T2_s, T3_s = list(map(day_to_sec, [T1, T2, T3]))
first_45 = mass_ratio(mass, (2 ** 1/2) / 2, k1_m, T1_s)
first_90 = mass_ratio(mass, 1, k1_m, T1_s)
sec_45 = mass_ratio(mass, (2 ** 1/2) / 2, k2_m, T2_s)
sec_90 = mass_ratio(mass, 1, k2_m, T2_s)
third_45 = mass_ratio(mass, (2 ** 1/2) / 2, k3_m, T3_s)
third_90 = mass_ratio(mass, 1, k3_m, T3_s)

star = ['DF Tau'] * 6
K = np.array([k1, k1, k2, k2, k3, k3])
T = np.array([T1, T1, T2, T2, T3, T3])
I = np.array([45, 90, 45, 90, 45, 90])
mass_ratios = np.array([first_45[0], first_90[0], sec_45[0], sec_90[0], third_45[0], third_90[0]])
masses = np.array([first_45[1], first_90[1], sec_45[1], sec_90[1], third_45[1], third_90[1]])

mr_DF_Tau = QTable([star, K, T, I, mass_ratios, masses],
            names=('star', 'semi major amplitude [km/s]', 'period [days]', 'angle of inclination [deg]', 'mass ratio', 'mass of secondary star [solar masses]'))

ascii.write(mr_DF_Tau, '/Users/zoeko/desktop/DFTau_mass_ratios.csv', format='csv')


'''HIP 20387 (RY TAU)'''
mass = solar_mass_to_kg(0.3)

# drop last data point

fname = "/Users/zoeko/desktop/research/RVdata/RYTAU.csv"

data = astropy.io.ascii.read(
    fname,
    names=[
         
         "Star",
        "JD",
        "RVs"
    ],
)

# load RV dataset
t1 = np.array(data['JD'])[:-1]# times of observations
rv1_data_38 = np.array(data['RVs'])[:-1] # radial velocities (in km/s)
# rv1_data_39 = np.array(data['RVs'])[:-1] # radial velocities (in km/s)
# rv1_data_40 = np.array(data['Order 40 RVs'])[:-1] # radial velocities (in km/s)
rv1_err = np.array([1.5] * 16) # uncertainty on rv measurement (in km/s) 

# plot data and error bars
fig, ax = plt.subplots(figsize=(7.1, 4.7))
plt.errorbar(t1, rv1_data_38, yerr=rv1_err, fmt=".b")
# plt.errorbar(t1, rv1_data_39, yerr=rv1_err, fmt=".g")
# plt.errorbar(t1, rv1_data_40, yerr=rv1_err, fmt=".r")

# plot orbital estimates
k2 = 2 # semi major amplitude [km/s]
T2 = 50 # period [days]
time = np.arange(t1[0], t1[-1] + 1)
fit2 = -(k2 * np.sin(2 * np.pi * ((time+45) + 10) / T2) - 17.5)
plt.plot(time, fit2, label = 'K=2 km/s T=50 days')

k1 = 2 # semi major amplitude [km/s]
T1 = 100 # period [days]
time = np.arange(t1[0], t1[-1] + 1)
fit1 = -(k1 * np.sin(2 * np.pi * (time + 13) / T1) - 17)
plt.plot(time, fit1, label = 'K=2 km/s T=100 days')

k3 = 2 # semi major amplitude [km/s]
T3 = 180 # period [days]
time = np.arange(t1[0], t1[-1] + 1)
fit3 = -(k3 * np.sin(2 * np.pi * ((time+68) + 20) / T3) - 17)
plt.plot(time, fit3, label = 'K=2 km/s T=180 days')

plt.xlabel("time [days]")
plt.ylabel("radial velocity [km/s]")
plt.title("RY Tau")
plt.legend()
fig.savefig('/Users/zoeko/desktop/RYTau_orb_est.pdf')
# plt.show()

# HIP 20387 table

# k and t unit conversion
k1_m, k2_m, k3_m = km_to_m(k1), km_to_m(k2), km_to_m(k3)
T1_s, T2_s, T3_s = day_to_sec(T1), day_to_sec(T2), day_to_sec(T3)
first_45 = mass_ratio(mass, 0.9063, k1_m, T1_s)
sec_45 = mass_ratio(mass, 0.9063, k2_m, T2_s)
third_45 = mass_ratio(mass, 0.9063, k3_m, T3_s)

star = ['HIP 20387'] * 3
K = np.array([k1, k2, k3])
T = np.array([T1, T2, T3])
I = np.array([65, 65, 65])
mass_ratios = np.array([first_45[0], sec_45[0], third_45[0]])
masses = np.array([first_45[1], sec_45[1], third_45[1]])

mr_HIP20387 = QTable([star, K, T, I, mass_ratios, masses],
            names=('star', 'semi major amplitude [km/s]', 'period [days]', 'angle of inclination [deg]', 'mass ratio', 'mass of secondary star [solar masses]'))

ascii.write(mr_HIP20387, '/Users/zoeko/desktop/RYTAU_mass_ratios.csv', format='csv')






'''AS 209'''
mass = solar_mass_to_kg(0.77)
fname = "/Users/zoeko/desktop/research/RVdata/AS209.csv"

data = astropy.io.ascii.read(
    fname,
    names=[
        "Star",
        "JD",
        "RVs"
    ],
)

# load RV dataset
t1 = np.array(data['JD']) # times of observations
rv1_data_38 = np.array(data['RVs']) # radial velocities (in km/s)
# rv1_data_39 = np.array(data['Order 39 RVs']) # radial velocities (in km/s)
# rv1_data_40 = np.array(data['Order 40 RVs']) # radial velocities (in km/s)
rv1_err = np.array([1] * 4) # uncertainty on rv measurement (in km/s) 

# plot data and error bars
fig, ax = plt.subplots(figsize=(7.1, 4.7))
plt.errorbar(t1, rv1_data_38, yerr=rv1_err, fmt=".b")
# plt.errorbar(t1, rv1_data_39, yerr=rv1_err, fmt=".g")
# plt.errorbar(t1, rv1_data_40, yerr=rv1_err, fmt=".r")

# plot orbital estimates
k1 = 1 # semi major amplitude [km/s]
T1 = 50 # period [days]
time = np.arange(t1[0], t1[-1] + 1)
fit1 = k1 * np.sin(2 * np.pi * (time+25) / T1) - 7
plt.plot(time, fit1, label = 'K=1 km/s T=50 days')

k2 = 1 # semi major amplitude [km/s]
T2 = 60 # period [days]
time = np.arange(t1[0], t1[-1] + 1)
fit2 = k2 * np.sin(2 * np.pi * ((time+5) + 10) / T2) - 7
plt.plot(time, fit2, label = 'K=1 km/s T=60 days')

k3 = 1 # semi major amplitude [km/s]
T3 = 70 # period [days]
time = np.arange(t1[0], t1[-1] + 1)
fit3 = k3 * np.sin(2 * np.pi * ((time+35) + 10) / T3) - 7.2
plt.plot(time, fit3, label = 'K=1 km/s T=70 days')

plt.xlabel("time [days]")
plt.ylabel("radial velocity [km/s]")
plt.title("AS 209")
plt.legend()
fig.savefig('/Users/zoeko/desktop/AS209_orb_est.pdf')
# plt.show()


# AS 209 table

# k and t unit conversion
k1_m, k2_m, k3_m = km_to_m(k1), km_to_m(k2), km_to_m(k3)
T1_s, T2_s, T3_s = day_to_sec(T1), day_to_sec(T2), day_to_sec(T3)
first_45 = mass_ratio(mass, 0.57358, k1_m, T1_s)
first_90 = mass_ratio(mass, 1, k1_m, T1_s)
sec_45 = mass_ratio(mass, 0.57358, k2_m, T2_s)
sec_90 = mass_ratio(mass, 1, k2_m, T2_s)
third_45 = mass_ratio(mass, 0.57358, k3_m, T3_s)
third_90 = mass_ratio(mass, 1, k3_m, T3_s)

star = ['HIP 82323'] * 3
K = np.array([k1, k2, k3])
T = np.array([T1, T2, T3])
I = np.array([35, 35, 35])
mass_ratios = np.array([first_45[0], sec_45[0], third_45[0]])
masses = np.array([first_45[1], sec_45[1], third_45[1]])

mr_HIP82323 = QTable([star, K, T, I, mass_ratios, masses],
            names=('star', 'semi major amplitude [km/s]', 'period [days]', 'angle of inclination [deg]', 'mass ratio', 'mass of secondary star [solar masses]'))

ascii.write(mr_HIP82323, '/Users/zoeko/desktop/AS209_mass_ratios.csv', format='csv')













'''AS 205'''
mass = solar_mass_to_kg(0.9)
fname = "/Users/zoeko/desktop/research/RVdata/AS205.csv"

data = astropy.io.ascii.read(
    fname,
    names=[
        "Star",
        "JD",
        "RVs"
    ],
)

# load RV dataset
t1 = np.array(data['JD']) # times of observations
rv1_data_38 = np.array(data['RVs']) # radial velocities (in km/s)
# rv1_data_39 = np.array(data['Order 39 RVs']) # radial velocities (in km/s)
# rv1_data_40 = np.array(data['Order 40 RVs']) # radial velocities (in km/s)
rv1_err = np.array([1] * 19) # uncertainty on rv measurement (in km/s) 

# plot data and error bars
fig, ax = plt.subplots(figsize=(7.1, 4.7))
plt.errorbar(t1, rv1_data_38, yerr=rv1_err, fmt=".b")
# plt.errorbar(t1, rv1_data_39, yerr=rv1_err, fmt=".g")
# plt.errorbar(t1, rv1_data_40, yerr=rv1_err, fmt=".r")

# plot orbital estimates
k1 = 3 # semi major amplitude [km/s]
T1 = 50 # period [days]
time = np.arange(t1[0], t1[-1] + 1)
fit1 = -k1 * np.sin(2 * np.pi * (time+25) / T1) - 7
plt.plot(time, fit1, label = 'K=3 km/s T=50 days')

k2 = 5 # semi major amplitude [km/s]
T2 = 60 # period [days]
time = np.arange(t1[0], t1[-1] + 1)
fit2 = -k2 * np.sin(2 * np.pi * ((time+5) + 10) / T2) - 6
plt.plot(time, fit2, label = 'K=5 km/s T=60 days')

k3 = 5 # semi major amplitude [km/s]
T3 = 70 # period [days]
time = np.arange(t1[0], t1[-1] + 1)
fit3 = -k3 * np.sin(2 * np.pi * ((time+35) + 10) / T3) - 5.2
plt.plot(time, fit3, label = 'K=5 km/s T=70 days')

plt.xlabel("time [days]")
plt.ylabel("radial velocity [km/s]")
plt.title("AS 205")
plt.legend()
fig.savefig('/Users/zoeko/desktop/AS205_orb_est.pdf')
# plt.show()


# AS 209 table

# k and t unit conversion
k1_m, k2_m, k3_m = km_to_m(k1), km_to_m(k2), km_to_m(k3)
T1_s, T2_s, T3_s = day_to_sec(T1), day_to_sec(T2), day_to_sec(T3)
first_45 = mass_ratio(mass, 0.3420, k1_m, T1_s)
first_90 = mass_ratio(mass, 1, k1_m, T1_s)
sec_45 = mass_ratio(mass, 0.3420, k2_m, T2_s)
sec_90 = mass_ratio(mass, 1, k2_m, T2_s)
third_45 = mass_ratio(mass, 0.3420, k3_m, T3_s)
third_90 = mass_ratio(mass, 1, k3_m, T3_s)

star = ['HIP 82323'] * 3
K = np.array([k1, k2, k3])
T = np.array([T1, T2, T3])
I = np.array([29, 20, 20])
mass_ratios = np.array([first_45[0], sec_45[0], third_45[0]])
masses = np.array([first_45[1], sec_45[1], third_45[1]])

mr_HIP82323 = QTable([star, K, T, I, mass_ratios, masses],
            names=('star', 'semi major amplitude [km/s]', 'period [days]', 'angle of inclination [deg]', 'mass ratio', 'mass of secondary star [solar masses]'))

ascii.write(mr_HIP82323, '/Users/zoeko/desktop/AS205_mass_ratios.csv', format='csv')





'''AB AUR'''
mass = solar_mass_to_kg(2.4)
fname = "/Users/zoeko/desktop/research/RVdata/ABAUR.csv"

data = astropy.io.ascii.read(
    fname,
    names=[
        "Star",
        "JD",
        "RVs"
    ],
)

# load RV dataset
t1 = np.array(data['JD']) # times of observations
rv1_data_38 = np.array(data['RVs']) # radial velocities (in km/s)
# rv1_data_39 = np.array(data['Order 39 RVs']) # radial velocities (in km/s)
# rv1_data_40 = np.array(data['Order 40 RVs']) # radial velocities (in km/s)
rv1_err = np.array([1] * 11) # uncertainty on rv measurement (in km/s) 

# plot data and error bars
fig, ax = plt.subplots(figsize=(7.1, 4.7))
plt.errorbar(t1, rv1_data_38, yerr=rv1_err, fmt=".b")
# plt.errorbar(t1, rv1_data_39, yerr=rv1_err, fmt=".g")
# plt.errorbar(t1, rv1_data_40, yerr=rv1_err, fmt=".r")

# plot orbital estimates
k1 = 3 # semi major amplitude [km/s]
T1 = 90 # period [days]
time = np.arange(t1[0], t1[-1] + 1)
fit1 = k1 * np.sin(2 * np.pi * (time+25) / T1) + 38
plt.plot(time, fit1, label = 'K=3 km/s T=90 days')

k2 = 5 # semi major amplitude [km/s]
T2 = 110 # period [days]
time = np.arange(t1[0], t1[-1] + 1)
fit2 = k2 * np.sin(2 * np.pi * ((time+5) + 10) / T2) + 35
plt.plot(time, fit2, label = 'K=5 km/s T=110 days')

k3 = 4 # semi major amplitude [km/s]
T3 = 70 # period [days]
time = np.arange(t1[0], t1[-1] + 1)
fit3 = k3 * np.sin(2 * np.pi * ((time+35+40) + 10) / T3) + 36
plt.plot(time, fit3, label = 'K=4 km/s T=70 days')

plt.xlabel("time [days]")
plt.ylabel("radial velocity [km/s]")
plt.title("AB AUR")
plt.legend()
fig.savefig('/Users/zoeko/desktop/ABAUR_orb_est.pdf')
# plt.show()


# AB AUR table

# k and t unit conversion
k1_m, k2_m, k3_m = km_to_m(k1), km_to_m(k2), km_to_m(k3)
T1_s, T2_s, T3_s = day_to_sec(T1), day_to_sec(T2), day_to_sec(T3)
first_45 = mass_ratio(mass, 0.3746, k1_m, T1_s)
first_90 = mass_ratio(mass, 1, k1_m, T1_s)
sec_45 = mass_ratio(mass, 0.3746, k2_m, T2_s)
sec_90 = mass_ratio(mass, 1, k2_m, T2_s)
third_45 = mass_ratio(mass, 0.3746, k3_m, T3_s)
third_90 = mass_ratio(mass, 1, k3_m, T3_s)

star = ['HIP 82323'] * 3
K = np.array([k1, k2, k3])
T = np.array([T1, T2, T3])
I = np.array([22, 22, 22])
mass_ratios = np.array([first_45[0], sec_45[0], third_45[0]])
masses = np.array([first_45[1], sec_45[1], third_45[1]])

mr_HIP82323 = QTable([star, K, T, I, mass_ratios, masses],
            names=('star', 'semi major amplitude [km/s]', 'period [days]', 'angle of inclination [deg]', 'mass ratio', 'mass of secondary star [solar masses]'))

ascii.write(mr_HIP82323, '/Users/zoeko/desktop/ABAUR_mass_ratios.csv', format='csv')






'''HD 245185'''
mass = solar_mass_to_kg(2.1)
fname = "/Users/zoeko/desktop/research/RVdata/HD245185.csv"

data = astropy.io.ascii.read(
    fname,
    names=[
        "Star",
        "JD",
        "RVs"
    ],
)

# load RV dataset
t1 = np.array(data['JD']) # times of observations
rv1_data_38 = np.array(data['RVs']) # radial velocities (in km/s)
# rv1_data_39 = np.array(data['Order 39 RVs']) # radial velocities (in km/s)
# rv1_data_40 = np.array(data['Order 40 RVs']) # radial velocities (in km/s)
rv1_err = np.array([1] * 6) # uncertainty on rv measurement (in km/s) 

# plot data and error bars
fig, ax = plt.subplots(figsize=(7.1, 4.7))
plt.errorbar(t1, rv1_data_38, yerr=rv1_err, fmt=".b")
# plt.errorbar(t1, rv1_data_39, yerr=rv1_err, fmt=".g")
# plt.errorbar(t1, rv1_data_40, yerr=rv1_err, fmt=".r")

# plot orbital estimates
k1 = 3 # semi major amplitude [km/s]
T1 = 90 # period [days]
time = np.arange(t1[0], t1[-1] + 1)
fit1 = k1 * np.sin(2 * np.pi * (time+25) / T1) + 8
plt.plot(time, fit1, label = 'K=3 km/s T=90 days')

k2 = 5 # semi major amplitude [km/s]
T2 = 110 # period [days]
time = np.arange(t1[0], t1[-1] + 1)
fit2 = -k2 * np.sin(2 * np.pi * ((time+5) + 10) / T2) + 13
plt.plot(time, fit2, label = 'K=5 km/s T=110 days')

k3 = 4 # semi major amplitude [km/s]
T3 = 70 # period [days]
time = np.arange(t1[0], t1[-1] + 1)
fit3 = -k3 * np.sin(2 * np.pi * ((time+35+40) + 10) / T3) + 23
plt.plot(time, fit3, label = 'K=4 km/s T=70 days')

plt.xlabel("time [days]")
plt.ylabel("radial velocity [km/s]")
plt.title("HD 245185")
plt.legend()
# fig.savefig('/Users/zoeko/desktop/HD245185_orb_est.pdf')
# plt.show()


# HD 245185 table

# k and t unit conversion
k1_m, k2_m, k3_m = km_to_m(k1), km_to_m(k2), km_to_m(k3)
T1_s, T2_s, T3_s = day_to_sec(T1), day_to_sec(T2), day_to_sec(T3)
first_45 = mass_ratio(mass, 0.743145, k1_m, T1_s)
first_90 = mass_ratio(mass, 1, k1_m, T1_s)
sec_45 = mass_ratio(mass, 0.743145, k2_m, T2_s)
sec_90 = mass_ratio(mass, 1, k2_m, T2_s)
third_45 = mass_ratio(mass, 0.743145, k3_m, T3_s)
third_90 = mass_ratio(mass, 1, k3_m, T3_s)

star = ['HD 245185'] * 3
K = np.array([k1, k2, k3])
T = np.array([T1, T2, T3])
I = np.array([48, 48, 48])
mass_ratios = np.array([first_45[0], sec_45[0], third_45[0]])
masses = np.array([first_45[1], sec_45[1], third_45[1]])

mr_HIP82323 = QTable([star, K, T, I, mass_ratios, masses],
            names=('star', 'semi major amplitude [km/s]', 'period [days]', 'angle of inclination [deg]', 'mass ratio', 'mass of secondary star [solar masses]'))

ascii.write(mr_HIP82323, '/Users/zoeko/desktop/HD245185_mass_ratios.csv', format='csv')





# combine tables
com = vstack([mr_36112, mr_142666, mr_VVSco, mr_HIP20387, mr_HIP82323])
ascii.write(com, '/Users/zoeko/desktop/flat_rv_mass_ratios.csv', format='csv')