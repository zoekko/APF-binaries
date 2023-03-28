# Create a detection threshold plot:
# Given an uncertainty of rv measurements and a time duration of observation, 
# determine the detection threshold

import numpy as np
import matplotlib.pylab as plt
from sympy import symbols, Eq, solve
import pandas as pd
import matplotlib

G = 6.67408 * 10 ** -11 #m3 kg-1 s-2

# unknown PDF - log likelihood function
def f(jds, P, K, phase, vel):
    '''calculate chi squared value
    of the rvs with the fit given 
    by these four parameters'''
    fit = K*np.sin(phase + 2*np.pi*jds/P) + vel
    return -1/2 * np.sum(((rvs - fit) ** 2) / (noise ** 2))

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

def mass_ratio(m1, sinI, semiamp, period):
    '''calculate mass ratio and mass of companion star given
    m1 in solar masses
    semiamp in km/s
    period in days'''
    x = symbols('x')
    P = day_to_sec(period)
    # print(P)
    K = km_to_m(semiamp)
    # print(K)
    mass = solar_mass_to_kg(m1)
    constant = (((2 * np.pi * G) / P) ** (1/3)) * sinI
    expr = ((x / ((mass + x) ** (2/3))) * constant) - K
    x = solve(expr)[0]
    # print('x is: ' + str(x))
    m2 = kg_to_solar_mass(solve(expr)[0]) # in solar masses
    m_ratio = m2 / m1
    return m_ratio, m2

def calc_K(m1, m2, P, sinI):
    '''calculate the semi-amplitude
    given the two masses, period, and inclination'''
    m1 = solar_mass_to_kg(m1)
    m2 = solar_mass_to_kg(m2)
    constant = ((2 * np.pi * G) / P) * (sinI ** 3)
    mass_ratio = (m2 ** 3) / ((m1 + m2) ** 2)
    K = ((constant * mass_ratio) ** (1/3)) / 1000
    return K

# first calculate the range of acceptable max K and min P values 
def calc_p(K, sigma_v, delta_t):
    # if (pd.isna(((np.pi * delta_t) / np.arccos(1 - (sigma_v / K))))):
    #     return 1
    if (K <= sigma_v / 2):
        return 1
    return (np.pi * delta_t) / np.arccos(1 - (sigma_v / K))
    
def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]

def mass_to_K(mass, secondary_masses, K_vals):
    # plt.scatter(secondary_masses, K_vals)
    # plt.show()
    closest_mass = find_nearest(secondary_masses, mass)
    index = secondary_masses.tolist().index(closest_mass)
    return np.round(K_vals[index], 1)

def threshold_plot(sigma_v, detla_t, primary_mass, sinI, K_min, K_max, num_points,
                    plot_both = False):

    K_vals = np.linspace(K_min, K_max, num_points)
    P_vals = np.array([calc_p(K, sigma_v, delta_t) for K in K_vals])

    # convert to mass
    secondary_masses = np.array([])
    for i in np.arange(len(K_vals)):
        K = K_vals[i]
        P = P_vals[i]
        secondary_mass = mass_ratio(primary_mass, sinI, K, P)[1]
        secondary_masses = np.append(secondary_masses, secondary_mass)

    idx = sum(np.array(P_vals) == 1)
    P_plot = P_vals[idx:]
    secondary_masses_plot = secondary_masses[idx:].tolist()

    if plot_both == True:
        fig, (ax1, ax2) = plt.subplots(2, 1, sharex=False)
        ax1.fill_between(K_vals, P_vals, max(P_vals))
        ax1.set_xlabel('K [km/s]')
        ax1.set_ylabel('P [days]')
        ax1.set_title('Detection Threshold given RV uncertainty of ' + str(sigma_v) + 
        ' km/s, timespan of observations: ' + str(delta_t) + ' days')

        # ax2.fill_between(secondary_masses, P_vals, 700)
        ax2.fill_between(np.array(secondary_masses_plot, dtype=float), P_plot, max(P_plot))
        rect1 = matplotlib.patches.Rectangle((0, 0), secondary_masses[idx], max(P_plot))
        ax2.add_patch(rect1)
        ax2.set_xlabel('Secondary Mass [M_solar]')
        ax2.set_ylabel('P [days]')
        plt.show()
    else:
        fig = plt.figure()
        ax1 = fig.add_subplot(111)

        ax1.fill_between(np.array(secondary_masses_plot, dtype=float), P_plot, y2=max(P_plot), color='brown')
        rect1 = matplotlib.patches.Rectangle((0, 0), secondary_masses[idx], max(P_plot), color='brown')
        ax1.add_patch(rect1)
        ax1.set_xlabel('Secondary Mass [$M_{\odot}$]')
        ax1.set_ylabel('P [days]')


        ax2 = ax1.twiny()
        ax2.set_xticks( ax1.get_xticks() )
        ax2.set_xbound(ax1.get_xbound())
        ax2.set_xticklabels([mass_to_K(x, secondary_masses, K_vals) for x in ax1.get_xticks()])
        ax2.set_xlabel('K [km/s]')
        plt.show()


sigma_v = 0.9 #km/s
delta_t = 101 #days
primary_mass = 2.5 #solar mass
sinI = 0.66
K_min = 0.5
K_max = 20
num_points = 20


threshold_plot(sigma_v, delta_t, primary_mass, sinI, K_min, K_max, num_points)