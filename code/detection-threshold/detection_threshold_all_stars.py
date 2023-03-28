# Create a figure and table summarizing the detection threshold of all stars
# Figure: create a figure of the detection threshold plot for every target
# Table: create a table of target name, mass, observation duration, and RV scatter

import numpy as np 
import matplotlib.pylab as plt 
from astropy.io import ascii
from astropy.table import Table, vstack
import pandas as pd
from sympy import symbols, Eq, solve
import matplotlib

plt.rcParams["figure.figsize"] = (9,12)

directory = '/Users/zoeko/desktop/research/separated/'


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
    closest_mass = find_nearest(secondary_masses, mass)
    if mass == 0 or closest_mass == 0:
        return 0.0
    index = secondary_masses.tolist().index(closest_mass)
    return np.round(K_vals[index], 1)


def threshold_plot(name, row, column, sigma_v, detla_t, primary_mass, sinI, K_min, K_max, num_points,
                    plot_both = False):

    df = pd.DataFrame(columns=['Target', 'K', 'P'])

    if name == 'CD2211432':
        name = 'CD-22 11432'

    K_vals = np.linspace(K_min, K_max, num_points)
    P_vals = np.array([calc_p(K, sigma_v, delta_t) for K in K_vals])
    for i in np.arange(len(K_vals)):
        to_append = pd.Series({'Target' : name, 'K' : K_vals[i], 'P' : P_vals[i]})
        df = pd.concat([df, to_append], ignore_index=True)
        # df = df.append({'Target' : name, 'K' : K_vals[i], 'P' : P_vals[i]}, 
        #         ignore_index = True)

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

    ax1 = axs[row_num, column_num]

    ax1.fill_between(np.array(secondary_masses_plot, dtype=float), P_plot, y2=max(P_plot), color='brown')
    rect1 = matplotlib.patches.Rectangle((0, 0), secondary_masses[idx], max(P_plot), color='brown')
    ax1.add_patch(rect1)
    ax1.set_xlabel('Secondary Mass [$M_{\odot}$]')
    ax1.set_ylabel('P [days]')
    ax1.set_title(name)
    ax1.set_xlim(0, primary_mass)
    ax1.set_ylim(0, max(P_plot))


    ax2 = ax1.twiny()
    ax2.set_xticks(ax1.get_xticks())
    ax2.set_xbound(ax1.get_xbound())
    ax2.set_xticklabels([mass_to_K(x, secondary_masses, K_vals) for x in ax1.get_xticks()])
    ax2.set_xlabel('K [km/s]')

    return df

df = pd.read_pickle(directory + 'table.pkl')


# # CREATE TABLE
# df['t'] = df['t'].astype(int)
# df['RV'] = np.round(df['RV'], 1)
# df.replace(45, 0, inplace=True)
# paper_df = df[['Target', 'Mass', 't', 'RV', 'Inclination']].reset_index(drop=True) 

# # print(paper_df)
# print(paper_df.to_latex(index=False))


row_num = 0
column_num = 0
fig, axs = plt.subplots(4, 3)

entire_df = pd.DataFrame()

for index, row in df.iterrows():
    target_name = row['Target']
    sigma_v = (row['RV'])
    delta_t = (row['t'])
    primary_mass = row['Mass']
    inclination = row['Inclination']
    sinI = np.sin(inclination * np.pi / 180)
    print(target_name)
    print(str(sigma_v) + ' ' + str(delta_t) + ' ' + str(primary_mass) + ' ' + str(sinI))
    
    df = threshold_plot(target_name, row_num, column_num, sigma_v, delta_t, primary_mass, sinI, 0.5, 20, 1000)
    if entire_df.empty:
        entire_df = df
    else:
        entire_df = pd.concat([entire_df, df])

    if row_num == 3:
        row_num = 0
        column_num += 1
    else:
        row_num += 1


for ax in axs.flat:
    ax.set(xlabel='Secondary Mass [$M_{\odot}$]', ylabel='P [days]')

fig.tight_layout()
# plt.savefig(directory + 'detection_threshold.pdf')
plt.show()  

# entire_df.to_pickle(directory + 'Ps_and_Ks.pkl')