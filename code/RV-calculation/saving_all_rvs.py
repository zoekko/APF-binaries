# Stack tables containing RVs for each star
# to create one table with all RV measurements
# write table as aastex

# SAVING ALL RVs:
# 1. Save individual RVs using "saving_rvs.py"
# 2. Stack all RV measurements together using this file.


import astropy
from astropy.table import Table, vstack
from astropy.table import Table, unique
from astropy.io import ascii

stars = [
    'LKHA330',     
    # # 'HD283447',
    'RYTAU',      
    'DFTAU',      
    # # 'DGTAU', 
    # # 'DHTAU', 
    # # 'GKTAU', 
    'ABAUR',      
    'HD36112',      
    # # 'HD245185', 
    'CQTAU',      
    # # 'TXORI', 
    # # 'W964771899', 
    'HD294268',      
    # # 'ZCMA', 
    'HD142666',      
    # # 'HD143006', 
    # # '2MJ16092217', 
    'AS205',      
    # # 'CD2211432', 
    'VVSCO',      
    # # '2MJ16262356', 
    # # '2MJ16392402', 
    'AS209',      
    # # 'HD163296', 
    # # 'V1515CYG', 
    ]

table = Table()
for star in stars:
    fname = "/Users/zoeko/desktop/research/medianRVdata/" + star + ".csv"
    data = astropy.io.ascii.read(
        fname,
        names=[
            "Target",
            "JD",
            "RV (km/s)"
        ],
    )
    table = vstack([table, data])

print(table)

# ascii.write(table, '/Users/zoeko/desktop/all_rvs.csv', format='csv')
# ascii.write(table, '/Users/zoeko/desktop/all_rvs.txt', format='aastex', formats = {'RV (km/s)': '%0.2f'})