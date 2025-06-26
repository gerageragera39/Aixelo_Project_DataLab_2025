import os

import pandas as pd

from src.fingerprints.generators.FP_Generator import generate_120_fingerprints, add_evergy

xyz_path_CO2 = '../mofs_CO2.xyz'
refcodes_path_CO2 = '../refcodes_CO2.csv'

xyz_path_H2O = '../mofs_H2O.xyz'
refcodes_path_H2O = '../refcodes_H2O.csv'

generate_120_fingerprints(refcodes_path_H2O, xyz_path_H2O, 'H2O')
generate_120_fingerprints(refcodes_path_CO2, xyz_path_CO2, 'CO2')



add_evergy('stoich120_fingerprints_CO2.csv', '../../average_energy_CO2.csv', '120_CO2')
add_evergy('stoich120_fingerprints_H2O.csv', '../../average_energy_H2O.csv', '120_H2O')
