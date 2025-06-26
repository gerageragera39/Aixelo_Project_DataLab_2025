from src.fingerprints.generators.FP_Generator import generate_45_fingerprints

xyz_path_QMOF = 'mofs.xyz'
refcodes_path_QMOF = 'refcodes.csv'

generate_45_fingerprints(refcodes_path_QMOF, xyz_path_QMOF, 'QMOF')
