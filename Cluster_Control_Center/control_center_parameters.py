import os, io, sys
models = {'QCD': 'QCD_Model', 'XY': 'XY_Model', 'TEE': 'TEE_Calc'}
print("NEW SIMULATION")
model = sys.argv[1]
y_tilde = float(sys.argv[2])
n = int(sys.argv[3])
lattice_size = int(sys.argv[4])
T_step = float(sys.argv[5])
measurements = int(sys.argv[6])
T_start = float(sys.argv[7])
T_end = float(sys.argv[8])
T_batch = float(sys.argv[9])
theta_coefficient = int(sys.argv[10])

partition = sys.argv[11]
partitions = {'medium': '4-00:00:00', 'long': '20-00:00:00', 'phi': '20-00:00:00', 'allcpu':'4:00:00', 'short': '2:00:00'}
time_limit = partitions[partition]

print("Populating calculation file...")
calculation_file = "N={0},{1}M,{2}dT,n={3}".format(lattice_size, measurements, T_step, n)
path = os.path.dirname(os.path.realpath(__file__))

# runs generate.py to populate the calculation file full of batch scripts
os.system('python3.6 important_scripts/generate.py {0} {1} {2} {3} "{4}" "{5}" "{6}" {7} {8} {9} {10} {11} {12} {13}'.format(lattice_size, measurements, T_step, n, path, model, models[model], y_tilde, T_start, T_end, T_batch, theta_coefficient, partition, time_limit))

print("Simulating on the above temperature ranges: {0} Model, y~{1}, n={2}, N={3}, dT={4}, M={5}".format(model, y_tilde, n, lattice_size, T_step, measurements))
os.system('chmod +x {0}/{1}/{2}dT/{3}/run_it_all.sh'.format(path, models[model], T_step, calculation_file))
os.system('sh {0}/{1}/{2}dT/{3}/run_it_all.sh'.format(path, models[model], T_step, calculation_file))