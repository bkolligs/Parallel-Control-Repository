import os, io
print("STEP 1: Enter basic simulation parameters below...")
if input("Would you like to input the parameters manually? ") == 'yes':
    # gets the model variety
    models = {'QCD': 'QCD_Model', 'XY': 'XY_Model', 'TEE': 'TEE_Calc'}

    model = input("Enter model to simulate: ")
    while model not in models:
        print("Valid models are: ", list(models.keys()))
        model = input("Invalid model name. Enter model to simulate: ")

    if model == 'XY':
        y_tilde = 0
    else:
        y_tilde = int(input("Enter y_tilde: "))

    # gets lattice number
    valid_n = [2, 3, 4]
    n = int(input("Enter replica number: "))
    while n not in valid_n:
        n = int(input("Invalid replica number. Enter replica number: "))

    # gets the lattice size
    lattice_size = int(input("Lattice size: "))
    while lattice_size % 8 != 0:
        lattice_size = int(input("Enter a multiple of 8 as a lattice size: "))

    # gets T_step
    T_step = float(input("Enter T_step: "))

    # gets measurements
    measurements = int(input("Enter monte carlo measurements: "))

    # gets theta coefficient
    if model == 'XY':
        theta_coefficient = 0
    else:
        theta_coefficient = int(input("Enter a theta coefficient: "))

    # partition on cluster
    partitions = {'medium': '4-00:00:00', 'long': '20-00:00:00', 'phi': '20-00:00:00', 'allcpu':'4:00:00', 'short': '2:00:00'}
    partition = input("Enter partition to run simulation on: ")
    while partition not in partitions:
        print("Valid partitions are: ", list(partitions.keys()))
        partition = input("Invalid partition. Enter cluster partition: ")

    time_limit = partitions[partition]


else:
    os.system("sh parameter_run.sh")
    quit()

calculation_file = "N={0},{1}M,{2}dT,n={3}".format(lattice_size, measurements, T_step, n)
path = os.path.dirname(os.path.realpath(__file__))

print("\nSTEP 2: Ready to populate the calculation file: ", calculation_file)
T_start = float(input("Enter a starting T: "))
T_end = float(input("Enter an ending T: "))
T_batch = float(input("Enter a temperature stepsize for T = {0}-{1}: ".format(T_start, T_end)))
# runs generate.py to populate the calculation file full of batch scripts
os.system('python3.6 important_scripts/generate.py {0} {1} {2} {3} "{4}" "{5}" "{6}" {7} {8} {9} {10} {11} {12} {13}'.format(lattice_size, measurements, T_step, n, path, model, models[model], y_tilde, T_start, T_end, T_batch, theta_coefficient, partition, time_limit))

print("\nCalculation file populated!")
while input("\nSTEP 3: Is the monte carlo script up to date? ") != 'yes':
    print("Update the script and enter 'yes' to continue...")

os.system('chmod +x {0}/{1}/{2}dT/{3}/run_it_all.sh'.format(path, models[model], T_step, calculation_file))

condition = input("Would you like to run the simulation now? ")

while condition != 'yes':
    if condition == 'no':
        break

    else:
        print("Enter 'yes' to run simulation")
if condition == 'yes':
    os.system('sh {0}/{1}/{2}dT/{3}/run_it_all.sh'.format(path, models[model], T_step, calculation_file))