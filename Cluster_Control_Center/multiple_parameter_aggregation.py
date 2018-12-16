import os, numpy, time, math, pylab, datetime, sys, shutil

models = {'QCD': 'QCD_Model', 'XY': 'XY_Model', 'TEE': 'TEE_Calc'}
model = input("What model data do you want to aggregate? ")
while model not in models:
    print("Valid models are: ", list(models.keys()))
    model = input("Invalid model name. Enter model to simulate: ")

data_directory = 'Aggregated on {0}'.format(datetime.date.today())  # name of folder to sift through
model_input = '{0}/Finished_Data'.format(models[model])  # folder where the data folders are
model_output = 'FINISHED DATA/{0} Aggregate Data'.format(models[model])  # folder where the aggregate data will go
if not os.path.exists(model_input):
    print("NOTIFICATION: No simulations have been ran for this model yet! Run a simulation and restart 'end_simulation'")
    quit()
path = os.path.dirname(os.path.realpath(__file__))
whole_path = '{0}/{1}'.format(path, model_output)
analyzed_output = '{0}/{1}/Analyzed Data'.format(path, models[model])

if not os.path.exists(whole_path):
    os.makedirs(whole_path)

contents = os.listdir(model_input)
if len(contents)==0:
    print("NOTIFICATION: All data has been analyzed. Run another simulation to produce data...")
# goes through each file in the data_directory and aggregates the data for each
for folder_name in contents:
    deconstruction = folder_name.split(',')
    prefix = deconstruction[0].split(';')[0]
    date_of_data = deconstruction[0].split(';')[1]
    measurements = int(deconstruction[0].split(';')[2])
    T_step = float(deconstruction[1])
    lattice_size = int(deconstruction[2])
    n = int(deconstruction[3].split('=')[1])
    y_tilde = float(deconstruction[4].split('~')[1])
    theta_coefficient = int(deconstruction[5].split('=')[1])

    print('\n' + folder_name +'\n')
    if model == 'QCD' or model == 'XY':
        os.system('python important_scripts/RMI_data_aggregator_worker.py {0} {1} {2} {3} {4} "{5}" "{6}" "{7}" "{8}" "{9}" "{10}" "{11}"'.format(lattice_size, T_step, n, y_tilde, theta_coefficient, date_of_data, model_input, model_output, folder_name, path, model, analyzed_output))
    if model == 'TEE':
        os.system('python important_scripts/TEE_data_aggregator_worker.py {0} {1} {2} {3} {4} "{5}" "{6}" "{7}" "{8}" "{9}" "{10}" "{11}"'.format(lattice_size, T_step, n, y_tilde, theta_coefficient, date_of_data, model_input, model_output, folder_name, path, model, analyzed_output))