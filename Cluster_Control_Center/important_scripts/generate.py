import os, io, numpy, sys, shutil
# T_min is the starting T, T_cutoff is the ending T. step_size is the RMI temp step size, and chunk is how many temperatures a singular call of RMI_Cluster.py should calculate.
N = int(sys.argv[1])
measurements = int(sys.argv[2])
T_size = float(sys.argv[3])
n = int(sys.argv[4])
path = sys.argv[5]
model = sys.argv[6]
model_folder = sys.argv[7]
y_tilde = float(sys.argv[8])
theta_coefficient = int(sys.argv[12])
partition = sys.argv[13]
time_limit = sys.argv[14]

# The Script name format depending on the model
scripts = {'XY': 'RMI_{0}_Cluster_n={1}.py'.format(model, n), 'QCD': 'RMI_{0}_Cluster_n={1}.py'.format(model, n), 'TEE': 'Multiple_Shapes.py'}

packed_folder = 'N={0},{1}M,{2}dT,n={3}'.format(N, measurements, T_size, n)
whole_path = '{0}/{1}/{2}dT/{3}'.format(path, model_folder, T_size , packed_folder)
folder_path = '{0}/{1}/{2}dT/{3}'.format(path, model_folder, T_size , packed_folder)

if not os.path.exists(folder_path):
    os.makedirs(folder_path)
else:
    shutil.rmtree(folder_path)           #removes all the subdirectories!
    os.makedirs(folder_path)


def float_array(T_min, T_max, T_step, check_num='yes'):
    number = int(round((T_max - T_min), 4) / T_step)
    if check_num == 'yes':
        print("Going to try this as num: ", number)
    array = numpy.linspace(T_min, T_max, number, endpoint=False)
    return array


def temperature_spreader(T_min, T_cutoff, step_size):
    range_list = []
    for T in float_array(T_min, T_cutoff, step_size, check_num='no'):
        if T == list(float_array(T_min, T_cutoff, step_size, check_num='no'))[-1]:
            range_list.append((T, T_cutoff))
        else:
            range_list.append((T, T + step_size))
    return range_list

T_start = float(sys.argv[9])
T_end = float(sys.argv[10])
T_batch = float(sys.argv[11])

temp_list = temperature_spreader(T_start, T_end, T_batch)

print("T_min, T_max pairs: \n", numpy.array(temp_list))

script_number = 1
for temp_pair in temp_list:
    with io.open('{0}/{4}dT/{1}/{2}{3}_submit.sh'.format(model_folder, packed_folder, model, script_number, T_size), 'w', newline='\n') as batchscript:
        write_T_min = round(temp_pair[0], 1)
        write_T_max = round(temp_pair[1], 1)
        # This generates the batch script!
        batchscript.write(
            "#!/bin/bash \n#SBATCH --job-name=N,n={0},{1}_{2}to{3} \n#SBATCH --output={9}/RMI_{4}{2}to{3}.txt \n#SBATCH --nodes 1 \n#SBATCH --cpus-per-task=20 \n#SBATCH --ntasks-per-node 1	 \n#SBATCH --time={11} \n#SBATCH --mem-per-cpu=MaxMemPerCPU \n#SBATCH --partition {10} \nsrun python3.6 {13}/{5}/Master\ Codes/{14} {0} {6} {7} {2} {3} {8} {12} {13}".format(N, n, write_T_min, write_T_max, model, model_folder, T_size, measurements, y_tilde, whole_path, partition, time_limit, theta_coefficient, path, scripts[model]))
    script_number += 1

# creates a script that allows me to run all the scripts
with io.open('{0}/{1}dT/{2}/run_it_all.sh'.format(model_folder, T_size, packed_folder), 'w', newline='\n') as run_file:
    run_file.write('#!/bin/bash\n')
    for script_number in range(1, len(temp_list) + 1):
        run_file.write('sbatch {2}/{0}{1}_submit.sh \nsleep 0.01\n'.format(model, script_number, whole_path))

# creates a script that allows me to cancel all the scripts
with io.open('{0}/{1}dT/{2}/emergency_halt.sh'.format(model_folder, T_size, packed_folder), 'w', newline='\n') as run_file:
    run_file.write('#!/bin/bash\n')
    for temp_pair in (temp_list):
        T_min = round(temp_pair[0], 1)
        T_max = round(temp_pair[1], 1)
        run_file.write('scancel N,n={0},{1}_{2}to{3} \nsleep 0.01\n'.format(N, n, T_min, T_max))

# Summary of task
print("\nCreated a folder named \n'{0}/{1}/{2}' \nwith {3} 'sbatch' scripts inside.".format(model_folder, T_size, packed_folder, script_number))
