#!/bin/bash
# FOR REFERENCE: 
# model = sys.argv[1]
# y_tilde = sys.argv[2]
# n = sys.argv[3]
# lattice_size = sys.argv[4]
# T_step = sys.argv[5]
# measurements = sys.argv[6]
# T_start = sys.argv[7]
# T_end = sys.argv[8]
# T_batch = sys.argv[9]
# theta_coefficient = int(sys.argv[10])
# partition = sys.argv[11]

module load Python/3.6.2/intel

python3.6 control_center_parameters.py TEE 1 2 8 0.05 2 0 4 1 2 'allcpu'
python3.6 control_center_parameters.py TEE 1 2 16 0.05 2 0 4 1 2 'allcpu'
python3.6 control_center_parameters.py TEE 1 2 24 0.05 2 0 4 1 2 'allcpu'
python3.6 control_center_parameters.py TEE 1 2 32 0.05 2 0 4 1 2 'allcpu'
python3.6 control_center_parameters.py TEE 1 2 40 0.05 2 0 4 1 2 'allcpu'
python3.6 control_center_parameters.py TEE 1 2 48 0.05 2 0 4 1 2 'allcpu'
python3.6 control_center_parameters.py TEE 1 2 56 0.05 2 0 4 1 2 'allcpu'
