import os, numpy, time, math, pylab, datetime, sys, shutil
# parameters for calculating RMI
lattice_size = int(sys.argv[1])
T_step = float(sys.argv[2])
n = int(sys.argv[3])
y_tilde = float(sys.argv[4])
theta_coefficient = int(sys.argv[5])
# This finds the correct files to aggregate
date_of_data = sys.argv[6]
model_input = sys.argv[7]
model_output = sys.argv[8]
folder_name = sys.argv[9]

path = sys.argv[10] # path of Control_Center

model = sys.argv[11]

analyzed_output = sys.argv[12]
if not os.path.exists(analyzed_output):
	os.makedirs(analyzed_output)

data_directory = '{0}/{1}'.format(model_input, folder_name)


# master plots for bringing everything together
Master_T_plot = []
Master_E_replica = []
Master_sigma_replica = []
Master_E_A_U_B = []
Master_sigma_A_U_B = []
Master_E_normal = []
Master_sigma_normal = []

magnetization = 'no'
Master_mag = []
Master_sigma_mag = []


contents = os.listdir(data_directory)

# reorders the listdir to go from 0 to 100 temperature.
def ordering(data_name):
	name_split = data_name.split(',')
	T_min = float(name_split[1])
	T_max = float(name_split[2])
	rank = T_min + T_max
	return float(rank)


# calculates RMI
def RMI_calc(Data, N_global, T_step, graph='no'):
	global date, n
	alpha = n
	t1 = time.time()
	T_plot = Data[0]
	# Gathers the replica data
	E_replica = Data[1]
	sigma_replica = Data[2]
	# Gathers the normal data
	E_A_U_B = Data[3]
	sigma_A_U_B = Data[4]

	E_normal = Data[5]
	sigma_normal = Data[6]
	# Calculating RMI for each T
	print('Working on Renyi Mutual Information...')
	count = len(E_A_U_B)

	RMI_plot = []
	RMI_sigma_plot = []
	deltaT = T_step
	# Calculates the RMI and the sigma for each RMI(T)
	for i in range(count):
		RMI = 0.0
		sigma_sigma_i = 0.0
		for j in range(i, count):
			term_j = deltaT * (2 * (E_replica[j]) - (E_A_U_B[j]) - alpha * E_normal[j]) / ((T_plot[j]) ** 2)
			RMI += term_j
			# Now to propagate the error from each E measurement...
			sigma_sigma_j = ((2 * deltaT) / ((T_plot[j] ** 2) * N_global * 2)) ** 2 * (sigma_replica[j] ** 2) + (deltaT / ((
				T_plot[j] ** 2) * N_global * 2)) ** 2 * (sigma_A_U_B[j] ** 2) + ((2 * deltaT) / ((T_plot[j] ** 2) * N_global * 2)) ** 2 * (sigma_normal[j] ** 2)
			sigma_sigma_i += sigma_sigma_j
		sigma_i = math.sqrt(sigma_sigma_i)
		RMI /= 2 * N_global
		RMI_plot.append(RMI)
		RMI_sigma_plot.append(sigma_i)
		if i % 100 == 0:
			print('Calculating RMI for T =', i * T_step)

	if graph == 'yes' or graph == 'plot':
		pylab.plot(T_plot, RMI_plot, 'b', linewidth=3)
		pylab.errorbar(T_plot, RMI_plot, yerr=RMI_sigma_plot, capsize=2, ecolor='r')
		pylab.title(r'RMI vs $T$; $T_{step}$' + ' = {0};'.format(T_step) + ' $T_{max}$' + ' = 100 ', fontsize=16)
		pylab.xlabel(r'$T$', fontsize=16)
		pylab.ylabel(r'$\frac{I_2(T)}{\ell}$', fontsize=16)
		pylab.xlim(0, 10)
		# pylab.ylim(0, 0.5)
		t_elapse = (time.time() - t1) / 60
		print('Done in {0:.3f} minutes'.format(t_elapse))
		if graph == 'plot':
			pylab.show()


	return T_plot, RMI_plot, RMI_sigma_plot


def RMI_calc_QCD(Data, N_global, T_step, graph='no'):
	global date, n
	alpha = n
	t1 = time.time()
	T_plot = Data[0]
	# Gathers the replica data
	E_replica = Data[1]
	sigma_replica = Data[2]
	# Gathers the normal data
	E_A_U_B = Data[3]
	sigma_A_U_B = Data[4]

	E_normal = Data[5]
	sigma_normal = Data[6]
	# Calculating RMI for each T
	print('Working on Renyi Mutual Information...')
	count = len(E_A_U_B)

	RMI_plot = []
	RMI_sigma_plot = []
	deltaT = T_step
	# Calculates the RMI and the sigma for each RMI(T)
	for i in range(count):
		RMI = 0.0
		sigma_sigma_i = 0.0
		for j in range(0, i):
			term_j = deltaT * (2 * (E_replica[j]) - (E_A_U_B[j]) - alpha * E_normal[j])
			RMI += term_j
			# Now to propagate the error from each E measurement...
			sigma_sigma_j = ((2 * deltaT) / (N_global * 2)) ** 2 * (sigma_replica[j] ** 2) + (deltaT / ( N_global * 2)) ** 2 * (sigma_A_U_B[j] ** 2) + ((alpha * deltaT) / ( N_global * 2)) ** 2 * (sigma_normal[j] ** 2)
			sigma_sigma_i += sigma_sigma_j
		sigma_i = math.sqrt(sigma_sigma_i)
		RMI /= 2 * N_global
		RMI_plot.append(RMI)
		RMI_sigma_plot.append(sigma_i)
		if i % 100 == 0:
			print('Calculating RMI for T =', i * T_step)

	if graph == 'yes' or graph == 'plot':
		pylab.plot(T_plot, RMI_plot, 'b', linewidth=3)
		pylab.errorbar(T_plot, RMI_plot, yerr=RMI_sigma_plot, capsize=2, ecolor='r')
		pylab.title(r'RMI vs $T$; $T_{step}$' + ' = {0};'.format(T_step) + ' $T_{max}$' + ' = 100 ', fontsize=16)
		pylab.xlabel(r'$T$', fontsize=16)
		pylab.ylabel(r'$\frac{I_2(T)}{\ell}$', fontsize=16)
		pylab.xlim(0, 10)
		# pylab.ylim(0, 0.5)
		t_elapse = (time.time() - t1) / 60
		print('Done in {0:.3f} minutes'.format(t_elapse))
		if graph == 'plot':
			pylab.show()

	return T_plot, RMI_plot, RMI_sigma_plot


# iterates over all chunk files
for data_files in sorted(contents, key=ordering):
	print(data_files)
	prefix = data_files.split(',')[0]
	suffix = ',' + data_files.split(',')[3] + ',' + data_files.split(',')[4] + ',' + data_files.split(',')[5]+ data_files.split(',')[6]+ data_files.split(',')[7]
	T_max = float(data_files.split(',')[2])
	data_chunk = numpy.loadtxt('{0}/{1}'.format(data_directory, data_files))
	# Gathers temperatures and adds them to the master list
	T_plot = list(data_chunk[0])
	Master_T_plot += T_plot
	# gathers the different energies and adds them to the master list
	E_replica = list(data_chunk[1])
	sigma_replica = list(data_chunk[2])
	Master_E_replica += E_replica
	Master_sigma_replica += sigma_replica

	E_A_U_B = list(data_chunk[3])
	sigma_A_U_B = list(data_chunk[4])
	Master_E_A_U_B += E_A_U_B
	Master_sigma_A_U_B += sigma_A_U_B

	E_normal = list(data_chunk[5])
	sigma_normal = list(data_chunk[6])
	Master_E_normal += E_normal
	Master_sigma_normal += sigma_normal

	if magnetization == 'yes':
		mag = list(data_chunk[7])
		sigma_mag = list(data_chunk[8])
		Master_mag += mag
		Master_sigma_mag += sigma_mag


if magnetization == 'yes':
	print('This file contains magnetization data!')
	pre_aggregate_data = numpy.array([Master_T_plot, Master_E_replica, Master_sigma_replica, Master_E_A_U_B, Master_sigma_A_U_B, Master_E_normal, Master_sigma_normal, Master_mag, Master_sigma_mag], float)
else:
	print('This file has no magnetization data')

	pre_aggregate_data = numpy.array(
		[Master_T_plot, Master_E_replica, Master_sigma_replica, Master_E_A_U_B, Master_sigma_A_U_B, Master_E_normal,
		 Master_sigma_normal], float)

print("The T_max is: ", T_max)
confirmation = numpy.arange(0 + T_step, T_max, T_step)
final_filename = prefix + ',0.0, {}'.format(T_max) + suffix
if len(confirmation) == len(Master_T_plot):
	print("The experimental and control T_plots have the same length! Calculating RMI...")

	if model == 'QCD':
		RMI_data = RMI_calc_QCD(pre_aggregate_data, lattice_size, T_step)
	elif model == 'XY':
		RMI_data = RMI_calc(pre_aggregate_data, lattice_size, T_step)

	RMI = RMI_data[1]
	RMI_sigma = RMI_data[2]
	RMI_add = numpy.array([RMI, RMI_sigma])
	aggregate_data = numpy.vstack((pre_aggregate_data, RMI_add))

	print("Saving a file...")

	numpy.savetxt(model_output + '/' + final_filename, aggregate_data, header='This data was aggregated on {}'.format(datetime.datetime.today()))
	print("Saved a file: {0}/{1}".format(model_output, final_filename))

	if os.path.exists('{0}/{1}'.format(analyzed_output, folder_name)):
		if input("NOTIFICATION: This data has already been analyzed. Overwrite it? ") == 'yes': 
			shutil.move('{0}/{1}'.format(path, data_directory), '{0}/{1}'.format(analyzed_output, folder_name))
			print("TASK COMPLETED: Moved '{0}'' to '{1}', and overwrote previous file.".format(data_directory, analyzed_output))
	else: 
		shutil.move('{0}/{1}'.format(path, data_directory), '{0}/{1}'.format(analyzed_output, folder_name))
		print('TASK COMPLETED: Moved {0} to {1}'.format(data_directory, analyzed_output))

	
else:
	print("\nThe experimental and control T_plots are different lengths! File save suspended!")
	print("It should be {0} but it is {1} instead.".format(len(confirmation), len(Master_T_plot)))
	print("E_normal length = ", len(Master_E_normal), "\nE_replica length = ", len(Master_E_replica), "\nE_AUB length = ", len(Master_E_A_U_B))

	condition = input("Continue with file save?\n")
	if condition == 'yes' or condition == 'y' or condition == 'Yes':
		print("Calculating RMI...")
		RMI_data = RMI_calc(pre_aggregate_data, lattice_size, T_step)
		RMI = RMI_data[1]
		RMI_sigma = RMI_data[2]
		RMI_add = numpy.array([RMI, RMI_sigma])
		aggregate_data = numpy.vstack((pre_aggregate_data, RMI_add))

		print("Saving a file...")

		numpy.savetxt(path + model_output + '/' + final_filename, aggregate_data,
					  header='This data was aggregated on {}'.format(datetime.datetime.today()))
		print("Saved a file: {0}/{1}".format(model_output, final_filename))

		if os.path.exists('{0}/{1}'.format(analyzed_output, folder_name)):
			if input("NOTIFICATION: This data has already been analyzed. Overwrite it? ") == 'yes': 
				shutil.move('{0}/{1}'.format(path, data_directory), '{0}/{1}'.format(analyzed_output, folder_name))
				print("TASK COMPLETED: Moved '{0}'' to '{1}', and overwrote previous file.".format(data_directory, analyzed_output))
		else: 
			shutil.move('{0}/{1}'.format(path, data_directory), '{0}/{1}'.format(analyzed_output, folder_name))
			print('TASK COMPLETED: Moved {0} to {1}'.format(data_directory, analyzed_output))
