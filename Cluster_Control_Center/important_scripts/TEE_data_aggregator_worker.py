import os, numpy, time, math, pylab, datetime, sys, shutil
label_size = 30
pylab.rcParams['xtick.labelsize'] = label_size
pylab.rcParams['ytick.labelsize'] = label_size
pylab.rcParams['legend.fontsize'] = label_size

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

path = sys.argv[10]

model = sys.argv[11]

analyzed_output = sys.argv[12]
if not os.path.exists(analyzed_output):
	os.makedirs(analyzed_output)

data_directory = '{0}/{1}'.format(model_input, folder_name)
# master plots for bringing everything together
Master_T_plot = []
Master_E_shape_1 = []
Master_sigma_shape_1 = []
Master_E_shape_2 = []
Master_sigma_shape_2 = []
Master_E_shape_3 = []
Master_sigma_shape_3 = []

Master_E_normal = []
Master_sigma_normal = []

contents = os.listdir(data_directory)


# reorders the listdir to go from 0 to 100 temperature.
def ordering(data_name):
	name_split = data_name.split(',')
	T_min = float(name_split[1])
	T_max = float(name_split[2])
	rank = T_min + T_max
	return float(rank)


def EE_calc(T_plot, E_interest, sigma_E_interest, E_normal, sigma_normal, T_step, n):
	global theta
	count = len(T_plot)
	deltaT = T_step

	S_plot = []
	# integrand = []
	S_sigma = []
	for i in range(count):
		S_A = 0.0
		sigma_sigma_i = 0.0
		# term_integrand = deltaT * ((E_interest[i]) - (n * E_normal[i]))
		# integrand.append(term_integrand)
		#
		for j in range(0, i):
			term_j = deltaT * ((E_interest[j]) - (n * E_normal[j]))
			S_A += term_j
			# error propagation:
			sigma_sigma_j = (deltaT * deltaT * (sigma_E_interest[j]**2 + 4 * (sigma_normal[j]**2)))  # this is wrong
			sigma_sigma_i += sigma_sigma_j
		sigma_i = numpy.sqrt(sigma_sigma_i)
		S_plot.append(S_A)
		S_sigma.append(sigma_i)

	# pylab.plot(T_plot, S_plot, 'b')
	# pylab.errorbar(T_plot, S_plot, yerr=S_sigma, ecolor='r')
	# pylab.show()
	return S_plot, S_sigma


def derivative(x_plot, y_plot, yerr=None, graph='show', g_title=r"Data vs It's Derivative"):

	step = x_plot[1] - x_plot[0]
	y_prime_plot = []
	y_prime_sigma = []

	count = len(x_plot)
	for index in range(count):
		dy = 0
		dy_sigma = 0
		if (count - 1) > index > 0:
			dy = (y_plot[index + 1] - y_plot[index - 1]) / (2 * step)
			if yerr is not None:
				dy_sigma = sqrt((yerr[index + 1] / (2 * step)) ** 2 + (yerr[index - 1] / (2 * step)) ** 2)
				y_prime_sigma.append(dy_sigma)

		y_prime_plot.append(dy)
	if graph == 'show':
		pylab.plot(x_plot, y_plot, 'b', label=r"$0^{th}$ order")
		pylab.plot(x_plot, y_prime_plot, 'r--', label=r"$1^{st}$ order")
		pylab.errorbar(x_plot, y_plot, yerr=yerr, capsize=2, color='b')
		# errorbar(x_plot, y_prime_plot, yerr=y_prime_sigma, capsize=2, color='r')

		pylab.title(g_title + " for N = Various")
		x_label, y_label = g_title.split('vs')
		pylab.xlabel(x_label)
		pylab.ylabel(y_label)
		pylab.xlim(0, 2)
		pylab.legend()
		pylab.show()

	return x_plot, y_plot, y_prime_plot


def TEE_calc(Data, T_step):
	global date, n
	t1 = time.time()
	T_plot = Data[0]
	# Gathers the replica data
	E_shape_1 = Data[1]
	sigma_shape_1 = Data[2]

	# Gathers the normal data
	E_shape_2 = Data[3]
	sigma_shape_2 = Data[4]

	E_shape_3 = Data[5]
	sigma_shape_3 = Data[6]

	E_normal = Data[7]
	sigma_normal = Data[8]
	print("Calculating EE for... ")
	# calculates the Renyi Entropy
	print("Shape 1...")
	shape_1_data = EE_calc(T_plot, E_shape_1, sigma_shape_1, E_normal, sigma_normal, T_step, 2)
	print("Shape 2...")
	shape_2_data = EE_calc(T_plot, E_shape_2, sigma_shape_2, E_normal, sigma_normal, T_step, 2)
	print("Shape 3...")
	shape_3_data = EE_calc(T_plot, E_shape_3, sigma_shape_3, E_normal, sigma_normal, T_step, 2)

	S_shape_1 = shape_1_data[0]
	S_shape_1_sigma = shape_1_data[1]
	S_shape_2 = shape_2_data[0]
	S_shape_2_sigma = shape_2_data[1]
	S_shape_3 = shape_3_data[0]
	S_shape_3_sigma = shape_3_data[1]


	# Calculating TEE for each T
	print('Working on Topological Entanglement Entropy...')
	count = len(T_plot)
	TEE_plot = []
	TEE_sigma_plot = []
	for T in range(count):
		S_i = -S_shape_1[T] + 2 * S_shape_2[T] - S_shape_3[T]
		TEE_plot.append(S_i)
		sigma = numpy.sqrt(S_shape_1_sigma[T]**2 + 4*S_shape_2_sigma[T]**2 + S_shape_3_sigma[T]**2)
		TEE_sigma_plot.append(sigma)
	graph = 'no'
	if graph == 'show':
		pylab.plot(T_plot, TEE_plot, 'b')
		pylab.errorbar(T_plot, TEE_plot, yerr=TEE_sigma_plot, ecolor='r')
		pylab.title(r"Topological Entanglement Entropy for $p =$" + '{}'.format(theta))
		pylab.xlim(0, 4)
		pylab.xlabel("T", fontsize = label_size)
		pylab.ylabel("TEE", fontsize=label_size)
		pylab.show()
	# prime = derivative(T_plot, TEE_plot)[1]

	return T_plot, TEE_plot, TEE_sigma_plot


# iterates over all chunk files
for data_files in sorted(contents, key=ordering):
	print(data_files)
	prefix = data_files.split(',')[0]
	suffix = ',' + data_files.split(',')[3] + ',' + data_files.split(',')[4] + ',' + data_files.split(',')[5] + ',' + \
			 data_files.split(',')[6] + ',' + data_files.split(',')[7]
	T_max = float(data_files.split(',')[2])

	data_chunk = numpy.loadtxt('{0}/{1}'.format(data_directory, data_files))
	# Gathers temperatures and adds them to the master list
	T_plot = list(data_chunk[0])
	Master_T_plot += T_plot
	# gathers the different energies and adds them to the master list
	E_shape_1 = list(data_chunk[1])
	sigma_shape_1 = list(data_chunk[2])
	Master_E_shape_1 += E_shape_1
	Master_sigma_shape_1 += sigma_shape_1

	E_shape_2 = list(data_chunk[3])
	sigma_shape_2 = list(data_chunk[4])
	Master_E_shape_2 += E_shape_2
	Master_sigma_shape_2 += sigma_shape_2

	E_shape_3 = list(data_chunk[5])
	sigma_shape_3 = list(data_chunk[6])
	Master_E_shape_3 += E_shape_3
	Master_sigma_shape_3 += sigma_shape_3

	E_normal = list(data_chunk[7])
	sigma_normal = list(data_chunk[8])
	Master_E_normal += E_normal
	Master_sigma_normal += sigma_normal

pre_aggregate_data = numpy.array([Master_T_plot, Master_E_shape_1, Master_sigma_shape_1, Master_E_shape_2, Master_sigma_shape_2, Master_E_shape_3, Master_sigma_shape_3, Master_E_normal, Master_sigma_normal])

print("The T_max is: ", T_max)
confirmation = numpy.arange(0 + T_step, T_max, T_step)
final_filename = prefix + ', 0.0, {}'.format(T_max) + suffix
print(final_filename)
if len(confirmation) == len(Master_T_plot):
	print("The experimental and control T_plots have the same length! Calculating TEE...")

	if model == 'TEE':
		TEE_data = TEE_calc(pre_aggregate_data, T_step)

	TEE = TEE_data[1]
	TEE_sigma = TEE_data[2]
	TEE_add = numpy.array([TEE, TEE_sigma])

	aggregate_data = numpy.vstack((pre_aggregate_data, TEE_add))

	print("Saving a file...")

	numpy.savetxt(model_output + '/' + final_filename, aggregate_data,
				  header='This data was aggregated on {}'.format(datetime.datetime.today()))

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
	print("E_normal length = ", len(Master_E_normal), "\nE_replica length = ", len(Master_E_shape_1),
		  "\nE_AUB length = ", len(Master_E_shape_2))

	condition = input("Continue with file save?\n")
	if condition == 'yes' or condition == 'y' or condition == 'Yes':
		print("Calculating RMI...")
		TEE_data = TEE_calc(pre_aggregate_data, T_step)
		TEE = TEE_data[0]
		TEE_sigma = TEE_data[1]
		TEE_add = numpy.array([TEE, TEE_sigma])
		aggregate_data = numpy.vstack((pre_aggregate_data, TEE_add))

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
