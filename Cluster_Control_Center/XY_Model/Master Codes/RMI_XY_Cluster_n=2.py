from numpy import ones, arange, sqrt, array, savetxt, vstack, zeros
from math import exp, pi, cos
from random import random, randrange
from multiprocessing import Pool
import time, sys, os, datetime, numpy
date = datetime.date.today()


N_global = int(sys.argv[1].split(',')[0])
T_step = float(sys.argv[2].split(',')[0])
output_path = sys.argv[8].split(',')[0]

if N_global == 16:
    tau_global = 10240
    tau_after = 1000
if N_global == 24:
    tau_global = 14000
    tau_after = 1200
if N_global == 32:
    tau_global = 21000
    tau_after = 4000
if N_global == 64:
    tau_global = 55000
    tau_after = 10000

E_measurements = int(sys.argv[3].split(',')[0])
print("Using {} Measurements".format(E_measurements))

# This model doesn't use these two parameters, but it makes the control center function smoother. 
y_tilde = float(sys.argv[6].split(',')[0])
theta_coefficient = int(sys.argv[7].split(',')[0])


# Normal XY
def XY_E(T):
    global N_global, E_measurements, tau_after
    J = 1
    N = N_global  # The lattice size: NxN
    tau = tau_global  # The correlation time
    if T > 20:
        tau = tau_after
    BM = E_measurements  # Number of independent measurements for the bootstrap analysis
    steps = 2 * tau * BM  # Number of times the program will run
    E = -2 * (N * N)  # Initial Value of Energy since all spins start pointed up at \theta_i = 0.0
    L = zeros([N, N], float)  # Generates the lattice where each entry is a value of \theta_i

    # JT = J / T  # The parameter J divided by T (temperature) Multiplying dE by this will potentially save time,
    # but if this is done, make sure to multiply dE by T again when doing E += dE/(N*N)

    print("N=", N, "; Normal XY-Model at T=", T)

    expE = 0.0  # Expectation value of E
    measurements = []  # List of Measurements
    # Main Monte Carlo cycle
    for x in range(steps + 1):
        i = randrange(0, N)
        j = randrange(0, N)  # Picks a random starting location

        # Decides an anticipated spin amount
        L_update = random() * 2 * pi
        # Calculates change in energy that would occur if this spin was accepted
        dE = 0.0
        # Starts calculating the nearest neighbor sum at location L[ i-1 , j]
        neighbor = i - 1
        if neighbor > -1:
            dE += cos(L_update - L[neighbor, j]) - cos(L[i, j] - L[neighbor, j])  # Checks if the neighbor is within the lattice
        else:
            dE += cos(L_update - L[N - 1, j]) - cos(L[i, j] - L[N - 1, j])  # Periodic boundary conditions
        neighbor = i + 1
        if neighbor < N:
            dE += cos(L_update - L[neighbor, j]) - cos(L[i, j] - L[neighbor, j])
        else:
            dE += cos(L_update - L[0, j]) - cos(L[i, j] - L[0, j])
        neighbor = j - 1
        if neighbor > -1:
            dE += cos(L_update - L[i, neighbor]) - cos(L[i, j] - L[i, neighbor])
        else:
            dE += cos(L_update - L[i, N - 1]) - cos(L[i, j] - L[i, N - 1])
        neighbor = j + 1
        if neighbor < N:
            dE += cos(L_update - L[i, neighbor]) - cos(L[i, j] - L[i, neighbor])
        else:
            dE += cos(L_update - L[i, 0]) - cos(L[i, j] - L[i, 0])
        dE *= -J
        # Calculates whether L[i,j] rotates
        R = exp(-dE / T)
        if R > 1 or random() < R:
            L[i, j] = L_update
            E += dE  # / (N * N)

        if x != 0 and x % (2 * tau) == 0:
            expE += E
            # print("at x = ",x,"  ", expE)
            measurements.append(E)  # Adds the measurement to the list
    expE /= BM
    # print(expE)

    # The Bootstrap Error Analysis
    resample = BM  # times to repeat re-sampling
    B_i = []  # for the calculation of <B> and sigma
    for y in range(resample):
        B = 0.0
        for z in range(int(BM)):
            n = randrange(0, BM)
            B += measurements[n]
        B /= BM
        B_i.append(B)

    # Now to calculate the Bootstrap sigma
    sigma_sigma = 0.0
    for w in range(resample):
        sigma_sigma += (B_i[w] - expE) ** 2
    sigma_sigma /= resample
    sigma_bootstrap = sqrt(sigma_sigma)

    return [T, expE, sigma_bootstrap]  # This will create a results matrix which can be plotted


def XY_A_U_B(T):
    global N_global, E_measurements, tau_after
    J = 1
    N = N_global  # The lattice size: NxN
    tau = tau_global  # The correlation time
    if T > 20:
        tau = tau_after
    BM = E_measurements  # Number of independent measurements for the bootstrap analysis
    steps = 2 * tau * BM  # Number of times the program will run
    E = -4 * (N * N)  # Initial Value of Energy since all spins start pointed up at \theta_i = 0.0
    L = zeros([N, N], float)  # Generates the lattice where each entry is a value of \theta_i

    # JT = J / T  # The parameter J divided by T (temperature) Multiplying dE by this will potentially save time,
    # but if this is done, make sure to multiply dE by T again when doing E += dE/(N*N)

    print("N=", N, "; A-union-B XY-Model at T=", T)

    expE = 0.0  # Expectation value of E
    measurements = []  # List of Measurements
    # Main Monte Carlo cycle
    for x in range(steps + 1):
        i = randrange(0, N)
        j = randrange(0, N)  # Picks a random starting location

        # Decides an anticipated spin amount
        L_update = random() * 2 * pi
        # Calculates change in energy that would occur if this spin was accepted
        dE = 0.0
        # Starts calculating the nearest neighbor sum at location L[ i-1 , j]
        neighbor = i - 1
        if neighbor > -1:
            dE += cos(L_update - L[neighbor, j]) - cos(
                L[i, j] - L[neighbor, j])  # Checks if the neighbor is within the lattice
        else:
            dE += cos(L_update - L[N - 1, j]) - cos(L[i, j] - L[N - 1, j])  # Periodic boundary conditions
        neighbor = i + 1
        if neighbor < N:
            dE += cos(L_update - L[neighbor, j]) - cos(L[i, j] - L[neighbor, j])
        else:
            dE += cos(L_update - L[0, j]) - cos(L[i, j] - L[0, j])
        neighbor = j - 1
        if neighbor > -1:
            dE += cos(L_update - L[i, neighbor]) - cos(L[i, j] - L[i, neighbor])
        else:
            dE += cos(L_update - L[i, N - 1]) - cos(L[i, j] - L[i, N - 1])
        neighbor = j + 1
        if neighbor < N:
            dE += cos(L_update - L[i, neighbor]) - cos(L[i, j] - L[i, neighbor])
        else:
            dE += cos(L_update - L[i, 0]) - cos(L[i, j] - L[i, 0])
        dE *= 2 * -J
        # Calculates whether L[i,j] rotates
        R = exp(-dE / T)
        if R > 1 or random() < R:
            L[i, j] = L_update
            E += dE  # / (N * N)

        if x != 0 and x % (2 * tau) == 0:
            expE += E
            # print("at x = ",x,"  ", expE)
            measurements.append(E)  # Adds the measurement to the list
    expE /= BM
    # print(expE)

    # The Bootstrap Error Analysis
    resample = BM  # times to repeat re-sampling
    B_i = []  # for the calculation of <B> and sigma
    for y in range(resample):
        B = 0.0
        for z in range(int(BM)):
            n = randrange(0, BM)
            B += measurements[n]
        B /= BM
        B_i.append(B)

    # Now to calculate the Bootstrap sigma
    sigma_sigma = 0.0
    for w in range(resample):
        sigma_sigma += (B_i[w] - expE) ** 2
    sigma_sigma /= resample
    sigma_bootstrap = sqrt(sigma_sigma)

    return [T, expE, sigma_bootstrap]  # This will create a results matrix which can be plotted


def XY_Replica_E(T):
    global N_global, E_measurements, tau_global, tau_after
    J = 1
    N = N_global  # The lattice size: NxN
    # A test to make things quicker; higher temperatures equilibrate faster
    tau = tau_global
    if T > 20:
        tau = tau_after
    BM = E_measurements  # Number of independent measurements for the bootstrap analysis
    steps = 2 * tau * BM  # Number of times the program will run
    E = -4 * (N * N)  # Initial Value of Energy since all spins start pointed up at \theta_i = 0.0
    boundary = N // 2
    L1 = zeros([N, N], float)  # Lattice 1 where each entry is a value of \theta_i
    L2 = zeros([N, N], float)  # Lattice 2
    A_1 = L1[:, 0:boundary]
    A_2 = L2[:, 0:boundary]
    B_1 = L1[:, boundary: N]
    B_2 = L2[:, boundary: N]

    print("N=", N, "; Replica XY-Model at T=", T)

    expE = 0.0  # Expectation value of E
    measurements = []  # List of Measurements
    # Main Monte Carlo cycle
    for x in range(steps + 1):
        i = randrange(0, N)
        j = randrange(0, N)  # Picks a random starting location

        # Decides an anticipated spin amount
        L1_update = random() * 2 * pi
        L2_update = random() * 2 * pi

        if j < boundary:
            L1_update = L2_update
        # Calculates change in energy that would occur if this spin was accepted
        dE = 0.0
        # Starts calculating the nearest neighbor sum at location L[ i-1 , j]
        neighbor = i - 1
        if neighbor > -1:
            dE += cos(L1_update - L1[neighbor, j]) - cos(L1[i, j] - L1[neighbor, j])
            dE += cos(L2_update - L2[neighbor, j]) - cos(L2[i, j] - L2[neighbor, j])
        else:
            dE += cos(L1_update - L1[N - 1, j]) - cos(L1[i, j] - L1[N - 1, j])  # Periodic boundary conditions
            dE += cos(L2_update - L2[N - 1, j]) - cos(L2[i, j] - L2[N - 1, j])
        neighbor = i + 1
        if neighbor < N:
            dE += cos(L1_update - L1[neighbor, j]) - cos(L1[i, j] - L1[neighbor, j])
            dE += cos(L2_update - L2[neighbor, j]) - cos(L2[i, j] - L2[neighbor, j])
        else:
            dE += cos(L1_update - L1[0, j]) - cos(L1[i, j] - L1[0, j])
            dE += cos(L2_update - L2[0, j]) - cos(L2[i, j] - L2[0, j])
        neighbor = j - 1
        if neighbor > -1:
            dE += cos(L1_update - L1[i, neighbor]) - cos(L1[i, j] - L1[i, neighbor])
            dE += cos(L2_update - L2[i, neighbor]) - cos(L2[i, j] - L2[i, neighbor])
        else:
            dE += cos(L1_update - L1[i, N - 1]) - cos(L1[i, j] - L1[i, N - 1])
            dE += cos(L2_update - L2[i, N - 1]) - cos(L2[i, j] - L2[i, N - 1])
        neighbor = j + 1
        if neighbor < N:
            dE += cos(L1_update - L1[i, neighbor]) - cos(L1[i, j] - L1[i, neighbor])
            dE += cos(L2_update - L2[i, neighbor]) - cos(L2[i, j] - L2[i, neighbor])
        else:
            dE += cos(L1_update - L1[i, 0]) - cos(L1[i, j] - L1[i, 0])
            dE += cos(L2_update - L2[i, 0]) - cos(L2[i, j] - L2[i, 0])
        dE *= -J

        # Calculates whether L[i,j] rotates
        R = exp(-dE / T)
        if R > 1 or random() < R:
            L1[i, j] = L1_update
            L2[i, j] = L2_update
            E += dE  # / (N * N)
        if x != 0 and x % (2 * tau) == 0:
            expE += E
            # print("at x = ",x,"  ", expE)
            measurements.append(E)  # Adds the measurement to the list
    expE /= BM
    # print(expE)

    # The Bootstrap Error Analysis
    resample = BM  # times to repeat re-sampling
    B_i = []  # for the calculation of <B> and sigma
    for y in range(resample):
        B = 0.0
        for z in range(int(BM)):
            n = randrange(0, BM)
            B += measurements[n]
        B /= BM
        B_i.append(B)

    # Now to calculate the Bootstrap sigma
    sigma_sigma = 0.0
    for w in range(resample):
        sigma_sigma += (B_i[w] - expE) ** 2
    sigma_sigma /= resample
    sigma_bootstrap = sqrt(sigma_sigma)

    # This is a test to make sure that A_1 and A_2 are indeed being updated the same.
    equivalence_test = 'no'
    if equivalence_test == 'yes':
        matches = 0.0
        for columns in range(0, boundary):
            for rows in range(N):
                if A_1[rows, columns] == A_2[rows, columns]:
                    matches += 1
        if matches == N * boundary:
            print("A_1 and A_2 are the same!")
        else:
            print("We messed up somewhere :(")

    return [T, expE, sigma_bootstrap]  # This will create a results matrix which can be plotted


def vary_temps_RMI(T_min, T_max, T_step, graph='no'):
    if T_min == 0:
        temps = arange(T_min + T_step, T_max, T_step)
    else:
        temps = arange(T_min, T_max, T_step)

    # I have to separate the core mapping to prevent a memory error
    cores = Pool()
    result1 = cores.map(XY_Replica_E, temps)
    cores.close()
    cores.join()

    cores = Pool()
    result2 = cores.map(XY_A_U_B, temps)
    cores.close()
    cores.join()

    cores = Pool()
    result3 = cores.map(XY_E, temps)
    cores.close()
    cores.join()

    replica = array(result1)
    A_U_B = array(result2)
    normal = array(result3)

    # Both Ising models are at the same temperature so,
    T_plot = normal[:, 0]  # Takes the first column of the results matrix
    #
    # Replica Ising
    E_replica = replica[:, 1]  # Second column
    sigma_replica = replica[:, 2]  # Third column

    # Normal Ising
    E_A_U_B = A_U_B[:, 1]  # Second column
    sigma_A_U_B = A_U_B[:, 2]  # Third column

    E_normal = normal[:, 1]
    sigma_normal = normal[:, 2]

    if graph == 'yes':
        plot(T_plot, E_replica, 'b', label='Replica XY')
        plot(T_plot, E_A_U_B, 'r', label='A U B')
        plot(T_plot, E_normal, 'g', label='Normal XY')

        title("Energy of the Three Models", fontsize=16)
        xlabel(r"$T$", fontsize=16)
        ylabel("Energy", fontsize=16)
        xlim(0, 10)
        legend()
        show()
    return T_plot, E_replica, sigma_replica, E_A_U_B, sigma_A_U_B, E_normal, sigma_normal


def RMI_calc(T_min, T_max, T_step, save_data='no', graph='no'):
    t1 = time.time()

    Data = vary_temps_RMI(T_min, T_max, T_step)

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
            term_j = deltaT * (2 * (E_replica[j]) - (E_A_U_B[j]) - 2 * E_normal[j]) / ((T_plot[j]) ** 2)
            RMI += term_j
            # Now to propagate the error from each E measurement...
            sigma_sigma_j = ((2 * deltaT) / ((T_plot[j] ** 2) * N_global * 2)) ** 2 * (sigma_replica[j] ** 2) + (deltaT / ((
                T_plot[j] ** 2) * N_global * 2)) ** 2 * (sigma_A_U_B[j] ** 2) + ((2 * deltaT) / ((T_plot[j] ** 2) * N_global * 2)) ** 2 * (sigma_normal[j] ** 2)
            sigma_sigma_i += sigma_sigma_j
        sigma_i = sqrt(sigma_sigma_i)
        RMI /= 2 * N_global
        RMI_plot.append(RMI)
        RMI_sigma_plot.append(sigma_i)
        if i % 100 == 0:
            print('Calculating RMI for T =', i * T_step)

    if save_data == 'yes' or save_data == 'dropbox':
        RMI_data = array([RMI_plot, RMI_sigma_plot])
        Data = array(Data)
        Data_txt = vstack((Data, RMI_data))
        t_elapse = (time.time() - t1) / 3600
        folder_path = '{0}/XY_Model/Finished Data/'.format(output_path)
        folder_name = 'Data from RMI XY; {0}; {1}, {2}, {3}, n=2, y~{4}, theta={5}'.format(date, E_measurements, T_step, N_global, y_tilde, theta_coefficient)
        if not os.path.exists(folder_path + folder_name):
            os.makedirs(folder_path + folder_name)
        savetxt('{6}{7}/RMI XY;{0};{1},{2},{3},{4},{5}.txt'.format(date, E_measurements, T_min, T_max, T_step, N_global, folder_path, folder_name), Data_txt, header='This data took {0:.3f} hours and was recorded on {1}. This was run on the PSU Cluster.'.format(t_elapse, datetime.datetime.today()))


    return T_plot, RMI_plot, RMI_sigma_plot


if __name__ == '__main__':
    t_start = time.time()


    # Main Program
    T_min = float(sys.argv[4].split(',')[0])
    T_max = float(sys.argv[5].split(',')[0])

    RMI_calc(T_min, T_max, T_step, save_data='yes')

    # End of Main Program

    t_elapse = (time.time() - t_start) / 3600
    print("Full Program done in {0:.3f} hours".format(t_elapse))

