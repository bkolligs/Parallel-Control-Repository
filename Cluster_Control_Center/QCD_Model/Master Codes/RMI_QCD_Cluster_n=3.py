from numpy import ones, arange, sqrt, array, savetxt, vstack, zeros
from math import exp, pi, cos
from random import random, randrange
from multiprocessing import Pool
import time, datetime, sys, os, numpy
date = datetime.date.today()

N_global = int(sys.argv[1].split(',')[0])
T_step = 0.05
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
E_measurements = 20000

y_tilde = 1
theta_coefficient = 2


# Normal XY
def QCD_E(T):
    global N_global, E_measurements, tau_after, y_tilde, theta_coefficient
    J = 1
    N = N_global  # The lattice size: NxN
    tau = tau_global  # The correlation time
    if T > 20:
        tau = tau_after
    BM = E_measurements  # Number of independent measurements for the bootstrap analysis
    steps = 2 * tau * BM  # Number of times the program will run
    E = -2 * (N * N) - y_tilde * (N * N)  # Initial Value of Energy since all spins start pointed up at \theta_i = 0.0
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
        dE += y_tilde * (cos(theta_coefficient * L[i, j]) - cos(theta_coefficient * L_update))

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


def QCD_A_U_B_3(T):
    global N_global, E_measurements, tau_after
    J = 1
    N = N_global  # The lattice size: NxN
    tau = tau_global  # The correlation time
    if T > 20:
        tau = tau_after
    BM = E_measurements  # Number of independent measurements for the bootstrap analysis
    steps = 2 * tau * BM  # Number of times the program will run
    E = -6 * (N * N) - 3 * y_tilde * (N * N)  # Initial Value of Energy since all spins start pointed up at \theta_i = 0.0
    L = zeros([N, N], float)  # Generates the lattice where each entry is a value of \theta_i

    # JT = J / T  # The parameter J divided by T (temperature) Multiplying dE by this will potentially save time,
    # but if this is done, make sure to multiply dE by T again when doing E += dE/(N*N)

    print("N=", N, "; A-union-B (n=3) XY-Model at T=", T)

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
        dE *= 3 * -J
        dE += 3 * y_tilde * (cos(theta_coefficient * L[i, j]) - cos(theta_coefficient * L_update))

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


def QCD_Replica_3_E(T):
    global N_global, E_measurements, tau_global, tau_after
    J = 1
    N = N_global  # The lattice size: NxN
    # A test to make things quicker; higher temperatures equilibrate faster
    tau = tau_global
    if N == 16:
        if 0 < T < 2.0:
            tau = 40000
        if 2.0 <= T < 3.0:
            tau = 13000
        if 3.0 <= T < 4.0:
            tau = 6200
        if 4.0 <= T < 5.0:
            tau = 5000
        if 5.0 <= T < 20:
            tau = 4300
        if T >= 20:
            tau = 1600
    if N == 24:
        if 0 < T < 2.0:
            tau = 100000
        if 2.0 <= T < 3.0:
            tau = 30000
        if 3.0 <= T < 4.0:
            tau = 15000
        if 4.0 <= T < 5.0:
            tau = 10000
        if 5.0 <= T < 20:
            tau = 6200
        if T >= 20:
            tau = 3000
    if N == 32:
        if 0 < T < 2.0:
            tau = 100000
        if 2.0 <= T < 3.0:
            tau = 37000
        if 3.0 <= T < 4.0:
            tau = 20000
        if 4.0 <= T < 5.0:
            tau = 16000
        if 5.0 <= T < 20:
            tau = 16000
        if T >= 20:
            tau = tau_after

    BM = E_measurements  # Number of independent measurements for the bootstrap analysis
    steps = 2 * tau * BM  # Number of times the program will run
    E = -6 * (N * N) - 3 * y_tilde * (N * N)  # Initial Value of Energy since all spins start pointed up at \theta_i = 0.0
    boundary = N // 2
    L1 = zeros([N, N], float)  # Lattice 1 where each entry is a value of \theta_i
    L2 = zeros([N, N], float)  # Lattice 2
    L3 = zeros([N, N], float) # lattice 3
    A_1 = L1[:, 0:boundary]
    A_2 = L2[:, 0:boundary]
    A_3 = L3[:, 0:boundary]
    B_1 = L1[:, boundary: N]
    B_2 = L2[:, boundary: N]
    B_3 = L3[:, boundary: N]

    print("N=", N, "; Replica (n=3) XY-Model at T=", T)

    expE = 0.0  # Expectation value of E
    measurements = []  # List of Measurements
    # Main Monte Carlo cycle
    for x in range(steps + 1):
        i = randrange(0, N)
        j = randrange(0, N)  # Picks a random starting location

        # Decides an anticipated spin amount
        L1_update = random() * 2 * pi
        L2_update = random() * 2 * pi
        L3_update = random() * 2 * pi

        if j < boundary:
            L1_update = L2_update = L3_update
        # Calculates change in energy that would occur if this spin was accepted
        dE = 0.0
        # Starts calculating the nearest neighbor sum at location L[ i-1 , j]
        neighbor = i - 1
        if neighbor > -1:
            dE += cos(L1_update - L1[neighbor, j]) - cos(L1[i, j] - L1[neighbor, j])
            dE += cos(L2_update - L2[neighbor, j]) - cos(L2[i, j] - L2[neighbor, j])
            dE += cos(L3_update - L3[neighbor, j]) - cos(L3[i, j] - L3[neighbor, j])

        else:
            dE += cos(L1_update - L1[N - 1, j]) - cos(L1[i, j] - L1[N - 1, j])  # Periodic boundary conditions
            dE += cos(L2_update - L2[N - 1, j]) - cos(L2[i, j] - L2[N - 1, j])
            dE += cos(L3_update - L3[N - 1, j]) - cos(L3[i, j] - L3[N - 1, j])
        neighbor = i + 1
        if neighbor < N:
            dE += cos(L1_update - L1[neighbor, j]) - cos(L1[i, j] - L1[neighbor, j])
            dE += cos(L2_update - L2[neighbor, j]) - cos(L2[i, j] - L2[neighbor, j])
            dE += cos(L3_update - L3[neighbor, j]) - cos(L3[i, j] - L3[neighbor, j])

        else:
            dE += cos(L1_update - L1[0, j]) - cos(L1[i, j] - L1[0, j])
            dE += cos(L2_update - L2[0, j]) - cos(L2[i, j] - L2[0, j])
            dE += cos(L3_update - L3[0, j]) - cos(L3[i, j] - L3[0, j])
        neighbor = j - 1
        if neighbor > -1:
            dE += cos(L1_update - L1[i, neighbor]) - cos(L1[i, j] - L1[i, neighbor])
            dE += cos(L2_update - L2[i, neighbor]) - cos(L2[i, j] - L2[i, neighbor])
            dE += cos(L3_update - L3[i, neighbor]) - cos(L3[i, j] - L3[i, neighbor])

        else:
            dE += cos(L1_update - L1[i, N - 1]) - cos(L1[i, j] - L1[i, N - 1])
            dE += cos(L2_update - L2[i, N - 1]) - cos(L2[i, j] - L2[i, N - 1])
            dE += cos(L3_update - L3[i, N - 1]) - cos(L3[i, j] - L3[i, N - 1])

        neighbor = j + 1
        if neighbor < N:
            dE += cos(L1_update - L1[i, neighbor]) - cos(L1[i, j] - L1[i, neighbor])
            dE += cos(L2_update - L2[i, neighbor]) - cos(L2[i, j] - L2[i, neighbor])
            dE += cos(L3_update - L3[i, neighbor]) - cos(L3[i, j] - L3[i, neighbor])

        else:
            dE += cos(L1_update - L1[i, 0]) - cos(L1[i, j] - L1[i, 0])
            dE += cos(L2_update - L2[i, 0]) - cos(L2[i, j] - L2[i, 0])
            dE += cos(L3_update - L3[i, 0]) - cos(L3[i, j] - L3[i, 0])

        dE *= -J
        dE += y_tilde * (cos(theta_coefficient * L1[i, j]) - cos(theta_coefficient * L1_update)) + y_tilde * (cos(theta_coefficient * L2[i, j]) - cos(theta_coefficient * L2_update)) + y_tilde * (cos(theta_coefficient * L3[i, j]) - cos(theta_coefficient * L3_update))
        # Calculates whether L[i,j] rotates
        R = exp(-dE / T)
        if R > 1 or random() < R:
            L1[i, j] = L1_update
            L2[i, j] = L2_update
            L3[i, j] = L3_update
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
                if A_1[rows, columns] == A_2[rows, columns] == A_3[rows, columns]:
                    matches += 1
        if matches == N * boundary:
            print("A_1 and A_2 are the same!")
        else:
            print("We messed up somewhere :(")

    return [T, expE, sigma_bootstrap]  # This will create a results matrix which can be plotted


def float_array(T_min, T_max, T_step, check_num='no'):
    number = int(round((T_max - T_min), 4) / T_step)
    if check_num == 'yes':
        print("Going to try this as num: ", number)
    array = numpy.linspace(T_min, T_max, number, endpoint=False)
    return array


def vary_temps_RMI(T_min, T_max, T_step):
    if T_min == 0:
        temps = float_array(T_min + T_step, T_max, T_step)
    else:
        temps = float_array(T_min, T_max, T_step)

    # I have to separate the core mapping to prevent a memory error
    cores = Pool()
    result1 = cores.map(QCD_Replica_3_E, temps)
    cores.close()
    cores.join()

    cores = Pool()
    result2 = cores.map(QCD_A_U_B_3, temps)
    cores.close()
    cores.join()

    cores = Pool()
    result3 = cores.map(QCD_E, temps)
    cores.close()
    cores.join()

    replica_3 = array(result1)
    A_U_B_3 = array(result2)
    normal = array(result3)

    # Both Ising models are at the same temperature so,
    T_plot = normal[:, 0]  # Takes the first column of the results matrix
    #
    E_normal = normal[:, 1]
    sigma_normal = normal[:, 2]

    E_replica_3 = replica_3[:, 1]
    sigma_replica_3 = replica_3[:, 2]

    E_A_U_B_3 = A_U_B_3[:, 1]
    sigma_A_U_B_3 = A_U_B_3[:, 2]

    return T_plot, E_replica_3, sigma_replica_3, E_A_U_B_3, sigma_A_U_B_3, E_normal, sigma_normal


def RMI_calc(T_min, T_max, T_step, save_data='no'):
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

    if save_data == 'yes':
        Data = array(Data)
        t_elapse = (time.time() - t1) / 3600
        folder_name = 'Data from RMI QCD; {0}; {1}, {2}, {3}, n=2'.format(date, E_measurements, T_step, N_global)
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)
        savetxt('{6}/RMI XY; {0}; {1}, {2}, {3}, {4}, {5}, n=3.txt'.format(date, E_measurements, T_min, T_max, T_step, N_global, folder_name), Data, header='This data took {0:.3f} hours and was recorded on {1}. This was run on the PSU Coeus Cluster.'.format(t_elapse, datetime.datetime.today()))
        print("This took {0:.3f} hours.".format(t_elapse))

    return T_plot


if __name__ == '__main__':
    t_start = time.time()

    # Main Program

    T_min = float(sys.argv[2].split(',')[0])
    T_max = float(sys.argv[3].split(',')[0])

    RMI_calc(T_min, T_max, T_step, save_data='yes')

    # End of Main Program

    t_elapse = (time.time() - t_start) / 3600
    print("Full Program done in {0:.3f} hours".format(t_elapse))

