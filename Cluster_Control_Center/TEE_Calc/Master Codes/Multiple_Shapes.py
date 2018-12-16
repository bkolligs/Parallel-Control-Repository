from numpy import ones, arange, sqrt, array, savetxt, vstack, zeros
from math import exp, pi, cos, sin
from random import random, randrange
from multiprocessing import Pool
import time, sys, os, datetime, numpy
date = datetime.date.today()

N_global = int(sys.argv[1].split(',')[0])
T_step = float(sys.argv[2].split(',')[0])
output_path = sys.argv[8].split(',')[0]

if N_global == 8:
    tau_global = 10000
if N_global == 16:
    tau_global = 15000
if N_global == 24:
    tau_global = 60000
if N_global == 32:
    tau_global = 100000
if N_global == 40:
	tau_global = 150000
if N_global == 48:
    tau_global = 200000
if N_global == 56:
    tau_global = 210000

E_measurements = int(sys.argv[3].split(',')[0])
print("Using {} Measurements".format(E_measurements))

y_tilde = float(sys.argv[6].split(',')[0])
theta_coefficient = int(sys.argv[7].split(',')[0])


def region(i_start, i_end):
    return list(range(int(i_start), int(i_end)))


def QCD_E(T):
    global N_global, E_measurements, tau_after, y_tilde, theta_coefficient
    kappa = 4 * pi
    #J = - (8 * T) / (pi * kappa)
    J = 1
    N = N_global  # The lattice size: NxN
    tau = tau_global  # The correlation time
    # if T > 20:
    #     tau = tau_after
    BM = E_measurements  # Number of independent measurements for the bootstrap analysis
    steps = 2 * tau * BM  # Number of times the program will run
    E = -2 * (N * N) - y_tilde * (N * N)  # Initial Value of Energy since all spins start pointed up at \theta_i = 0.0
    m_1 = N*N # Initial value of magnetization
    m_2 = 0
    L = zeros([N, N], float)  # Generates the lattice where each entry is a value of \theta_i

    # JT = J / T  # The parameter J divided by T (temperature) Multiplying dE by this will potentially save time,
    # but if this is done, make sure to multiply dE by T again when doing E += dE/(N*N)

    print("N=", N, "; Normal QCD XY-Model at T=", T)

    expE = 0.0  # Expectation value of E
    expM = 0.0
    measurements = []  # List of Measurements
    M_measurements = []
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
        R = exp(-dE * T)
        if R > 1 or random() < R:
            m_1 = m_1 + cos(L_update) - cos(L[i, j])
            m_2 = m_2 + sin(L_update) - sin(L[i, j])
            L[i, j] = L_update
            E += dE  # / (N * N)

        if x != 0 and x % (2 * tau) == 0:
            expE += E
            M = sqrt(m_1**2 + m_2**2)
            expM += M
            # print("at x = ",x,"  ", expE)
            measurements.append(E)  # Adds the measurement to the list
            M_measurements.append(M)
    expE /= BM
    expM /= BM
    # print(expE)

    # The Bootstrap Error Analysis
    resample = BM  # times to repeat re-sampling
    B_i = []  # for the calculation of <B> and sigma
    M_i = []
    for y in range(resample):
        B = 0.0
        M_error = 0.0
        for z in range(int(BM)):
            n = randrange(0, BM)
            B += measurements[n]
            M_error += M_measurements[n]
        B /= BM
        M_error /= BM
        B_i.append(B)
        M_i.append(M_error)

    # Now to calculate the Bootstrap sigma
    sigma_sigma = 0.0
    sigma_sigma_M = 0.0
    for w in range(resample):
        sigma_sigma += (B_i[w] - expE) ** 2
        sigma_sigma_M += (M_i[w] - expM) ** 2
    sigma_sigma /= resample
    sigma_sigma_M /= resample
    sigma_bootstrap = sqrt(sigma_sigma)
    sigma_bootstrap_M = sqrt(sigma_sigma_M)

    return [T, expE, sigma_bootstrap, M, sigma_bootstrap_M]  # This will create a results matrix which can be plotted


# The parallel sections
def Region_1(T):
    global N_global, E_measurements, tau_global, tau_after
    J = 1
    N = N_global  # The lattice size: NxN
    # A test to make things quicker; higher temperatures equilibrate faster
    tau = tau_global

    BM = E_measurements  # Number of independent measurements for the bootstrap analysis
    steps = 2 * tau * BM  # Number of times the program will run
    E = -4 * (N * N) - 2 * y_tilde * (N * N) # Initial Value of Energy since all spins start pointed up at \theta_i = 0.0
    boundary = N // 2
    L1 = zeros([N, N], float)  # Lattice 1 where each entry is a value of \theta_i
    L2 = zeros([N, N], float)  # Lattice 2
    A_1 = [list(range(int(N/8), int(N/8 + .75 * N))), list(range(int(N/8), int(N/8 + N/4)))]
    A_2 = [list(range(int(N/8), int(N/8) + int(.75 * N))), list(range(int(N/8 + N/2), int(3*N/8 + N/2)))]
    B_1 = L1[:, boundary: N]
    B_2 = L2[:, boundary: N]

    lattice_test = 'no'
    if lattice_test == 'yes':
        print("PERFORMING LATTICE TEST...DO NOT COLLECT DATA!")
        for i in range(N):
            for j in range(N):
                if i in A_1[0] and j in A_1[1]:
                    L1[i, j] = 1
                    L2[i, j] = 1
                if i in A_2[0] and j in A_2[1]:
                    L1[i, j] = 2
                    L2[i, j] = 2
        print(L1)
        print(L2)

    print("N=", N, "; Region 1 at T=", T)

    expE = 0.0  # Expectation value of E
    measurements = []  # List of Measurements
    # Main Monte Carlo cycle
    for x in range(steps + 1):
        i = randrange(0, N)
        j = randrange(0, N)  # Picks a random starting location

        # Decides an anticipated spin amount
        L1_update = random() * 2 * pi
        L2_update = random() * 2 * pi

        if i in A_1[0] and j in A_1[1]:
                    L1_update = L2_update
        if i in A_2[0] and j in A_2[1]:
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
        dE += y_tilde * (cos(theta_coefficient * L1[i, j]) - cos(theta_coefficient * L1_update)) + y_tilde * (cos(theta_coefficient * L2[i, j]) - cos(theta_coefficient * L2_update))

        # Calculates whether L[i,j] rotates
        R = exp(-dE * T)
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
        for columns in range(N):
            for rows in range(N):
                if L1[rows, columns] == L2[rows, columns]:
                    matches += 1
                # if rows in A_1[0] and columns in A_1[1]:
                #     if L1[rows, columns] == L2[rows, columns]:
                #         matches += 1
                # if rows in A_2[0] and columns in A_2[1]:
                #     if L1[rows, columns] == L2[rows, columns]:
                #         matches += 1
        print(matches)
        if matches == 2 * int(.75 * N * N/4):
            print("A_1 and A_2 match!")
        else:
            print("We messed up somewhere :(")

    return [T, expE, sigma_bootstrap]  # This will create a results matrix which can be plotted


# The big U
def Region_2(T):
    global N_global, E_measurements, tau_global, tau_after
    J = 1
    N = N_global  # The lattice size: NxN
    # A test to make things quicker; higher temperatures equilibrate faster
    tau = tau_global

    BM = E_measurements  # Number of independent measurements for the bootstrap analysis
    steps = 2 * tau * BM  # Number of times the program will run
    E = -4 * (N * N) - 2 * y_tilde * (N * N) # Initial Value of Energy since all spins start pointed up at \theta_i = 0.0
    boundary = N // 2
    L1 = zeros([N, N], float)  # Lattice 1 where each entry is a value of \theta_i
    L2 = zeros([N, N], float)  # Lattice 2

    A_1 = [region(N/8, N/8 + .75 * N), region(N/8 + N/2, 3*N/8 + N/2) + region(N/8, N/8 + N/4)]
    A_2 = [region(N/8, 3*N/8), region(3*N/8, 5*N/8)]

    # A_2 = [list(range(int(N/8), int(N/8 + .75 * N))), list(range(int(N/8), int(N/8 + N/4))) + list(range(int(N/8 + N/2), int(3*N/8 + N/2)))]

    lattice_test = 'no'
    if lattice_test == 'yes':
        print("PERFORMING LATTICE TEST...DO NOT COLLECT DATA!")
        for i in range(N):
            for j in range(N):
                if i in A_1[0] and j in A_1[1]:
                    L1[i, j] = 1
                    L2[i, j] = 1
                if i in A_2[0] and j in A_2[1]:
                    L1[i, j] = 2
                    L2[i, j] = 2
        print(L1)
        print(L2)

    print("N=", N, "; Region 2 at T=", T)

    expE = 0.0  # Expectation value of E
    measurements = []  # List of Measurements
    # Main Monte Carlo cycle
    for x in range(steps + 1):
        i = randrange(0, N)
        j = randrange(0, N)  # Picks a random starting location

        # Decides an anticipated spin amount
        L1_update = random() * 2 * pi
        L2_update = random() * 2 * pi

        if i in A_1[0] and j in A_1[1]:
            L1_update = L2_update
        if i in A_2[0] and j in A_2[1]:
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
        dE += y_tilde * (cos(theta_coefficient * L1[i, j]) - cos(theta_coefficient * L1_update)) + y_tilde * (cos(theta_coefficient * L2[i, j]) - cos(theta_coefficient * L2_update))

        # Calculates whether L[i,j] rotates
        R = exp(-dE * T)
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
        for columns in range(N):
            for rows in range(N):
                if L1[rows, columns] == L2[rows, columns]:
                    matches += 1
                # if rows in A_1[0] and columns in A_1[1]:
                #     if L1[rows, columns] == L2[rows, columns]:
                #         matches += 1
                # if rows in A_2[0] and columns in A_2[1]:
                #     if L1[rows, columns] == L2[rows, columns]:
                #         matches += 1
        print(matches)
        if matches == 2 * int(.75 * N * N/4) + N*N/16:
            print("A_1 and A_2 match!")
        else:
            print("We messed up somewhere :(")

    return [T, expE, sigma_bootstrap]  # This will create a results matrix which can be plotted


# The nice box
def Region_3(T):
    global N_global, E_measurements, tau_global, tau_after
    J = 1
    N = N_global  # The lattice size: NxN
    # A test to make things quicker; higher temperatures equilibrate faster
    tau = tau_global

    BM = E_measurements  # Number of independent measurements for the bootstrap analysis
    steps = 2 * tau * BM  # Number of times the program will run
    E = -4 * (N * N) - 2 * y_tilde * (N * N) # Initial Value of Energy since all spins start pointed up at \theta_i = 0.0
    boundary = N // 2
    L1 = zeros([N, N], float)  # Lattice 1 where each entry is a value of \theta_i
    L2 = zeros([N, N], float)  # Lattice 2

    A_1 = [region(N/8, N/8 + .75 * N), region(N/8 + N/2, 3*N/8 + N/2) + region(N/8, N/8 + N/4)]
    A_2 = [region(N/8, 3*N/8) + region(N - 3*N/8, N - N/8), region(3*N/8, 5*N/8)]

    # A_2 = [list(range(int(N/8), int(N/8 + .75 * N))), list(range(int(N/8), int(N/8 + N/4))) + list(range(int(N/8 + N/2), int(3*N/8 + N/2)))]

    lattice_test = 'no'
    if lattice_test == 'yes':
        print("PERFORMING LATTICE TEST...DO NOT COLLECT DATA!")
        for i in range(N):
            for j in range(N):
                if i in A_1[0] and j in A_1[1]:
                    L1[i, j] = 1
                    L2[i, j] = 1
                if i in A_2[0] and j in A_2[1]:
                    L1[i, j] = 2
                    L2[i, j] = 2
        print(L1)
        print(L2)
        print(A_2)

    print("N=", N, "; Replica Region 3 at T=", T)

    expE = 0.0  # Expectation value of E
    measurements = []  # List of Measurements
    # Main Monte Carlo cycle
    for x in range(steps + 1):
        i = randrange(0, N)
        j = randrange(0, N)  # Picks a random starting location

        # Decides an anticipated spin amount
        L1_update = random() * 2 * pi
        L2_update = random() * 2 * pi

        if i in A_1[0] and j in A_1[1]:
            L1_update = L2_update
        if i in A_2[0] and j in A_2[1]:
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
        dE += y_tilde * (cos(theta_coefficient * L1[i, j]) - cos(theta_coefficient * L1_update)) + y_tilde * (cos(theta_coefficient * L2[i, j]) - cos(theta_coefficient * L2_update))

        # Calculates whether L[i,j] rotates
        R = exp(-dE * T)
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
        for columns in range(N):
            for rows in range(N):
                if L1[rows, columns] == L2[rows, columns]:
                    matches += 1
                # if rows in A_1[0] and columns in A_1[1]:
                #     if L1[rows, columns] == L2[rows, columns]:
                #         matches += 1
                # if rows in A_2[0] and columns in A_2[1]:
                #     if L1[rows, columns] == L2[rows, columns]:
                #         matches += 1
        print(matches)
        if matches == 2 * int(.75 * N * N/4) + N*N/8:
            print("A_1 and A_2 match!")
        else:
            print("We messed up somewhere :(")

    return [T, expE, sigma_bootstrap]  # This will create a results matrix which can be plotted


def vary_temps_RMI(T_min, T_max, T_step):
    if T_min == 0:
        temps = arange(T_min + T_step, T_max, T_step)
    else:
        temps = arange(T_min, T_max, T_step)

    # I have to separate the core mapping to prevent a memory error
    cores = Pool()
    result1 = cores.map(Region_1, temps)
    cores.close()
    cores.join()

    cores = Pool()
    result2 = cores.map(Region_2, temps)
    cores.close()
    cores.join()

    cores = Pool()
    result3 = cores.map(Region_3, temps)
    cores.close()
    cores.join()

    cores = Pool()
    result4 = cores.map(QCD_E, temps)
    cores.close()
    cores.join()

    shape_1 = array(result1)
    shape_2 = array(result2)
    shape_3 = array(result3)
    normal = array(result4)

    # Both Ising models are at the same temperature so,
    T_plot = normal[:, 0]  # Takes the first column of the results matrix

    E_shape_1 = shape_1[:, 1]  # Second column
    sigma_shape_1 = shape_1[:, 2]  # Third column

    E_shape_2 = shape_2[:, 1]  # Second column
    sigma_shape_2 = shape_2[:, 2]  # Third column

    E_shape_3 = shape_3[:, 1]  # Second column
    sigma_shape_3 = shape_3[:, 2]  # Third column

    E_normal = normal[:, 1]
    sigma_normal = normal[:, 2]

    return T_plot, E_shape_1, sigma_shape_1, E_shape_2, sigma_shape_2, E_shape_3, sigma_shape_3, E_normal, sigma_normal


def Topological_Entropy(T_min, T_max, T_step, save_data='no'):
    global output_path
    t1 = time.time()

    Data = vary_temps_RMI(T_min, T_max, T_step)

    T_plot = Data[0]

    if save_data == 'yes':
        Data = array(Data)
        t_elapse = (time.time() - t1) / 3600
        folder_path = '{0}/TEE_Calc/Finished_Data/'.format(output_path)
        folder_name = 'Data from TEE QCD; {0}; {1}, {2}, {3}, n=2, y~{4}, theta={5}'.format(date, E_measurements, T_step, N_global, y_tilde, theta_coefficient)
        if not os.path.exists(folder_path + folder_name):
            os.makedirs(folder_path + folder_name)
        savetxt('{8}{6}/RMI TEE; {0}; {1}, {2}, {3}, {4}, {5}, n=2, y~{7}, theta={9}.txt'.format(date, E_measurements, T_min, T_max, T_step, N_global, folder_name, y_tilde, folder_path, theta_coefficient), Data, header='This data took {0:.3f} hours and was recorded on {1}. This was run on the PSU Cluster.'.format(t_elapse, datetime.datetime.today()))

    return T_plot


if __name__ == '__main__':
    t_start = time.time()

    # Main Program
    T_min = float(sys.argv[4].split(',')[0])
    T_max = float(sys.argv[5].split(',')[0])

    Topological_Entropy(T_min, T_max, T_step, save_data='yes')

    # End of Main Program

    t_elapse = (time.time() - t_start) / 3600
    print("Full Program done in {0:.3f} hours".format(t_elapse))

