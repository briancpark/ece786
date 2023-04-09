# generate random input for proj3b

import numpy as np
import os
import sys


# make a random symmetric 2x2 matrix
U_0 = np.random.rand(4)
U_1 = np.random.rand(4)
U_2 = np.random.rand(4)
U_3 = np.random.rand(4)
U_4 = np.random.rand(4)
U_5 = np.random.rand(4)


# choose a number betweeen 7 and 30
qubits = np.random.randint(7, 10)

a = np.random.rand(2 ** qubits)

# choose 6 numbers between 0 and qubits; they must be unique
gates = np.random.choice(qubits, 6, replace=False)

gates.sort()

# write everything to a temp input file
input_file = open("temp_input.txt", "w")


# Write U's in row major 2x2 format with 3 decimal places and spaces between each number
input_file.write(f"{U_0[0]} {U_0[1]}\n{U_0[2]} {U_0[3]}\n\n")
input_file.write(f"{U_1[0]} {U_1[1]}\n{U_1[2]} {U_1[3]}\n\n")
input_file.write(f"{U_2[0]} {U_2[1]}\n{U_2[2]} {U_2[3]}\n\n")
input_file.write(f"{U_3[0]} {U_3[1]}\n{U_3[2]} {U_3[3]}\n\n")
input_file.write(f"{U_4[0]} {U_4[1]}\n{U_4[2]} {U_4[3]}\n\n")
input_file.write(f"{U_5[0]} {U_5[1]}\n{U_5[2]} {U_5[3]}\n\n")


for i in range(2 ** qubits):
    input_file.write(f"{a[i]}\n")

input_file.write(f"\n")

for i in range(6):
    input_file.write(f"{gates[i]}\n")

input_file.close()

os.system("make clean")
os.system("make")
os.system("./quamsimV1 temp_input.txt > v1.out")
os.system("./quamsimV2 temp_input.txt > v2.out")
os.system("diff v1.out v2.out")
os.system("rm temp_input.txt")
os.system("rm v1.out")
os.system("rm v2.out")