# compare the output of two files, with some tolerance on floating point numbers

# read the two files
import sys
import re
import math
import os

correct_file = open(sys.argv[1], 'r')
my_file = open(sys.argv[2], 'r')

# read the two files into lists of lines
correct_lines = correct_file.readlines()
my_lines = my_file.readlines()

for i in range(len(correct_lines)):
    correct_line = correct_lines[i]
    my_line = my_lines[i]
  
    # get the floating point numbers in each line
    correct_num = float(re.findall(r"[-+]?\d*\.\d+|\d+", correct_line)[0])
    my_num = float(re.findall(r"[-+]?\d*\.\d+|\d+", my_line)[0])
    
    # compare the numbers
    if math.fabs(correct_num - my_num) > 0.0001:
        # print("Error: line " + str(i+1) + " is incorrect")
        print("Correct output is " + str(correct_num) + ", but your output is " + str(my_num) + ".")