import matplotlib.pyplot as plt
import numpy as np


v1_global_reads = (
    8272
    + 16464
    + 24736
    + 32928
    + 41200
    + 49392
    + 57664
    + 65856
    + 82320
    + 90512
    + 106976
)
v1_global_writes = (
    8192 + 16384 + 24576 + 32768 + 40960 + 49152 + 57344 + 65536 + 73728 + 81920 + 90112
)

v2_global_reads = 19168
v2_global_writes = 32768

v3_global_reads = 19168
v3_global_writes = 65536

print("Version 1 Global Reads: ", v1_global_reads)
print("Version 1 Global Writes: ", v1_global_writes)
print("Version 1 Total: ", v1_global_reads + v1_global_writes)

print("Version 2 Global Reads: ", v2_global_reads)
print("Version 2 Global Writes: ", v2_global_writes)
print("Version 2 Total: ", v2_global_reads + v2_global_writes)

print("Version 3 Global Reads: ", v3_global_reads)
print("Version 3 Global Writes: ", v3_global_writes)
print("Version 3 Total: ", v3_global_reads + v3_global_writes)

# Plot the data in a bar chart
plt.figure(figsize=(10, 10))
x_ticks = ["Naive", "SharedMem", "SharedMem+ThreadCoarsening"]
x = np.arange(len(x_ticks))
offset = 0.2
plt.bar(
    x - offset,
    [v1_global_reads, v2_global_reads, v3_global_reads],
    width=0.2,
    label="Global Reads",
)
plt.bar(
    x,
    [v1_global_writes, v2_global_writes, v3_global_writes],
    width=0.2,
    label="Global Writes",
)
plt.bar(
    x + offset,
    [
        v1_global_reads + v1_global_writes,
        v2_global_reads + v2_global_writes,
        v3_global_reads + v3_global_writes,
    ],
    width=0.2,
    label="Total",
)
plt.legend()
plt.xlabel("Version")
plt.xticks(x, x_ticks)
plt.ylabel("Number of Global Accesses")
plt.title("Global Reads/Writes for Different Versions")
plt.savefig("global_reads_writes.png")
plt.clf()

V1_time = 0.0243405 * 1000
V2_time = 0.00600864 * 1000
V3_time = 0.00637728 * 1000

plt.figure(figsize=(10, 10))
plt.bar(x_ticks, [V1_time, V2_time, V3_time])
plt.xticks(x_ticks)
plt.xlabel("Version")
plt.ylabel("Time (ms)")
plt.title("Execution Time for Different Versions")
plt.savefig("quantum_time.png")
