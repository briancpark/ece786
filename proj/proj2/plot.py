import matplotlib.pyplot as plt
import numpy as np

benchmarks = [
    "BFS",
    "LIB",
    "LPS",
    "NN",
    "NQU",
]


SM6_TITANX_IPC = [
    136518 / 1771,
    490197 / 1715,
    757054 / 1242,
    84493 / 1440,
    225476 / 6839,
]
SM7_TITANV_IPC = [
    131260 / 781,
    349650 / 1131,
    485776 / 818,
    137822 / 550,
    122987 / 3167,
]

# create barplot of ipc side by side
bar1 = np.arange(len(benchmarks))
bar2 = [x + 0.25 for x in bar1]
bar3 = [x + 0.25 for x in bar2]
plt.figure(figsize=(10, 5))
plt.bar(bar1, SM6_TITANX_IPC, width=0.25, label="SM6 TitanX")
plt.bar(bar2, SM7_TITANV_IPC, width=0.25, label="SM7 TitanV")
plt.xticks([r + 0.125 for r in range(len(benchmarks))], benchmarks)
plt.title("IPC of SM6 TitanX and SM7 TitanV")
plt.xlabel("Benchmark")
plt.ylabel("IPC")
plt.legend()
plt.show()
plt.savefig("ipc.png", dpi=300)