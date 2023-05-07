# ECE 786

ECE 786 is Advanced Computer Architecture: Data Parallel Processors at NCSU. I took this offering in Spring 2023 under Huiyang Zhou. This repo contains my development setup and notes.

## GPGPU-Sim

A lot of the projects were done under GPGPU-Sim. Here is the setup required to run GPGPU-Sim. You will need Docker container, and there is no need to have a CUDA capable GPU. Simulation is all done through the CPU. It is preferred to use Docker container, since it replaces the CUDA libraries with the GPGPU-Sim libraries.

```sh
docker pull pli11/gpgpusim:cuda_10_1
docker run -w /root -it pli11/gpgpusim:cuda_10_1 /bin/bash
cd ~/gpgpu-sim_distribution/
git checkout tags/v4.0.1 -b v401
```

When making changes and running GPGPU-Sim, you will need to make sure to run the following commands:

```sh
cd gpgpu-sim_distribution
source setup_environment
make -j
```

When reconnecting to the Docker container, you will need to run the following commands:

```sh
docker exec -it ContainerID /bin/bash
```

## Project 1: CUDA Programming and GPGPU-Sim

A programming assignment to get your feet wet with CUDA and GPGPU-Sim. The assignment is to implement a simple matrix multiplication kernel for quantum simulation and run it on GPGPU-Sim. Then we do simple analysis of the GPGPU-Sim project of cache misses and performance analysis.

Specifically, we implement and optimize this kernel for quantum simulation:

```math
\left(\begin{array}{l}
a_{b_{n-1}, \ldots, b_{t+1}, 0, b_{t-1}, \ldots, b_0}^{\prime} \\
a_{b_{n-1}^{\prime}, \ldots, b_{t+1}, 1, b_{t-1}, \ldots, b_0}
\end{array}\right)=\left[\begin{array}{ll}
U_{0,0} & U_{0,1} \\
U_{1,0} & U_{1,1}
\end{array}\right]\left(\begin{array}{l}
a_{b_{n-1}, \ldots, b_{t+1}, 0, b_{t-1}, \ldots, b_0} \\
a_{b_{n-1}, \ldots, b_{t+1}, 1, b_{t-1}, \ldots, b_0}
\end{array}\right)
```

## Project 2: GPGPU-Sim

We benchmark a variety of kernels on GPGPU-Sim from ISPASS benchmarks. We analyze the IPC as well as the branch divergence of each of the kernels. We also analyze the number of global and shared memory accesses that each kernel makes.

## Project 3: Load Bypass and Quantum Simulation Optimizations

We implement a load bypass implementation based on certain contitions of the address accesses. This was another case study in how load bypass can affect cache performance.

Separately, we implement a quantum simulation optimization that now supports multiple qubits and gates. This was entirely done in CUDA, so we wrote a naive kernel, then optimized it with software optimizations such as utilizing shared memory and thread coarsening. Later, the number of shared and global memory accesses was analyzed under GPGPU-Sim as well as end-to-end performance on a real GPU.

## Final Project: Cache Bypassing Analysis and Performance Study

We implement a profile-based cache bypassing mechanism based on my professor's paper: Locality-Driven Dynamic GPU Cache Bypassing. After implementing, we do a thorough analysis on the mechanism under various benchmarks from ISPASS and Rodinia.

## Requirements for Plotting

Install the dependencies for plotting.

```sh
pip3 install -r requirements.txt
```
