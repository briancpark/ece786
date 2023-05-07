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

A programming assignment to get your feet wet with CUDA and GPGPU-Sim. The assignment is to implement a simple matrix multiplication kernel for quantum simulation and run it on GPGPU-Sim.

Specifically, we optimize this kernel for quantum simulation:
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

```sh

## Project 2: GPGPU-Sim

## Project 3: Load Bypass and Quantum Simulation Optimizations

## Final Project: Cache Bypassing Analysis and Performance Study

## Requirements for Plotting

Install the dependencies for plotting.

```sh
pip3 install -r requirements.txt
```
