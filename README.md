# Distributed Matrix Multiplication using MPI

![Matrix Multiplication](https://upload.wikimedia.org/wikipedia/commons/e/e5/MatrixLabelled.svg)

## 📖 Table of Contents

- [Overview](#-overview)
- [Implementation Details](#-implementation-details)
- [Prerequisites](#-prerequisites)
- [Installation](#-installation)
- [Usage](#-usage)
- [Project Structure](#-project-structure)
  - [Serial Implementation](#serial-implementation)
  - [Distributed MPI Implementation](#distributed-mpi-implementation)
- [Performance Analysis](#-performance-analysis)
  - [Scalability Testing](#scalability-testing)
  - [Benchmarking Report](#benchmarking-report)
  - [Plotting Performance Metrics](#plotting-performance-metrics)
- [Observations](#-observations)
- [Conclusion](#-conclusion)
- [GitHub Repository](#-github-repository)
- [References](#-references)

## 🚀 Overview

Matrix multiplication is a fundamental operation in scientific computing, data analysis, and machine learning. As the size of matrices increases, the computation becomes prohibitively time-consuming on a single machine.

This project demonstrates how to accelerate matrix multiplication by distributing the computational workload across multiple processes using the **Message Passing Interface (MPI)**. It includes a parallel MPI-based implementation and a standard serial version for performance comparison.

## ✨ Implementation Details

- **Parallel Processing**: MPI-based matrix multiplication using `mpi4py`.
- **Serial Fallback**: Includes a standard NumPy implementation that runs if `mpi4py` is not detected.
- **Performance Measurement**: Measures and compares execution time between serial and parallel versions.
- **Data Distribution**: Handles matrix partitioning and broadcasting of data among processes.
- **Scalability**: Supports basic scalability testing across a variable number of processes.

## ✅ Prerequisites

- Python 3.x
- NumPy
- `mpi4py` (MPI for Python)
- An MPI implementation (e.g., Open MPI) and its command-line runners (`mpiexec` or `mpirun`).

## 🛠️ Installation

1.  **Install MPI Libraries** (Example for Debian/Ubuntu):
    ```bash
    sudo apt update
    sudo apt install openmpi-bin openmpi-common libopenmpi-dev -y
    ```

2.  **Install Python Dependencies**:
    ```bash
    pip install mpi4py numpy matplotlib
    ```

3.  **Verify MPI Installation**:
    Check your Open MPI installation to ensure it's available in your PATH.
    ```bash
    mpiexec --version
    ```

## 💻 Usage

-   **Run the serial version**:
    ```bash
    python serial_matrix_multiplication.py
    ```

-   **Run the distributed MPI version**:
    Replace `<num_processes>` with the number of processes you want to use (e.g., 4).
    ```bash
    mpiexec -n <num_processes> python mpi_matrix_multiplication.py
    ```

## 📂 Project Structure

### Serial Implementation

The `serial_matrix_multiplication.py` script provides a baseline for performance comparison. It multiplies two matrices using only NumPy.

```python:serial_matrix_multiplication.py
import numpy as np
import time

N = 500
A = np.random.rand(N, N)
B = np.random.rand(N, N)

start_time = time.time()
C = np.dot(A, B)
end_time = time.time()

print("Matrix multiplication completed.")
print(f"Execution Time (Serial): {end_time - start_time:.4f} seconds")
```

### Distributed MPI Implementation

The `mpi_matrix_multiplication.py` script contains the core logic for the distributed computation. It scatters parts of the matrix across different processes, performs local computations, and gathers the results. It automatically falls back to a serial implementation if `mpi4py` is not available.

```python:mpi_matrix_multiplication.py
import numpy as np
import time
import sys
from typing import Optional, Tuple


def check_mpi_availability() -> bool:
    """Check if MPI is available and return availability status."""
    try:
        from mpi4py import MPI
        return True
    except ImportError:
        print("mpi4py module is not installed. Running in serial mode.")
        return False


def create_random_matrices(size: int) -> Tuple[np.ndarray, np.ndarray]:
    """Create two random matrices of given size."""
    np.random.seed(42)  # For reproducible results
    A = np.random.rand(size, size)
    B = np.random.rand(size, size)
    return A, B


def print_matrix(name: str, matrix: np.ndarray) -> None:
    """Print a matrix with a descriptive name."""
    print(f"\n{name}:")
    print(matrix)


def serial_matrix_multiplication(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Perform matrix multiplication using numpy's dot product."""
    return np.dot(A, B)


def mpi_matrix_multiplication() -> None:
    """Perform matrix multiplication using MPI parallel processing."""
    from mpi4py import MPI
    
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    N = 4  # Matrix size (NxN)
    
    # Ensure N is divisible by number of processes
    if N % size != 0:
        if rank == 0:
            print(f"Error: Matrix size {N} must be divisible by number of processes {size}")
        return
    
    # Master process initializes matrices
    if rank == 0:
        A, B = create_random_matrices(N)
        C = np.zeros((N, N))
        
        print_matrix("Matrix A", A)
        print_matrix("Matrix B", B)
        start_time = time.time()
    else:
        A = B = C = None
    
    # Distribute matrix A across processes
    local_A = np.zeros((N // size, N))
    comm.Scatter([A, MPI.DOUBLE], [local_A, MPI.DOUBLE], root=0)
    
    # Broadcast matrix B to all processes
    if rank != 0:
        B = np.empty((N, N))
    comm.Bcast([B, MPI.DOUBLE], root=0)
    
    # Perform local matrix multiplication
    local_C = np.dot(local_A, B)
    
    # Gather results back to master process
    comm.Gather([local_C, MPI.DOUBLE], [C, MPI.DOUBLE] if rank == 0 else None, root=0)
    
    # Master process displays results
    if rank == 0:
        end_time = time.time()
        print_matrix("Resultant Matrix C (A × B)", C)
        print(f"\nExecution Time: {end_time - start_time:.4f} seconds")


def main() -> None:
    """Main function to run matrix multiplication."""
    print("Matrix Multiplication Calculator")
    print("=" * 40)
    
    if check_mpi_availability():
        mpi_matrix_multiplication()
    else:
        # Fallback to serial execution
        N = 4
        A, B = create_random_matrices(N)
        
        print_matrix("Matrix A", A)
        print_matrix("Matrix B", B)
        
        start_time = time.time()
        C = serial_matrix_multiplication(A, B)
        end_time = time.time()
        
        print_matrix("Resultant Matrix C (A × B)", C)
        print(f"\nExecution Time (Serial): {end_time - start_time:.4f} seconds")


if __name__ == "__main__":
    main()
```

## 📊 Performance Analysis

To evaluate the effectiveness of the parallel implementation, we can measure execution time, speedup, and efficiency.

### Scalability Testing

Run the MPI script with a varying number of processes to collect performance data.

| Processes | Command                                              |
| :-------- | :--------------------------------------------------- |
| 1         | `mpiexec -n 1 python mpi_matrix_multiplication.py`   |
| 2         | `mpiexec -n 2 python mpi_matrix_multiplication.py`   |
| 4         | `mpiexec -n 4 python mpi_matrix_multiplication.py`   |
| 8         | `mpiexec -n 8 python mpi_matrix_multiplication.py`   |

### Benchmarking Report

Here is a sample report for a **500x500** matrix.

| Matrix Size | Processes | Serial Time (s) | Parallel Time (s) | Speedup | Efficiency |
| :---------- | :-------- | :-------------- | :---------------- | :------ | :--------- |
| 500×500     | 1         | 5.60            | 5.60              | 1.00x   | 100%       |
| 500×500     | 2         | 5.60            | 3.00              | 1.87x   | 93.5%      |
| 500×500     | 4         | 5.60            | 1.80              | 3.11x   | 77.8%      |
| 500×500     | 8         | 5.60            | 1.20              | 4.67x   | 58.4%      |

-   **Speedup**: `Serial Time / Parallel Time`
-   **Efficiency**: `Speedup / Number of Processes`

### Plotting Performance Metrics

You can visualize the performance metrics using the `plot_metrics.py` script. Update the `execution_time` list with your benchmark results.

```python:plot_metrics.py
import matplotlib.pyplot as plt

# Data collected from benchmarks
processes = [1, 2, 4, 8]
execution_time = [5.6, 3.0, 1.8, 1.2]  # Update with your results
speedup = [execution_time[0] / t for t in execution_time]

# Plot Execution Time
plt.figure(figsize=(8, 5))
plt.plot(processes, execution_time, marker='o', color='blue')
plt.title('Execution Time vs Number of Processes')
plt.xlabel('Number of Processes')
plt.ylabel('Execution Time (seconds)')
plt.xticks(processes)
plt.grid(True)
plt.savefig('execution_time_plot.png')
plt.show()

# Plot Speedup
plt.figure(figsize=(8, 5))
plt.plot(processes, speedup, marker='s', color='green')
plt.title('Speedup vs Number of Processes')
plt.xlabel('Number of Processes')
plt.ylabel('Speedup')
plt.xticks(processes)
plt.grid(True)
plt.savefig('speedup_plot.png')
plt.show()
```

Run the script to generate the charts:
```bash
python plot_metrics.py
```

This will produce `execution_time_plot.png` and `speedup_plot.png`.

## 💡 Observations

-   Parallel computation significantly reduces execution time, especially for large matrices.
-   Communication overhead (scattering and gathering data) can become a bottleneck, especially for small matrices or a large number of processes.
-   Scalability is not perfectly linear, as efficiency tends to decrease when more processes are added due to this overhead.

## ✅ Conclusion

This mini-project successfully demonstrates the benefits of distributed computing for an intensive task like matrix multiplication. It provides hands-on experience in partitioning data, coordinating processes with MPI, and measuring performance metrics like speedup and efficiency—all crucial concepts in high-performance computing (HPC).

## 🔗 GitHub Repository

The source code for this project is available on GitHub:
[https://github.com/rajpurohitrajkumar/mpi_matrix_multiplication](https://github.com/rajpurohitrajkumar/mpi_matrix_multiplication)

## 📚 References

-   [MPI Forum Documentation](https://www.mpi-forum.org/docs/)
-   [mpi4py Documentation](https://mpi4py.readthedocs.io/en/stable/)
-   [Python Official Website](https://www.python.org/)
-   [Matrix Mathematics (Wikipedia)](https://en.wikipedia.org/wiki/Matrix_(mathematics))
