#!/usr/bin/env python3
"""
MPI Matrix Multiplication Implementation

This script demonstrates parallel matrix multiplication using MPI.
It can run in both MPI and serial modes depending on availability.
"""

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
