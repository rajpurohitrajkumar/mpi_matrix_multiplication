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
