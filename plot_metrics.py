# Data collected from benchmarks
processes = [1, 2, 4, 8]
execution_time = [5.6, 3.0, 1.8, 1.2]
speedup = [execution_time[0]/t for t in execution_time]

# Plot Execution Time
plt.figure(figsize=(8, 5))
plt.plot(processes, execution_time, marker='o', color='blue')
plt.title('Execution Time vs Number of Processes')
plt.xlabel('Number of Processes')
plt.ylabel('Execution Time (seconds)')
plt.xticks(processes)
plt.grid(True)
plt.savefig('execution_time_plot.png')  # saves the figure
plt.show()

# Plot Speedup
plt.figure(figsize=(8, 5))
plt.plot(processes, speedup, marker='s', color='green')
plt.title('Speedup vs Number of Processes')
plt.xlabel('Number of Processes')
plt.ylabel('Speedup')
plt.xticks(processes)
plt.grid(True)
plt.savefig('speedup_plot.png')  # saves the figure
plt.show()
