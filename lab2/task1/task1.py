# TASK: Signal Processing Lab - Generating and Visualizing Finite Sequences
# This program creates two discrete sequences and displays them using stem plots

import numpy as np
import matplotlib.pyplot as plt

# TASK: Define two finite sequences a(n) and b(n) with lengths 5 and 4 respectively
a = np.array([-2, 0, 1, -1, 3])
b = np.array([1, 2, 0, -1])

# TASK: Create time indices for both sequences
d = 5
n = np.arange(1, d + 1)  # Time indices for sequence a: [1, 2, 3, 4, 5]
c = 4
l = np.arange(1, c + 1)  # Time indices for sequence b: [1, 2, 3, 4]

# TASK: Create a figure with two subplots arranged vertically
plt.figure(figsize=(10, 8))

# TASK: Plot the first sequence a(n) using stem plot in the top subplot
plt.subplot(2, 1, 1)
plt.stem(n, a, basefmt="b-")
plt.xlabel("Time index n")
plt.ylabel("Amplitude")
plt.title("Sequence a = [-2, 0, 1, -1, 3]")
plt.grid(True)

# TASK: Plot the second sequence b(n) using stem plot in the bottom subplot
plt.subplot(2, 1, 2)
plt.stem(l, b, basefmt="b-")
plt.xlabel("Time index n")
plt.ylabel("Amplitude")
plt.title("Sequence b = [1, 2, 0, -1]")
plt.grid(True)

# TASK: Adjust layout and display the plots
plt.tight_layout()  # Optimize spacing between subplots
plt.show()  # Display the figure with both sequences
