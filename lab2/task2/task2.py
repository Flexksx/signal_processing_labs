# TASK: Signal Processing Lab - Discrete Convolution of Two Sequences
# This program performs convolution of two sequences and displays the result

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

# TASK: Perform convolution operation using NumPy's convolve function
# The convolution of a and b will have length (len(a) + len(b) - 1) = 8
convolution_result = np.convolve(a, b)

# TASK: Define time indices for the convolution result
m = 8  # Length of convolution result (a+b-1) = (5+4-1) = 8
k = np.arange(1, m + 1)  # Time indices for convolution result: [1, 2, 3, 4, 5, 6, 7, 8]

# TASK: Create a figure with three subplots arranged vertically
plt.figure(figsize=(10, 12))

# TASK: Plot the first sequence a(n) using stem plot in the top subplot
plt.subplot(3, 1, 1)
plt.stem(n, a, basefmt="b-")
plt.xlabel("Time index n")
plt.ylabel("Amplitude")
plt.title("Sequence a = [-2, 0, 1, -1, 3]")
plt.grid(True)

# TASK: Plot the second sequence b(n) using stem plot in the middle subplot
plt.subplot(3, 1, 2)
plt.stem(l, b, basefmt="b-")
plt.xlabel("Time index n")
plt.ylabel("Amplitude")
plt.title("Sequence b = [1, 2, 0, -1]")
plt.grid(True)

# TASK: Plot the convolution result c(n) using stem plot in the bottom subplot
plt.subplot(3, 1, 3)
plt.stem(k, convolution_result, basefmt="r-")
plt.xlabel("Time index k")
plt.ylabel("Amplitude")
plt.title("Convolution c = conv(a, b)")
plt.grid(True)

# TASK: Adjust layout and display the plots
plt.tight_layout()
plt.show()

# TASK: Print the convolution result for reference
print("Convolution result c = conv(a, b):")
print(convolution_result)
