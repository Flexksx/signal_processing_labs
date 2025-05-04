# TASK 8: Convolution of two signals
# This program computes the convolution of an input signal and a system's impulse response

import numpy as np
import matplotlib.pyplot as plt

# TASK: Define the input signal and the system's impulse response
a = np.array([1, 4, 2])  # Input signal
b = np.array([1, 2, 3, 4, 5, 4, 3, 3, 2, 2, 1, 1])  # Impulse response

# TASK: Compute the full convolution of signals a and b
# The convolution length will be len(a) + len(b) - 1 = 3 + 12 - 1 = 14
c_full = np.convolve(a, b)

# TASK: Create time indices for the convolution result
m_full = np.arange(1, len(c_full) + 1)  # Time indices from 1 to 14

# TASK: Create a figure to visualize the signals and their convolution
plt.figure(figsize=(12, 10))

# Plot the input signal a
plt.subplot(3, 1, 1)
plt.stem(np.arange(1, len(a) + 1), a)
plt.xlabel("Time index n")
plt.ylabel("Amplitude")
plt.title("Input Signal a = [1, 4, 2]")
plt.grid(True)

# Plot the impulse response b
plt.subplot(3, 1, 2)
plt.stem(np.arange(1, len(b) + 1), b)
plt.xlabel("Time index n")
plt.ylabel("Amplitude")
plt.title("Impulse Response b = [1, 2, 3, 4, 5, 4, 3, 3, 2, 2, 1, 1]")
plt.grid(True)

# Plot the convolution result
plt.subplot(3, 1, 3)
plt.stem(m_full, c_full)
plt.xlabel("Time index n")
plt.ylabel("Amplitude")
plt.title("Convolution c = conv(a, b) (length 14)")
plt.grid(True)

plt.tight_layout()
plt.show()

# TASK: Print the numerical results for reference
print("Input signal a:")
print(a)
print("\nImpulse response b:")
print(b)
print("\nConvolution result c_full (length 14):")
print(c_full)
