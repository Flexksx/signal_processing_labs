# TASK: Signal Processing Lab - Comparing Convolution Methods and Calculating Error
# This program compares direct convolution with convolution via FFT and computes the error

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

# TASK: Determine the length of convolution and FFT
m = 8  # Length of convolution result (a+b-1) = (5+4-1) = 8
k = np.arange(1, m + 1)  # Time indices for results: [1, 2, 3, 4, 5, 6, 7, 8]

# TASK: Compute direct convolution
c_direct = np.convolve(a, b)  # Direct convolution using np.convolve

# TASK: Compute convolution via FFT method
AE = np.fft.fft(a, m)  # FFT of sequence a with length 8
BE = np.fft.fft(b, m)  # FFT of sequence b with length 8
p = AE * BE  # Element-wise multiplication of frequency components
y1 = np.fft.ifft(p)  # Inverse FFT of the product

# TASK: Calculate the error between the two convolution methods
error = c_direct - np.real(y1)  # Error between direct convolution and FFT method

# TASK: Create a figure with three subplots arranged vertically
plt.figure(figsize=(10, 12))

# TASK: Plot the direct convolution result c in the first subplot
plt.subplot(3, 1, 1)
plt.stem(k, c_direct, basefmt="b-")
plt.xlabel("Time index k")
plt.ylabel("Amplitude")
plt.title("Direct Convolution: c = conv(a, b)")
plt.grid(True)

# TASK: Plot the convolution via FFT y1 in the second subplot
plt.subplot(3, 1, 2)
plt.stem(k, np.real(y1), basefmt="r-")
plt.xlabel("Time index k")
plt.ylabel("Amplitude")
plt.title("Convolution via FFT: y1 = ifft(fft(a) * fft(b))")
plt.grid(True)

# TASK: Plot the error between the two methods in the third subplot
plt.subplot(3, 1, 3)
plt.stem(k, error, basefmt="g-")
plt.xlabel("Time index k")
plt.ylabel("Amplitude")
plt.title("Error: c - y1")
plt.grid(True)

# TASK: Adjust layout and display the plots
plt.tight_layout()
plt.show()

# TASK: Print numerical results for detailed analysis
print("Direct convolution result (c):")
print(c_direct)
print("\nConvolution via FFT (y1):")
print(np.real(y1))
print("\nError between methods (error = c - y1):")
print(error)
print("\nMaximum absolute error:")
print(np.max(np.abs(error)))
