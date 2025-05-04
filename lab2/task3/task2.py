# TASK: Signal Processing Lab - Discrete Fourier Transform and Convolution Property
# This program demonstrates the convolution property of Fourier Transform where
# the product of Fourier transforms equals the Fourier transform of convolution

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

# TASK: Compute Fast Fourier Transform (FFT) of both sequences with length m
AE = np.fft.fft(a, m)  # FFT of sequence a with length 8
BE = np.fft.fft(b, m)  # FFT of sequence b with length 8

# TASK: Compute the product of the Fourier transforms
p = AE * BE  # Element-wise multiplication of frequency components

# TASK: Compute convolution directly for comparison
convolution_result = np.convolve(a, b)

# TASK: Compute inverse FFT of the product to verify convolution property
convolution_via_fft = np.fft.ifft(p)

# TASK: Create a figure with multiple subplots
plt.figure(figsize=(12, 15))

# TASK: Plot the first sequence a(n)
plt.subplot(4, 1, 1)
plt.stem(n, a, basefmt="b-")
plt.xlabel("Time index n")
plt.ylabel("Amplitude")
plt.title("Sequence a = [-2, 0, 1, -1, 3]")
plt.grid(True)

# TASK: Plot the second sequence b(n)
plt.subplot(4, 1, 2)
plt.stem(l, b, basefmt="b-")
plt.xlabel("Time index n")
plt.ylabel("Amplitude")
plt.title("Sequence b = [1, 2, 0, -1]")
plt.grid(True)

# TASK: Plot the product of Fourier transforms (magnitude)
plt.subplot(4, 1, 3)
plt.stem(k, np.abs(p), basefmt="r-")
plt.xlabel("Frequency index k")
plt.ylabel("Magnitude")
plt.title("Product of Fourier Transforms: |AE * BE|")
plt.grid(True)

# TASK: Plot the direct convolution result for comparison
plt.subplot(4, 1, 4)
plt.stem(k, convolution_result, basefmt="g-")
plt.xlabel("Time index k")
plt.ylabel("Amplitude")
plt.title("Convolution Result: conv(a, b)")
plt.grid(True)

# TASK: Adjust layout and display the plots
plt.tight_layout()
plt.show()

# TASK: Print numerical results for comparison
print("Fourier Transform of sequence a:")
print(AE)
print("\nFourier Transform of sequence b:")
print(BE)
print("\nProduct of Fourier Transforms (p = AE * BE):")
print(p)
print("\nDirect convolution result:")
print(convolution_result)
print("\nConvolution via inverse FFT of product:")
print(convolution_via_fft.real)  # Taking real part as results should be real
