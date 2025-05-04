# TASK: Signal Processing Lab - Verifying Convolution Property through Inverse Fourier Transform
# This program demonstrates how the inverse FFT of the product of Fourier transforms
# equals the convolution of the original sequences

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

# TASK: Compute inverse FFT of the product to get convolution result
y1 = np.fft.ifft(p)  # This should be equivalent to the convolution of a and b

# TASK: Compute convolution directly for comparison
convolution_result = np.convolve(a, b)

# TASK: Create a figure with multiple subplots
plt.figure(figsize=(12, 15))

# TASK: Plot the first sequence a(n)
plt.subplot(5, 1, 1)
plt.stem(n, a, basefmt="b-")
plt.xlabel("Time index n")
plt.ylabel("Amplitude")
plt.title("Sequence a = [-2, 0, 1, -1, 3]")
plt.grid(True)

# TASK: Plot the second sequence b(n)
plt.subplot(5, 1, 2)
plt.stem(l, b, basefmt="b-")
plt.xlabel("Time index n")
plt.ylabel("Amplitude")
plt.title("Sequence b = [1, 2, 0, -1]")
plt.grid(True)

# TASK: Plot the product of Fourier transforms (magnitude)
plt.subplot(5, 1, 3)
plt.stem(k, np.abs(p), basefmt="r-")
plt.xlabel("Frequency index k")
plt.ylabel("Magnitude")
plt.title("Product of Fourier Transforms: |AE * BE|")
plt.grid(True)

# TASK: Plot the inverse FFT of the product (y1)
plt.subplot(5, 1, 4)
plt.stem(k, np.real(y1), basefmt="g-")  # Taking real part as results should be real
plt.xlabel("Time index k")
plt.ylabel("Amplitude")
plt.title("Inverse FFT of Product: y1 = ifft(AE * BE)")
plt.grid(True)

# TASK: Plot the direct convolution result for comparison
plt.subplot(5, 1, 5)
plt.stem(k, convolution_result, basefmt="m-")
plt.xlabel("Time index k")
plt.ylabel("Amplitude")
plt.title("Direct Convolution: conv(a, b)")
plt.grid(True)

# TASK: Adjust layout and display the plots
plt.tight_layout()
plt.show()

# TASK: Print numerical results for comparison
print("Convolution through Inverse FFT (y1 = ifft(AE * BE)):")
print(np.real(y1))  # Taking real part as results should be real
print("\nDirect convolution result:")
print(convolution_result)
print("\nDifference between methods (should be near zero):")
print(np.real(y1) - convolution_result)
