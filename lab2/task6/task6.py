# TASK: Signal Processing Lab - Comparing Convolution Methods with Large Mathematical Signals
# This program compares the time performance of direct and FFT-based convolution

import numpy as np
import matplotlib.pyplot as plt
import time


# TASK: Define functions to generate various signals (similar to those in Lab #1)
def square(t):
    """Generate a square wave signal"""
    return np.sign(np.sin(t))


def sawtooth(t):
    """Generate a sawtooth wave signal"""
    return 2 * (t / (2 * np.pi) - np.floor(t / (2 * np.pi) + 0.5))


def cosine(t):
    """Generate a cosine wave signal"""
    return np.cos(t)


# TASK: Function to measure time for direct convolution
def measure_direct_convolution(a, b):
    """Measure time for direct convolution"""
    print("Measuring time for direct convolution...")
    start_time = time.time()

    # Direct convolution calculation
    c_direct = np.convolve(a, b)

    end_time = time.time()
    time_direct = end_time - start_time

    print(f"Direct convolution completed in {time_direct:.6f} seconds")
    return c_direct, time_direct


# TASK: Function to measure time for FFT-based convolution
def measure_fft_convolution(a, b, m):
    """Measure time for FFT-based convolution"""
    print("Measuring time for FFT-based convolution...")
    start_time = time.time()

    # FFT of both signals
    AE = np.fft.fft(a, m)
    BE = np.fft.fft(b, m)

    # Multiply in frequency domain
    p = AE * BE

    # Inverse FFT to get result
    y1 = np.fft.ifft(p)

    end_time = time.time()
    time_fft = end_time - start_time

    print(f"FFT-based convolution completed in {time_fft:.6f} seconds")
    return y1, time_fft


# TASK: Compare convolution methods for signals of specified size
def compare_convolution_methods(signal_size):
    """Compare direct and FFT-based convolution for signals of given size"""
    print(f"\nComparing convolution methods for signals of length {signal_size}...")

    # Generate time indices
    n = np.arange(signal_size)
    l = np.arange(signal_size)

    # Generate signals using mathematical functions (as in Lab #1)
    a = 2 * square(20 * np.pi * n / signal_size + 1)
    b = 3 * sawtooth(20 * np.pi * l / signal_size + 1)

    # Determine convolution length and next power of 2
    conv_length = len(a) + len(b) - 1
    m = 2 ** int(np.ceil(np.log2(conv_length)))  # Next power of 2

    print(f"Signal lengths: a={len(a)}, b={len(b)}")
    print(f"Theoretical convolution length: {conv_length}")
    print(f"FFT length (next power of 2): {m}")

    # Only perform direct convolution for smaller signals (up to 2^14)
    if signal_size <= 2**14:
        c_direct, time_direct = measure_direct_convolution(a, b)
    else:
        print("Direct convolution would be too slow for this signal size, skipping...")
        time_direct = None
        c_direct = None

    # Perform FFT-based convolution
    y1, time_fft = measure_fft_convolution(a, b, m)

    # Compare performance
    if time_direct is not None:
        speedup = time_direct / time_fft
        print(f"FFT method is {speedup:.2f}x faster than direct convolution")

        # Calculate error between methods
        error = c_direct - np.real(y1[:conv_length])
        max_error = np.max(np.abs(error))
        print(f"Maximum absolute error: {max_error}")

    # Plot sample segments of signals and results
    plot_sample_signals(a, b, y1, signal_size)

    return time_direct, time_fft


# TASK: Plot sample segments of large signals
def plot_sample_signals(a, b, y1, signal_size):
    """Plot sample segments of signals and convolution result"""
    # Determine sample size for plotting (up to 1000 points)
    sample_size = min(1000, signal_size)

    # Create figure with 3 subplots
    fig, axs = plt.subplots(3, 1, figsize=(12, 10))

    # Plot first signal
    axs[0].stem(
        np.arange(sample_size), a[:sample_size], "b-", markerfmt="bo", basefmt="b-"
    )
    axs[0].set_title(f"First {sample_size} points of Signal a (square wave)")
    axs[0].set_xlabel("Time index n")
    axs[0].set_ylabel("Amplitude")
    axs[0].grid(True)

    # Plot second signal
    axs[1].stem(
        np.arange(sample_size), b[:sample_size], "r-", markerfmt="ro", basefmt="r-"
    )
    axs[1].set_title(f"First {sample_size} points of Signal b (sawtooth wave)")
    axs[1].set_xlabel("Time index n")
    axs[1].set_ylabel("Amplitude")
    axs[1].grid(True)

    # Plot convolution result (sample)
    axs[2].stem(
        np.arange(sample_size),
        np.real(y1[:sample_size]),
        "g-",
        markerfmt="go",
        basefmt="g-",
    )
    axs[2].set_title(f"First {sample_size} points of Convolution Result")
    axs[2].set_xlabel("Time index n")
    axs[2].set_ylabel("Amplitude")
    axs[2].grid(True)

    plt.tight_layout()
    plt.show()


# TASK: Run comparison with progressively larger signals
def main():
    """Main function to run convolution comparisons"""
    # List of signal sizes to test (powers of 2)
    signal_sizes = [2**10, 2**12, 2**14, 2**16]

    # Results storage
    results = []

    # Run comparisons for each signal size
    for size in signal_sizes:
        time_direct, time_fft = compare_convolution_methods(size)
        results.append((size, time_direct, time_fft))

    # Display summary of results
    print("\nSummary of Convolution Method Performance:")
    print("-" * 60)
    print(
        f"{'Signal Size':<15} {'Direct Time (s)':<20} {'FFT Time (s)':<20} {'Speedup':<10}"
    )
    print("-" * 60)

    for size, time_direct, time_fft in results:
        if time_direct is not None:
            speedup = time_direct / time_fft
            print(
                f"{size:<15} {time_direct:<20.6f} {time_fft:<20.6f} {speedup:<10.2f}x"
            )
        else:
            print(f"{size:<15} {'too slow':<20} {time_fft:<20.6f} {'-':<10}")

    print("-" * 60)

    # Plot performance comparison
    plot_performance_comparison(results)


# TASK: Plot performance comparison of methods
def plot_performance_comparison(results):
    """Plot performance comparison of convolution methods"""
    sizes = [r[0] for r in results]

    # Extract times (only for sizes where both methods were run)
    direct_times = []
    fft_times = []
    size_labels = []

    for size, time_direct, time_fft in results:
        if time_direct is not None:
            direct_times.append(time_direct)
            fft_times.append(time_fft)
            size_labels.append(str(size))

    # Create figure
    plt.figure(figsize=(10, 6))

    # Bar positions
    x = np.arange(len(size_labels))
    width = 0.35

    # Create bars
    plt.bar(x - width / 2, direct_times, width, label="Direct Convolution")
    plt.bar(x + width / 2, fft_times, width, label="FFT-based Convolution")

    # Add labels and title
    plt.xlabel("Signal Size")
    plt.ylabel("Computation Time (seconds)")
    plt.title("Performance Comparison: Direct vs. FFT-based Convolution")
    plt.xticks(x, size_labels)
    plt.legend()

    # Add value labels on top of bars
    for i, v in enumerate(direct_times):
        plt.text(i - width / 2, v + 0.01, f"{v:.3f}s", ha="center")

    for i, v in enumerate(fft_times):
        plt.text(i + width / 2, v + 0.01, f"{v:.3f}s", ha="center")

    plt.tight_layout()
    plt.show()


# TASK: Run the main function
if __name__ == "__main__":
    main()
