# TASK: Signal Processing Lab - Comprehensive Comparison of Convolution Methods
# This program compares direct and FFT-based convolution for progressively larger signals,
# saves plots to files, and records performance metrics to results.txt

import numpy as np
import matplotlib.pyplot as plt
import time
import sys
import os
from datetime import datetime


# TASK: Create a class to redirect console output to both console and file
class OutputRedirector:
    """Redirect console output to both console and file"""

    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log_file = open(filename, "w")

    def write(self, message):
        self.terminal.write(message)
        self.log_file.write(message)
        self.log_file.flush()

    def flush(self):
        self.terminal.flush()
        self.log_file.flush()

    def close(self):
        self.log_file.close()


# TASK: Define signal generation functions
def square(t):
    """Generate a square wave signal"""
    return np.sign(np.sin(t))


def sawtooth(t):
    """Generate a sawtooth wave signal"""
    return 2 * (t / (2 * np.pi) - np.floor(t / (2 * np.pi) + 0.5))


def cosine(t):
    """Generate a cosine wave signal"""
    return np.cos(t)


def generate_signals(signal_size, signal_type="mixed"):
    """Generate test signals of specified size and type"""
    # Time indices
    n = np.linspace(0, 20 * np.pi, signal_size)

    if signal_type == "square_sawtooth":
        # Square and sawtooth waves (as in the example)
        a = 2 * square(n + 1)
        b = 3 * sawtooth(n + 1)
    elif signal_type == "cosine_square":
        # Cosine and square waves
        a = 2 * cosine(n + 1)
        b = 3 * square(n + 1)
    elif signal_type == "mixed":
        # Mixed signals with some randomness for variety
        a = 2 * square(n + 1) + 0.5 * cosine(2 * n)
        b = 3 * sawtooth(n + 1) + 0.5 * np.random.randn(signal_size)
    else:
        # Default to random signals
        a = np.random.randn(signal_size)
        b = np.random.randn(signal_size)

    return a, b


# TASK: Measure time for direct convolution
def measure_direct_convolution(a, b, max_size_for_direct=2**18):
    """Measure time for direct convolution with size limit"""
    if len(a) > max_size_for_direct:
        print(
            f"SKIPPING direct convolution - signal size {len(a)} exceeds limit {max_size_for_direct}"
        )
        return None, None

    print(f"Measuring time for direct convolution of signals with length {len(a)}...")
    start_time = time.time()

    # Direct convolution calculation
    c_direct = np.convolve(a, b)

    end_time = time.time()
    time_direct = end_time - start_time

    print(f"Direct convolution completed in {time_direct:.6f} seconds")
    return c_direct, time_direct


# TASK: Measure time for FFT-based convolution
def measure_fft_convolution(a, b):
    """Measure time for FFT-based convolution"""
    # Determine convolution length and next power of 2
    conv_length = len(a) + len(b) - 1
    m = 2 ** int(np.ceil(np.log2(conv_length)))  # Next power of 2

    print(
        f"Measuring time for FFT-based convolution of signals with length {len(a)}..."
    )
    print(f"Using FFT length of {m} (next power of 2 >= {conv_length})")

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
    return y1, time_fft, m


# TASK: Run comprehensive comparison for various signal sizes
def run_comprehensive_comparison():
    """Run comprehensive comparison for various signal sizes"""
    # Record start time
    start_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"Starting comprehensive convolution method comparison at {start_datetime}")
    print("=" * 80)

    # List of signal sizes to test (powers of 2, from 2^17 to 2^19)
    signal_sizes = [2**17, 2**18, 2**19]

    # Set maximum size for direct convolution (to avoid excessive runtime)
    max_size_for_direct = 2**18  # Skip direct convolution for sizes above this

    # Results storage
    results = []

    # Create directory for plots if it doesn't exist
    if not os.path.exists("plots"):
        os.makedirs("plots")

    # Run comparisons for each signal size
    for size_idx, size in enumerate(signal_sizes):
        print(f"\nTest {size_idx + 1}/{len(signal_sizes)}: Signal size = {size}")
        print("-" * 80)

        # Generate signals
        print(f"Generating signals of length {size}...")
        a, b = generate_signals(size, "square_sawtooth")

        # Save small segments of signals for verification
        segment_size = min(1000, size)
        plt.figure(figsize=(12, 8))

        plt.subplot(2, 1, 1)
        plt.stem(np.arange(segment_size), a[:segment_size], markerfmt="", basefmt="b-")
        plt.title(f"First {segment_size} points of Signal a (square wave)")
        plt.grid(True)

        plt.subplot(2, 1, 2)
        plt.stem(np.arange(segment_size), b[:segment_size], markerfmt="", basefmt="r-")
        plt.title(f"First {segment_size} points of Signal b (sawtooth wave)")
        plt.grid(True)

        plt.tight_layout()
        plt.savefig(f"plots/signals_size_{size}.png")
        plt.close()

        # Measure FFT-based convolution first (always possible)
        y1, time_fft, fft_length = measure_fft_convolution(a, b)

        # Try direct convolution if size is not too large
        if size <= max_size_for_direct:
            c_direct, time_direct = measure_direct_convolution(a, b)

            # Compare results if both methods were run
            if c_direct is not None:
                # Calculate error between methods (for the actual convolution length)
                conv_length = len(a) + len(b) - 1
                error = c_direct - np.real(y1[:conv_length])
                max_error = np.max(np.abs(error))
                print(f"Maximum absolute error between methods: {max_error}")

                # Calculate speedup
                speedup = time_direct / time_fft
                print(f"FFT method is {speedup:.2f}x faster than direct convolution")

                # Save comparison plots
                segment_size = min(1000, conv_length)
                plt.figure(figsize=(12, 10))

                plt.subplot(3, 1, 1)
                plt.stem(
                    np.arange(segment_size),
                    c_direct[:segment_size],
                    markerfmt="",
                    basefmt="g-",
                )
                plt.title(f"First {segment_size} points of Direct Convolution Result")
                plt.grid(True)

                plt.subplot(3, 1, 2)
                plt.stem(
                    np.arange(segment_size),
                    np.real(y1[:segment_size]),
                    markerfmt="",
                    basefmt="m-",
                )
                plt.title(
                    f"First {segment_size} points of FFT-based Convolution Result"
                )
                plt.grid(True)

                plt.subplot(3, 1, 3)
                plt.stem(
                    np.arange(segment_size),
                    error[:segment_size],
                    markerfmt="",
                    basefmt="r-",
                )
                plt.title(f"Error between Methods (max error: {max_error:.2e})")
                plt.grid(True)

                plt.tight_layout()
                plt.savefig(f"plots/convolution_comparison_size_{size}.png")
                plt.close()
        else:
            time_direct = None
            speedup = None

        # Save results
        results.append(
            {
                "size": size,
                "time_direct": time_direct,
                "time_fft": time_fft,
                "speedup": speedup,
                "fft_length": fft_length,
            }
        )

        # Save FFT result plot
        segment_size = min(1000, len(y1))
        plt.figure(figsize=(10, 6))
        plt.stem(
            np.arange(segment_size),
            np.real(y1[:segment_size]),
            markerfmt="",
            basefmt="m-",
        )
        plt.title(
            f"First {segment_size} points of FFT-based Convolution Result (size {size})"
        )
        plt.grid(True)
        plt.savefig(f"plots/fft_convolution_size_{size}.png")
        plt.close()

    # Plot performance comparison
    plot_performance_comparison(results)

    # Print summary table
    print("\nSummary of Convolution Method Performance:")
    print("=" * 80)
    print(
        f"{'Signal Size':<12} {'Direct Time (s)':<16} {'FFT Time (s)':<16} {'Speedup':<10} {'FFT Length':<12}"
    )
    print("-" * 80)

    for result in results:
        if result["time_direct"] is not None:
            print(
                f"{result['size']:<12} {result['time_direct']:<16.6f} {result['time_fft']:<16.6f} "
                f"{result['speedup']:<10.2f}x {result['fft_length']:<12}"
            )
        else:
            print(
                f"{result['size']:<12} {'skipped':<16} {result['time_fft']:<16.6f} "
                f"{'N/A':<10} {result['fft_length']:<12}"
            )

    print("=" * 80)

    # Record end time
    end_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"Completed comprehensive convolution method comparison at {end_datetime}")


# TASK: Plot and save performance comparison
def plot_performance_comparison(results):
    """Plot and save performance comparison of convolution methods"""
    sizes = [str(r["size"]) for r in results]
    fft_times = [r["time_fft"] for r in results]

    # Extract direct times (only for sizes where direct convolution was run)
    direct_times = []
    direct_sizes = []

    for r in results:
        if r["time_direct"] is not None:
            direct_times.append(r["time_direct"])
            direct_sizes.append(str(r["size"]))

    # Create figure for all FFT times
    plt.figure(figsize=(10, 6))
    plt.bar(sizes, fft_times, color="blue")
    plt.xlabel("Signal Size")
    plt.ylabel("Computation Time (seconds)")
    plt.title("FFT-based Convolution Performance")

    # Add value labels on top of bars
    for i, v in enumerate(fft_times):
        plt.text(i, v + 0.1, f"{v:.3f}s", ha="center")

    plt.grid(axis="y")
    plt.tight_layout()
    plt.savefig("plots/fft_performance.png")
    plt.close()

    # Create figure comparing methods (only for sizes where both were run)
    if direct_times:
        # Get corresponding FFT times
        comparable_fft_times = [
            r["time_fft"] for r in results if r["time_direct"] is not None
        ]

        plt.figure(figsize=(10, 6))

        # Bar positions
        x = np.arange(len(direct_sizes))
        width = 0.35

        # Create bars
        plt.bar(
            x - width / 2, direct_times, width, label="Direct Convolution", color="red"
        )
        plt.bar(
            x + width / 2,
            comparable_fft_times,
            width,
            label="FFT-based Convolution",
            color="blue",
        )

        # Add labels and title
        plt.xlabel("Signal Size")
        plt.ylabel("Computation Time (seconds)")
        plt.title("Performance Comparison: Direct vs. FFT-based Convolution")
        plt.xticks(x, direct_sizes)
        plt.legend()

        # Add value labels on top of bars
        for i, v in enumerate(direct_times):
            plt.text(i - width / 2, v + 0.1, f"{v:.3f}s", ha="center")

        for i, v in enumerate(comparable_fft_times):
            plt.text(i + width / 2, v + 0.1, f"{v:.3f}s", ha="center")

        plt.grid(axis="y")
        plt.tight_layout()
        plt.savefig("plots/method_comparison.png")
        plt.close()

        # Create speedup plot
        speedups = [r["speedup"] for r in results if r["speedup"] is not None]

        plt.figure(figsize=(10, 6))
        plt.bar(direct_sizes, speedups, color="green")
        plt.xlabel("Signal Size")
        plt.ylabel("Speedup Factor (x times)")
        plt.title("FFT Speedup over Direct Convolution")

        # Add value labels on top of bars
        for i, v in enumerate(speedups):
            plt.text(i, v + 0.5, f"{v:.2f}x", ha="center")

        plt.grid(axis="y")
        plt.tight_layout()
        plt.savefig("plots/speedup_factor.png")
        plt.close()


# TASK: Main function
def main():
    """Main function to run the program"""
    # Redirect output to both console and file
    output_redirector = OutputRedirector("results.txt")
    sys.stdout = output_redirector

    try:
        # Print system information
        print("=" * 80)
        print("Signal Processing Lab - Comprehensive Convolution Method Comparison")
        print("=" * 80)
        print(f"Date and Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Python version: {sys.version}")
        print(f"NumPy version: {np.__version__}")
        print("=" * 80)

        # Run the comprehensive comparison
        run_comprehensive_comparison()

    finally:
        # Restore stdout and close the file
        sys.stdout = sys.stdout.terminal
        output_redirector.close()
        print(
            "Results have been saved to results.txt and plots have been saved to the plots directory."
        )


# TASK: Run the main function
if __name__ == "__main__":
    main()
