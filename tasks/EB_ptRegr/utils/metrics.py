import numpy as np
import hist
from scipy.stats import poisson

def effSigmaInterval(hist):
    """
    Finds the smallest interval that contains 68% of the entries in a 1D histogram.

    This function uses a sliding window to find the contiguous block of bins
    with the highest density (sum of counts) that contains at least 68% of the
    total histogram entries. The interval returned is the smallest physical
    range (smallest width) that satisfies this condition.

    Args:
        hist (hist.Hist): A 1D histogram object from the boost-histogram library.

    Returns:
        tuple: A tuple (min_edge, max_edge) representing the smallest interval.
    """
    # 1. Get the bin counts and edges from the histogram
    counts = hist.values()
    edges = hist.axes[0].edges

    # 2. Calculate the target number of entries (68% of the total)
    total_entries = np.sum(counts)
    target_entries = 0.68 * total_entries

    # 3. Use a sliding window to find the smallest interval
    min_width = float('inf')
    best_interval = None

    current_sum = 0
    start = 0

    for end in range(len(counts)):
        current_sum += counts[end]

        # Shrink the window from the left until the sum is just above the target
        while current_sum >= target_entries:
            width = edges[end + 1] - edges[start]

            # If the current interval is smaller, update the best one
            if width < min_width:
                min_width = width
                best_interval = (edges[start], edges[end + 1])

            # Move the left edge of the window to find a new interval
            current_sum -= counts[start]
            start += 1

    return best_interval


def get_bootstrap_uncertainty(original_hist, num_bootstraps=1000):
    """
    Calculates the uncertainty of the effSigmaInterval width using the
    bootstrap method.

    Args:
        original_hist (hist.Hist): The original 1D histogram.
        num_bootstraps (int): The number of bootstrap samples to generate.

    Returns:
        tuple: A tuple (mean_width, std_dev_width, confidence_interval).
    """
    original_counts = original_hist.values()
    widths = []

    # Get the original width to compare
    original_interval = effSigmaInterval(original_hist)
    original_width = original_interval[1] - original_interval[0]

    for _ in range(num_bootstraps):
        # Generate new bin counts by drawing from a Poisson distribution
        # with the original counts as the mean. This simulates statistical
        # fluctuations in the histogram.
        bootstrapped_counts = poisson.rvs(original_counts)

        # Create a new histogram with the bootstrapped counts
        bootstrapped_hist = hist.Hist(original_hist.axes[0], data=bootstrapped_counts)

        # Calculate the interval for the bootstrapped histogram
        bootstrapped_interval = effSigmaInterval(bootstrapped_hist)

        # If an interval was found, calculate its width and store it
        if bootstrapped_interval:
            width = bootstrapped_interval[1] - bootstrapped_interval[0]
            widths.append(width)

    if not widths:
        return (original_width, 0, (original_width, original_width))

    # Calculate the mean, standard deviation, and 68% confidence interval
    mean_width = np.mean(widths)
    std_dev_width = np.std(widths)

    # Calculate the 16th and 84th percentiles for the 68% confidence interval
    lower_bound = np.percentile(widths, 16)
    upper_bound = np.percentile(widths, 84)

    return std_dev_width
