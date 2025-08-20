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

def smallest_interval_68(arr):
    arr = np.sort(arr)
    n = len(arr)
    k = int(np.floor(0.68 * n))  # number of points to include

    # Sliding window over sorted values
    min_width = np.inf
    best_interval = (None, None)

    for i in range(n - k):
        low = arr[i]
        high = arr[i + k]
        width = high - low
        if width < min_width:
            min_width = width
            best_interval = (low, high)

    return best_interval