"""
Experiment configuration parameters.

For full reproduction of paper results, use:
- N_REALIZATIONS = 1000
- N_SURROGATES = 1000

For faster testing:
- N_REALIZATIONS = 100
- N_SURROGATES = 100
"""

# Sample sizes
N_LARGE = 65536  # 2^16
N_SMALL = 4096   # 2^12

# Number of realizations/surrogates (reduce for faster execution)
N_REALIZATIONS = 100  # Paper uses 1000
N_SURROGATES = 100    # Paper uses 1000

# Cascade parameters
SIGMA_W = 0.3

# WTMM parameters
Q_MIN = -10
Q_MAX = 10
Q_STEP = 1

# Dependence measure parameters
MAX_LAG = 21
KSG_K = 5

# IAFT parameters
IAFT_ITERATIONS = 100

# Parallel processing
N_JOBS = -1  # Use all cores

# Random seeds
MASTER_SEED = 42
