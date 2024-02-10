RANDOM_INT_LIMIT = 65535

MAXIMUM_ERRORS_PER_ITERATION = 512
MAXIMUM_INFOS_PER_ITERATION = 512

# Device capability for pytorch
MINIMUM_DEVICE_CAPABILITY = 5  # Maxwell
MINIMUM_DEVICE_CAPABILITY_TORCH_COMPILE = 7  # Volta

# FORCE the gpu to be present
DEVICE = "cuda:0"
CPU = "cpu"

# File to save last status of the benchmark when log helper not active
TMP_CRASH_FILE = "/tmp/maximal_crash_file.txt"


FLOAT_ERROR_THRESHOLD = None

# range for random generation
GENERATOR_MAX_ABS_VALUE_GEMM = 10
GENERATOR_MIN_ABS_VALUE_GEMM = -GENERATOR_MAX_ABS_VALUE_GEMM
