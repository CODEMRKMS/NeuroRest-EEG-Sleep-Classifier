# Authors: Shadab, Aryan, Adina

from pathlib import Path

# Where the Sleep-EDF files (and the processed .npz cache) live.
# Override on the command line with --data-dir.
DATA_DIR = Path(__file__).resolve().parent / 'data' / 'sleep-edf'

# Sleep-EDF Expanded, Sleep-Cassette study. Used only if DATA_DIR has no EDF files.
BASE_URL = 'https://physionet.org/files/sleep-edfx/1.0.0/sleep-cassette/'
NUM_SUBJECTS = 20  # first N subjects: SC4001E0 ... SC4192E0

# Preprocessing (as in the interim report)
CHANNELS = ('EEG Fpz-Cz', 'EEG Pz-Oz')
EPOCH_SECONDS = 30
SAMPLES_PER_EPOCH = 3000  # 30 s at 100 Hz
CLASS_NAMES = ('S1', 'S2', 'S3', 'S4', 'REM')

# Per-recording, per-channel z-score. Its tag is part of the normalised cache file names, so
# bump it whenever the normalisation changes. Un-normalised files keep the plain <id>.npz name.
NORM_VERSION = 'zscore-v1'

# Subject-wise split (fractions of subjects, not epochs)
SEED = 42
VAL_FRACTION = 0.1
TEST_FRACTION = 0.2

RESULTS_DIR = Path(__file__).resolve().parent / 'results'

BATCH_SIZE = 64
SHUFFLE_BUFFER_SIZE = 10000
