# ---------------
# All Labels
# ---------------
BUSINESS = 0
ENTERTAINMENT = 1
POLITICS = 2
SPORT = 3
TECH = 4
NO_LABEL = -1

DEFAULT_PROBABILITY = 0.2
PROBABILITY_THRESHOLD = 0.65
MAX_ITERATIONS = 2

LABEL_DICT = {
    0: "business",
    1: "entertainment",
    2: "politics",
    3: "sport",
    4: "tech",
    -1: "no_label"
}

LABEL_REV_DICT = {
    "business": 0,
    "entertainment": 1,
    "politics": 2,
    "sport": 3,
    "tech": 4,
    "no_label": -1
}

# ---------------
# Document counts
# ---------------
TRAIN_LABELLED_DOCUMENTS_FRACTION = 0.8
TRAIN_UNLABELLED_DOCUMENTS_FRACTION = 0.2

# ---------------
# All Paths
# ---------------
DATA_DIR = 'data/'
BBC_DATA_DIR = 'data/bbc/'
TEST_DATA_DIR = 'data/test/'
TRAIN_DATA_DIR = 'data/train/'
LABELLED_TRAIN_DATA_DIR = 'data/train/labelled/'
LABEL_DIR_NAMES = ["business", "entertainment", "politics", "sport", "tech"]
UNLABELLED_TRAIN_DATA_DIR = 'data/train/unlabelled/'

# ---------------
# CODE DECISION
# ---------------
CREATE_FRESH_DATA = True
TAKE_HUMAN_HELP = False
TEXT_FILE_ENCODING = "windows-1252"
