import os.path

_DATASETS_FOLDER_NAME = 'datasets'
_MODELS_FOLDER_NAME = 'models'
_IMAGES_FOLDER_NAME = 'images'
_ENCODINGS_FOLDER_NAME = 'encodings'

ROOT_DIR = os.path.realpath(os.path.join(__file__, '..'))

DATASETS_DIR = os.path.join(ROOT_DIR, _DATASETS_FOLDER_NAME)
MODELS_DIR = os.path.join(ROOT_DIR, _MODELS_FOLDER_NAME)
IMAGES_DIR = os.path.join(ROOT_DIR, _IMAGES_FOLDER_NAME)
ENCODINGS_DIR = os.path.join(ROOT_DIR, _ENCODINGS_FOLDER_NAME)

LABELS_DICT = {
    8: 'learnpython',
    3: 'Python',
    2: 'ProgrammerHumor',
    1: 'MachineLearning',
    6: 'datascience',
    7: 'learnmachinelearning',
    0: 'BusinessIntelligence',
    4: 'bioinformatics',
    5: 'dataengineering'
}