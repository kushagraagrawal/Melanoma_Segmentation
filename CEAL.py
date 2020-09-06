from __future__ import print_function
from keras.callbacks import ModelCheckpoint
from data import load_train_data
from utils import *

create_paths()
log_file = open(global_path + "logs/log_file.txt", 'a')

X_train, y_train = load_train_data()
labeled_index = np.arange(0, nb_labeled)
unlabeled_index = np.arange(nb_labeled, len(X_train))

model = get_unet(dropout=True)