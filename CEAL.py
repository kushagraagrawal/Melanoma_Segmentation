from __future__ import print_function
import tensorflow as tf
# from keras.callbacks import ModelCheckpoint
from data import load_train_data
from utils import *
import os

create_paths()
log_file = open(global_path + "logs/log_file.txt", 'a')

X_train, y_train = load_train_data()
labeled_index = np.arange(0, nb_labeled)
unlabeled_index = np.arange(nb_labeled, len(X_train))

model = get_unet(dropout=True)
if os.path.exists(initial_weights_path):
    model.load_weights(initial_weights_path)

if initial_train:
    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(initial_weights_path, monitor='loss', save_best_only=True)

    if apply_augmentation:
        for initial_epoch in range(0, nb_initial_epochs):
            history = model.fit_generator(
                data_generator().flow(X_train[labeled_index], y_train[labeled_index], batch_size=32, shuffle=True),
                steps_per_epoch=len(labeled_index), epochs=1, verbose=1, callbacks=[model_checkpoint]
            )
        
            model.save(initial_weights_path)
            log(history, initial_epoch, log_file)

    else:
        history = model.fit(X_train[labeled_index], y_train[labeled_index], batch_size=32, epochs=nb_initial_epochs, verbose=1, shuffle=True, callbacks=[model_checkpoint])

        log(history, 0, log_file)
else:
    model.load_weights(initial_weights_path)

model_checkpoint = tf.keras.callbacks.ModelCheckpoint(final_weights_path, monitor='loss', save_best_only=True)

for iteration in range(1, nb_iterations+1):
    if iteration == 1:
        weights = initial_weights_path
    else:
        weights = final_weights_path
    
    X_labeled_train, y_labeled_train, labeled_index, unlabeled_index = compute_train_sets(X_train, y_train, labeled_index, unlabeled_index, weights, iteration)

    history = model.fit(X_labeled_train, y_labeled_train, batch_size=32, epochs=nb_active_epochs, verbose=1, shuffle=True, callbacks=[model_checkpoint])

    log(history, iteration, log_file)
    model.save(global_path + "models/active_model" + str(iteration) + ".h5")

log_file.close()