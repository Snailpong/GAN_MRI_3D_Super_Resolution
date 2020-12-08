from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow import keras
import argparse
import datetime

from data_generator import *
from utils import psnr
from model import *

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("N_TRAIN_DATA", type=int)
    parser.add_argument("N_TEST_DATA", type=int)
    parser.add_argument("BATCH_SIZE", type=int)
    parser.add_argument("EPOCHS", type=int)
    args = parser.parse_args()


    SCALE = 4 
    DATA_DIR = "./data/"
    FILE_PATH = "./models/srcnn_{}x.hdf5".format(SCALE)
    TRAIN_PATH = "train"
    TEST_PATH = "Set5"

    N_TRAIN_DATA = args.N_TRAIN_DATA
    N_TEST_DATA = args.N_TEST_DATA
    BATCH_SIZE = args.BATCH_SIZE
    EPOCHS = args.EPOCHS

    total_train = get_data('../NISR/train', SCALE, N_TRAIN_DATA, True)
    total_test = get_data('../NISR/test', SCALE, N_TEST_DATA, True)

    train_data_generator = train_data_generator_3d(total_train[0], total_train[1], BATCH_SIZE, N_TRAIN_DATA)

    test_x, test_y = next(
        train_data_generator_3d(total_test[0], total_test[1], BATCH_SIZE, N_TEST_DATA)
    )

    mirrored_strategy = tf.distribute.MirroredStrategy()
    with mirrored_strategy.scope():

        model = Generator()
        model.build((1, 60, 60, 60, 1))
        #model = keras.models.load_model('./srcnn_{}x'.format(SCALE), custom_objects={'psnr':psnr})
    
        model.summary()

        save_checkpoint = ModelCheckpoint(
            FILE_PATH,
            monitor="val_loss",
            verbose=1,
            save_best_only=True,
            save_weights_only=False,
            mode="auto",
            period=1,
        )

        optimizer = keras.optimizers.Adam(learning_rate=1e-4)

        model.compile(loss="mean_squared_error", optimizer=optimizer, metrics=[psnr])
        # model.load_weights(FILE_PATH)

        log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=10)


        model.fit(
            train_data_generator,
            validation_data=(test_x, test_y),
            steps_per_epoch= N_TRAIN_DATA,
            epochs=EPOCHS,
            callbacks=[save_checkpoint, tensorboard_callback],
        )


    model.save('srcnn_{}x'.format(SCALE))
