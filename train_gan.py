from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow import keras
import argparse
import datetime
import time

from data_generator import *
from utils import psnr
from model import *


@tf.function
def train_step(lr_batch, hr_batch):

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(lr_batch, training=True)
        
        real_output = discriminator(hr_batch, training=True)
        fake_output = discriminator(generated_images, training=True)

        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)

    grad_disc = gen_tape.gradient(gen_loss, generator.trainable_variables)
    grad_gen = gen_tape.gradient(gen_loss, generator.trainable_variables)

    generator_optimizer.apply_gradients(zip(grad_gen, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(grad_disc, generator.trainable_variables))


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

    optimizer_gen = keras.optimizers.Adam(learning_rate=1e-4)
    optimizer_disc = keras.optimizers.Adam(learning_rate=1e-4)


    generator = Generator()
    discriminator = Discriminator()

    generator.build((None, None, None, None, 1))
    discriminator.build((None, None, None, None, 1))

    for epoch in range(EPOCHS):
        start = time.time()

        for step in range(100):

            train_step()

    #model = keras.models.load_model('./srcnn_{}x'.format(SCALE), custom_objects={'psnr':psnr})


    save_checkpoint = ModelCheckpoint(
        FILE_PATH,
        monitor="val_loss",
        verbose=1,
        save_best_only=True,
        save_weights_only=False,
        mode="auto",
        period=1,
    )

    

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
