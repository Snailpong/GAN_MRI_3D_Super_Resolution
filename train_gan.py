from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow import keras
import argparse
import datetime
import time
import os
from tqdm import tqdm

from data_generator import *
from utils import psnr
from model import *


@tf.function
def train_step(batch_lr, batch_hr):

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        batch_es = generator(batch_lr, training=True)
        
        real_output = discriminator(batch_hr, training=True)
        fake_output = discriminator(batch_es, training=True)

        loss_gen = generator_loss(batch_hr, batch_es, fake_output)
        loss_disc = discriminator_loss(real_output, fake_output)

    grad_gen = gen_tape.gradient(loss_gen, generator.trainable_variables)
    grad_disc = disc_tape.gradient(loss_disc, discriminator.trainable_variables)

    optimizer_gen.apply_gradients(zip(grad_gen, generator.trainable_variables))
    optimizer_disc.apply_gradients(zip(grad_disc, discriminator.trainable_variables))

    return float(tf.reduce_mean(loss_gen)), float(tf.reduce_mean(loss_disc))


def test_step(batch_lr, batch_hr):
    batch_es = generator(batch_lr, training=False)
        
    real_output = discriminator(batch_hr, training=False)
    fake_output = discriminator(batch_es, training=False)

    loss_gen = generator_loss(batch_hr, batch_es, fake_output)
    loss_disc = discriminator_loss(real_output, fake_output)

    return tf.reduce_mean(loss_gen), tf.reduce_mean(loss_disc)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("N_TRAIN_DATA", type=int)
    parser.add_argument("N_TEST_DATA", type=int)
    parser.add_argument("BATCH_SIZE", type=int)
    parser.add_argument("EPOCHS", type=int)
    args = parser.parse_args()

    DATA_DIR = "./data/"
    FILE_PATH = "./models/model.hdf5"
    TRAIN_PATH = "train"
    TEST_PATH = "Set5"

    N_TRAIN_DATA = args.N_TRAIN_DATA
    N_TEST_DATA = args.N_TEST_DATA
    BATCH_SIZE = args.BATCH_SIZE
    EPOCHS = args.EPOCHS

    total_train = get_data('../NISR/train', N_TRAIN_DATA)
    total_test = get_data('../NISR/test', N_TEST_DATA)

    train_data_generator = train_data_generator_3d(total_train[0], total_train[1], BATCH_SIZE, N_TRAIN_DATA)
    test_data_generator = train_data_generator_3d(total_test[0], total_test[1], BATCH_SIZE, N_TEST_DATA)

    optimizer_gen = keras.optimizers.Adam(learning_rate=1e-4)
    optimizer_disc = keras.optimizers.Adam(learning_rate=1e-5)

    tf.keras.backend.clear_session()
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True

    generator = Generator()
    discriminator = Discriminator()

    generator.build((None, None, None, None, 1))
    discriminator.build((None, 60, 60, 60, 1))

    generator.summary()
    discriminator.summary()

    for epoch in range(EPOCHS):
        start = time.time()

        pbar = tqdm(range(50), ncols=100)

        for step in pbar:
            #print('.', end='', flush=True)
            batch_lr, batch_hr = next(train_data_generator)
            loss_gen, loss_disc = train_step(batch_lr, batch_hr)
            pbar.set_description_str('Epoch {}/{} '.format(epoch+1, EPOCHS))
            pbar.set_postfix_str('gen_loss: {}, disc_loss: {}'.format(round(float(loss_gen), 4), round(float(loss_disc), 4)))

        batch_lr, batch_hr = next(train_data_generator)
        loss_gen, loss_disc = test_step(batch_lr, batch_hr)

        generator.save_weights('./models/generator')
        discriminator.save_weights('./models/discriminator')

        print('Validation - gen_loss: {}, disc_loss: {}, time: {}'.format(round(float(loss_gen), 4), round(float(loss_disc), 4), time.time() - start))

        

    #model = keras.models.load_model('./srcnn_{}x'.format(SCALE), custom_objects={'psnr':psnr})


    # save_checkpoint = ModelCheckpoint(
    #     FILE_PATH,
    #     monitor="val_loss",
    #     verbose=1,
    #     save_best_only=True,
    #     save_weights_only=False,
    #     mode="auto",
    #     period=1,
    # )

    

    # model.compile(loss="mean_squared_error", optimizer=optimizer, metrics=[psnr])
    # # model.load_weights(FILE_PATH)

    # log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    # tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=10)


    # model.fit(
    #     train_data_generator,
    #     validation_data=(test_x, test_y),
    #     steps_per_epoch= N_TRAIN_DATA,
    #     epochs=EPOCHS,
    #     callbacks=[save_checkpoint, tensorboard_callback],
    # )


    # model.save('srcnn_{}x'.format(SCALE))
