import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow import keras
import argparse
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import os
import time

from data_generator import *
from utils import psnr
from model import *

if __name__ == "__main__":
    #tf.get_logger().setLevel('WARNING')

    DATA_DIR = "./data/"
    FILE_PATH = './models/generator'
    TRAIN_PATH = "train"
    TEST_PATH = "Set5"

    current_hour = time.strftime('%m%d%H', time.localtime(time.time()))
    result_dir = './result/{}/'.format(current_hour)
    if not os.path.isdir(result_dir):
        os.mkdir(result_dir)

    generator = Generator()
    # model = tf.keras.models.load_model('srcnn_{}x'.format(SCALE), custom_objects={'psnr': [psnr]})
    generator.load_weights('./models/generator')
    

    file_list = make_dataset('../NISR/test')
    for file_idx, file in enumerate(file_list):
        file_name = file.split('/')[-1].split('.')[0]
        print('\r{} / {} {}'.format(file_idx + 1, len(file_list), file_name), end='')
        raw_image = nib.load(file).get_fdata()
        clipped_image = clip_image(raw_image)
        im = clipped_image
        
        im_HR = im / im.max()
        im_LR = get_lr(im_HR)

        ni_img = nib.Nifti1Image(im_LR * im.max(), np.eye(4))
        nib.save(ni_img, '{}/{}_lr.nii.gz'.format(result_dir, file_name))
        
        slice_area = crop_slice(im_HR)
        im_HR_slice = im_HR[slice_area]
        im_LR_slice = im_LR[slice_area]

        im_LR_input = im_LR_slice[np.newaxis, :, :, :, np.newaxis]

        timer = time.time()

        im_SR = generator.predict(im_LR_input)[0, :, :, :, 0]

        print(time.time() - timer)

        im_SR = np.clip(im_SR, 0, 1)

        output_img = np.zeros(raw_image.shape)
        output_img[slice_area] = im_SR
        output_img = output_img * im.max()
        output_img[np.where(raw_image == 0)] = 0
        ni_img = nib.Nifti1Image(output_img, np.eye(4))
        nib.save(ni_img, '{}/{}_result.nii.gz'.format(result_dir, file_name))

        print(peak_signal_noise_ratio(im_HR, im_LR), peak_signal_noise_ratio(im_HR, output_img / im.max()))
        print(structural_similarity(im_HR, im_LR), structural_similarity(im_HR, output_img / im.max()))

        if file_idx == 0:
            break
