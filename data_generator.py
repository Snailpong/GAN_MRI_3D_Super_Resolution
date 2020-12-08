from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array
import numpy as np
import os
from scipy.ndimage import zoom
import nibabel as nib


def make_dataset(dir):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir
    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if fname.endswith('.nii.gz'):
                path = os.path.join(root, fname)
                images.append(path)
    return images


def clip_image(im):
    clip_value = np.sort(im.ravel())[int(np.prod(im.shape) * 0.999)]
    im = np.clip(im, 0, clip_value)
    return im


def get_lr(im):
    imgfft = np.fft.fftn(im)

    x_area = y_area = z_area = 50

    x_center = imgfft.shape[0] // 2
    y_center = imgfft.shape[1] // 2
    z_center = imgfft.shape[2] // 2

    imgfft_shift = np.fft.fftshift(imgfft)
    imgfft_shift2 = imgfft_shift.copy()

    imgfft_shift[x_center-x_area : x_center+x_area, y_center-y_area : y_center+y_area, z_center-z_area : z_center+z_area] = 0
    imgfft_shift2 = imgfft_shift2 - imgfft_shift

    imgifft3 = np.fft.ifftn(imgfft_shift2)
    lr = abs(imgifft3)
    return lr


def crop_slice(array):
    for i in range(array.shape[0]):
        if not np.all(array[i, :, :] == 0):
            x_use1 = i
            break
    for i in reversed(range(array.shape[0])):
        if not np.all(array[i, :, :] == 0):
            x_use2 = i 
            break
    for i in range(array.shape[1]):
        if not np.all(array[:, i, :] == 0):
            y_use1 = i
            break
    for i in reversed(range(array.shape[1])):
        if not np.all(array[:, i, :] == 0):
            y_use2 = i
            break
    for i in range(array.shape[2]):
        if not np.all(array[:, :, i] == 0):
            z_use1 = i
            break
    for i in reversed(range(array.shape[2])):
        if not np.all(array[:, :, i] == 0):
            z_use2 = i
            break

    area = (slice(x_use1, x_use2), slice(y_use1, y_use2), slice(z_use1, z_use2))
    return area


def get_data(data_dir, max_files):
    images = make_dataset(data_dir)
    image_list_hr = []
    image_list_lr = []
    for file_idx, file_name in enumerate(images):
        print('\r{} / {}'.format(file_idx + 1, len(images)), end='')
        raw_image = nib.load(file_name).get_fdata().astype('float32')
        clipped_image = clip_image(raw_image)
        im = clipped_image

        im_HR = im / im.max()
        im_LR = get_lr(im_HR)

        slice_area = crop_slice(im_HR)
        im_HR_slice = im_HR[slice_area]
        im_LR_slice = im_LR[slice_area]

        image_list_hr.append(im_HR_slice)
        image_list_lr.append(im_LR_slice)

        if file_idx == max_files - 1:
            break

    print()
    return image_list_hr, image_list_lr


def train_data_generator_3d(image_list_hr, image_list_lr, batch_size, data_size):
    i = 0
    while True:
        batch_hr = []
        batch_lr = []
        while len(batch_lr) != batch_size:
            xa = np.random.randint(image_list_hr[i].shape[0] - 60)
            ya = np.random.randint(image_list_hr[i].shape[1] - 60)
            za = np.random.randint(image_list_hr[i].shape[2] - 60)

            if image_list_hr[i][xa+30, ya+30, za+30] != 0:
                batch_hr.append(image_list_hr[i][xa:xa+60, ya:ya+60, za:za+60, np.newaxis])
                batch_lr.append(image_list_lr[i][xa:xa+60, ya:ya+60, za:za+60, np.newaxis])
                i = (i + 1) % data_size

        yield np.array(batch_lr), np.array(batch_hr)
