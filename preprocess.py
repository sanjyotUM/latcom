from datetime import datetime
from pathlib import Path

import tensorflow as tf
# import tensorflow_addons as tfa


def get_fname(path):
    """
    Extract filename from string ``path``

    Parameters
    ----------
    path : str
        Absolute file path

    Returns
    -------
    str

    """

    return path.split('\\')[-1]


def truncate_filenames(df):
    """
    Extracts filename from full paths in label dataframe

    Parameters
    ----------
    df : Pandas dataframe
        Dataframe with ``right``, ``left``, and ``center`` columns

    Returns
    -------
    Pandas dataframe

    """

    return (
        df
        .assign(
            right=lambda x: x['right'].apply(get_fname),
            left=lambda x: x['left'].apply(get_fname),
            center=lambda x: x['center'].apply(get_fname),
        )
    )


def remove_reverse_commands(df):
    """
    Remove reverse data labels

    Parameters
    ----------
    df : Pandas dataframe
        Label data with ``reverse`` column

    Returns
    -------
    Pandas dataframe

    """

    return df[df['reverse'] == 0.0]


def extract_datetime(fname):
    """
    Extract the datetime from the file name

    Parameters
    ----------
    fname : str
        File name encoding the datetime information

    Returns
    -------
    datetime

    """

    data = fname.split('.jpg')[0].split('_')[1:]
    data = [int(x) for x in data]
    return datetime(*data)


def downsample_zero_steering(df, ratio=0.9):
    """
    Remove ``ratio`` fraction of straight driving samples to reduce bias

    Parameters
    ----------
    df : pandas dataframe
        Dataframe with the column ``steering``

    ratio : float
        Percentage downsampling of straight driving data

    Returns
    -------
    Pandas dataframe

    """

    zero_label_index = df[df['steering'] == 0.0].sample(
        frac=1 - ratio,
        random_state=42).index  # '0.0' label downsampled index

    other_label_index = df[df['steering'] != 0.0].index  # All other labels

    keep_index = zero_label_index.append(other_label_index)

    df = df.loc[keep_index].sample(
        frac=1.0, random_state=42)  # Combine labels and shuffle
    return df


def preprocess_labels(df, downsample_ratio):
    """
    Preprocess filenames and downsample straight driving data

    Parameters
    ----------
    df : Pandas dataframe
        Label data

    downsample_ratio : float
        Percentage reduction in straight driving data

    Returns
    -------
    Pandas dataframe

    """
    df_new = (
        df
        .pipe(truncate_filenames)
        .pipe(remove_reverse_commands)
        .pipe(downsample_zero_steering, ratio=downsample_ratio)
    )
    # print(f'\nOriginal size: {len(df)} | New size: {len(df_new)}')
    # print(f'Size reduced to {round(len(df_new) * 100.0 / len(df), 2)}%')
    return df_new


def preprocess_image(img_path,
                     orig_shape=(210, 800),
                     final_shape=(40, 80)):
    """
    Read, crop and preprocess the image

    Parameters
    ----------
    img_path : str
        Absolute path of the jpg image

    orig_shape : tuple
        (height, width) of original image

    final_shape : tuple
        (height, width) of final image


    Returns
    -------
    Image tensor

    """

    height, width = orig_shape
    crop_y = 30
    rightside_width_cut = 700
    crop_window = tf.constant([crop_y, 0, height - crop_y, rightside_width_cut], dtype=tf.int32)

    if isinstance(img_path, Path):
        img_path = str(img_path)
    img = tf.io.read_file(img_path)  # Read
    return preprocess_opened_image(img, final_shape=final_shape, crop_window=crop_window)


def preprocess_opened_image(img, crop_window, final_shape):
    img = tf.io.decode_and_crop_jpeg(img, crop_window=crop_window)  # Decode and crop
    # img = tfa.image.gaussian_filter2d(img, filter_shape=(3, 3))  # Blur
    img = tf.image.resize(img, size=final_shape)  # Resize
    img = tf.divide(img, tf.constant(255.0))  # Normalize
    # img = tf.image.rgb_to_yuv(img)  # Convert to YUV space
    return img


if __name__ == '__main__':
    pass
