import tensorflow as tf
from preprocess import preprocess_image


def construct_dataset(label_data,
                      data_dir,
                      batch_size=128,
                      validation_split_ratio=0.2):
    """
    Construct training and validation tf.data.Dataset

    Parameters
    ----------
    label_data : Pandas dataframe
        Dataframe with ``center`` and ``steering`` columns.

    data_dir : Pathlib path
        Path to data directory

    batch_size : int

    validation_split_ratio : float
        Fraction of data to put into validation dataset

    Returns
    -------
    Tuple of training and validation tf dataset

    """

    x_paths = list(str(data_dir / 'IMG') + '/' + label_data['center'])  # Center image paths
    y = label_data['steering'].values  # Corresponding steering angles
    dataset_size = len(x_paths)

    x_paths = tf.data.Dataset.from_tensor_slices(x_paths)
    y = tf.data.Dataset.from_tensor_slices(y)

    # Add image preprocessing to pipeline
    X = x_paths.map(preprocess_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    tfdata = tf.data.Dataset.zip((X, y))
    tfdata = tfdata.prefetch(tf.data.experimental.AUTOTUNE)  # Parallelize data processing and training

    validation_count = int(validation_split_ratio * dataset_size)
    training_count = dataset_size - validation_count

    valid = tfdata.take(validation_count).batch(batch_size)
    train = tfdata.skip(validation_count).batch(batch_size)

    print(f'Training samples: {training_count}\nValidation samples: {validation_count}')
    return train, valid


if __name__ == '__main__':
    pass
