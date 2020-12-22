import os
import sys
from datetime import datetime
from pathlib import Path
import itertools

from PIL import Image
import numpy as np
import pandas as pd
import tensorflow as tf
import h5py
from preprocess import preprocess_image

ce491_path = os.environ['CE491_PATH']
sys.path.append('{}/neural_net'.format(ce491_path))
import config

conf = config.Config().config

init_height = conf['lc_init_img_height']
init_width = conf['lc_init_img_width']
fin_height = conf['lc_fin_img_height']
fin_width = conf['lc_fin_img_width']
crop_y_neural_net = conf['lc_crop_y_start']
rightside_width_cut = conf['lc_rightside_width_cut']

H5_IMAGE_INDEX_BOUND = {
    '2016-01-30--11-24-51.h5': {'lower': 4500, 'upper': 45800},
}


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

    # print(f'Training samples: {training_count}\nValidation samples: {validation_count}')
    return train, valid


class CommaData:
    """
    Handle creating training and validation comma.ai TF dataset creation.

    The function `get_train_valid_datasets` will return a 4 size tuple containing
    training and validation datasets as well as training and validation index over
    the original h5 data. The function will split the data into training
    and validation sets randomly. Every time the training data generator function
    is called by Tensorflow's `from_generator` function, it will shuffle the training
    data within itself.

    """

    def __init__(self, filepaths,
                 label_shift_steps=None,
                 start_index=None,
                 end_index=None,
                 img_shape=(160, 320, 3),
                 validation_split=0.15,
                 batch_size=64,
                 seed=42):
        """
        Initialize

        Parameters
        ----------
        filepaths : dict
            Dictionary containing file path objects for image and label data in
            `img_path` and `lbl_path` keys respectively

        label_shift_steps : int
            Number of steps to shift the image pointer array. Every positive
            1 step shifts the labels 10 ms in future (or effectively shifts
            images 10 ms in past)

        start_index : int
            Index from where to start reading data

        end_index : int
            Index at where to end reading data

        img_shape : tuple
            3 size tuple of image shape (channels first)

        validation_split: float
            Fraction of data to put into validation set

        batch_size : int
            Batch size of tf data

        seed : int
            Random seed
        """

        self.filepaths = filepaths
        self.steps = int(label_shift_steps) if label_shift_steps else None
        self.start_index = int(start_index) if start_index else None
        self.end_index = int(end_index) if end_index else None
        self.img_shape = img_shape
        self.validation_split = validation_split
        self.batch_size = batch_size
        self.seed = seed
        self.lbl_path = self.filepaths['lbl_path']
        self.img_path = self.filepaths['img_path']
        self.steerdt = self.load_h5_data(self.lbl_path, 'steering_angle')
        self.ptrdt = self.load_h5_data(self.lbl_path, 'cam1_ptr')
        self.orig_label_count = len(self.steerdt)
        self.steer_shrink = 1000.0  # To scale steering from -1 to 1
        self.index = None
        self.train_index = None
        self.valid_index = None
        self.generator_seed = 0

    def get_steering(self, raw_steer):
        return raw_steer / self.steer_shrink

    @staticmethod
    def get_image(raw_image):
        return np.moveaxis(raw_image, 0, -1) / 255.0

    @staticmethod
    def load_h5_data(h5path, dataset_name):
        with h5py.File(h5path, 'r') as f:
            data = f[dataset_name][:]
        return data

    def shift_pointer(self, ptr, fill_val=-1):
        ptr = np.roll(ptr, self.steps)
        if self.steps < 0:
            ptr[self.steps:] = fill_val
        elif self.steps > 0:
            ptr[:self.steps] = fill_val
        return ptr

    def print_info(self):
        pass
        # print(f'Data size: {len(self.index)}')
        # print(f'Training size: {len(self.train_index)}')
        # print(f'Validation size: {len(self.valid_index)}')
        # if self.steps:
        #     print(f'Shifting labels by {self.steps} steps ({self.steps * 10} ms)')

    def filter_index(self, index):
        """
        Filter index to only contain values between the lower and upper indexes of image data
        as specifed in `H5_IMAGE_INDEX_BOUND` dict. Also remove the data which has steering
        angle magnitude over 1000 since that data represents large turns at intersections.

        Parameters
        ----------
        index : Numpy array

        Returns
        -------
        Numpy array
            Filtered index

        """
        orig_count = len(index)
        image_lower_bound_index = H5_IMAGE_INDEX_BOUND[self.img_path.name]['lower']
        image_upper_bound_index = H5_IMAGE_INDEX_BOUND[self.img_path.name]['upper']
        steer_low_bound_index = np.where(self.ptrdt == image_lower_bound_index)[0].min()
        steer_upper_bound_index = np.where(self.ptrdt == image_upper_bound_index)[0].max()
        index = index[(index >= steer_low_bound_index)
                      & (index <= steer_upper_bound_index)]  # Low and high bounds on steering angle data index
        index = np.array([x for x in index
                          if not abs(self.steerdt[x]) > 1000])  # Filter out steering angles more than 1000
        final_count = len(index)
        # print(f'Labels filtered from {orig_count} to {final_count}')
        return index

    def create_index(self, shuffle=True):
        """
        Create index on steering h5 dataset which has been filtered

        Parameters
        ----------
        shuffle : bool
            Whether to shuffle the dataset or not

        Returns
        -------
        None
        """

        index = np.arange(0, self.orig_label_count)
        index = self.filter_index(index)  # Filter index to filter out garbage data
        np.random.seed(self.seed)  # Make the shuffle deterministic
        self.index = np.random.permutation(index) if shuffle else index

    def create_train_valid_index(self, shuffle=True):
        """
        Shuffle and split index into training and validation indices

        Returns
        -------
        None
        """
        assert self.index is not None

        index_size = len(self.index)
        validation_size = int(index_size * self.validation_split)
        self.valid_index = self.index[:validation_size]
        self.train_index = self.index[validation_size:]

    def make_gen(self, type):
        """
        Create a callable generator (image, steering) to make it compatible with TF `from_generator` call

        Parameters
        ----------
        type : str
            Whether 'train' or 'valid'

        Returns
        -------
        2 size tuple (image, steering)

        """

        assert self.train_index is not None
        assert self.valid_index is not None

        if type == 'train':
            self.generator_seed += 1
            np.random.seed(self.generator_seed)
            this_index = np.random.permutation(self.train_index)  # Shuffle training data within itself on each call
        elif type == 'valid':
            this_index = self.valid_index
        else:
            raise ValueError

        def dummy(lbl_path=self.lbl_path, img_path=self.img_path, index=this_index):
            with h5py.File(lbl_path, 'r') as lbl:
                with h5py.File(img_path, 'r') as img:

                    # h5 data
                    imagedt = img['X']
                    steerdt = lbl['steering_angle'][:]
                    ptrdt = lbl['cam1_ptr'][:]

                    # shift pointer to serve labels in future
                    if self.steps:
                        ptrdt = self.shift_pointer(ptrdt)

                    for i in index:
                        ptr = ptrdt[i].astype(int)

                        if ptr == -1:  # Ignore edge case data after shifting pointer
                            continue

                        steer = steerdt[i]
                        image = imagedt[ptr]
                        yield self.get_image(image), self.get_steering(steer)

        return dummy

    def make_dataset(self, generator):
        """
        Create TF dataset from callable generator

        Parameters
        ----------
        generator : callable function
            Returns (image, steering) generator when called

        Returns
        -------
        TF dataset

        """
        return tf.data.Dataset.from_generator(
            generator,
            output_types=(tf.float64, tf.float64),
            output_shapes=(tf.TensorShape(list(self.img_shape)), tf.TensorShape([])),
        ).prefetch(tf.data.experimental.AUTOTUNE).batch(self.batch_size)

    def get_train_valid_datasets(self):
        """
        Generate TF datasets for training and validation along with the index of training
        and validation datasets.

        Returns
        -------
        4 size tuple (train_dataset, valid_dataset, training_index, validation_index)

        """

        # create index, shuffle and split into train/valid
        self.create_index()
        self.create_train_valid_index()
        self.print_info()

        # create callable generators
        train_gen_func = self.make_gen('train')
        valid_gen_func = self.make_gen('valid')

        # create TF datasets
        train_dataset = self.make_dataset(train_gen_func)
        valid_dataset = self.make_dataset(valid_gen_func)
        return train_dataset, valid_dataset, self.train_index, self.valid_index

    def get_train_valid_datasets_from_indices(self, indices):
        """
        Generate TF datasets for training and validation from the given training and validation
        data index present in the `indices` dictionary

        Parameters
        ----------
        indices : dict
            Dictionary with keys `train` and `valid` containing respective indices

        Returns
        -------
        2 size tuple (train_dataset, valid_dataset)

        """
        self.train_index, self.valid_index = indices['train'], indices['valid']

        # create callable generators
        train_gen_func = self.make_gen('train')
        valid_gen_func = self.make_gen('valid')

        # create TF datasets
        train_dataset = self.make_dataset(train_gen_func)
        valid_dataset = self.make_dataset(valid_gen_func)
        return train_dataset, valid_dataset


class CE491data:
    def __init__(self, data_dir, summary_filename, split_info, time_shift=None, batch_size=128, shuffle_buffer_size=None, cache=False):
        self.data_dir = data_dir
        self.summary_filename = summary_filename
        self.split_info = split_info
        self.time_shift = int(time_shift)
        self.batch_size = batch_size
        self.shuffle_buffer_size = shuffle_buffer_size
        self.cache = cache
        self.train_dirs = None
        self.valid_dirs = None
        self.train_lbl = None
        self.valid_lbl = None
        self.summary = self.get_summary()

    @staticmethod
    def parse_date(dtstr):
        dtinfo = [int(x) for x in dtstr.strip('.jpg').split('-')]
        return datetime(*dtinfo)

    def get_summary(self):
        return (
            pd.read_csv(self.data_dir / 'summary.csv', parse_dates=['Start time', 'End time'])
                .assign(datetime=lambda x: x['Directory'].apply(self.parse_date))
                .assign(date=lambda x: x['datetime'] - pd.to_timedelta(x['datetime'].dt.time.astype(str)))
                .assign(
                    start_ts=lambda x: x['date'] + pd.to_timedelta(x['Start time'].dt.time.astype(str)),
                    end_ts=lambda x: x['date'] + pd.to_timedelta(x['End time'].dt.time.astype(str))
            )
        )

    def img_preprocess(self, *args, **kwargs):
        return preprocess_image(*args, **kwargs)

    def load_lbl_file(self, filepath):
        parent = str(filepath.parent)
        columns = ['filename', 'steering', 'throttle']
        lbl = (
            pd.read_csv(filepath, names=columns)
            .assign(datetime=lambda x: x['filename'].apply(self.parse_date))
            .assign(filename=lambda x: parent + '/' + x['filename'])
        )
        return lbl

    def shift_labels(self, df):

        """ Shift the labels `time_shift` milliseconds into the future """

        df = df.sort_values(by='datetime')
        left = df[['filename', 'datetime']]
        right = (
            df.assign(
                datetime_shift=lambda x: x['datetime'] - pd.Timedelta('{} ms'.format(self.time_shift))
            )
            .rename(columns={'datetime': 'future_datetime'})
            [['datetime_shift', 'future_datetime', 'steering', 'throttle']]
        )

        merged = (
            pd.merge_asof(left, right,
                          left_on='datetime',
                          right_on='datetime_shift',
                          tolerance=pd.Timedelta('20 ms'),
                          direction='nearest')
            .assign(delta=lambda x: x['future_datetime'] - x['datetime'])
            .assign(delta=lambda x: x['delta'].dt.seconds*1000.0 + x['delta'].dt.microseconds/1000.0)  # Milliseconds
        )
        print("Time shift executed: Mean={:.1f} Low={:.1f} High={:.1f}".format(merged['delta'].mean(),
                                                                               merged['delta'].min(),
                                                                               merged['delta'].max()))
        return merged.dropna()

    def filter_lbl_using_summary(self, lblpath, lbldf):
        directory, user = lblpath.parent.name, lblpath.parent.parent.name

        metadata = self.summary[(self.summary['User'] == user)
                                & (self.summary['Directory'] == directory)].iloc[0]

        start = metadata['start_ts'].to_pydatetime()
        end = metadata['end_ts'].to_pydatetime()
        filtered = lbldf[(lbldf['datetime'] >= start) & (lbldf['datetime'] < end)]
        return filtered

    def get_lbl_dfs(self):

        def parse_dirs(dir_list):
            return [self.data_dir.joinpath(x) for x in dir_list]

        def parse_lbl_filenames(dirs):
            return [list(x.glob('*.csv'))[0] for x in dirs]

        def parse_dirs_and_lbl(dir_list):
            dirs = parse_dirs(dir_list)
            lbl_filenames = parse_lbl_filenames(dirs)
            lbl_dfs = {x: self.load_lbl_file(x) for x in lbl_filenames}
            if self.time_shift:
                print('Shifting labels by {} ms.'.format(self.time_shift))
                lbl_dfs = {x: self.shift_labels(df) for x, df in lbl_dfs.items()}
            return dirs, lbl_dfs

        def filter_if_image_exists(df):
            def check_file_existence(row):
                return Path(row['filename']).exists()

            df['exists'] = df.apply(check_file_existence, axis=1)
            init_len = len(df)
            df = df[df['exists'] == True]
            fin_len = len(df)
            if fin_len < init_len:
                print('Images not found for %d labels. Skipping them.' % (init_len - fin_len))
            return df

        # Read label data, time shift labels if required
        self.train_dirs, train_lbl_dfs = parse_dirs_and_lbl(self.split_info['train'])
        self.valid_dirs, valid_lbl_dfs = parse_dirs_and_lbl(self.split_info['valid'])

        # Filter data according to the summary file description and combine into one dataframe
        train_lbl_df = pd.concat([
            self.filter_lbl_using_summary(path, df)
            for path, df in train_lbl_dfs.items()
        ])
        valid_lbl_df = pd.concat([
            self.filter_lbl_using_summary(path, df)
            for path, df in valid_lbl_dfs.items()
        ])

        # Remove data if image does not exist for the label
        train_lbl_df = filter_if_image_exists(train_lbl_df)[['filename', 'steering']]
        valid_lbl_df = filter_if_image_exists(valid_lbl_df)[['filename', 'steering']]
        return train_lbl_df, valid_lbl_df

    def get_train_valid_datasets(self):
        trainlbl, validlbl = self.get_lbl_dfs()

        def get_dataset_from_pandas(df, datatype, cache=self.cache):
            x = tf.data.Dataset.from_tensor_slices(df['filename'])
            y = tf.data.Dataset.from_tensor_slices(df['steering'])
            x = x.map(
                self.img_preprocess,
                num_parallel_calls=4
            )
            ds = tf.data.Dataset.zip((x, y))

            if cache:
                ds = ds.cache()

            if datatype == 'train':
                ds = ds.shuffle(
                    len(df) if self.shuffle_buffer_size is None else self.shuffle_buffer_size,
                    reshuffle_each_iteration=True
                )

            ds = ds.prefetch(4).batch(self.batch_size)
            return ds

        train = get_dataset_from_pandas(trainlbl, datatype='train')
        valid = get_dataset_from_pandas(validlbl, datatype='valid')
        return train, valid, trainlbl, validlbl

    def get_train_valid_generators(self):

        final_shape = (fin_height, fin_width)
        h, w = final_shape

        def load_img(filename, orig_shape=(init_height, init_width), final_shape=final_shape):
            height, width = orig_shape
            crop_y = crop_y_neural_net
            box = (0, crop_y, rightside_width_cut, height)
            return np.asarray(Image.open(filename).crop(box).resize(reversed(final_shape))) / 255.0

        def get_generator_from_df(df, batch_size, training_type):
            x, y = df['filename'].values, df['steering'].values
            if training_type == 'training':
                iterator = itertools.cycle(range(0, len(x), batch_size))
            else:
                iterator = range(0, len(x), batch_size)

            for gi in iterator:
                adaptive_batch_size = min(batch_size, len(x) - gi)
                x_batch = np.zeros((adaptive_batch_size, h, w, 3))
                y_batch = np.zeros((adaptive_batch_size, 1))

                for i in range(adaptive_batch_size):
                    x_batch[i, :] = load_img(x[gi + i])
                    y_batch[i, :] = y[gi + i]
                yield x_batch, y_batch

        traindf, validdf = self.get_lbl_dfs()
        print('Training data count: %d\nValidation data count: %d' % (len(traindf), len(validdf)))
        train_gen = get_generator_from_df(traindf.sample(frac=1), self.batch_size, training_type='training')
        valid_gen = get_generator_from_df(validdf, self.batch_size, training_type='validation')
        return train_gen, valid_gen, traindf, validdf


if __name__ == '__main__':
    pass
