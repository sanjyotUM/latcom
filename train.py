from pathlib import Path
import argparse

import tensorflow as tf
from data import CommaData
from models import load_nvidia
from utils import get_episode_name, save_indices, save_history

parser = argparse.ArgumentParser(description='Train model and record performance.')

# Data params
parser.add_argument('--h5file', help='H5 file name', default='2016-01-30--11-24-51.h5', type=str)
parser.add_argument('--timeshift', help='Number of milliseconds to shift labels in future', default=None, type=int)
parser.add_argument('--start-index', help='Index to start reading data from', default=0, type=int)
parser.add_argument('--end-index', help='Index to start reading data from', type=int)
parser.add_argument('--batch-size', help='Training batch size', default=128, type=int)

# Model params
parser.add_argument('--learning-rate', help='Training learning rate', default=1e-4, type=float)

# Training params
parser.add_argument('--epochs', help='Training epochs', default=5, type=int)
parser.add_argument('--workers', help='Number of threads creating batches', default=10, type=int)
parser.add_argument('--queue-size', help='Maximum queue size of batches to create', default=20, type=int)
parser.add_argument('--cpu-threads', help='If using cpu, how many threads to use', default=None, type=int)

args = parser.parse_args()

if args.timeshift:
    args.timeshift = args.timeshift // 10  # Convert milliseconds to steps

if args.cpu_threads:
    tf.config.threading.set_intra_op_parallelism_threads(args.cpu_threads)
    tf.config.threading.set_inter_op_parallelism_threads(args.cpu_threads)

project_dir = Path.cwd()
lbl_dir = project_dir / 'dataset' / 'raw' / 'log'
img_dir = project_dir / 'dataset' / 'raw' / 'camera'
img_path = img_dir / args.h5file
lbl_path = lbl_dir / args.h5file

model_dir = project_dir / 'models'
result_dir = project_dir / 'results'
model_dir.mkdir(exist_ok=True)
result_dir.mkdir(exist_ok=True)

episode_name = get_episode_name(args)
model_name = episode_name + '.h5'
model_path = model_dir / model_name
episode_dir = result_dir / episode_name
episode_dir.mkdir(exist_ok=True)

filepaths = {
    'lbl_path': lbl_path,
    'img_path': img_path
}

comma = CommaData(
    filepaths=filepaths,
    label_shift_steps=args.timeshift,
    start_index=args.start_index,
    end_index=args.end_index,
    img_shape=(160, 320, 3),
    validation_split=0.15,
    batch_size=args.batch_size
)

train, valid, train_index, valid_index = comma.get_train_valid_datasets()
save_indices(train_index, valid_index, episode_dir)

nvidia = load_nvidia(
    learning_rate=args.learning_rate,
    input_shape=(160, 320, 3)
)

log_dir = project_dir / 'logs' / episode_name
log_dir.mkdir(parents=True, exist_ok=True)
tensorboard_callback = tf.keras.callbacks.TensorBoard(
    log_dir=str(log_dir),
    histogram_freq=1,
    update_freq='batch',
    write_images=True,
)
print(f'\nLog dir: {log_dir}\n')

history = nvidia.fit(
    train,
    epochs=args.epochs,
    validation_data=valid,
    workers=args.workers,
    max_queue_size=args.queue_size,
    callbacks=[tensorboard_callback]
)

nvidia.save(str(model_path))
print(f'Model saved at: {model_path}')
save_history(history.history, episode_dir)
