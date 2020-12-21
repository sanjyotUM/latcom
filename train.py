import argparse
import cPickle as pickle
import json
from pathlib import Path
from datetime import datetime, timedelta
import random

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
tf.get_logger().setLevel('ERROR')

from preprocess import preprocess_image
from data import CE491data
from models import load_nvidia

parser = argparse.ArgumentParser(description='Train model and record performance.')
parser.add_argument('--epochs', help='Number of epochs to run training', default=3, type=int)
parser.add_argument('--batch_size', help='Number of records in a mini batch', default=64, type=int)
parser.add_argument('--time_shift', help='Number of milliseconds to shift labels in future', default=0, type=int)
parser.add_argument('--lr', help='Learning rate', default=1e-4, type=float)
parser.add_argument('--height', help='Input image height in pixels', default=40, type=int)
parser.add_argument('--width', help='Input image width in pixels', default=80, type=int)

args = parser.parse_args()

project_dir = Path.cwd()
data_dir = project_dir / 'dataset'
model_dir = project_dir / 'models'
if not model_dir.exists():
    model_dir.mkdir()

summary_filename = 'summary.csv'
split_info = {
    'train': [
        'yild5350/2019-12-13-14-54-15',
        'yild5350/2019-12-14-17-01-33',
        # 'Jaku6779/2019-12-07-18-14-00',
    ],
    'valid': [
        'Jaku6779/2019-12-16-19-31-06',
        'yild5350/2019-12-15-17-13-05',
        # 'jack4815/2019-12-15-21-09-25',
    ]
}

ce = CE491data(
    data_dir=data_dir,
    summary_filename=summary_filename,
    time_shift=args.time_shift,
    batch_size=args.batch_size,
    split_info=split_info,
    cache=True
)
train_gen, valid_gen, t, v = ce.get_train_valid_generators()

nvidia = load_nvidia(input_shape=(args.height, args.width, 3), learning_rate=args.lr)
nvidia.summary()

history = nvidia.fit_generator(
    train_gen,
    epochs=args.epochs,
    steps_per_epoch=np.ceil(len(t)/float(args.batch_size)),
    verbose=1,
    validation_data=valid_gen,
    max_queue_size=20,
    workers=8,
    validation_steps=np.ceil(len(v)/float(args.batch_size)),
    # callbacks=[tensorboard_callback],
    use_multiprocessing=True
)

model_name = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
results_dir = project_dir / 'models' / model_name
results_dir.mkdir()
weight_path = results_dir / 'weights.h5'
model_config_path = results_dir / 'config.json'
history_path = results_dir / 'history.pkl'
params_path = results_dir / 'params.json'

# Save all files
nvidia.save_weights(weight_path)
json_config = nvidia.to_json()
with open(str(model_config_path), 'w') as json_file:
    json_file.write(json_config)
with open(str(history_path), 'wb') as f:
    pickle.dump(history, f)
with open(str(params_path), 'w') as json_file:
    json.dump(vars(args), json_file)

print('Model saved at: {}'.format(results_dir))
