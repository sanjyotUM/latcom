import pickle
import h5py
import numpy as np
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm


def print_summary(args):
    print('-' * 40)
    print('TRAINING SUMMARY')
    print('-' * 40)
    print(f'H5 file: {args.h5file}')
    print(f'Time shift: {args.timeshift if args.timeshift else 0}')
    print(f'Start index: {args.start_index}')
    print(f'End index: {args.end_index}')
    print(f'Batch size: {args.batch_size}')
    print(f'Learning rate: {args.learning_rate}')
    print(f'Epochs: {args.epochs}')
    print(f'Workers: {args.workers}')
    print(f'Queue size: {args.queue_size}')
    print('-' * 40)


def get_episode_name(args, summary=True):
    if summary:
        print_summary(args)

    # TODO: Include time part of h5file name in p1
    p1 = args.h5file[:10]
    p2 = f't{args.timeshift if args.timeshift else 0}'
    p3 = f's{args.start_index}'
    p4 = f'e{args.end_index}'
    p5 = f'b{args.batch_size}'
    p6 = f'lr{args.learning_rate}'
    p7 = f'ep{args.epochs}'
    p8 = f'w{args.workers}'
    p9 = f'q{args.queue_size}'
    return '_'.join([p1, p2, p3, p4, p5, p6, p7, p8, p9])


def save_indices(train_index, valid_index, episode_dir):
    indices = {
        'train': train_index,
        'valid': valid_index
    }
    index_file = str(episode_dir / 'indices.pkl')
    with open(index_file, 'wb') as f:
        pickle.dump(indices, f)
    print(f'Indices saved at: {index_file}')


def save_history(history, episode_dir):
    hist_file = str(episode_dir / 'history.pkl')
    with open(hist_file, 'wb') as f:
        pickle.dump(history, f)
    print(f'History saved at: {hist_file}')


def profile_images(img_h5_path, lbl_h5_path, dest_dir, skip=10, pack=50, show=False):
    import gc
    save_dir = dest_dir / img_h5_path.name
    save_dir.mkdir(parents=True, exist_ok=True)

    with h5py.File(img_h5_path, 'r') as img:
        with h5py.File(lbl_h5_path, 'r') as lbl:
            x = img['X']
            steerdt = lbl['steering_angle'][:]
            ptrdt = lbl['cam1_ptr'][:]

            size = len(x)

            skip_index = [(img_ix, steerdt[np.where(ptrdt == img_ix)]) for img_ix in range(0, size, skip)]
            img_pack_index = [skip_index[i: i + pack] for i in range(0, len(skip_index), pack)]
            print(f'Generating {len(img_pack_index)} images')
            print(f'Saving at {save_dir}')

            for pack_ix, pack in enumerate(tqdm((img_pack_index))):
                fname = save_dir / f'{str(pack_ix)}.jpg'
                if fname.exists():
                    continue
                fig, ax = plt.subplots(10, 5, figsize=(21, 18), sharex=True, sharey=True)
                ax = ax.flatten()

                for img_no, data in enumerate(pack):
                    img_ix, steer = data
                    steer = np.mean(steer)
                    img = np.moveaxis(x[img_ix], 0, -1)
                    ax[img_no].imshow(img)
                    ax[img_no].set_title(img_ix)
                    ax[img_no].tick_params(which='both', bottom=False, labelbottom=False, left=False, labelleft=False)
                    ax[img_no].annotate(f'Act: {round(steer, 1)}', (100, 130), color='white', size=14,
                                        family='monospace', )
                if show:
                    plt.show()
                    break
                plt.savefig(fname, bbox_inches='tight', pad_inches=0.5)
                plt.cla()
                plt.clf()
                plt.close('all')
                plt.close(fig)
                gc.collect()
