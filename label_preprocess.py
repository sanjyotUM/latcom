def get_fname(path):
    """ Extract filename from string ``path``"""

    return path.split('\\')[-1]


def truncate_filenames(df):
    """ Extracts filename from full paths in label dataframe """

    return (
        df
        .assign(
            right = lambda x: x['right'].apply(get_fname),
            left = lambda x: x['left'].apply(get_fname),
            center = lambda x: x['center'].apply(get_fname),
        )
    )


def remove_reverse_commands(df):
    """ Remove reverse commands """

    return df[df['reverse'] == 0.0]


def downsample_zero_steering(df, ratio=0.9):
    """ Remove ``ratio`` fraction of straight driving samples to reduce bias """

    zero_label_index = df[df['steering'] == 0.0].sample(
        frac=1 - ratio,
        random_state=42).index  # '0.0' label downsampled index

    other_label_index = df[df['steering'] != 0.0].index  # All other labels

    keep_index = zero_label_index.append(other_label_index)

    df = df.loc[keep_index].sample(
        frac=1.0, random_state=42)  # Combine labels and shuffle
    return df


def preprocess_labels(df, downsample_ratio):
    """ Clean and choose the labels important """
    df_new = (
        df
        .pipe(truncate_filenames)
        .pipe(remove_reverse_commands)
        .pipe(downsample_zero_steering, ratio=downsample_ratio)
    )
    print(f'\nOriginal size: {len(df)} | New size: {len(df_new)}')
    print(f'Size reduced to {round(len(df_new) * 100.0 / len(df), 2)}%')
    return df_new


if __name__ == '__main__':
    pass


