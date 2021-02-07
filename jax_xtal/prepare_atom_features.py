import os
import json

import pandas as pd
import numpy as np
from mendeleev import get_table, element
from sklearn.preprocessing import KBinsDiscretizer, OneHotEncoder


def get_group(el):
    number = el.atomic_number
    if (number >= 57) and (number <= 71):
        return 3
    elif (number >= 89) and (number <= 103):
        return 3
    else:
        return el.group_id


def get_period(el):
    number = el.atomic_number
    if (number >= 57) and (number <= 71):
        return 8
    elif (number >= 89) and (number <= 103):
        return 9
    else:
        return el.period


def create_dataframe():
    all_data = {}
    for number in range(1, 118 + 1):
        el = element(number)
        data = {
            # Group number
            'group': get_group(el),
            # Period number
            'period': get_period(el),
            # Pauling's electronegativity
            'electronegativity': el.en_pauling,
            # Cordero's covalent radius
            'covalent_radius': el.covalent_radius_cordero,
            # valence electrons
            'valence_electrons': el.nvalence(),
            # first ionization energy
            'first_ionization_energy': el.ionenergies.get(1),
            # electron affinity
            'electron_affinity': el.electron_affinity,
            # block
            'block': el.block,
            # atomic volume
            'atomic_volume': el.atomic_volume,
        }
        all_data[number] = data
    df = pd.DataFrame(all_data).transpose()

    # fill with mean
    df = df.fillna(df.mean())

    # log scale
    df['first_ionization_energy'] = np.log(df['first_ionization_energy'])
    df['atomic_volume'] = np.log(df['atomic_volume'])

    return df


def encode(df):
    label_cols = ['group', 'period', 'valence_electrons', 'block']
    numeric_cols = ['electronegativity', 'covalent_radius', 'first_ionization_energy', 'electron_affinity', 'atomic_volume']
    numeric_col_bins = [10, 10, 10, 10, 10]

    encoded = []

    for col in label_cols:
        encoder = OneHotEncoder(sparse=False)
        transformed = encoder.fit_transform(df[col].to_numpy()[:, None])
        print(col, transformed.shape)
        encoded.append(transformed)

    for col, bins in zip(numeric_cols, numeric_col_bins):
        encoder = KBinsDiscretizer(n_bins=bins, encode='onehot-dense')
        transformed = encoder.fit_transform(df[col].to_numpy()[:, None])
        print(col, transformed.shape)
        encoded.append(transformed)

    onehotted = np.hstack(encoded).astype(int)
    dataset = {}
    for i, arr in enumerate(onehotted):
        dataset[i + 1] = list(map(int, list(arr)))
    return dataset


if __name__ == '__main__':
    df = create_dataframe()
    dataset = encode(df)

    output = os.path.join('data', 'atom_init.json')
    with open(output, 'w') as f:
        json.dump(dataset, f, indent=2)
