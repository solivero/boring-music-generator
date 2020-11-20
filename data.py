import numpy as np
import os
from tensorflow.keras.utils import to_categorical 
from tensorflow.data import Dataset 
from params import sequence_length, batch_size, train_split

def read_file_as_chord_seq(filename):
    with open(f'{clt_dir}/{filename}') as file:
        lines = file.readlines()
        raw = [line[:-1].split('\t') for line in lines][:-1]
        table = np.zeros((len(raw), 6), dtype=np.float32)
        romans = np.empty(len(raw), dtype="U10")
        for i, row in enumerate(raw):
            # [start] [end] [Roman numeral] [chromatic root] [diatonic root] [key] [absolute root]
            table[i][0] = float(row[0])
            table[i][1] = float(row[1])
            romans[i] = row[2]
            table[i][2] = int(row[3])
            table[i][3] = int(row[4])
            table[i][4] = int(row[5])
            table[i][5] = int(row[6])
        chord_series = table[:, 3]
    return chord_series, romans

clt_dir = './rs200_harmony_table'
filenames = os.listdir(clt_dir)

sequences = [read_file_as_chord_seq(filename) for filename in filenames]
numeric_seqs, roman_seqs = zip(*sequences)
roman_sequence = np.concatenate(roman_seqs)
chords_used = np.unique(roman_sequence)
roman_to_numerical = { roman: i for i, roman in enumerate(chords_used) }
roman_as_num = np.array([roman_to_numerical[roman] for roman in roman_sequence])
unique, counts = np.unique(roman_as_num, return_counts=True)
common_chords, = np.where(counts > 100)
common_numerals = chords_used[common_chords]
vocab_size = len(common_numerals)
print(common_numerals)
common_mask = np.isin(roman_as_num, common_chords)
print("Common chords percent", len(common_mask.nonzero()[0]) / len(roman_as_num))
filtered_chords = np.where(common_mask, roman_as_num, 1) #TODO find cat of root chord. Never assume
filtered_romans = np.where(common_mask, roman_sequence, 'I')
chord_idx_map = { chord: i for i, chord in enumerate(common_chords)}
chord_seq = np.array([chord_idx_map[chord] for chord in filtered_chords])
og_seq = common_chords[chord_seq]

def roman_seq_to_chord_embeddings(roman_seq):
    roman_as_num = np.array([roman_to_numerical[roman] for roman in roman_seq])
    common_mask = np.isin(roman_as_num, common_chords)
    filtered_chords = np.where(common_mask, roman_as_num, 1)
    song_chord_seq = np.array([chord_idx_map[chord] for chord in filtered_chords])
    return to_categorical(song_chord_seq, num_classes=len(common_chords))

def make_dataset(filenames=filenames, batch_size=batch_size, train_split=train_split):
    chord_seqs = []
    target_chords = []
    for filename in filenames:
        numerical, roman_seq = read_file_as_chord_seq(filename)
        song_chords = roman_seq_to_chord_embeddings(roman_seq)
        if len(song_chords) - sequence_length <= 0:
            continue
        sequences = [song_chords[i:i+sequence_length] for i in range(len(song_chords)-sequence_length)]
        chord_seqs.append(sequences)
        targets = song_chords[sequence_length:]
        target_chords.append(targets)


    features = np.concatenate(chord_seqs, axis=0)
    labels = np.concatenate(target_chords, axis=0)
    print(features.shape, labels.shape)
    split_idx = int(train_split * len(labels))
    full_dataset = Dataset.from_tensor_slices((features, labels)).shuffle(500)

    train_dataset = full_dataset.take(split_idx) \
        .batch(batch_size, drop_remainder=True)

    val_dataset = full_dataset.skip(split_idx) \
        .batch(batch_size, drop_remainder=True)

    print(train_dataset.element_spec)
    print(len(train_dataset))
    return train_dataset, val_dataset