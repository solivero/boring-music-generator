import numpy as np
import data
from model import make_model
from params import sequence_length
from tensorflow.keras.models import load_model

def predict_chords(model, warmup_chords_cat, num_chords=16):
    warmup_batch = np.expand_dims(warmup_chords_cat, 0)
    # warmup_chords = np.argmax(warmup_batch, axis=-1)[0]
    sequence = np.copy(warmup_batch)
    pred_chords = []
    model.reset_states()
    for i in range(num_chords):
        prediction = model.predict(warmup_batch)[0]
        pred_chord = np.random.choice(len(prediction), p=prediction)
        pred_chords.append(pred_chord)
        sequence[:, :-1] = sequence[:, 1:]
        sequence[:, -1] = prediction[0]
    return pred_chords


from music21 import stream, duration, note, instrument, roman
from datetime import datetime

def make_score(roman_numerals):
    roman_chords = map(roman.RomanNumeral, roman_numerals)
    chords_part = stream.Part()
    chords_part.insert(0, instrument.Piano())
    bass_part = stream.Part()
    bass_part.insert(0, instrument.Bass())
    for roman_chord in roman_chords:
        roman_chord.duration = duration.Duration(8)
        chords_part.append(roman_chord)
        bass_pitch = roman_chord.bass().transpose(-12)
        bass_note = note.Note(bass_pitch)
        bass_note.duration = duration.Duration(4)
        bass_part.append(bass_note)
        bass_note = note.Note(bass_pitch)
        bass_note.duration = duration.Duration(4)
        bass_part.append(bass_note)
    score = stream.Score((chords_part, bass_part))
    score.show('midi')
    score.write('midi', f'output/score-{datetime.now()}.mid')
    return score

if __name__ == "__main__":
    warmup_song = 'billie_jean_tdc.clt'
    numerical_seq, roman_seq = data.read_file_as_chord_seq(warmup_song)
    song_chords = data.roman_seq_to_chord_embeddings(roman_seq)
    warmup_seq = song_chords[:sequence_length]
    model = load_model('saved_model')
    num_chords = 512
    chords = predict_chords(model, warmup_seq, num_chords=num_chords)
    roman_numerals = data.common_numerals[chords]
    score = make_score(roman_numerals)
    print(roman_numerals)
