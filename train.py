from model import make_model
from tensorflow.keras.callbacks import EarlyStopping
from data import common_chords, make_dataset
from params import sequence_length, batch_size, patience, epochs


def train(model, train_dataset, val_dataset, epochs=epochs, patience=patience):
    print("About to train model")
    print(model.summary())

    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=epochs,
        callbacks=[EarlyStopping(patience=2)]
    )

    loss, accuracy, precision, recall = model.evaluate(val_dataset, batch_size=batch_size)
    model.save(f'models/seq_{sequence_length}-loss_{loss}')
    return history, loss, accuracy, precision, recall

if __name__ == "__main__":
    model = make_model(len(common_chords))
    train_dataset, val_dataset = make_dataset()
    train(model, train_dataset, val_dataset)