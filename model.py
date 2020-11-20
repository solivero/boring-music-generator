from tensorflow.keras import layers, models, metrics
from tensorflow import losses
from params import sequence_length, lstm1_size, lstm2_size, dropout

def make_model(vocabulary_size, lstm1_size=lstm1_size, lstm2_size=lstm2_size, dropout=dropout):
    model = models.Sequential((
        layers.LSTM(lstm1_size, return_sequences=True, input_shape=(sequence_length, vocabulary_size)),
        layers.Dropout(dropout),
        layers.LSTM(lstm2_size),
        layers.Dropout(dropout),
        layers.Dense(vocabulary_size, activation='softmax'),
    ))
    model.compile(
        optimizer='adam',
        loss=losses.CategoricalCrossentropy(),
        metrics=[metrics.CategoricalAccuracy(), metrics.Precision(), metrics.Recall()]
    )
    return model