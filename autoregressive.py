import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

def main():
    clt_dir = './rs200_harmony_table'
    filename = '1999_dt.clt'
    with open(f'{clt_dir}/{filename}') as file:
        lines = file.readlines()
        raw = [line[:-1].split('\t') for line in lines][:-1]
        table = np.zeros((len(raw), 6))
        for i, row in enumerate(raw):
            # [start] [end] [Roman numeral] [chromatic root] [diatonic root] [key] [absolute root]
            table[i][0] = float(row[0])
            table[i][1] = float(row[1])
            table[i][2] = int(row[3])
            table[i][3] = int(row[4])
            table[i][4] = int(row[5])
            table[i][5] = int(row[6])
        print(table)


class FeedBack(tf.keras.Model):
  def __init__(self, units):
    super().__init__()
    self.units = units
    self.lstm_cell = tf.keras.layers.LSTMCell(units)
    # Also wrap the LSTMCell in an RNN to simplify the `warmup` method.
    self.lstm_rnn = tf.keras.layers.RNN(self.lstm_cell, return_state=True)
    self.dense = tf.keras.layers.Dense(num_features)
    self.softmax = tf.keras.layers.Softmax()

  def generate(self, inputs, out_steps=16, training=None):
    # Use a TensorArray to capture dynamically unrolled outputs.
    predictions = []
    # Initialize the lstm state
    prediction, state = self.warmup(inputs)
   # Insert the first prediction
    predictions.append(prediction)

    # Run the rest of the prediction steps
    for n in range(1, out_steps):
        # Use the last prediction as input.
        x = prediction
        # Execute one lstm step.
        x, state = self.lstm_cell(x, states=state,
                                training=training)
        # Convert the lstm output to a prediction.
        prediction = self.dense(x)
        # Add the prediction to the output
        self.softmax(prediction)
        predictions.append(prediction)

    # predictions.shape => (time, batch, features)
    predictions = tf.stack(predictions)
    # predictions.shape => (batch, time, features)
    predictions = tf.transpose(predictions, [1, 0, 2])
    normalized = self.softmax(predictions)
    chords = np.argmax(normalized.numpy(), axis=-1) + 1
    return chords

  def call(self, inputs, training=None):
    # Execute one lstm step.
    x, *state = self.lstm_rnn(inputs, training=training)
    # Convert the lstm output to a prediction.
    prediction = self.dense(x, training=training)
    #normalized = self.softmax(prediction, training=training)
    return prediction
    
  def warmup(self, inputs):
    # inputs.shape => (batch, time, features)
    # x.shape => (batch, lstm_units)
    x, *state = self.lstm_rnn(inputs)

    # predictions.shape => (batch, features)
    prediction = self.dense(x)
    return prediction, state

if __name__ == "__main__":
    main()