import tensorflow as tf
import numpy as np
import os
import pandas as pd

sq_len = 30
label_ma = 8
include_all = True
hidden_state_size = 16

min_lr, max_lr = 0.0001, 0.001
step_size = 12
num_epochs = step_size*2*8

path = 'E:/oanda_new_short_history'
onlyfiles = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]

x = None
for i in range(len(onlyfiles)):
    data = pd.read_csv(path + '/' + onlyfiles[i])
    data.columns = ['Date', 'Time', 'Open', 'High', 'Low', 'Close', 'Volume']
    del data['Time']

    if x is None:
        x = data
    else:
        x = x.merge(data, on='Date')

del x['Date']
closes = x.iloc[:, list(range(3, x.shape[1], 5))]
print(x, closes)
x = x.to_numpy()
closes = closes.to_numpy()

def moving_average(a, window, shift=0):
    ma = []
    for m in range(window, a.shape[0]+1):
        ma.append(np.mean(a[m-window:m]))

    return np.roll(ma, -shift)

X = []
Y = []
for i in range(x.shape[1]//5):
    ma = moving_average(closes[:, i], label_ma)
    ma_shifted = moving_average(closes[:,  i], label_ma, label_ma//2)
    for j in range(sq_len, x.shape[0]-(label_ma//2)-label_ma+1):
        if include_all:
            sq = x[j-sq_len:j, :]
            sq -= np.mean(sq, axis=0)
            sq /= np.std(sq, axis=0)

            one_hot = np.zeros([sq_len, 28 * 4])
            one_hot[:, i * 5:(i + 1) * 5] = 1

            X.append(np.concatenate([sq, one_hot], axis=1))
        else:
            sq = x[j - sq_len:j, i*5:(i+1)*5]
            sq -= np.mean(sq, axis=0)
            sq /= np.std(sq, axis=0)

            X.append(sq)

        Y.append(ma[j] > ma_shifted[j])

X = np.array(X)
Y = np.array(Y)

print(X.shape, Y.shape)

class NAC_Layer(tf.keras.layers.Layer):
  def __init__(self, num_outputs):
    super(NAC_Layer, self).__init__()
    self.num_outputs = num_outputs

  def build(self, input_shape):
    shape = [int(input_shape[-1]), self.num_outputs]
    self.W_hat = self.add_weight("W_hat",
                                    shape=shape, initializer=tf.keras.initializers.VarianceScaling(0.0001))

    self.M_hat = self.add_weight("M_hat",
                                    shape=shape, initializer=tf.keras.initializers.VarianceScaling(0.0001))

  def call(self, x):
    W = tf.nn.tanh(self.W_hat) * tf.nn.sigmoid(self.M_hat)
    return tf.matmul(x, tf.cast(W, 'float32'))

class NALU_Layer(tf.keras.layers.Layer):
  def __init__(self, num_outputs):
    super(NALU_Layer, self).__init__()
    self.num_outputs = num_outputs
    self.nac = NAC_Layer(self.num_outputs)
    self.eps = 1e-7

  def build(self, input_shape):
    shape = [int(input_shape[-1]), self.num_outputs]
    self.G = self.add_weight("G", shape=shape, initializer=tf.keras.initializers.VarianceScaling(0.0001))

  def call(self, x):
    g = tf.nn.sigmoid(tf.matmul(x, self.G))
    y1 = g * self.nac(x)
    y2 = (1 - g) * tf.exp(self.nac(tf.math.log(tf.abs(x) + self.eps)))
    return y1 + y2

class NALU_RNN_Cell(tf.keras.layers.Layer):
    def __init__(self, state_size):
        super(NALU_RNN_Cell, self).__init__()
        self.state_size = state_size
        self.nalu_x = NALU_Layer(self.state_size)
        self.nalu_s = NALU_Layer(self.state_size)

    def call(self, x, s):
        output = self.nalu_x(x) + self.nalu_s(s[0])
        return output, [output]

model = tf.keras.Sequential([tf.keras.Input([sq_len, 252 if include_all else 5]), tf.keras.layers.RNN(NALU_RNN_Cell(hidden_state_size), return_sequences=True), tf.keras.layers.RNN(NALU_RNN_Cell(1)), tf.keras.layers.Activation('sigmoid')])
#model = tf.keras.Sequential([tf.keras.Input([sq_len, 225]), tf.keras.layers.RNN(NALU_RNN_Cell(hidden_state_size), return_sequences=True), tf.keras.layers.LSTM(1), tf.keras.layers.Activation('sigmoid')])
#model = tf.keras.Sequential([tf.keras.Input([sq_len, 225]), tf.keras.layers.LSTM(hidden_state_size, return_sequences=True), tf.keras.layers.LSTM(1), tf.keras.layers.Activation('sigmoid')])
model.summary()

def clr_schedule(epoch):
  cycle = tf.math.floor(1 + epoch / (2 * step_size))
  x = tf.math.abs(epoch / step_size - 2 * cycle + 1)
  lr = min_lr + (max_lr - min_lr) * tf.maximum(0, 1 - x)

  return lr
lr_scheduler = tf.keras.callbacks.LearningRateScheduler(clr_schedule)

checkpoint_path = "training_0/cp.ckpt"
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 save_best_only=True,
                                                 monitor='val_accuracy',
                                                 verbose=2)

model.compile(loss=tf.keras.losses.binary_crossentropy, optimizer=tf.keras.optimizers.Adam(), metrics=['accuracy'])

model.fit(X, Y, validation_split=0.2, epochs=num_epochs, verbose=2, callbacks=[cp_callback, lr_scheduler])

#print(model.predict(X))
