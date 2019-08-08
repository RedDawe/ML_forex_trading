import numpy as np
import tensorflow as  tf
import matplotlib.pyplot as plt
import datetime
import time
import sys
from termcolor import cprint
from os import listdir
from os.path import isfile, join
import backtrader as bt
import pandas as pd
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()


with tf.name_scope('Data_parameters'):
    beginning_cut = 200
    ending_cut = 200

    beginning_space = 51
    end_cv = 730

    examples = 1

    include = [18, 19, 20, 21]

    constant_seed = 0

    num_minibatches = None

    init_from_files = True
    save_every_cycle= False

    use_swn = True

    mypath = 'C:\duckascopy_forex_website_data'


with tf.name_scope('Hyper_parameters'):
    # LEARNING RATE
    exp_lr = False
    stepsize_mul = 20  # useless if stepsize != None, integer between 2 and 10
    stepsize = None  # inferred if == None
    real_low = 0.00001  # 0.0001 DO NOT CHANGE THIS, THERE'S A BIG CHANCE SOMETHING ELSE IS WRONG
    real_high = 0.0001  # 0.001 ACTUALLY ONE MORE ZERO SEEMS TO WORK JUST FINE, FOR THE FINAL TRAINING CONSIDER ADDING EVEN ONE MORE

    num_epochs = None
    num_epochs_mul = 2


def get_data():
    min_length = 1446

    onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
    my_data = np.zeros((6000, len(onlyfiles), 4))
    for j in range(len(onlyfiles)):
        current_data = np.genfromtxt(mypath + '\\' + onlyfiles[j], delimiter=',')
        my_data[1:current_data.shape[0], j, 0] = current_data[1:, 2]
        my_data[1:current_data.shape[0], j, 1] = current_data[1:, 3]
        my_data[1:current_data.shape[0], j, 2] = current_data[1:, 4]
        my_data[1:current_data.shape[0], j, 3] = current_data[1:, 5]
        my_data[0, j, 0] = current_data.shape[0] - 1
        my_data[0, j, 1] = current_data.shape[0] - 1
        my_data[0, j, 2] = current_data.shape[0] - 1
        my_data[0, j, 3] = current_data.shape[0] - 1

    ###############
    deleter = []
    for jiterator in range(my_data.shape[1]):
        if my_data[0, jiterator, 0] < min_length:
            deleter.append(jiterator)
            continue
        #for iterator in range(1, int(my_data[0, jiterator, 0]) + 1):
        #    if my_data[iterator, jiterator].any() <= too_low_value_forex:
        #        deleter.append(jiterator)
        #        break
    for jiterator in reversed(deleter):
        my_data = np.delete(my_data, jiterator, 1)
    ################

    return my_data
def create_inds(plane):
    #high
    #low
    #close
    #volume

    plane = np.tile(plane, [1, 1, 8])

    if use_swn:
        def normalize(src, pos, omega=beginning_cut):
            for jiterator in range(plane.shape[1]):
                for iterator in range(1+omega, int(plane[0, jiterator, 0])+2):
                    mean = np.mean(plane[iterator-omega:iterator, jiterator, src])
                    sigma = np.std(plane[iterator - omega:iterator, jiterator, src])
                    if sigma == 0:
                        sigma = 1
                    plane[iterator-1, jiterator, pos] = (plane[iterator-1, jiterator, src] - mean) / sigma

        def normalize(src, pos, omega=beginning_cut):
            for jiterator in range(plane.shape[1]):
                for iterator in range(1+omega, int(plane[0, jiterator, 0])+2):
                    min = np.min(plane[iterator-omega:iterator, jiterator, src])
                    max = np.max(plane[iterator - omega:iterator, jiterator, src])
                    if max - min == 0:
                        max = 1e-7
                    plane[iterator-1, jiterator, pos] = (plane[iterator-1, jiterator, src]-(min + max)/2)/(max-min)

        normalize(2, 18)
        normalize(1, 19)
        normalize(0, 20)
        normalize(3, 21)

    else:


        #feature normalization
        src = 2
        pos = 18

        means = []
        sigmas = []
        for jiterator in range(plane.shape[1]):
            mean = np.mean(plane[1:int(plane[0, jiterator, 0]) + 1 - end_cv, jiterator, src])
            sigma = np.std(plane[1:int(plane[0, jiterator, 0]) + 1 - end_cv, jiterator, src])
            plane[1:int(plane[0, jiterator, 0]) + 1, jiterator, pos] = (plane[1:int(plane[0, jiterator, 0]) + 1, jiterator, src] - mean) / sigma
            means.append(mean)
            sigmas.append(sigma)
        np.save('ms.npy', np.array(np.stack([means, sigmas])))
        ###

        # feature normalization
        src = 1
        pos = 19

        means = []
        sigmas = []
        for jiterator in range(plane.shape[1]):
            mean = np.mean(plane[1:int(plane[0, jiterator, 0]) + 1 - end_cv, jiterator, src])
            sigma = np.std(plane[1:int(plane[0, jiterator, 0]) + 1 - end_cv, jiterator, src])
            plane[1:int(plane[0, jiterator, 0]) + 1, jiterator, pos] = (plane[1:int(plane[0, jiterator, 0]) + 1, jiterator, src] - mean) / sigma
            means.append(mean)
            sigmas.append(sigma)
        np.save('ms1.npy', np.array(np.stack([means, sigmas])))
        ###

        # feature normalization
        src = 0
        pos = 20

        means = []
        sigmas = []
        for jiterator in range(plane.shape[1]):
            mean = np.mean(plane[1:int(plane[0, jiterator, 0]) + 1 - end_cv, jiterator, src])
            sigma = np.std(plane[1:int(plane[0, jiterator, 0]) + 1 - end_cv, jiterator, src])
            plane[1:int(plane[0, jiterator, 0]) + 1, jiterator, pos] = (plane[1:int(plane[0, jiterator, 0]) + 1, jiterator, src] - mean) / sigma
            means.append(mean)
            sigmas.append(sigma)
        np.save('ms2.npy', np.array(np.stack([means, sigmas])))
        ###

        # feature normalization
        src = 3
        pos = 21

        means = []
        sigmas = []
        for jiterator in range(plane.shape[1]):
            mean = np.mean(plane[1:int(plane[0, jiterator, 0]) + 1 - end_cv, jiterator, src])
            sigma = np.std(plane[1:int(plane[0, jiterator, 0]) + 1 - end_cv, jiterator, src])
            plane[1:int(plane[0, jiterator, 0]) + 1, jiterator, pos] = (plane[1:int(plane[0, jiterator, 0]) + 1, jiterator, src] - mean) / sigma
            means.append(mean)
            sigmas.append(sigma)
        np.save('ms3.npy', np.array(np.stack([means, sigmas])))
        ###

    # labels
    p = 14
    pp = 10
    for j in range(plane.shape[1]):
        for i in range(int(p/2)+1, int(plane[0, j, 0])-int(p/2)+2):
            plane[i, j, 22] = np.sum(plane[int(i-p/2):int(i+p/2), j, 2])/p
        for i in reversed(range(1, int(plane[0, j, 0]))):
            plane[i, j, 16] = 1 if plane[i+1, j, 22] > plane[i, j, 22] else 0
        for i in range(int(pp/2)+1, int(plane[0, j, 0])-int(pp/2)+2):
            plane[i, j, 22] = np.sum(plane[int(i-pp/2):int(i+pp/2), j, 22])/pp
        for i in reversed(range(1, int(plane[0, j, 0]))):
            plane[i, j, 17] = 1 if plane[i+1, j, 22] > plane[i, j, 22] else 0
    ###

    # label weights
    labels_pos = 17
    weights_pos = 16

    for j in range(plane.shape[1]):
        start_of_a_trade = 1
        position = -1
        for i in range(1, int(plane[0, j, 0])+1):
            if position != plane[i, j, labels_pos]:
                if position == 1:
                    plane[start_of_a_trade:i, j, weights_pos] = plane[i, j, 2] / plane[start_of_a_trade, j, 2] - 1
                else:
                    plane[start_of_a_trade:i, j, weights_pos] = plane[start_of_a_trade, j, 2] / plane[i, j, 2] - 1

                position = plane[i, j, labels_pos]
                start_of_a_trade = i
        for i in range(1, int(plane[0, j, 0])+1):
            weighted = 1 if plane[i, j, weights_pos] > 0.035 else 0
            direction = plane[i, j, labels_pos]

            if weighted == 0:
                weighted_direction = 0
            elif direction == 0:
                weighted_direction = -1
            else:
                weighted_direction = 1

            plane[i, j, labels_pos] = weighted_direction
    ###

    #loss weights
    src = 17
    pos = 16

    for j in range(plane.shape[1]):
        plane[1:int(plane[0, j, 0]) + 1, j, pos] = np.where(plane[1:int(plane[0, j, 0]) + 1, j, src] == 0, 2, 1)
    ###

    #plt.plot(plane[1000:1200, 16, 2])
    #plt.plot(plane[1000:1200, 16, 18])
    #akalo = plane[1000:1200, 16, 2]
    #plt.plot((akalo - (np.min(akalo) + np.max(akalo))/2)/(np.max(akalo) - np.min(akalo)))
    #plt.plot(plane[10:4000, 16, labels_pos]*0.1+plane[10:4000, 16, 2])
    #plt.show()

    #print(sum([1 if i == 0 else 0 for i in plane[1:int(plane[0, 0, 0]), 0, labels_pos]]))
    #sys.exit()
    answers = np.expand_dims(plane[:, :, 17], -1)
    return np.stack(tuple([plane[:, :, p] for p in include]), axis=-1), answers, plane
load_from_csv = True
if load_from_csv:
    X = get_data()
    X, Y, D = create_inds(X)
    np.savez('data', X=X, Y=Y, D=D)
else:
    theta = np.load('data.npz')
    X = theta['X']
    Y = theta['Y']
    D = theta['D']

with tf.name_scope('Hyper_parameter_calculations'):
    if constant_seed is not None:
        tf.set_random_seed(constant_seed)
        np.random.seed(constant_seed)
    if num_minibatches is None:
        num_minibatches = int(np.ceil((X.shape[1]) / examples))
    if exp_lr:
        low = 0
        high = np.log10(real_high) - np.log10(real_low)
        if not (real_low * 10 ** high == real_high and low == 0):
            print('real_low * 10 ** high != real_high and low == 0')
            print('probably numerical instability, switching off exp_lr should help')
            sys.exit()
    else:
        low = real_low
        high = real_high
    if stepsize is None:
        stepsize = stepsize_mul * num_minibatches
    if num_epochs is None:
        num_epochs = stepsize * 2 * num_epochs_mul
    if num_epochs % (stepsize * 2) != 0:
        # print(colored('!!! The last cycle of learning rate will not be completed. !!!', 'blue'))
        cprint("!!! The last cycle of learning rate will not be completed. !!!", 'white', 'on_grey')
        # print("!!! The last cycle of learning rate will not be completed. !!!")
    if num_epochs / (stepsize * 2) < 4 and stepsize_mul != 2:
        print('Consider making stepzise_mul smaller.')
    print('Num_minibatches =', num_minibatches, 'of', X.shape[1], 'possible.')
    print('Step_size =', stepsize)
    print('Num_epochs =', num_epochs)

with tf.name_scope('Network'):
    cnn_act = tf.nn.elu
    keep_prob = 0.9

    x = tf.placeholder(tf.float32, (examples, None, len(include)))
    y = tf.placeholder(tf.int32, (examples, None, 1))
    epochCounter = tf.placeholder(tf.int32)
    training = tf.placeholder(tf.bool)

    a = x
    a_hat = y
    """
    #for dense_block in range(6):
    for dense_block in range(12):
        #a_d = tf.layers.conv1d(inputs=tf.pad(a, paddings=tf.constant([[0, 0], [2, 0], [0, 0]])), filters=2, kernel_size=3, strides=1, padding='valid', activation=cnn_act, kernel_initializer=tf.contrib.layers.variance_scaling_initializer())
        a_d = tf.layers.dropout(a, rate=1 - keep_prob, training=training)
        a_d = tf.layers.conv1d(inputs=tf.pad(a_d, paddings=tf.constant([[0, 0], [8, 0], [0, 0]])), filters=4, kernel_size=9, strides=1, padding='valid', activation=cnn_act, kernel_initializer=tf.contrib.layers.variance_scaling_initializer(factor=1))
        a = tf.concat([a_d, a], -1)
    """
    #a = tf.layers.dropout(a, rate=0.9, training=training)

    units = 16
    layers = 1
    RNN = tf.contrib.cudnn_rnn.CudnnLSTM(layers, units, dropout=1 - keep_prob)
    #print(RNN)
    a, _ = RNN(tf.reshape(a, [-1, examples, len(include)]))
    a = tf.reshape(a, [examples, -1, units])



    a = tf.layers.dense(a, 3, activation=None, kernel_initializer=tf.contrib.layers.variance_scaling_initializer(mode='FAN_AVG', factor=1))

    logits = a[:, beginning_space:, :]
    labels = a_hat[:, beginning_space:, :]
    m = tf.to_float(tf.shape(labels)[1])

    to_be_smoothed_logits = logits
    logits = tf.nn.softmax(logits)

    labels = tf.one_hot(labels + 1, 3)[:, :, 0, :]

    J = tf.losses.softmax_cross_entropy(logits=to_be_smoothed_logits, onehot_labels=labels, label_smoothing=0.)

    with tf.name_scope('Learning_rate'):
        cycle = tf.floor(1 + epochCounter / 2 / stepsize)
        scale = tf.abs(epochCounter / stepsize - 2 * cycle + 1)
        alpha = low + (high - low) * tf.maximum(tf.constant(0.0, dtype=tf.float64), (1 - scale))
        if exp_lr:
            alpha = real_low * 10 ** alpha

    optimizer = tf.train.AdamOptimizer(learning_rate=alpha).minimize(J)

leverage = 20
my_commission= 0
def profit(cv = False):
    beginning_grounds = beginning_space + 1 + beginning_cut

    if cv:
        start = datetime.datetime(2016, 4, 9).date()
        end = datetime.datetime(2018, 10, 26).date()
        duration = 930
    else:
        start = datetime.datetime(1980, 1, 1).date()
        end = datetime.datetime(2016, 4, 9).date()
        duration = 2749

    cerebro = bt.Cerebro()

    onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))] [4:5]
    for j in range(len(onlyfiles)):
        data = bt.feeds.GenericCSVData(dataname=mypath + '/' + str(onlyfiles[j]), nullvalue=0.0, dtformat=('%d.%m.%Y %H:%M:%S.000'), open=1, high=2, low=3, close=4, volume=5, openinterest=-1, timeframe=bt.TimeFrame.Days)
        cerebro.adddata(data)


    class Strategy_HLC(bt.SignalStrategy):
        params = (('risk', 0.02), ('atr_mul', 1.5), ('threshold', 0.3))

        def __init__(self):
            _ = bt.indicators.SimpleMovingAverage(period=beginning_grounds)

            self.stop_loss = [[None, None, None]] * len(onlyfiles)

            self.printed = False

            self.atrs = []
            for i in range(len(onlyfiles)):
                self.atrs.append(bt.indicators.AverageTrueRange(self.datas[i]))


        def next(self):
            global my_commission

            if self.broker.get_value() < starting_value * 0.75 and not self.printed:
                print('Dropped bellow treshold')
                self.printed = True
            if self.datas[0].datetime.date() > end:
                for i in range(len(onlyfiles)):
                    the_size = self.getposition(data=self.datas[i]).size
                    if the_size > 0:
                        position = 1
                    elif the_size == 0:
                        position = 0
                    elif the_size < 0:
                        position = -1
                    if position != 0:
                        self.close(data=self.datas[i])
            elif self.datas[0].datetime.date() > start:
                for i in range(len(onlyfiles)):

                    the_size = self.getposition(data=self.datas[i]).size
                    if the_size > 0:
                        position = 1
                    elif the_size == 0:
                        position = 0
                    elif the_size < 0:
                        position = -1

                    data_list_close = []
                    data_list_low = []
                    data_list_high = []
                    data_list_volume = []
                    for j in range(beginning_grounds):
                        data_list_close.append(self.datas[i].close[-j])
                        data_list_low.append(self.datas[i].low[-j])
                        data_list_high.append(self.datas[i].high[-j])
                        data_list_volume.append(self.datas[i].volume[-j])
                    data_list_close.reverse()
                    data_list_low.reverse()
                    data_list_high.reverse()
                    data_list_volume.reverse()
                    data_arr = np.stack([data_list_close, data_list_low, data_list_high], axis=-1).reshape(1, -1, 3)
                    data_arr = np.concatenate([data_arr, np.array(data_list_volume).reshape(1, -1, 1)], axis=-1)

                    if not use_swn:
                        means_and_sigmas = np.load('ms.npy')
                        mean = means_and_sigmas[0, i]
                        sigma = means_and_sigmas[1, i]
                        data_arr[:, :, 0] = (data_arr[:, :, 0] - mean) / sigma

                        means_and_sigmas = np.load('ms1.npy')
                        mean = means_and_sigmas[0, i]
                        sigma = means_and_sigmas[1, i]
                        data_arr[:, :, 1] = (data_arr[:, :, 1] - mean) / sigma

                        means_and_sigmas = np.load('ms2.npy')
                        mean = means_and_sigmas[0, i]
                        sigma = means_and_sigmas[1, i]
                        data_arr[:, :, 2] = (data_arr[:, :, 2] - mean) / sigma

                        means_and_sigmas = np.load('ms3.npy')
                        mean = means_and_sigmas[0, i]
                        sigma = means_and_sigmas[1, i]
                        data_arr[:, :, 3] = (data_arr[:, :, 3] - mean) / sigma


                    normalized_data_arr = np.copy(data_arr)

                    if False:
                        omega = beginning_cut
                        for pos in range(data_arr.shape[2]):
                            for j in range(1+omega, data_arr.shape[1]+1):
                                mean = np.mean(data_arr[:, j-omega:j, pos])
                                sigma = np.std(data_arr[:, j-omega:j, pos])
                                if sigma == 0:
                                    sigma = 1
                                normalized_data_arr[:, j-1, pos] = (data_arr[:, j-1, pos] - mean) / sigma


                    if use_swn:
                        omega = beginning_cut
                        for pos in range(data_arr.shape[2]):
                            for j in range(1+omega, data_arr.shape[1]+1):
                                max = np.max(data_arr[:, j-omega:j, pos])
                                min = np.min(data_arr[:, j-omega:j, pos])
                                if max == 0:
                                    max = 1e-7
                                normalized_data_arr[:, j - 1, pos] = (data_arr[:, j-1, pos] - (min + max) / 2) / (max - min)



                    lgt = sess.run(logits, feed_dict={x: normalized_data_arr, training: False})[0, -1, :]
                    lgt = np.argmax(lgt) - 1
                    if lgt == 0:
                        lgt = 0.5
                    elif lgt == 1:
                        lgt = 1
                    elif lgt == -1:
                        lgt = 0

                    #size = int((self.broker.get_value() * self.params.risk) / self.datas[i].close[0])
                    size = int((self.broker.get_value() * self.params.risk) / (self.atrs[i][0] * self.params.atr_mul))
                    size = abs(size)


                    if abs(self.broker.get_cash()*leverage / self.datas[i].close[0]) < size or abs(self.broker.get_value()*leverage / self.datas[i].close[0]) < size:
                        print('Got insufficient funds somewhere in the process', i, abs(self.broker.get_cash()*leverage / self.datas[i].close[0]), abs(self.broker.get_value()*leverage / self.datas[i].close[0]), size)


                    if lgt < 0.5 - self.params.threshold :
                        sl_price = self.datas[i].close[0] + self.atrs[i][0] * self.params.atr_mul
                        so_price = self.datas[i].close[0] - self.atrs[i][0]

                        if position == -1 and lgt == 0 and self.stop_loss[i][-1].status == 4 and self.stop_loss[i][-2].status == 5:
                            self.stop_loss[i][-2] = self.buy(data=self.datas[i], trailamount=1.5*self.atrs[i][0], size=the_size, exectype=bt.Order.StopTrail)
                        if position == 0:
                            mainside = self.sell(data=self.datas[i], transmit=False, size=size, exectype=bt.Order.Market)
                            lowside = self.buy(data=self.datas[i], price = sl_price, size=mainside.size, exectype=bt.Order.Stop, transmit=False, parent=mainside)
                            highside = self.buy(data=self.datas[i], price=so_price, size=mainside.size/2, exectype=bt.Order.Limit, transmit=True, parent=mainside)
                            self.stop_loss[i] = [mainside, lowside, highside]
                        if position == 1:
                            self.cancel(self.stop_loss[i][-2])
                            self.close(data=self.datas[i])

                            mainside = self.sell(data=self.datas[i], transmit=False, size=size, exectype=bt.Order.Market)
                            lowside = self.buy(data=self.datas[i], price = sl_price, size=mainside.size, exectype=bt.Order.Stop, transmit=False, parent=mainside)
                            highside = self.buy(data=self.datas[i], price=so_price, size=mainside.size/2, exectype=bt.Order.Limit, transmit=True, parent=mainside)
                            self.stop_loss[i] = [mainside, lowside, highside]

                    elif lgt > 0.5 + self.params.threshold :
                        sl_price = self.datas[i].close[0] - self.atrs[i][0] * self.params.atr_mul
                        so_price = self.datas[i].close[0] + self.atrs[i][0]

                        if position == 1 and lgt == 1 and self.stop_loss[i][-1].status == 4 and self.stop_loss[i][-2].status == 5:
                            self.stop_loss[i][-2] = self.sell(data=self.datas[i], trailamount=1.5*self.atrs[i][0], size=the_size, exectype=bt.Order.StopTrail)
                        if position == 0:
                            mainside = self.buy(data=self.datas[i], transmit=False, size=size, exectype=bt.Order.Market)
                            lowside = self.sell(data=self.datas[i], price=sl_price, size=mainside.size, exectype=bt.Order.Stop, transmit=False, parent=mainside)
                            highside = self.sell(data=self.datas[i], price=so_price, size=mainside.size/2, exectype=bt.Order.Limit, transmit=True, parent=mainside)
                            self.stop_loss[i] = [mainside, lowside, highside]
                        if position == -1:
                            self.cancel(self.stop_loss[i][-2])
                            self.close(data=self.datas[i])

                            mainside = self.buy(data=self.datas[i], transmit=False, size=size, exectype=bt.Order.Market)
                            lowside = self.sell(data=self.datas[i], price=sl_price, size=mainside.size, exectype=bt.Order.Stop, transmit=False, parent=mainside)
                            highside = self.sell(data=self.datas[i], price=so_price, size=mainside.size/2, exectype=bt.Order.Limit, transmit=True, parent=mainside)
                            self.stop_loss[i] = [mainside, lowside, highside]

                    else:
                        if position != 0:
                            self.cancel(self.stop_loss[i][-2])
                            self.close(data=self.datas[i])

    cerebro.addstrategy(Strategy_HLC)

    #cerebro.broker.setcommission(commission=0.001)
    cerebro.broker.setcommission(leverage=20)

    starting_value = cerebro.broker.get_value()

    cerebro.run()


    print(cerebro.broker.get_value() - my_commission, my_commission)


    cerebro.plot()

    return ((cerebro.broker.get_value() - my_commission) / starting_value) ** (1 / (duration / 365.2422)) # percentage profit per year


saver = tf.train.Saver()
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
path = 'D:/tmp/model_LSTM.ckpt'

with tf.Session(config=config) as sess:
    if init_from_files:
        saver.restore(sess, path)
    else:
        sess.run(tf.global_variables_initializer())

    for epoch in range(num_epochs):
        costs = []
        for iteration in range(num_minibatches):
            _, cost = sess.run([optimizer, J], feed_dict={
                x: X[1+beginning_cut:int(X[0, iteration, 0])-ending_cut-end_cv, iteration, :].reshape(examples, -1, len(include)),
                y: Y[1+beginning_cut:int(X[0, iteration, 0])-ending_cut-end_cv, iteration, 0].reshape(examples, -1, 1),
                epochCounter: epoch,
                training: True
            })

            costs.append(cost)

        print('On epoch', epoch, 'cost =', np.mean(costs))


        if epoch % (stepsize * 2) == 0 or epoch + 1 == num_epochs:
            costs = []
            for iteration in range(num_minibatches):
                cost = sess.run(J, feed_dict={
                    x: X[1+beginning_cut:int(X[0, iteration, 0])-ending_cut-end_cv, iteration, :].reshape(examples, -1, len(include)),
                    y: Y[1+beginning_cut:int(X[0, iteration, 0])-ending_cut-end_cv, iteration, 0].reshape(examples, -1, 1),
                    epochCounter: epoch,
                    training: False
                })

                costs.append(cost)


            print('Validation cost =', np.mean(costs), ', profits = ', profit(cv = False))

            costs = []
            for iteration in range(num_minibatches):
                cost = sess.run(J, feed_dict={
                    x: X[int(X[0, iteration, 0])-ending_cut-end_cv-beginning_space:int(X[0, iteration, 0])-ending_cut, iteration, :].reshape(examples, -1, len(include)),
                    y: Y[int(X[0, iteration, 0])-ending_cut-end_cv-beginning_space:int(X[0, iteration, 0])-ending_cut, iteration, 0].reshape(examples, -1, 1),
                    epochCounter: epoch,
                    training: False
                })

                costs.append(cost)

            print('Cross validation cost =', np.mean(costs), ', profits = ', profit(cv = True))

            if save_every_cycle:
                saver.save(sess, path)