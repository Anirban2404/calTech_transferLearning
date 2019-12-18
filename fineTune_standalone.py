#!/usr/bin/python3
import os, urllib.request
import mxnet as mx

def get_iterators(batch_size, data_shape=(3, 224, 224)):
    train = mx.io.ImageRecordIter(
        path_imgrec         = './caltech-256-10-train1.rec',
        data_name           = 'data',
        label_name          = 'softmax_label',
        batch_size          = batch_size,
        data_shape          = data_shape,
        shuffle             = True,
        rand_crop           = True,
        rand_mirror         = True)
    val = mx.io.ImageRecordIter(
        path_imgrec         = './caltech-256-10-val1.rec',
        data_name           = 'data',
        label_name          = 'softmax_label',
        batch_size          = batch_size,
        data_shape          = data_shape,
        rand_crop           = False,
        rand_mirror         = False)
    return (train, val)

import logging
head = '%(asctime)-15s %(message)s'
logging.basicConfig(level=logging.DEBUG, format=head)

def fit(symbol, arg_params, aux_params, train, val, batch_size, num_gpus):
    devs = [mx.gpu(i) for i in range(num_gpus)]
    #print(devs)
    mod = mx.mod.Module(symbol=sym, context=devs)

    ########
    tic = time.time()
    print("Saving Model..")
    # save model
    checkpoint = mx.callback.do_checkpoint('cal_Tech-resnet-50', period=1)
    toc = time.time()
    print("Model Saving Time: ", toc - tic, " seconds")
    #######

    mod.fit(train, 
        eval_data=val,
        num_epoch=1,
        arg_params=arg_params,
        aux_params=aux_params,
        allow_missing=True,
        batch_end_callback = mx.callback.Speedometer(batch_size, 10),
        kvstore='device',
        optimizer='sgd',
        optimizer_params={'learning_rate':0.01},
        initializer=mx.init.Xavier(rnd_type='gaussian', factor_type="in", magnitude=2),
        eval_metric='acc',
        epoch_end_callback=checkpoint)
    
    metric = mx.metric.create('acc')
    #print(mod.score(val_dataiter, ['mse', 'acc']))
    return mod.score(val, metric)


# @@@ AUTOT2EST_OUTPUT_IGNORED_CELL
num_gpus = 1
batch_size = 16

import time
tic = time.time()
sym, arg_params, aux_params = mx.model.load_checkpoint('calTech', 0)
toc = time.time()
print("calTech loading time: ", toc-tic, " seconds")

(train, val) = get_iterators(batch_size)

mod_score = fit(sym, arg_params, aux_params, train, val, batch_size, num_gpus)
print("Model Score: ", mod_score)

assert mod_score[0][1] > 0.7, "Low training accuracy."
