#!/usr/bin/python3
import os, urllib.request
def download(url):
    filename = url.split("/")[-1]
    if not os.path.exists(filename):
        urllib.request.urlretrieve(url, filename)

#download('http://data.mxnet.io/data/caltech-256/caltech-256-60-train.rec')
#download('http://data.mxnet.io/data/caltech-256/caltech-256-60-val.rec')

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

def get_model(prefix, epoch):
    download(prefix+'-symbol.json')
    download(prefix+'-%04d.params' % (epoch,))


def get_fine_tune_model(symbol, arg_params, num_classes, layer_name='flatten0'):
    """
    symbol: the pre-trained network symbol
    arg_params: the argument parameters of the pre-trained model
    num_classes: the number of classes for the fine-tune datasets
    layer_name: the layer name before the last fully-connected layer
    """
    all_layers = sym.get_internals()
    net = all_layers[layer_name+'_output']
    net = mx.symbol.FullyConnected(data=net, num_hidden=num_classes, name='fc1')
    net = mx.symbol.SoftmaxOutput(data=net, name='softmax')
    new_args = dict({k:arg_params[k] for k in arg_params if 'fc1' not in k})
    return (net, new_args)


import logging
head = '%(asctime)-15s %(message)s'
logging.basicConfig(level=logging.DEBUG, format=head)

def fit(symbol, arg_params, aux_params, train, val, batch_size, num_gpus):
    devs = [mx.gpu(i) for i in range(num_gpus)]
    #print(devs)
    mod = mx.mod.Module(symbol=new_sym, context=devs)

    ########
    tic = time.time()
    print("Saving Model..")
    # save model
    checkpoint = mx.callback.do_checkpoint('calTech-resnet-18', period=1)
    toc = time.time()
    print("Model Saving Time: ", toc - tic, " seconds")
    #######

    mod.fit(train, 
        eval_data=val,
        num_epoch=20,
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
num_classes = 256
batch_per_gpu = 16
num_gpus = 1

batch_size = batch_per_gpu * num_gpus

import time
get_model('http://data.mxnet.io/models/imagenet/resnet/50-layers/resnet-50', 0)
get_model('http://data.mxnet.io/models/imagenet/resnet/34-layers/resnet-34', 0)
get_model('http://data.mxnet.io/models/imagenet/resnet/18-layers/resnet-18', 0)
get_model('http://data.mxnet.io/models/imagenet/inception-bn/Inception-BN', 126)

tic = time.time()
sym, arg_params, aux_params = mx.model.load_checkpoint('resnet-18', 0)
toc = time.time()
print("resnet-18 model loading time: ", toc-tic, " seconds")

(new_sym, new_args) = get_fine_tune_model(sym, arg_params, num_classes)
(train, val) = get_iterators(batch_size)

mod_score = fit(new_sym, new_args, aux_params, train, val, batch_size, num_gpus)
print("Model Score: ", mod_score)

assert mod_score[0][1] > 0.7, "Low training accuracy."
