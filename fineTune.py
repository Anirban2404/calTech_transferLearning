#!/usr/bin/python3
import os, urllib.request
import mxnet as mx
import os
import argparse
import time
import logging
head = '%(asctime)-15s %(message)s'
#logging.basicConfig(level=logging.DEBUG, format=head)
logging.basicConfig(filename="de_worker_{}.log".format(time.time()), level=logging.DEBUG)

def get_iterators(data_train, data_val, batch_size, data_shape=(3, 224, 224)):
    train = mx.io.ImageRecordIter(
        path_imgrec         = data_train,
        data_name           = 'data',
        label_name          = 'softmax_label',
        batch_size          = batch_size,
        data_shape          = data_shape,
        shuffle             = True,
        rand_crop           = True,
        rand_mirror         = True)
    val = mx.io.ImageRecordIter(
        path_imgrec         = data_val,
        data_name           = 'data',
        label_name          = 'softmax_label',
        batch_size          = batch_size,
        data_shape          = data_shape,
        rand_crop           = False,
        rand_mirror         = False)
    return (train, val)

def fit(symbol, arg_params, aux_params, train, val, batch_size, num_gpus, num_epochs, kv_store):
    devs = [mx.gpu(i) for i in range(num_gpus)]
    #print(devs)
    if num_gpus == 0:
        devs = mx.cpu(0)
    mod = mx.mod.Module(symbol=sym, context=devs)

    ########
    tic = time.time()
    print("Saving Model..")
    # save model
    checkpoint = mx.callback.do_checkpoint('cal_Tech-inception', period=1)
    toc = time.time()
    print("Model Saving Time: ", toc - tic, " seconds")
    #######

    mod.fit(train, 
        eval_data=val,
        num_epoch=num_epochs,
        arg_params=arg_params,
        aux_params=aux_params,
        allow_missing=True,
        batch_end_callback = mx.callback.Speedometer(batch_size, 10),
        kvstore=kv_store,
        optimizer='sgd',
        optimizer_params={'learning_rate':0.01},
        initializer=mx.init.Xavier(rnd_type='gaussian', factor_type="in", magnitude=2),
        eval_metric='acc',
        epoch_end_callback=checkpoint)
    
    metric = mx.metric.create('acc')
    #print(mod.score(val_dataiter, ['mse', 'acc']))
    return mod.score(val, metric)

if __name__ == '__main__':
    # @@@ AUTOT2EST_OUTPUT_IGNORED_CELL
    parser = argparse.ArgumentParser(description="train calTech",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('-f', '--de-filename', type=str, help='output filename for profile dump')
    parser.add_argument('-j', '--de-influxdb', type=str, help='influxdb address IP:PORT')
    parser.add_argument('-l', '--de-task-id', type=str, help='task-id for logging purposes')
    parser.add_argument('-u', '--de-worker-id', type=str, help='worker id for logging puposes')
    
    parser.add_argument('-gpus', '--num_gpus', type=int, help='number of gpus')
    parser.add_argument('-b', '--batch_size', type=int, help='batch size')
    parser.add_argument('-n', '--network', type=str, help='the neural network to use')
    parser.add_argument('-le', '--load_epoch', type=int, help='load epoch')
    parser.add_argument('-e', '--num_epochs', type=int, help='number of epochs')
    parser.add_argument('-dt', '--data_train', type=str, help='location of training data')
    parser.add_argument('-dv', '--data_val', type=str, help='location of validation data')
    parser.add_argument('-kv-store', '--kv_store', type=str, help='kv store')
    
    parser.set_defaults(
        # network
        network        = 'calTech-resnet-50',
        num_layers     = 50,
        # data
        data_train     = os.environ['DE_PROFILER_DATA_TRAIN'], #'./caltech-256-10-train1.rec'
        data_val       = os.environ['DE_PROFILER_DATA_VAL'],   #'./caltech-256-10-val1.rec'
        num_classes    = 256,
        num_examples   = os.environ['DE_PROFILER_NUM_EXAMPLES'],
        data_shape    = '3,224,224',
        pad_size       = 4,
        batch_size     = 16,
        # train
        lr_step_epochs = '80,90',
        num_gpus       = 1,
        load_epoch     = 0,
        num_epochs     = 1,
        kv_store       = 'device'
    )

    args = parser.parse_args()
    print(args)

    args.num_gpus = int(args.num_gpus)
    args.batch_size = int(args.batch_size)

    tic = time.time()
    sym, arg_params, aux_params = mx.model.load_checkpoint(args.network, args.load_epoch)
    toc = time.time()
    print("calTech loading time: ", toc-tic, " seconds")

    (train, val) = get_iterators(args.data_train, args.data_val, args.batch_size)
    # train
    mod_score = fit(sym, arg_params, aux_params, train, val, args.batch_size, args.num_gpus, args.num_epochs, args.kv_store)
    print("Model Score: ", mod_score)

    assert mod_score[0][1] > 0.7, "Low training accuracy."
