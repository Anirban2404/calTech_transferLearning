#wget http://www.vision.caltech.edu/Image_Datasets/Caltech256/256_ObjectCategories.tar
tar -xf 256_ObjectCategories.tar

# size in 10 -- change it if needed
mkdir -p caltech_256_train_10_$1
for i in 256_ObjectCategories/*; do
    c=`basename $i`
    mkdir -p caltech_256_train_10_$1/$c
    for j in `ls $i/*.jpg | shuf | head -n 10`; do
        mv $j caltech_256_train_10_$1/$c/
    done
done
mkdir -p caltech_256_val_3_$1
for i in 256_ObjectCategories/*; do
    c=`basename $i`
    mkdir -p caltech_256_val_3_$1/$c
    for j in `ls $i/*.jpg | shuf | head -n 3`; do
        mv $j caltech_256_val_3_$1/$c/
    done
done

python3 /home/de-worker-tx2/incubator-mxnet/tools/im2rec.py --list --recursive caltech-256-10-train$1 caltech_256_train_10_$1/
python3 /home/de-worker-tx2/incubator-mxnet/tools/im2rec.py --list --recursive caltech-256-10-val$1 caltech_256_val_3_$1/
python3 /home/de-worker-tx2/incubator-mxnet/tools/im2rec.py --resize 256 --quality 90 --num-thread 6 caltech-256-10-val$1 caltech_256_val_3_$1/
python3 /home/de-worker-tx2/incubator-mxnet/tools/im2rec.py --resize 256 --quality 90 --num-thread 6 caltech-256-10-train$1 caltech_256_train_10_$1/
