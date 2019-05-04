mkdir -p ${HOME}/exps/fpovi/
cd ../ffn
python bnn_stein_f.py -layers 1000 1000 1000 -lr 5e-4 -batch_size 1000 -n_epoch 1000 -dataset mnist -dir ${HOME}/exps/fpovi/mnist-adv -save -production 
python attack.py -ckptd ${HOME}/exps/fpovi/mnist-adv
