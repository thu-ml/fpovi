cd ../conv
python bayes_convnet.py -gpu 2,3,4,5,6,7 -n_particles_per_dev 2 -mm_npt 2 -variational f_svgd -mm_jitter 1e-3 -mm_nc 2 -mm_n_inp 4 -max_epoch 200
# after training: python attack.py ${train_log_dir} (default is /tmp/train_log/cifar_10_f_svgd/somedate)
