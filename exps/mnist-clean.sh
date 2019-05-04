cd ../ffn
python bnn_stein_f.py -layers 400 400 -lr 2e-4 -batch_size 100 -n_epoch 1000 -valid -dataset mnist -logits_w_sd 1 -n_mm_sample 2 -mm_jitter 10.0 -mm_n_particles 30 -n_particles 5
