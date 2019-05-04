set -e
cd ../ffn
model_params="-layers 50 50 -n_particles 50 -n_epoch 5000 -fix_variance 0.2"
psvi_config="-lr 1e-3"
dump_prefix='/tmp/dupm'
for method in 'gfsf' 'pisgld' 'wsgld' 'svgd'
do
    python bnn_stein_f.py -dataset sine -dump_pred_dir ${dump_prefix}/f${method}50.bin $model_params $psvi_config -psvi_method=$method -mm_n_particles 400 
    python bnn_stein.py   -dataset sine -dump_pred_dir ${dump_prefix}/${method}50.bin  $model_params $psvi_config -psvi_method=$method
done
python hmc.py -dataset sine $model_params -dump_pred_dir ${dump_prefix}/hmc.bin -lr 1e-3
cd ../exps
python synthetic_vis.py -figs 14
