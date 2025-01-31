dt=`date '+%Y%m%d_%H%M%S'`
mkdir result_model
python train.py --model_fn ./result_model/nsmc.ensemble_model.${dt}.pth --train_fn ./data/review.sorted.refined.tok.shuf.train.tsv --gpu_id 0 --batch_size 128 --n_epochs 10 --word_vec_size 256 --dropout .3 --rnn --hidden_size 512 --n_layers 4 --cnn --window_size 3 4 5 6 7 8 --n_filters 128 128 128 128 128