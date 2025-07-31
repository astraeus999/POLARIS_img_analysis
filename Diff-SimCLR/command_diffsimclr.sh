CUDA_VISIBLE_DEVICES=0,1,2
python main_supcon.py --resume --method SimCLR --dataset path --epochs 300 --lr_decay_epochs 90 --batch_size 4 --data_folder ./SupContrast/images