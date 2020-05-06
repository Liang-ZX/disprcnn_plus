echo 'preparing offline predictions...'
python -m torch.distributed.launch --nproc_per_node $NGPUS tools/test_net.py --config-file configs/kitti/vob/mask.yaml DATALOADER.NUM_WORKERS 0
# split predictions
python tools/split_predictions.py --prediction_path models/kitti/vob/mask/inference/kitti_%s_vob/predictions.pth --split_path data/kitti/object/split_set/%s_set.txt
# generate rois
echo 'generating rois...'
python tools/kitti_object/generate_psmnet_input_inf.py --output_dir data/vob_roi --prediction_template models/kitti/vob/mask/inference/kitti_%s_vob/predictions.pth --mask_disp_sub_path vob
# train iDispNet
echo 'train iDispNet...'
python -m torch.distributed.launch --nproc_per_node $NGPUS tools/kitti_object/train_idispnet_fa.py --data_dir data/vob_roi --mode train_oc --model_dir models/kitti/vob/idispnet --load_model models/PSMNet/pretrained_model_KITTI2015.tar