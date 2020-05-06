python -m torch.distributed.launch --nproc_per_node 8 tools/test_net.py --config-file configs/kitti/vob/mask.yaml --ckpt models/kitti/vob/smrcnn.pth OUTPUT_DIR models/kitti/vob DATALOADER.NUM_WORKERS 0
python tools/split_predictions.py --prediction_path models/kitti/vob/inference/kitti_%s_vob/predictions.pth --split_path data/kitti/object/split_set/%s_set.txt
# generate rois
echo 'generating rois...'
python tools/kitti_object/generate_psmnet_input_inf.py --output_dir data/vob_roi --prediction_template models/kitti/vob/inference/kitti_%s_vob/predictions.pth --mask_disp_sub_path vob

python -m torch.distributed.launch --nproc_per_node 8 tools/test_net.py --config-file configs/kitti/vob/idispnet.yaml MODEL.DISPNET.TRAINED_MODEL models/kitti/vob/idispnet.pth SOLVER.OFFLINE_2D_PREDICTIONS models/kitti/vob/inference/kitti_%s_vob/predictions DATALOADER.NUM_WORKERS 0
python tools/split_predictions.py --prediction_path models/kitti/vob/idispnet/inference/kitti_%s_vob/predictions.pth --split_path data/kitti/object/split_set/%s_set.txt

python -m torch.distributed.launch --nproc_per_node 8 tools/test_net.py --config-file configs/kitti/vob/idispnet.yaml MODEL.DISPNET.TRAINED_MODEL models/kitti/vob/idispnet.pth OUTPUT_DIR models/kitti/vob SOLVER.OFFLINE_2D_PREDICTIONS models/kitti/vob/inference/kitti_val_vob/predictions DATALOADER.NUM_WORKERS 0
python tools/split_predictions.py --prediction_path models/kitti/vob/inference/kitti_val_vob/predictions.pth --split_path data/kitti/object/split_set/val_set.txt
python -m torch.distributed.launch --nproc_per_node 8 tools/test_net.py --config-file configs/kitti/vob/rcnn.yaml --ckpt  models/kitti/vob/pointrcnn.pth DATALOADER.NUM_WORKERS 0