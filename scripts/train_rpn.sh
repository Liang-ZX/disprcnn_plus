# load idispnet, inference on train/val set
echo 'preparing offline predictions...'
python -m torch.distributed.launch --nproc_per_node 8 tools/test_net.py --config-file configs/kitti/vob/idispnet.yaml
python tools/split_predictions.py --prediction_path models/kitti/vob/idispnet/inference/kitti_%s_vob/predictions.pth --split_path data/kitti/object/split_set/%s_set.txt
# train RPN
echo 'train RPN...'
python -m torch.distributed.launch --nproc_per_node $NGPUS tools/train_net.py --config-file configs/kitti/vob/rpn.yaml