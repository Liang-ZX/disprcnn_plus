import cv2
import torch
import torch.utils.data
import zarr
from tqdm import tqdm
import os
from PIL import Image
import pickle
import numpy as np
from disprcnn.structures.bounding_box import BoxList
from disprcnn.structures.bounding_box_3d import Box3DList
from disprcnn.structures.calib import Calib
from disprcnn.structures.disparity import DisparityMap
from disprcnn.structures.segmentation_mask import SegmentationMask
from disprcnn.utils.kitti_utils import load_calib, load_image_2, load_label_2, load_label_3
from disprcnn.utils.stereo_utils import align_left_right_targets
from disprcnn.modeling.sassd_module.datasets.kitti_utils import Calibration, Sassd_object


class KITTIObjectDataset(torch.utils.data.Dataset):
    CLASSES = (
        "__background__",
        "car",
        'dontcare'
    )
    NUM_TRAINING = 7481
    NUM_TRAIN = 3712
    NUM_VAL = 3769
    NUM_TESTING = 7518

    def __init__(self, root, split, transforms=None, filter_empty=False, offline_2d_predictions_path='',
                 mask_disp_sub_path='vob', remove_ignore=True):
        """
        :param root: '.../kitti/
        :param split: ['train','val']
        :param transforms:
        :param filter_empty:
        :param offline_2d_predictions_path:
        """
        self.root = root
        self.split = split
        cls = KITTIObjectDataset.CLASSES
        self.remove_ignore = remove_ignore
        self.class_to_ind = dict(zip(cls, range(len(cls))))
        self.transforms = transforms
        self.mask_disp_sub_path = mask_disp_sub_path
        # make cache or read cached annotation
        self.annotations = self.read_annotations()
        self.infos = self.read_info()
        self._imgsetpath = os.path.join(self.root, "object/split_set/%s_set.txt")

        with open(self._imgsetpath % self.split) as f:
            self.ids = f.readlines()
        self.ids = [x.strip("\n") for x in self.ids]
        if filter_empty:
            ids = []
            for i in self.ids:
                if self.annotations['left'][int(i)]['labels'].sum() != 0:
                    ids.append(i)
            self.ids = ids
        self.truncations_list, self.occlusions_list = self.get_truncations_occluded_list()
        if '%s' in offline_2d_predictions_path:
            self.offline_2d_predictions_dir = offline_2d_predictions_path % split
        else:
            self.offline_2d_predictions_dir = offline_2d_predictions_path
        # print('using dataset of length', self.__len__())

    def __getitem__(self, index):
        imgs = self.get_image(index)
        targets = self.get_ground_truth(index)
        if self.transforms is not None:
            imgs, targets = self.transforms(imgs, targets)
        if self.split != 'test':
            for view in ['left', 'right']:
                labels = targets[view].get_field('labels')
                targets[view] = targets[view][labels == 1]  # remove not cars
            l, r = align_left_right_targets(targets['left'], targets['right'], thresh=0.15)
            if self.split == 'val' and self.remove_ignore:
                l, r = self.remove_ignore_cars(l, r)
            targets['left'] = l
            targets['right'] = r
        if self.offline_2d_predictions_dir != '':
            lp, rp = self.get_offline_prediction(index)
            return imgs, targets, index, self.ids[index], lp, rp
        else:
            return imgs, targets, index, self.ids[index]

    def get_image(self, index):
        img_id = self.ids[index]
        split = 'training' if self.split != 'test' else 'testing'
        left_img = Image.open(os.path.join(self.root, 'object', split, 'image_2', img_id + '.png'))
        right_img = Image.open(os.path.join(self.root, 'object', split, 'image_3', img_id + '.png'))
        imgs = {'left': left_img, 'right': right_img}
        return imgs

    def get_ground_truth(self, index):
        img_id = self.ids[index]
        if self.split != 'test':
            left_annotation = self.annotations['left'][int(img_id)]
            right_annotation = self.annotations['right'][int(img_id)]
            info = self.get_img_info(index)
            height, width = info['height'], info['width']
            # left target
            left_target = BoxList(left_annotation["boxes"], (width, height), mode="xyxy")
            left_target.add_field("labels", left_annotation["labels"])
            left_target.add_field("alphas", left_annotation['alphas'])
            boxes_3d = Box3DList(left_annotation["boxes_3d"], (width, height), mode='ry_lhwxyz')
            left_target.add_field("box3d", boxes_3d)
            left_target.add_map('disparity', self.get_disparity(index))
            left_target.add_field('masks', self.get_mask(index))
            left_target.add_field('truncation', torch.tensor(self.truncations_list[int(img_id)]))
            left_target.add_field('occlusion', torch.tensor(self.occlusions_list[int(img_id)]))
            left_target.add_field('image_size', torch.tensor([[width, height]]).repeat(len(left_target), 1))
            left_target.add_field('masks', self.get_mask(index))
            left_target.add_field('calib', Calib(self.get_calibration(index), (width, height)))
            left_target.add_field('sassd_calib', self.get_sassd_calib(index))
            left_target.add_field('sassd_objects', self.read_sassd_object(index))
            left_target.add_field('index', torch.full((len(left_target), 1), index, dtype=torch.long))
            left_target.add_field('imgid', torch.full((len(left_target), 1), int(img_id), dtype=torch.long))
            left_target = left_target.clip_to_image(remove_empty=True)
            # right target
            right_target = BoxList(right_annotation["boxes"], (width, height), mode="xyxy")
            right_target.add_field("labels", right_annotation["labels"])
            right_target.add_field("alphas", right_annotation['alphas'])
            boxes_3d = Box3DList(right_annotation["boxes_3d"], (width, height), mode='ry_lhwxyz')
            right_target.add_field("box3d", boxes_3d)
            right_target = right_target.clip_to_image(remove_empty=True)
            target = {'left': left_target, 'right': right_target}
            return target
        else:
            fakebox = torch.tensor([[0, 0, 0, 0]])
            info = self.get_img_info(index)
            height, width = info['height'], info['width']
            # left target
            left_target = BoxList(fakebox, (width, height), mode="xyxy")
            left_target.add_field('image_size', torch.tensor([[width, height]]).repeat(len(left_target), 1))
            left_target.add_field('calib', Calib(self.get_calibration(index), (width, height)))
            left_target.add_field('index', torch.full((len(left_target), 1), index, dtype=torch.long))
            left_target.add_field('masks', self.get_mask(index))
            left_target.add_map('disparity', self.get_disparity(index))
            left_target.add_field('imgid', torch.full((len(left_target), 1), int(img_id), dtype=torch.long))
            # right target
            right_target = BoxList(fakebox, (width, height), mode="xyxy")
            target = {'left': left_target, 'right': right_target}
            return target

    def __len__(self):
        return len(self.ids)

    def get_img_info(self, index):
        img_id = self.ids[index]
        return self.infos[int(img_id)]

    def map_class_id_to_class_name(self, class_id):
        return KITTIObjectDataset.CLASSES[class_id]

    def read_sassd_object(self, index):
        if self.split == 'test':
            split = 'testing'
            return None
        else:
            imgid = self.ids[index]
            split = 'training'
            calib_dir = os.path.join(self.root, 'object', split, 'label_2')
            absolute_path = os.path.join(calib_dir, imgid + '.txt')
            return Sassd_object(absolute_path)

    def read_annotations(self):
        double_view_annotations = {}
        if self.split == 'test':
            split = 'testing'
            return {'left': [], 'right': []}
        else:
            split = 'training'
        for view in [2, 3]:
            annodir = os.path.join(self.root, f"object/{split}/label_{view}")
            anno_cache_path = os.path.join(annodir, 'annotations.pkl')
            if os.path.exists(anno_cache_path):
                annotations = pickle.load(open(anno_cache_path, 'rb'))
            else:
                print('generating', anno_cache_path)
                annotations = []
                for i in tqdm(range(7481)):
                    if view == 2:
                        anno_per_img = load_label_2(self.root, 'training', i)
                    else:
                        anno_per_img = load_label_3(self.root, 'training', i)
                    num_objs = len(anno_per_img)
                    label = np.zeros((num_objs), dtype=np.int32)
                    boxes = np.zeros((num_objs, 4), dtype=np.float32)
                    boxes_3d = np.zeros((num_objs, 7), dtype=np.float32)
                    alphas = np.zeros((num_objs), dtype=np.float32)
                    ix = 0
                    for anno in anno_per_img:
                        cls, truncated, occluded, alpha, x1, \
                        y1, x2, y2, h, w, l, x, y, z, ry = anno.cls.name, anno.truncated, anno.occluded, anno.alpha, anno.x1, anno.y1, anno.x2, anno.y2, \
                                                           anno.h, anno.w, anno.l, \
                                                           anno.x, anno.y, anno.z, anno.ry
                        cls_str = cls.lower().strip()
                        if self.split == 'training':
                            # regard car and van as positive
                            cls_str = 'car' if cls_str in ['car', 'van'] else '__background__'
                        else:  # val
                            # return 'dontcare' in validation phase
                            if cls_str != 'car':
                                cls_str = '__background__'
                        cls = self.class_to_ind[cls_str]
                        label[ix] = cls
                        alphas[ix] = float(alpha)
                        boxes[ix, :] = [float(x1), float(y1), float(x2), float(y2)]
                        boxes_3d[ix, :] = [ry, l, h, w, x, y, z]
                        ix += 1
                    label = label[:ix]
                    alphas = alphas[:ix]
                    boxes = boxes[:ix, :]
                    boxes_3d = boxes_3d[:ix, :]
                    P2 = load_calib(self.root, 'training', i).P2
                    annotations.append({'labels': torch.tensor(label),
                                        'boxes': torch.tensor(boxes, dtype=torch.float32),
                                        'boxes_3d': torch.tensor(boxes_3d),
                                        'alphas': torch.tensor(alphas),
                                        'P2': torch.tensor(P2).float(),
                                        })
                pickle.dump(annotations, open(anno_cache_path, 'wb'))
            if view == 2:
                double_view_annotations['left'] = annotations
            else:
                double_view_annotations['right'] = annotations
        return double_view_annotations

    def read_info(self):
        split = 'training' if self.split != 'test' else 'testing'
        infopath = os.path.join(self.root,
                                f'object/{split}/infos.pkl')
        if not os.path.exists(infopath):
            infos = []
            total = 7481 if self.split != 'test' else 7518
            for i in tqdm(range(total)):
                img = load_image_2(self.root, split, i)
                infos.append({"height": img.height, "width": img.width, 'size': img.size})
            pickle.dump(infos, open(infopath, 'wb'))
        else:
            with open(infopath, 'rb') as f:
                infos = pickle.load(f)
        return infos

    def get_truncations_occluded_list(self):
        if self.split == 'test':
            return [], []
        annodir = os.path.join(self.root, f"object/training/label_2")
        truncations_occluded_cache_path = os.path.join(annodir, 'truncations_occluded.pkl')
        if os.path.exists(truncations_occluded_cache_path):
            truncations_list, occluded_list = pickle.load(open(truncations_occluded_cache_path, 'rb'))
        else:
            truncations_list, occluded_list = [], []
            print('generating', truncations_occluded_cache_path)
            for i in tqdm(range(7481)):
                anno_per_img = load_label_2(self.root, 'training', i)
                truncations_list_per_img = []
                occluded_list_per_img = []
                for anno in anno_per_img:
                    truncated, occluded = float(anno.truncated), float(anno.occluded)
                    truncations_list_per_img.append(truncated)
                    occluded_list_per_img.append(occluded)
                truncations_list.append(truncations_list_per_img)
                occluded_list.append(occluded_list_per_img)
            pickle.dump([truncations_list, occluded_list],
                        open(truncations_occluded_cache_path, 'wb'))
        return truncations_list, occluded_list

    def get_offline_prediction(self, index):
        imgid = self.ids[index]
        pred = pickle.load(open(os.path.join(
            self.offline_2d_predictions_dir, str(imgid) + '.pkl'), 'rb'))
        lp, rp = pred['left'], pred['right']
        return lp, rp

    def get_mask(self, index):
        imgid = self.ids[index]
        split = 'training' if self.split != 'test' else 'testing'
        imginfo = self.get_img_info(index)
        width = imginfo['width']
        height = imginfo['height']
        if split == 'training':
            mask = zarr.load(
                os.path.join(self.root, 'object', split, self.mask_disp_sub_path, 'mask_2', imgid + '.zarr')) != 0
            mask = SegmentationMask(mask, (width, height), mode='mask')
        else:
            mask = SegmentationMask(np.zeros((height, width)), (width, height), mode='mask')
        return mask

    def get_disparity(self, index):
        imgid = self.ids[index]
        split = 'training' if self.split != 'test' else 'testing'
        if split == 'training':
            path = os.path.join(self.root, 'object', split,
                                self.mask_disp_sub_path, 'disparity_2',
                                imgid + '.png')
            disp = cv2.imread(path, 2).astype(np.float32) / 256
            disp = DisparityMap(disp)
        else:
            imginfo = self.get_img_info(index)
            width = imginfo['width']
            height = imginfo['height']
            disp = DisparityMap(np.ones((height, width)))
        return disp

    def get_calibration(self, index):
        imgid = self.ids[index]
        split = 'training' if self.split != 'test' else 'testing'
        calib = load_calib(self.root, split, imgid)
        return calib

    def get_sassd_calib(self, index):
        imgid = self.ids[index]
        split = 'training' if self.split != 'test' else 'testing'
        calib_dir = os.path.join(self.root, 'object', split, 'calib')
        absolute_path = os.path.join(calib_dir, imgid+'.txt')
        return Calibration(absolute_path)

    def remove_ignore_cars(self, l, r):
        if len(l) == 0 and len(r) == 0:
            return l, r

        heights = l.heights / l.height * l.get_field('image_size')[0, 1]
        truncations = l.get_field('truncation').tolist()
        occlusions = l.get_field('occlusion').tolist()
        keep = []
        for i, (height, truncation, occlusion) in enumerate(zip(heights, truncations, occlusions)):
            if height >= 40 and truncation <= 0.15 and occlusion <= 0:
                keep.append(i)
            elif height >= 25 and truncation <= 0.3 and occlusion <= 1:
                keep.append(i)
            elif height >= 25 and truncation <= 0.5 and occlusion <= 2:
                keep.append(i)
        l = l[keep]
        r = r[keep]
        return l, r


class KITTIObjectDatasetPOB(KITTIObjectDataset):

    def __init__(self, root, split, transforms=None, filter_empty=False, offline_2d_predictions_path='', ):
        super().__init__(root, split, transforms, filter_empty, offline_2d_predictions_path, 'pob')


class KITTIObjectDatasetVOB(KITTIObjectDataset):

    def __init__(self, root, split, transforms=None, filter_empty=False, offline_2d_predictions_path='',
                 remove_ignore=True):
        # print('using dataset', self.__class__.__name__)
        super().__init__(root, split, transforms, filter_empty, offline_2d_predictions_path, 'vob',
                         remove_ignore)
