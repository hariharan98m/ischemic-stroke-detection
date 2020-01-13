import numpy as np
from dataloader import SISSDataset, Rescale, RandomCrop, RandomRotate, ToMultiFloatMaskValues, Normalize, FinalRoIAlignExperiment
from torchvision import transforms, models, datasets
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib import colors
import skimage
import torch
from torchvision.ops import RoIAlign, MultiScaleRoIAlign
import pdb

def show_single_img(image, label):
    """Show image"""
    cmap = 'gray'
    if label:
        cmap = 'binary'
    plt.imshow(image, cmap = cmap)


def viz_sample(sample):
    fig = plt.figure(figsize=(20, 5))
    # fig.suptitle('Lesion Properties')

    for slice_, scan in enumerate(['DWI', 'FLAIR', 'T1', 'T2', 'Lesion Mask']):
        ax = plt.subplot(1, 5, slice_ + 1)
        show_single_img(sample[:, :, slice_], scan == 'label')
        plt.tight_layout()
        ax.set_title(scan)
        ax.axis('off')
    plt.savefig('fig.png')


class FinalRoIAlignExperiment(object):

    def __init__(self):
        self.valid_anchor_boxes = self.get_valid_anchor_boxes().astype(np.float32)
        # extracting RoI of size 56x56 on 224x224
        self.scale_base_roi_align = RoIAlign((56, 56), spatial_scale=1.0, sampling_ratio=2)  # base map is of size 56x56
        self.scale_1_roi_align = RoIAlign((28, 28), spatial_scale=1.0,
                                          sampling_ratio=2)  # wil look as 28x28 on the 112x112 image
        self.scale_2_roi_align = RoIAlign((14, 14), spatial_scale=1.0,
                                          sampling_ratio=2)  # will look at 14x14 on the 56x56 image
        self.scale_3_roi_align = RoIAlign((7, 7), spatial_scale=1.0,
                                          sampling_ratio=2)  # will look at 7x7 on the 28x28 image

    def get_valid_anchor_boxes(self):

        fe_size = 224 // 8  # (224 to 28) is 224/8 = 28
        ctr = np.zeros((fe_size * fe_size, 2))
        sub_sample_ratio = 224 // 28  # it is the height and width stride of the anchor centres.
        # we want to visit each and every point of the feature map and create a set of anchors.
        # for our case, 28x28 points on the original map will be the anchor centres.

        # Aspect Ratio of an anchor box is basically width/height. aspect ratio will always be 1 (square box)
        # Scales are bigger as the anchor box is from the base box (i.e. 512 x 512 box is twice as big as 256 x 256).
        # for scale, its better to go at 8 times.

        import math
        ar = 1.0
        scale = 56 / sub_sample_ratio  # to ensure that we get 25 on 112, 12.5 on 56 and 6.25 on 28

        # every 1x1 pixel on the 28x28 map corresponds to 8x8 on the original image. need to get the center of every 8x8 region.
        width_b = scale * math.sqrt(ar) * sub_sample_ratio
        height_b = scale * sub_sample_ratio / math.sqrt(ar)

        # Generate all the center points for all the boxes.
        ctr = np.zeros((fe_size * fe_size, 2))
        ctr_x = np.arange(sub_sample_ratio, (fe_size + 1) * sub_sample_ratio, sub_sample_ratio)
        ctr_y = ctr_x.copy()

        index = 0
        for x in range(len(ctr_x)):
            for y in range(len(ctr_y)):
                ctr[index, 0] = ctr_x[x] - sub_sample_ratio / 2
                ctr[index, 1] = ctr_y[y] - sub_sample_ratio / 2
                index += 1

        anchors = np.zeros((fe_size * fe_size, 4))
        anchors[:, 0] = ctr[:, 0] - height_b / 2.
        anchors[:, 1] = ctr[:, 1] - width_b / 2.
        anchors[:, 2] = ctr[:, 0] + height_b / 2.
        anchors[:, 3] = ctr[:, 1] + width_b / 2.
        # %%
        valid_anchor_boxes_indices = np.where(
            (anchors[:, 0] >= 0) &
            (anchors[:, 1] >= 0) &
            (anchors[:, 2] <= 224) &
            (anchors[:, 3] <= 224)
        )[0]
        valid_anchor_boxes = anchors[valid_anchor_boxes_indices]

        return valid_anchor_boxes

    def iou(self, box1, box2):
        xa1, ya1, xa2, ya2 = box1
        anchor_area = (ya2 - ya1) * (xa2 - xa1)
        xb1, yb1, xb2, yb2 = box2
        box_area = (yb2 - yb1) * (xb2 - xb1)
        inter_x1 = max([xb1, xa1])
        inter_y1 = max([yb1, ya1])
        inter_x2 = min([xb2, xa2])
        inter_y2 = min([yb2, ya2])
        if (inter_x1 < inter_x2) and (inter_y1 < inter_y2):
            iter_area = (inter_y2 - inter_y1 + 1) * \
                        (inter_x2 - inter_x1 + 1)
            iou = iter_area / \
                  (anchor_area + box_area - iter_area)
        else:
            iou = 0.
        return iou

    def get_max_ious_boxes_labels(self, scans, label224):
        max_boxes = 10
        mask = label224

        # If there is some lesion on the mask, that is, if
        if len(np.unique(mask)) != 1:
            masked_labels = skimage.measure.label(mask)

            # instances are encoded as different colors
            obj_ids = np.unique(masked_labels)
            # first id is the background, so remove it
            obj_ids = obj_ids[1:]

            # split the color-encoded mask into a set
            # of binary masks
            masks = masked_labels == obj_ids[:, None, None]

            # get bounding box coordinates for each mask
            num_objs = len(obj_ids)
            boxes = []
            for i in range(num_objs):
                pos = np.where(masks[i])
                xmin = np.min(pos[0])
                xmax = np.max(pos[0])
                ymin = np.min(pos[1])
                ymax = np.max(pos[1])
                boxes.append([xmin, ymin, xmax, ymax])

            # only choose the top 10 boxes from this.
            ious = np.empty((len(self.valid_anchor_boxes), len(boxes)), dtype=np.float32)
            ious.fill(0)
            for num1, i in enumerate(self.valid_anchor_boxes):
                for num2, j in enumerate(boxes):
                    ious[num1, num2] = self.iou(i, j)

            # choose the highest valued bounding boxes
            patches_for_objs = max_boxes // num_objs
            maxarg_ious = np.argsort(ious, axis=0)[::-1]

            selected_ious_args = []
            for obj in range(num_objs):
                obj_max_indices = maxarg_ious[:patches_for_objs, obj].tolist()
                maxarg_ious = np.delete(maxarg_ious, obj_max_indices, axis=0)
                selected_ious_args.extend(obj_max_indices)

            # Return, the selected anchor boxes coords and the class_labels
            sel_anchors = self.valid_anchor_boxes[selected_ious_args]
            # and the all ones class labels
            class_labels = [1.0] * max_boxes

            return sel_anchors, class_labels

        # so there's no lesion at all in any part of the mask
        else:
            # box_for_scan_area
            cornerVal = scans[0, 0, 0]
            pos = np.where(scans[0, :, :] != cornerVal)
            if len(pos[0]):
                x1_scan = np.min(pos[0])
                x2_scan = np.max(pos[0])
                y1_scan = np.min(pos[1])
                y2_scan = np.max(pos[1])
            else:
                return None

            box = (x1_scan, y1_scan, x2_scan, y2_scan)
            iou_vals = np.empty((len(self.valid_anchor_boxes)), dtype=np.float32)

            for index, anchor_box in enumerate(self.valid_anchor_boxes):
                iou_vals[index] = self.iou(anchor_box, box)

            maxarg_ious = np.argsort(iou_vals, axis=0)[::-1][:max_boxes]

            # Wont work as there s no way an entire anchor box in filled in this brain region
            # filter valid bounding boxes

            # valid_anchor_boxes_indices = np.where(
            #     (self.valid_anchor_boxes[:, 0] >= x1_scan) &
            #     (self.valid_anchor_boxes[:, 1] >= y1_scan) &
            #     (self.valid_anchor_boxes[:, 2] <= x2_scan) &
            #     (self.valid_anchor_boxes[:, 3] <= y2_scan)
            # )[0]

            sel_anchors = self.valid_anchor_boxes[maxarg_ious]
            class_labels = [0.0] * max_boxes

            return sel_anchors, class_labels

    def __call__(self, sample):

        # scans and the labelled mask are in (4, 224, 224) and (1,224,224)
        scans, label224 = sample[:, :, :-1], np.round(sample[:, :, -1])

        # consistent datatypes
        scans, label224 = scans.astype(np.float32), label224.astype(np.float32)

        # get the data objs into an object of shape (4, h, w) and (1, h, w)
        scans, label224 = scans.transpose((2, 0, 1)), label224[np.newaxis, ...]

        # get 10 anchor boxes formatted.
        max_iou_boxes_labels = self.get_max_ious_boxes_labels(scans, label224)

        if max_iou_boxes_labels is not None:
            anchor_boxes, class_labels = max_iou_boxes_labels

            scans, label224 = torch.from_numpy(scans), torch.from_numpy(label224)

            # image of size 224.
            # every anchor box has a size of 56x56
            cut_boxes = torch.from_numpy(anchor_boxes)
            base, scale1, scale2, scale3 = self.scale_base_roi_align(label224.unsqueeze(dim=0), [cut_boxes]), \
                                           self.scale_1_roi_align(label224.unsqueeze(0), [cut_boxes]), \
                                           self.scale_2_roi_align(label224.unsqueeze(0), [cut_boxes]), \
                                           self.scale_3_roi_align(label224.unsqueeze(0), [cut_boxes])
            # will return boxes of shape (batch_size, 1, h, w)

            # items to return from here, are the anchor boxes for all the cuts, the class_labels, the image scan 224,
            # the anchor cut labels of 56(base), 28 for 112, 14 for 56x56, 7 for 28x28

            return cut_boxes, torch.tensor(class_labels), scans, (base, scale1, scale2, scale3)
        else:
            return None

scale = Rescale(int(1.05 * 230))
crop = RandomCrop(224)
rotate = RandomRotate(20.0)
norm = Normalize()
exp5 = FinalRoIAlignExperiment()
final_transform = transforms.Compose([scale,
                               rotate,
                               crop,
                               norm,
                               exp5])

final_dataset = SISSDataset(
    num_slices = 153,
    num_scans = 2,
    root_dir = Path.cwd().parents[0],
    transform = final_transform,
    train = True
)

pdb.set_trace()

for i in range(50):
    sample = final_dataset[i]
    if sample is not None:
        cut_boxes, labels, scans, (base, scale1, scale2, scale3) = sample
        print(cut_boxes.dtype, labels.dtype, scans.dtype, (base.dtype, scale1.dtype, scale2.dtype, scale3.dtype))
        print(cut_boxes.size(), labels.size(), scans.size(), (base.size(), scale1.size(), scale2.size(), scale3.size()))
        # print(base, '\n\n')
    else:
        print(None)


#############################
from torch.utils.data._utils.collate import default_collate

batch_size = 10
def my_collate(batch):
    batch = list (filter (lambda x:(x is not None and x[0].size()[0]== batch_size), batch))
    try:
        collated = default_collate(batch)
        return collated
    except:
        for i in batch:
            cut_boxes, labels, scans, (base, scale1, scale2, scale3) = i
            print(i, ': ', cut_boxes.shape, labels.shape, scans.shape,
                  (base.shape, scale1.shape, scale2.shape, scale3.shape))

def filter_collate(batch):
    batch = list( filter(lambda x: (x is not None and x[0].size()[0] == batch_size), batch) )
    return default_collate(batch)

#
# dataloader = torch.utils.data.DataLoader(final_dataset, 16, True, collate_fn= my_collate)
# for batch in dataloader:
#     cut_boxes, labels, scans, (base, scale1, scale2, scale3) = batch
#     print(cut_boxes.shape, labels.shape, scans.shape, (base.shape, scale1.shape, scale2.shape, scale3.shape))
#     break
#
# for i, sample in enumerate(final_dataset):
#     if sample is not None:
#         cut_boxes, labels, scans, (base, scale1, scale2, scale3) = sample
#         print(i, ': ', cut_boxes.shape, labels.shape, scans.shape, (base.shape, scale1.shape, scale2.shape, scale3.shape))

dataloader = torch.utils.data.DataLoader(final_dataset, 16, True, collate_fn= my_collate)
for i, batch in enumerate(dataloader):
    cut_boxes, labels, scans, (base, scale1, scale2, scale3) = batch
    split_boxes_list = torch.split(batch, split_size_or_sections=1, dim=0)

    print(i, ': ', cut_boxes.shape, labels.shape, scans.shape, (base.shape, scale1.shape, scale2.shape, scale3.shape))


#############################


# final_transform_wo_exp5 = transforms.Compose([scale,
#                                rotate,
#                                crop,
#                                norm])
# dataset = SISSDataset(
#     num_slices = 153,
#     num_scans = 2,
#     root_dir = Path.cwd().parents[0],
#     transform = final_transform_wo_exp5,
#     train = True
# )
#
# sample = dataset[46]
# viz_sample(sample)
#
# scans, label224 = sample[:, :, :-1], np.round(sample[:, :, -1])
#
# print('unique values in label224: ', np.unique(label224))
# # consistent datatypes
# scans, label224 = scans.astype(np.float32), label224.astype(np.float32)
#
# # get the data objs into an object of shape (4, h, w) and (1, h, w)
# scans, label224 = scans.transpose((2, 0, 1)), label224[np.newaxis, ...]
# max_boxes = 10
# mask = label224
# print('unique values in mask: ', np.unique(mask))
#
# # If there is some lesion on the mask, that is, if
# if len(np.unique(mask)) != 1:
#     masked_labels = skimage.measure.label(mask)
#
#     # instances are encoded as different colors
#     obj_ids = np.unique(masked_labels)
#
#     print('obj ids:', obj_ids)
#     # first id is the background, so remove it
#     obj_ids = obj_ids[1:]
#
#     # split the color-encoded mask into a set
#     # of binary masks
#     masks = masked_labels == obj_ids[:, None, None]
#
#     # get bounding box coordinates for each mask
#     num_objs = len(obj_ids)
#
#     print(num_objs)
