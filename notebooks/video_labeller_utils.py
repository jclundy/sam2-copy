
import os
import numpy as np
# import torch
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image
from sam2.build_sam import build_sam2_video_predictor

def propagate_in_video(predictor, inference_state):
    video_segments = {}  # video_segments contains the per-frame segmentation results
    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
        video_segments[out_frame_idx] = {
            out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
            for i, out_obj_id in enumerate(out_obj_ids)
        }
    return video_segments

def obtain_bounding_boxes(frame_names, video_segments, label_array):
    for out_frame_idx in range(0, len(frame_names)):
        for out_obj_id, out_mask in video_segments[out_frame_idx].items():
            coords = get_bounding_box_from_mask(out_mask)
            cx, cy, w, h = convert_points_to_center_and_dimensions(coords)           
            label_array[out_frame_idx][out_obj_id][0] = cx
            label_array[out_frame_idx][out_obj_id][1] = cy
            label_array[out_frame_idx][out_obj_id][2] = w
            label_array[out_frame_idx][out_obj_id][3] = h

def show_mask(mask, ax, obj_id=None, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        cmap = plt.get_cmap("tab10")
        cmap_idx = 0 if obj_id is None else obj_id
        color = np.array([*cmap(cmap_idx)[:3], 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_points(coords, labels, ax, marker_size=200):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)


def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))

def get_bounding_box_from_mask(mask):
    coords = np.argwhere(mask == True)
    if(coords.size == 0):
        return [(0, 0), (0, 0)] 
    
    xmin = np.min(coords[:,2])
    xmax = np.max(coords[:,2])

    ymin = np.min(coords[:,1])
    ymax = np.max(coords[:,1])
    return [(xmin, ymin), (xmax, ymax)]

def draw_bounding_box(coords, ax):
    xmin = coords[0][0]
    xmax = coords[1][0]

    ymin = coords[0][1]
    ymax = coords[1][1]
    height = ymax - ymin
    width = xmax - xmin
    ax.add_patch(Rectangle((xmin,ymin),width,height,
                edgecolor='red',
                facecolor='none',
                lw=0.5))

def draw_bounding_box_from_scaled_label(cx, cy, w, h, imageSize, ax):
    img_w = imageSize[0]
    img_h = imageSize[1]
    xmin = (cx - w/2) * img_w
    ymin = (cy - h/2) * img_h
    width = w * img_w
    height = h * img_h
    ax.add_patch(Rectangle((xmin,ymin),width,height,
            edgecolor='red',
            facecolor='none',
            lw=0.5))
    
def convert_points_to_center_and_dimensions(coords):
    xmin = coords[0][0]
    xmax = coords[1][0]

    ymin = coords[0][1]
    ymax = coords[1][1]

    center_x = (xmin+xmax)/2
    center_y = (ymin + ymax)/2
    height = ymax - ymin
    width = xmax - xmin
    return center_x, center_y, width, height

def normalize_bounding_box_parameters(cx, cy, w, h, img_w, img_h):
    new_cx = cx / img_w
    new_cy = cy / img_h
    new_w = w / img_w
    new_h = h / img_h
    return new_cx, new_cy, new_w, new_h

def write_label_data_to_file(frame_name, output_dir, frame_labels):
    basename= os.path.splitext(os.path.basename(frame_name))[0]
    save_file_path = os.path.join(output_dir, basename + ".txt")
    print_buffer = []
    for object_idx in range(len(frame_labels)):
        label = frame_labels[object_idx]
        cx = label[0]
        cy = label[1]
        w = label[2]
        h = label[3]
        class_id = 1
        if(object_idx) == 0:
            class_id = 0

        # negative bounding box - not detected
        if(cx > 0 ):
            print_buffer.append("{} {:.3f} {:.3f} {:.3f} {:.3f}".format(class_id, cx, cy, w, h))
            
    print("\n".join(print_buffer), file= open(save_file_path, "w"))