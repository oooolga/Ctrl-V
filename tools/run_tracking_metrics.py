from ultralytics import YOLO
TRACKER = YOLO("yolov8x.pt")

import numpy as np
import os, sys, pickle, torch
from PIL import Image, ImageSequence
import cv2
from ctrlv.datasets import KittiDataset, VKittiDataset, BDD100KDataset, COCO_LABELS_LOOKUP
import matplotlib.pyplot as plt

def get_video_loader(video_dir):

    def sort_name(text):
        text = text[text.find('videos_')+len('videos_'):]
        num = text[:text.find('_')]
        return int(num)

    def load_frames(image: Image, mode='RGB',size=(256,256)):
        return np.array([
            np.array(frame.resize(size).convert(mode))
            for frame in ImageSequence.Iterator(image)
        ])

    f_gt_vid = []
    f_gen_vid = []
    for fname in os.listdir(vid_dir):
        if fname.startswith('generated_videos'):
            f_gen_vid.append(fname)
        if fname.startswith('gt_videos'):
            f_gt_vid.append(fname)
    f_gt_vid.sort(key=sort_name)
    f_gen_vid.sort(key=sort_name)
    assert(len(f_gt_vid) == len(f_gen_vid))

    all_gt = []
    all_gen = []
    file_names = []
    for idx,(fgt,fgen) in enumerate(zip(f_gt_vid, f_gen_vid)):
        assert (sort_name(fgen) == sort_name(fgt))
        file_names.append(fgen)
        gt_gif = cv2.VideoCapture(os.path.join(vid_dir,fgt))
        gen_gif = cv2.VideoCapture(os.path.join(vid_dir,fgen))
        all_gt.append(gt_gif)
        all_gen.append(gen_gif)
    return file_names, all_gt, all_gen


def track_video(cv_gif, classes, conf=0.6, iou=0.4, save_path=None, prefix='gt'):
    
    if_next, frame = cv_gif.read()
    i = 0
    ret = []
    while if_next:
        result = TRACKER(frame, conf=conf, iou=iou, classes=classes)
        # result = TRACKER.track(source=frame, conf=conf, iou=iou, show=False, persist=False, classes=classes, imgsz=(320, 512))
        # Visualize the results on the frame
        if save_path is not None:
            annotated_frame = result[0].plot()
            annotated_frame = Image.fromarray(annotated_frame[:, :, ::-1])
            annotated_frame.save(os.path.join(f"{save_path}/{prefix}_result{i}.jpg"))
        ret.append(result[0].boxes)
        if_next, frame = cv_gif.read()
        i += 1
        if i == 16:
            break
    return ret

def write_bbox_to_file(vid_dir, dataset):
    if dataset == 'kitti':
        label_lookup = KittiDataset.TO_COCO_LABELS
        orig_H, orig_W = 375, 1242
    elif dataset == 'vkitti':
        label_lookup = VKittiDataset.TO_COCO_LABELS
        orig_H, orig_W = 375, 1242
    elif dataset == 'bdd100k':
        label_lookup = BDD100KDataset.TO_COCO_LABELS
        orig_H, orig_W = 720, 1280
    classes = list(set(label_lookup.values()))

    pkl_path = f"/{os.path.join(*vid_dir.split('/')[:-6])}/gt_labels_ctrl_eval"
    
    plot_path = f"{pkl_path[:pkl_path.rfind('/')]}/plots"
    os.makedirs(plot_path, exist_ok=True)
    from ultralytics.utils.metrics import ConfusionMatrix, DetMetrics, box_iou

    # cms = {interval:ConfusionMatrix(nc=len(classes), conf=0.0, iou_thres=interval/100, task='detect') for interval in range(50, 100, 5)}
    # cms_gt = {interval:ConfusionMatrix(nc=len(classes), conf=0.0, iou_thres=interval/100, task='detect') for interval in range(50, 100, 5)}
    cms = {interval:ConfusionMatrix(nc=1, conf=0.0, iou_thres=interval/100, task='detect') for interval in range(50, 100, 5)}
    cms_gt = {interval:ConfusionMatrix(nc=1, conf=0.0, iou_thres=interval/100, task='detect') for interval in range(50, 100, 5)}
    
    # classes_lookup = {v:i for i, v in enumerate(classes)}
    # class_names = [COCO_LABELS_LOOKUP[i] for i in classes_lookup.keys()]

    f_names, gt_gifs, gen_gifs = get_video_loader(vid_dir)
    generated_detections = []
    gt_detections = []
    gt_labels = []

    for f, (f_name, gt, gen) in enumerate(zip(f_names, gt_gifs, gen_gifs)):

        gt_bboxes = track_video(gt, classes=classes, iou=0.35, conf=0.10, save_path=plot_path if f<=50 else None, prefix=f'gt_{f}')
        gen_bboxes = track_video(gen, classes=classes, iou=0.35, conf=0.10, save_path=plot_path if f<=50 else None, prefix=f'gen_{f}')
        label_file = open(os.path.join(pkl_path, f"sample_{f_name.split('_')[2]}.pkl"), 'rb')
        labels = pickle.load(label_file)

        # process label files
        for frame_labels, gt_bbox, gen_bbox in zip(labels, gt_bboxes, gen_bboxes):
            
            # import pdb; pdb.set_trace()
            gt_bb = gt_bbox.xyxyn
            gt_bb[:,0] = gt_bb[:,0]*orig_W
            gt_bb[:,1] = gt_bb[:,1]*orig_H
            gt_bb[:,2] = gt_bb[:,2]*orig_W
            gt_bb[:,3] = gt_bb[:,3]*orig_H
            
            for i, label in enumerate(frame_labels):
                # if label_lookup[label['id_type']] not in classes:
                #     frame_labels.remove(label)
                # else:
                #     # to coco labels
                #     frame_labels[i]['id_type'] = label_lookup[label['id_type']]
                frame_labels[i]['id_type'] = 0
            
            data_label_bboxes = torch.empty((len(frame_labels), 4), dtype=torch.int64).to('cuda')
            data_label_cls = torch.empty((len(frame_labels)), dtype=torch.int64).to('cuda')
            for i, label in enumerate(frame_labels):
                data_label_bboxes[i] = torch.tensor(label['bbox'])
                # data_label_cls[i] = label['id_type']
                data_label_cls[i] = 0
            
            # filter
            # filter = torch.any(box_iou(gt_bb.cuda(), label_bboxes) > 0.8, dim=1)
            label_bboxes = gt_bb.cuda()# [filter] # label_bboxes[filter]
            label_cls = gt_bbox.cls.cuda()# label_cls[filter]

            gen_bb = gen_bbox.xyxyn

            gen_bb[:,0] = gen_bb[:,0]*orig_W
            gen_bb[:,1] = gen_bb[:,1]*orig_H
            gen_bb[:,2] = gen_bb[:,2]*orig_W
            gen_bb[:,3] = gen_bb[:,3]*orig_H
            temp = gen_bbox.data.clone()
            if temp.shape[1] == 7:
                temp = temp[:,torch.arange(7)!=4]
            temp[:, :4] = gen_bb
            
            # filter: size must be larger than ~ 5% of the image size
            temp = temp[torch.logical_and(gen_bbox.xywhn[:, 2] >= 0.08, gen_bbox.xywhn[:, 3] >= 0.08),::]
            filter = torch.logical_and(gt_bbox.xywhn[:, 2] >= 0.08, gt_bbox.xywhn[:, 3] >= 0.08)
            label_bboxes = label_bboxes[filter]
            label_cls = label_cls[filter].cuda()
            temp2 = gt_bbox.data.clone().cuda()[filter]
            if temp2.shape[1] == 7:
                temp2 = temp2[:,torch.arange(7)!=4]
            temp2[:, :4] = label_bboxes
            
            for c in range(len(label_cls)):
                label_cls[c] = 0# classes_lookup[int(label_cls[c])]
            # for c in range(len(data_label_cls)):
            #     try:
            #         data_label_cls[c] = classes_lookup[int(data_label_cls[c])]
            #     except Exception:
            #         import pdb; pdb.set_trace()
            for c in range(len(temp)):
                temp[c, -1] = 0# classes_lookup[int(temp[c, -1])]
            for interval in range(50, 100, 5):
                cms[interval].process_batch(temp.cuda(), label_bboxes.cuda(), label_cls.cuda())
            
            temp2[:,-1] = 0# label_cls
            for interval in range(50, 100, 5):
                cms_gt[interval].process_batch(temp2, data_label_bboxes.cuda(), data_label_cls.cuda())
            
            generated_detections.append(temp.detach().cpu().numpy())
            gt_detections.append(temp2.detach().cpu().numpy())
            gt_label = torch.cat([data_label_bboxes, data_label_cls.unsqueeze(1)], dim=1)
            gt_labels.append(gt_label.detach().cpu().numpy())
    
    with open(os.path.join(plot_path, 'detections.pickle'), 'wb') as handle:
        pickle.dump({
            'generated_detections': generated_detections,
            'gt_detections': gt_detections,
            'gt_labels': gt_labels
        }, handle, protocol=pickle.HIGHEST_PROTOCOL)

def get_tp_fp_tn(data, iou_thres, confidence_thres, run_gt_stats=True):
    from ultralytics.utils.metrics import ConfusionMatrix, DetMetrics, box_iou
    cm = ConfusionMatrix(nc=1, conf=confidence_thres, iou_thres=iou_thres, task='detect')
    for gen, gt in zip(data['generated_detections'] if not run_gt_stats else data['gt_detections'],
                       data['gt_detections'] if not run_gt_stats else data['gt_labels']):
        if gt.shape[1] == 5:
            gt_bbox = gt[:, :4]
            gt_label = gt[:, -1]
        else:
            filter = gt[:, -2] >= 0.6
            gt_bbox = gt[filter, :4]
            gt_label = gt[filter, -1]
        cm.process_batch(torch.tensor(gen), torch.tensor(gt_bbox), torch.tensor(gt_label))
    
    tp, fp, fn = cm.matrix[0,0], cm.matrix[0,1], cm.matrix[1,0]
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    return tp, fp, fn, recall, precision

def get_map(data,plot=False,dataset=None):
    all_ap = []
    if plot:
        ax = plt.subplot()
        ax.set_xlim(0.0,1.05)
        ax.set_ylim(0.0,1.05)
    for iou_th in np.arange(0.50,1.00,0.05):
        print(f'----------iou_th = {iou_th}')
        ap, rp_pairs = get_ap_iou(data,iou_threshold=iou_th)
        all_ap.append(ap)
        if plot:
            line = plot_rp(rp_pairs=rp_pairs,ax=ax)
            line.set_label(f'{iou_th:.2f}')
    if plot:
        ax.legend(loc='lower right', title="IOU cutoff", ncols=5)
        plt.savefig(f'{dataset}_map.png',bbox_inches='tight')
    return sum(all_ap)/len(all_ap), all_ap

def get_ap_iou(data,iou_threshold):
    rp_pairs = []
    for idx, conf in enumerate(np.arange(0.00, 1.01, 0.01)):
        print(conf)
        tp, fp, fn, recall, precision = get_tp_fp_tn(data, iou_thres=iou_threshold, confidence_thres=conf)
        if precision != precision:
            continue
        rp_pairs.append([recall, precision])
    ap = get_ap_from_rp(rp_pairs)
    return ap, rp_pairs

def get_ap_from_rp(rp_pairs):
    rp_pairs.sort(key=lambda x: x[0], reverse=True)
    last_r, max_p = rp_pairs[0]
    area = 0.0
    for r, p in rp_pairs:
        if p > max_p:
            area += (last_r - r) * max_p
            max_p = p
            last_r = r
    area += p * r # add the last little missing bin from r = 0 to r = minimum r in the list
    return area

def plot_rp(rp_pairs, ax):
    rp_pairs = np.array(rp_pairs)
    line = ax.plot(rp_pairs[:,0],rp_pairs[:,1])
    return line[0]


if __name__ == '__main__':
    vid_dir = sys.argv[1] # make sure to pass the path to the video directory as absolute path
    if vid_dir[-1] != "/":
        vid_dir += "/"
    dataset = sys.argv[2]

    detection_path = f"/{os.path.join(*vid_dir.split('/')[:-6])}/plots/detections.pickle"

    if not os.path.exists(detection_path):
        write_bbox_to_file(vid_dir, dataset)
    
    with open(detection_path, 'rb') as handle:
        data = pickle.load(handle)
    
    mean_ap, all_aps = get_map(data,plot=True,dataset=dataset)
    print(f"Mean AP: {mean_ap}")
    print(f"APs: {all_aps}")