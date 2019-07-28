import os
import argparse
import numpy as np
# from utils.cython_bbox import bbox_overlaps
from model.utils.cython_bbox import bbox_overlaps
import pickle
import torch
# from mmcv.image import imread, imwrite
import cv2
# from nms2 import py_cpu_nms_m_consider_area2
# import ipdb
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description='constructing tubes')
    parser.add_argument(
        '--data_dir',
        type=str,
        default="/home/linhaojie/data/TSD-Person"
    )
    parser.add_argument(
        '--output_path',
        type=str,
        default="/home/linhaojie/data/tubes/temp_tubes.pkl"
    )
    parser.add_argument(
        '--vis_dir',
        type=str,
        default=None
    )


    return parser.parse_args()


def nms(boxes, scores, overlap=0.5, top_k=200):
    """Apply non-maximum suppression at test time to avoid detecting too many
    overlapping bounding boxes for a given object.
    Args:
        boxes: (tensor) The location preds for the img, Shape: [num_priors,4].
        scores: (tensor) The class predscores for the img, Shape:[num_priors].
        overlap: (float) The overlap thresh for suppressing unnecessary boxes.
        top_k: (int) The Maximum number of box preds to consider.
    Return:
        The indices of the kept boxes with respect to num_priors.
    """

    keep = scores.new(scores.size(0)).zero_().long()
    if boxes.numel() == 0:
        return keep
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    area = torch.mul(x2 - x1, y2 - y1)
    v, idx = scores.sort(0)  # sort in ascending order
    # I = I[v >= 0.01]
    idx = idx[-top_k:]  # indices of the top-k largest vals
    xx1 = boxes.new()
    yy1 = boxes.new()
    xx2 = boxes.new()
    yy2 = boxes.new()
    w = boxes.new()
    h = boxes.new()

    # keep = torch.Tensor()
    count = 0
    while idx.numel() > 0:
        i = idx[-1]  # index of current largest val
        # keep.append(i)
        keep[count] = i
        count += 1
        if idx.size(0) == 1:
            break
        idx = idx[:-1]  # remove kept element from view
        # load bboxes of next highest vals
        torch.index_select(x1, 0, idx, out=xx1)
        torch.index_select(y1, 0, idx, out=yy1)
        torch.index_select(x2, 0, idx, out=xx2)
        torch.index_select(y2, 0, idx, out=yy2)
        # store element-wise max with next highest score
        xx1 = torch.clamp(xx1, min=x1[i])
        yy1 = torch.clamp(yy1, min=y1[i])
        xx2 = torch.clamp(xx2, max=x2[i])
        yy2 = torch.clamp(yy2, max=y2[i])
        w.resize_as_(xx2)
        h.resize_as_(yy2)
        w = xx2 - xx1
        h = yy2 - yy1
        # check sizes of xx1 and xx2.. after each iteration
        w = torch.clamp(w, min=0.0)
        h = torch.clamp(h, min=0.0)
        inter = w*h
        # IoU = i / (area(a) + area(b) - i)
        rem_areas = torch.index_select(area, 0, idx)  # load remaining areas)
        union = (rem_areas - inter) + area[i]
        IoU = inter/union  # store result in iou
        # keep only elements with an IoU <= overlap
        idx = idx[IoU.le(overlap)]
    return keep, count


def score_of_edge(v1, v2, iouth, costtype):
    """
    :param v1: live paths
    :param v2:  frames
    :param iouth:
    :param costtype:
    :return:
    """
    # Number of detections at frame t
    N2 = v2['boxes'].shape[0]
    score = np.zeros((1, N2))
    iou = bbox_overlaps(np.ascontiguousarray(v2['boxes'], dtype=np.float),
                        np.ascontiguousarray(v1['boxes'][-1].reshape(1, -1), dtype=np.float))
    for i in range(0, N2):
        if iou.item(i) >= iouth:
            scores2 = v2['scores'][i]
            scores1 = v1['scores'][-1]
            # if len(v1['allScores'].shape)<2:
            #    v1['allScores'] = v1['allScores'].reshape(1,-1)
            score_similarity = np.sqrt(
                np.sum(((v1['allScores'][-1, :].reshape(1, -1) - v2['allScores'][i, :].reshape(1, -1)) ** 2)))
            if costtype == 'score':
                score[:, i] = scores2
            elif costtype == 'scrSim':
                score[:, i] = 1.0 - score_similarity
            elif costtype == 'scrMinusSim':
                score[:, i] = scores2 + (1. - score_similarity)
    return score


def getPathCount(live_paths):
    if len(live_paths)>0 and 'boxes' in live_paths[0]:
        lp_count = len(live_paths)
    else:
        lp_count = 0
    return lp_count


def sort_live_paths(live_paths, path_order_score, dead_paths, dp_count, gap):
    inds = path_order_score.flatten().argsort()[::-1]
    sorted_live_paths = []
    lpc = 0
    for lp in range(getPathCount(live_paths)):
        olp = inds[lp]
        if live_paths[olp]['lastfound'] < gap:
            sorted_live_paths.append({'boxes': None, 'scores': None, 'allScores': None,
                                      'pathScore': None, 'foundAt': None, 'count': None, 'lastfound': None})
            sorted_live_paths[lpc]['boxes'] = live_paths[olp]['boxes']
            sorted_live_paths[lpc]['scores'] = live_paths[olp]['scores']
            sorted_live_paths[lpc]['allScores'] = live_paths[olp]['allScores']
            sorted_live_paths[lpc]['pathScore'] = live_paths[olp]['pathScore']
            sorted_live_paths[lpc]['foundAt'] = live_paths[olp]['foundAt']
            sorted_live_paths[lpc]['count'] = live_paths[olp]['count']
            sorted_live_paths[lpc]['lastfound'] = live_paths[olp]['lastfound']
            lpc += 1
        else:
            dead_paths.append({'boxes': None, 'scores': None, 'allScores': None,
                               'pathScore': None, 'foundAt': None, 'count': None, 'lastfound': None})
            dead_paths[dp_count]['boxes'] = live_paths[olp]['boxes']
            dead_paths[dp_count]['scores'] = live_paths[olp]['scores']
            dead_paths[dp_count]['allScores'] = live_paths[olp]['allScores']
            dead_paths[dp_count]['pathScore'] = live_paths[olp]['pathScore']
            dead_paths[dp_count]['foundAt'] = live_paths[olp]['foundAt']
            dead_paths[dp_count]['count'] = live_paths[olp]['count']
            dead_paths[dp_count]['lastfound'] = live_paths[olp]['lastfound']
            dp_count = dp_count + 1
    return sorted_live_paths, dead_paths, dp_count


def fill_gaps(paths, gap):
    gap_filled_paths = []
    if len(paths)>0 and 'boxes' in paths[0]:
        g_count = 0
        for lp in range(getPathCount(paths)):
            if len(paths[lp]['foundAt']) > gap:
                gap_filled_path_dict = {'start': None, 'end': None,
                                        'pathScore': None, 'foundAt': None,
                                        'count': None, 'lastfound': None,
                                        'boxes': [], 'scores': [], 'allScores': None}
                gap_filled_paths.append(gap_filled_path_dict)
                gap_filled_paths[g_count]['start'] = paths[lp]['foundAt'][0]
                gap_filled_paths[g_count]['end'] = paths[lp]['foundAt'][-1]
                gap_filled_paths[g_count]['pathScore'] = paths[lp]['pathScore']
                gap_filled_paths[g_count]['foundAt'] = paths[lp]['foundAt']
                gap_filled_paths[g_count]['count'] = paths[lp]['count']
                gap_filled_paths[g_count]['lastfound'] = paths[lp]['lastfound']
                count = 0
                i = 0
                while i < len(paths[lp]['scores']):
                    diff_found = paths[lp]['foundAt'][i] - paths[lp]['foundAt'][max(i, 0)]
                    if count == 0 or diff_found == 0:
                        gap_filled_paths[g_count]['boxes'].append(paths[lp]['boxes'][i])
                        gap_filled_paths[g_count]['scores'].append(paths[lp]['scores'][i])
                        if count == 0:
                            gap_filled_paths[g_count]['allScores'] = paths[lp]['allScores'][i, :].reshape(1, -1)
                        else:
                            gap_filled_paths[g_count]['allScores'] = \
                                np.concatenate((gap_filled_paths[g_count]['allScores'],
                                                paths[lp]['allScores'][i, :].reshape(1, -1)), axis=0)
                        i += 1
                        count += 1
                    else:
                        for d in range(diff_found):
                            gap_filled_paths[g_count]['boxes'].append(paths[lp]['boxes'][i])
                            gap_filled_paths[g_count]['scores'].append(paths[lp]['scores'][i])
                            assert gap_filled_paths[g_count]['allScores'].shape[
                                       0] > 1, 'allScores shape dim==0 must be >1'
                            gap_filled_paths[g_count]['allScores'] = \
                                np.concatenate((gap_filled_paths[g_count]['allScores'],
                                                paths[lp]['allScores'][i, :].reshape(1, -1)), axis=0)
                            count += 1
                        i += 1
                g_count += 1
    return gap_filled_paths


def incremental_linking(frames, iouth, costtype, jumpgap, threshgap):
    """

    :param frames:
        action_frames[frame_index]['boxes'] = boxes
        action_frames[frame_index]['scores'] = scores
        action_frames[frame_index]['allScores'] = allscores
    :param iouth:
    :param costtype:
    :param jumpgap:
    :param threshgap:
    :return:
    """

    num_frames = len(frames)
    # online path building

    live_paths = []  # Stores live paths
    dead_paths = []  # Store the paths that have been terminated
    dp_count = 0
    for t in range(num_frames):
        num_box = frames[t]['boxes'].shape[0]
        # if first frame, start paths
        if t == 0:
            # Start a path for each box in first frame
            for b in range(num_box):
                live_paths.append({'boxes': [], 'scores': [], 'allScores': None,
                                   'pathScore': None, 'foundAt': [], 'count': 1, 'lastfound': 0})

                live_paths[b]['boxes'].append(frames[t]['boxes'][b, :])  # bth box x0,y0,x1,y1 at frame t
                live_paths[b]['scores'].append(frames[t]['scores'][b])  # action score of bth box at frame t
                live_paths[b]['allScores'] = frames[t]['allScores'][b, :].reshape(1,-1)  # scores for all action for bth box at frame t
                live_paths[b]['pathScore'] = frames[t]['scores'][b]  # current path score at frame t
                live_paths[b]['foundAt'].append(0)  # frame box was found in
                live_paths[b]['count'] = 1  # current box count for bth box tube
                live_paths[b]['lastfound'] = 0  # diff between current frame and last frame in bth path
        else:
            # Link each path to detections at frame t
            lp_count = getPathCount(live_paths)  # total paths at time t
            edge_scores = np.zeros((lp_count, num_box))  # (path count) x (number of boxes in frame t)
            for lp in range(lp_count):  # for each path, get linking (IoU) score with detections at frame t
                edge_scores[lp, :] = score_of_edge(live_paths[lp], frames[t], iouth, costtype)

            dead_count = 0
            covered_boxes = np.zeros(num_box)
            path_order_score = np.zeros((1, lp_count))
            for lp in range(lp_count):
                # Check whether path has gone stale
                if live_paths[lp]['lastfound'] < jumpgap:
                    # IoU scores for path lp
                    box_to_lp_score = edge_scores[lp, :]
                    if np.sum(box_to_lp_score) > 0.0:  # check if there's at least one match to detection in this frame
                        maxInd = np.argmax(box_to_lp_score)
                        m_score = np.max(box_to_lp_score)
                        live_paths[lp]['count'] = live_paths[lp]['count'] + 1
                        #lpc = live_paths[lp]['count']
                        # Add detection to live path lp
                        live_paths[lp]['boxes'].append(frames[t]['boxes'][maxInd, :])
                        live_paths[lp]['scores'].append(frames[t]['scores'][maxInd])
                        live_paths[lp]['allScores'] = \
                            np.vstack((live_paths[lp]['allScores'], frames[t]['allScores'][maxInd, :].reshape(1, -1)))
                        # Keep running sum of the path lp
                        live_paths[lp]['pathScore'] += m_score
                        # Record the frame at which the detections were added to path lp
                        live_paths[lp]['foundAt'].append(t)
                        # Reset when we last added to path lp
                        live_paths[lp]['lastfound'] = 0
                        # Squash detection since it's been assigned
                        edge_scores[:, maxInd] = 0
                        covered_boxes[maxInd] = 1
                    else:
                        # if we have no match of this path with a detection at frame t, record the miss
                        live_paths[lp]['lastfound'] += 1
                    scores = sorted(live_paths[lp]['scores'])
                    num_sc = len(scores)
                    path_order_score[:,lp] = np.mean(np.asarray(scores[int(max(0, num_sc - jumpgap-1)):num_sc]))
                else:
                    # If the path is stale, increment the dead_count
                    dead_count += 1

            # Sort the path based on score of the boxes and terminate dead path
            live_paths, dead_paths, dp_count = sort_live_paths(live_paths, path_order_score, dead_paths, dp_count,
                                                               jumpgap)
            lp_count = getPathCount(live_paths)

            # start new paths using boxes that are not assigned
            if np.sum(covered_boxes) < num_box:
                for b in range(num_box):
                    if not covered_boxes.flatten()[b]:
                        live_paths.append({'boxes': [], 'scores': [], 'allScores': None,
                                           'pathScore': None, 'foundAt': [], 'count': 1, 'lastfound': 0})
                        live_paths[lp_count]['boxes'].append(frames[t]['boxes'][b, :])  # bth box x0,y0,x1,y1 at frame t
                        live_paths[lp_count]['scores'].append(
                            frames[t]['scores'][b])  # action score of bth box at frame t
                        live_paths[lp_count]['allScores'] = frames[t]['allScores'][b, :].reshape(1,
                                                                                                 -1)  # scores for all action for bth box at frame t
                        live_paths[lp_count]['pathScore'] = frames[t]['scores'][b]  # current path score at frame t
                        live_paths[lp_count]['foundAt'].append(t)  # frame box was found in
                        live_paths[lp_count]['count'] = 1  # current box count for bth box tube
                        live_paths[lp_count]['lastfound'] = 0  # last frame box was found
                        lp_count += 1
        #print(t)
        #for i in range(len(live_paths)): print(live_paths[i]['pathScore'])
        #for i in range(len(live_paths)): print(live_paths[i]['scores'])

    live_paths = fill_gaps(live_paths, threshgap)
    dead_paths = fill_gaps(dead_paths, threshgap)
    lp_count = getPathCount(live_paths)
    lp = lp_count
    if len(dead_paths) > 0 and 'boxes' in dead_paths[0]:
        for dp in range(len(dead_paths)):
            live_paths.append({'boxes': None, 'scores': None, 'allScores': None,
                               'pathScore': None, 'foundAt': None, 'count': None, 'lastfound': None,
                               'start':None, 'end':None})
            live_paths[lp]['start'] = dead_paths[dp]['start']
            live_paths[lp]['end'] = dead_paths[dp]['end']
            live_paths[lp]['boxes'] = dead_paths[dp]['boxes']
            live_paths[lp]['scores'] = dead_paths[dp]['scores']
            live_paths[lp]['allScores'] = dead_paths[dp]['allScores']
            live_paths[lp]['pathScore'] = dead_paths[dp]['pathScore']
            live_paths[lp]['foundAt'] = dead_paths[dp]['foundAt']
            live_paths[lp]['count'] = dead_paths[dp]['count']
            live_paths[lp]['lastfound'] = dead_paths[dp]['lastfound']
            lp += 1
    return live_paths


def dofilter(frames, action_index, frame_index, nms_thresh, score_ths=0.3):
    # filter out least likely detections for actions
    scores = frames[frame_index]['scores'][:, action_index]
    pick = np.where(scores > score_ths)
    scores = scores[pick]
    boxes = frames[frame_index]['boxes'][pick, :].squeeze(0)
    allscores = frames[frame_index]['scores'][pick, :].squeeze(0)
    # sort in descending order
    pick = np.argsort(scores)[::-1]
    # pick at most 50
    to_pick = min(50, len(pick))
    pick = pick[:to_pick]
    scores = scores[pick]
    boxes = boxes[pick, :]
    allscores = allscores[pick, :]
    # Perform nms on picked boxes
    dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32)
    if len(boxes)==0 or len(scores) == 0 or len(allscores)==0:
        return boxes, scores, allscores

    pick, counts = nms(torch.from_numpy(boxes), torch.from_numpy(scores), nms_thresh)  # idsn - ids after nms
    pick = pick[:counts]
    pick = pick[:counts].cpu().numpy()
    boxes = boxes[pick, :]
    scores = scores[pick]
    allscores = allscores[pick, :]
    return boxes, scores, allscores


def genActionPaths(frames, action_index, nms_thresh, iouth, costtype, gap, score_ths=0.1):
    '''
    :param frames:
    :param action_index:
    :param nms_thresh:
    :param iouth:
    :param costtype:
    :param gap:
    :return:
    '''
    action_frames = []
    for frame_index in range(len(frames)):
        boxes, scores, allscores = dofilter(frames, action_index, frame_index, nms_thresh, score_ths)
        action_frames.append({'boxes': None, 'scores': None, 'allScores': None})
        action_frames[frame_index]['boxes'] = boxes
        action_frames[frame_index]['scores'] = scores
        action_frames[frame_index]['allScores'] = allscores
    paths = incremental_linking(action_frames, iouth, costtype, gap, gap)
    return paths


def interpolate(path):
    # interpolate boxes, scores, foundAt, count
    #foundAt =  list(range(path['start'], path['end']+1))
    scores = []
    boxes = []
    count = path['count']
    for i in range(count-1):
        if path['foundAt'][i+1] != path['foundAt'][i] + 1:
            n = path['foundAt'][i+1] - path['foundAt'][i] + 1

            scores_temp = np.linspace(path['scores'][i], path['scores'][i+1], n)
            scores.extend(scores_temp[:-1])

            boxes_temp = [np.zeros(4) for i in range(n)]
            boxes_x1_list = np.linspace(path['boxes'][i][0], path['boxes'][i+1][0], n)
            boxes_y1_list = np.linspace(path['boxes'][i][1], path['boxes'][i+1][1], n)
            boxes_x2_list = np.linspace(path['boxes'][i][2], path['boxes'][i+1][2], n)
            boxes_y2_list = np.linspace(path['boxes'][i][3], path['boxes'][i+1][3], n)

            for ibox in range(n):
                boxes_temp[ibox][0] = boxes_x1_list[ibox]
                boxes_temp[ibox][1] = boxes_y1_list[ibox]
                boxes_temp[ibox][2] = boxes_x2_list[ibox]
                boxes_temp[ibox][3] = boxes_y2_list[ibox]
            boxes.extend(boxes_temp[:-1])
        else:
            scores.append(path['scores'][i])
            boxes.append(path['boxes'][i])

    # last frame
    scores.append(path['scores'][-1])
    boxes.append(path['boxes'][-1])

    path['boxes'] = boxes
    path['scores'] = scores
    path['foundAt'] = list(range(path['start'], path['end']+1))
    path['count'] = path['end'] - path['start'] + 1

    return path


def visual_path(paths, video_id, frames_dir, output_dir):

    print("processing {} video".format(video_id))
    for ipath in range(len(paths)):
        print("processing {} path".format(ipath))
        for iframe in range(len(paths[ipath]['foundAt'])):
            img_id = str(paths[ipath]['foundAt'][iframe]).zfill(5)
            if not os.path.exists(os.path.join(output_dir, os.path.basename("{}-{}.png".format(video_id, img_id)))):
                frame_path = os.path.join(frames_dir, "{}-{}.png".format(video_id, img_id))
            else:
                frame_path = os.path.join(output_dir, "{}-{}.png".format(video_id, img_id))
            img = imread(frame_path)
            bbox = paths[ipath]['boxes'][iframe]
            bbox_int = bbox.astype(np.int32)
            left_top = (bbox_int[0], bbox_int[1])
            right_bottom = (bbox_int[2], bbox_int[3])
            cv2.rectangle(
                img, left_top, right_bottom, (0, 255, 0), thickness=1)
            cv2.putText(img, str("person-{}".format(ipath)), (bbox_int[0], bbox_int[1] - 2),
                        cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 0))
            imwrite(img, os.path.join(output_dir, os.path.basename(frame_path)))


def main():

    args = parse_args()
    # detectron
    detectron_mask = pickle.load(open("/home/linhaojie/det_rslt/mask.pkl", "rb"), encoding="bytes")
    detectron_mask_class = [0]

    # jjw
    wad = pickle.load(open("/home/linhaojie/det_rslt/wad_rslt.pkl", "rb"))

    videos_paths = {}

    ######## nms_max ###############
    thresh_iou = 0.3
    thresh_obj = 0.5
    ###############################
    if args.vis_dir:
        if not os.path.exists(args.vis_dir):
            os.mkdir(args.vis_dir)

    video_ids = os.listdir(args.data_dir)
    print(video_ids)
    for video_id in video_ids:

        frames = []
        frames_num = len(os.listdir(os.path.join(args.data_dir, video_id)))

        for i in range(frames_num):

            ########## adding dets ######################
            dets = []

            jjw_key = "{}-{}.png".format(video_id, str(i).zfill(5))
            # append detectron maskrcnn result
            for c in detectron_mask_class:
                if jjw_key in detectron_mask.keys():
                    dets.append(detectron_mask[jjw_key][c])
                else:
                    dets.append(np.array([]).reshape(-1, 5))

            # wad
            if jjw_key in wad.keys() and wad[jjw_key].__len__() > 0:
                dets.append(wad[jjw_key])
            else:
                dets.append(np.array([]).reshape(-1, 5))

            dets = np.vstack(dets)
            #########################################

            ###### filter dets by scores ###########
            score_ths = 0.5
            keep = np.where(dets[:, -1]>score_ths)
            dets = dets[keep]
            #######################################


            ##### filter dets by seg info #########
            seg_prob = cv2.imread(os.path.join('/home/linhaojie/seg_rslt', jjw_key))
            keep_dets = []
            for det in dets:
               idet = det.astype(np.int)
               point1 = (idet[3], idet[0])
               point2 = (idet[3], idet[2])
               if seg_prob[point1].mean() > 0 or seg_prob[point2].mean() > 0:
                   keep_dets.append(det)
            keep_dets = np.array(keep_dets).reshape(-1, 5)
            dets = keep_dets
            #######################################

            ##### filter by nms_m #################
            dets = py_cpu_nms_m_consider_area2(dets, thresh=thresh_iou, op="max", thresh_obj=thresh_obj)
            #######################################

            boxes = dets[:, :-1]
            scores = dets[:, -1].reshape(-1, 1)

            frames.append({
                "scores": np.hstack((scores, np.ones(scores.shape)-scores)),
                "boxes": boxes
            })

        paths = []
        for path in genActionPaths(frames, 0, nms_thresh=0.3, iouth=0.1, costtype="score", gap=3):
            if path['count'] != (path['end'] - path['start'] + 1):
                print("interpolating video {}" .format(video_id))
                paths.append(interpolate(path))
            else:
                paths.append(path)

        videos_paths[video_id] = paths
        if args.vis_dir:
            visual_path(paths, video_id, os.path.join(args.data_dir, video_id), os.path.join(args.vis_dir, video_id))
    with open(args.output_path.format(thresh_iou, thresh_obj), "wb") as f:
            pickle.dump(videos_paths, f, protocol=2)


if __name__ == "__main__":
    main()

