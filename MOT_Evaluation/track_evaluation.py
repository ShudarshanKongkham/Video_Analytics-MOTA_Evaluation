import os
import numpy as np
from scipy.optimize import linear_sum_assignment
from easydict import EasyDict as edict
import matplotlib.pyplot as plt

# Import custom utility functions (ensure these are in your project)
from utils.io import read_txt_to_struct, extract_valid_gt_data, print_metrics
from utils.bbox import bbox_overlap
from utils.measurements import clear_mot_hungarian, idmeasures

def preprocessingDB(trackDB, gtDB, distractor_ids, iou_thres=0.5, minvis=0):
    """
    Preprocess the computed trajectory data by matching computed boxes to ground truth.
    Removes distractors and low visibility data in both trackDB and gtDB.

    Parameters:
    - trackDB: ndarray, computed trajectory data
    - gtDB: ndarray, ground-truth data
    - distractor_ids: ndarray, IDs of distractor objects in the sequence
    - iou_thres: float, bounding box overlap threshold
    - minvis: float, minimum visibility of ground truth boxes

    Returns:
    - trackDB: ndarray, preprocessed tracking data
    - gtDB: ndarray, preprocessed ground truth data
    """
    track_frames = np.unique(trackDB[:, 0])
    gt_frames = np.unique(gtDB[:, 0])
    nframes = min(len(track_frames), len(gt_frames))
    res_keep = np.ones((trackDB.shape[0], ), dtype=float)

    for frame_num in range(1, nframes + 1):
        # Get indices of detections in the current frame
        res_in_frame_idx = np.where(trackDB[:, 0] == frame_num)[0]
        res_in_frame_data = trackDB[res_in_frame_idx, :]
        gt_in_frame_idx = np.where(gtDB[:, 0] == frame_num)[0]
        gt_in_frame_data = gtDB[gt_in_frame_idx, :]

        res_num = res_in_frame_data.shape[0]
        gt_num = gt_in_frame_data.shape[0]
        overlaps = np.zeros((res_num, gt_num), dtype=float)

        # Compute overlaps between detections and ground truth boxes
        for gid in range(gt_num):
            overlaps[:, gid] = bbox_overlap(
                res_in_frame_data[:, 2:6], gt_in_frame_data[gid, 2:6])

        # Perform assignment using the Hungarian algorithm
        matched_indices = linear_sum_assignment(1 - overlaps)
        for res_idx, gt_idx in zip(*matched_indices):
            # Discard pairs with overlap lower than threshold
            if overlaps[res_idx, gt_idx] < iou_thres:
                continue

            # Discard result box if matched to a distractor or low visibility object
            if (gt_in_frame_data[gt_idx, 1] in distractor_ids or
                gt_in_frame_data[gt_idx, 8] < minvis):
                res_keep[res_in_frame_idx[res_idx]] = 0

        # Check for duplicate IDs in the same frame
        frame_id_pairs = res_in_frame_data[:, :2]
        uniq_frame_id_pairs = np.unique(frame_id_pairs, axis=0)
        has_duplicates = uniq_frame_id_pairs.shape[0] < frame_id_pairs.shape[0]
        assert not has_duplicates, f'Duplicate ID in frame {frame_num}.'

    # Keep only the valid detections
    keep_idx = np.where(res_keep == 1)[0]
    print(f'[TRACK PREPROCESSING]: Remaining {len(keep_idx)}/{len(res_keep)} computed boxes after removing distractors and low visibility boxes.')
    trackDB = trackDB[keep_idx, :]

    # Preprocess ground truth data
    valid_gt_idx = np.array([
        i for i in range(gtDB.shape[0])
        if gtDB[i, 1] not in distractor_ids and gtDB[i, 8] >= minvis
    ])
    print(f'[GT PREPROCESSING]: Remaining {len(valid_gt_idx)}/{gtDB.shape[0]} ground truth boxes after removing distractors and low visibility boxes.')
    gtDB = gtDB[valid_gt_idx, :]

    return trackDB, gtDB

def evaluate_sequence(trackDB, gtDB, distractor_ids, iou_thres=0.5, minvis=0):
    """
    Evaluate a single sequence by computing tracking metrics.

    Parameters:
    - trackDB: ndarray, tracking result data
    - gtDB: ndarray, ground-truth data
    - distractor_ids: ndarray, IDs of distractor objects
    - iou_thres: float, bounding box overlap threshold
    - minvis: float, minimum tolerable visibility

    Returns:
    - metrics: list, computed evaluation metrics
    - extra_info: EasyDict, additional information from evaluation
    """
    trackDB, gtDB = preprocessingDB(trackDB, gtDB, distractor_ids, iou_thres, minvis)
    mme, c, fp, g, missed, d, M, allfps = clear_mot_hungarian(trackDB, gtDB, iou_thres)

    gt_frames = np.unique(gtDB[:, 0])
    gt_ids = np.unique(gtDB[:, 1])
    st_ids = np.unique(trackDB[:, 1])
    f_gt = len(gt_frames)
    n_gt = len(gt_ids)
    n_st = len(st_ids)

    FN = sum(missed)
    FP = sum(fp)
    IDS = sum(mme)
    MOTP = (sum(sum(d)) / sum(c)) * 100  # Multiple Object Tracking Precision
    MOTAL = (1 - (FP + FN + np.log10(IDS + 1)) / sum(g)) * 100
    MOTA = (1 - (FP + FN + IDS) / sum(g)) * 100  # Multiple Object Tracking Accuracy
    recall = sum(c) / sum(g) * 100
    precision = sum(c) / (FP + sum(c)) * 100
    FAR = FP / f_gt  # False Alarm Rate

    # Compute Mostly Tracked, Partially Tracked, Mostly Lost
    MT_stats = np.zeros(n_gt, dtype=float)
    for i in range(n_gt):
        gt_id = gt_ids[i]
        gt_indices = np.where(gtDB[:, 1] == gt_id)[0]
        gt_length = len(gt_indices)
        gt_frames_tmp = gtDB[gt_indices, 0].astype(int)
        st_length = sum(1 if i in M[int(f - 1)].keys() else 0 for f in gt_frames_tmp)
        ratio = float(st_length) / gt_length

        if ratio >= 0.8:
            MT_stats[i] = 3  # Mostly Tracked
        elif ratio < 0.2:
            MT_stats[i] = 1  # Mostly Lost
        else:
            MT_stats[i] = 2  # Partially Tracked

    ML = np.sum(MT_stats == 1)
    PT = np.sum(MT_stats == 2)
    MT = np.sum(MT_stats == 3)

    # Compute Fragments
    fr = np.zeros(n_gt, dtype=int)
    M_arr = np.zeros((f_gt, n_gt), dtype=int)
    for i in range(f_gt):
        for gid in M[i].keys():
            M_arr[i, gid] = M[i][gid] + 1
    for i in range(n_gt):
        occurrences = np.where(M_arr[:, i] > 0)[0]
        discontinuities = np.where(np.diff(occurrences) != 1)[0]
        fr[i] = len(discontinuities)
    FRA = np.sum(fr)

    # Compute ID metrics
    idmetrics = idmeasures(gtDB, trackDB, iou_thres)

    metrics = [
        idmetrics.IDF1, idmetrics.IDP, idmetrics.IDR, recall,
        precision, FAR, n_gt, MT, PT, ML, FP, FN, IDS, FRA,
        MOTA, MOTP, MOTAL
    ]

    extra_info = edict()
    extra_info.mme = IDS
    extra_info.c = sum(c)
    extra_info.fp = FP
    extra_info.g = sum(g)
    extra_info.missed = FN
    extra_info.d = d
    extra_info.f_gt = f_gt
    extra_info.n_gt = n_gt
    extra_info.n_st = n_st
    extra_info.ML = ML
    extra_info.PT = PT
    extra_info.MT = MT
    extra_info.FRA = FRA
    extra_info.idmetrics = idmetrics
    extra_info.metrics = metrics  # Store metrics for plotting
    return metrics, extra_info

def evaluate_benchmark(all_info):
    """
    Evaluate the entire benchmark by summarizing all metrics across sequences.

    Parameters:
    - all_info: list of EasyDict, additional information from each sequence evaluation

    Returns:
    - metrics: list, summarized evaluation metrics
    """
    f_gt = sum(info.f_gt for info in all_info)
    n_gt = sum(info.n_gt for info in all_info)
    n_st = sum(info.n_st for info in all_info)
    c = sum(info.c for info in all_info)
    g = sum(info.g for info in all_info)
    fp = sum(info.fp for info in all_info)
    missed = sum(info.missed for info in all_info)
    ids = sum(info.mme for info in all_info)
    MT = sum(info.MT for info in all_info)
    PT = sum(info.PT for info in all_info)
    ML = sum(info.ML for info in all_info)
    FRA = sum(info.FRA for info in all_info)
    overlap_sum = sum(sum(sum(info.d)) for info in all_info)
    idmetrics_list = [info.idmetrics for info in all_info]

    # Summarize ID metrics
    IDTP = sum(m.IDTP for m in idmetrics_list)
    IDFP = sum(m.IDFP for m in idmetrics_list)
    IDFN = sum(m.IDFN for m in idmetrics_list)
    nbox_gt = sum(m.nbox_gt for m in idmetrics_list)
    nbox_st = sum(m.nbox_st for m in idmetrics_list)

    IDP = IDTP / (IDTP + IDFP) * 100 if (IDTP + IDFP) > 0 else 0
    IDR = IDTP / (IDTP + IDFN) * 100 if (IDTP + IDFN) > 0 else 0
    IDF1 = 2 * IDTP / (nbox_gt + nbox_st) * 100 if (nbox_gt + nbox_st) > 0 else 0
    FAR = fp / f_gt if f_gt > 0 else 0
    MOTP = (overlap_sum / c) * 100 if c > 0 else 0
    MOTAL = (1 - (fp + missed + np.log10(ids + 1)) / g) * 100 if g > 0 else 0
    MOTA = (1 - (fp + missed + ids) / g) * 100 if g > 0 else 0
    recall = c / g * 100 if g > 0 else 0
    precision = c / (fp + c) * 100 if (fp + c) > 0 else 0

    metrics = [
        IDF1, IDP, IDR, recall, precision, FAR, n_gt,
        MT, PT, ML, fp, missed, ids, FRA, MOTA, MOTP, MOTAL
    ]
    return metrics

def evaluate_tracking(sequences, track_dir, gt_dir):
    """
    Evaluate tracking results against ground truth data for given sequences.

    Parameters:
    - sequences: list of str, names of sequences to evaluate
    - track_dir: str, directory containing tracking results
    - gt_dir: str, directory containing ground truth data
    """
    all_info = []
    for seqname in sequences:
        print(f'\nEvaluating sequence: {seqname}')
        track_res = os.path.join(track_dir, seqname, 'res.txt')
        gt_file = os.path.join(gt_dir, seqname, 'gt.txt')
        assert os.path.exists(track_res), f'Tracking result {track_res} does not exist.'
        assert os.path.exists(gt_file), f'Ground truth file {gt_file} does not exist.'

        # Read tracking results and ground truth data
        trackDB = read_txt_to_struct(track_res)
        gtDB = read_txt_to_struct(gt_file)

        # Preprocess ground truth data
        gtDB, distractor_ids = extract_valid_gt_data(gtDB)

        # Evaluate sequence
        metrics, extra_info = evaluate_sequence(trackDB, gtDB, distractor_ids)
        print_metrics(f'{seqname} Evaluation', metrics)
        extra_info.seq_name = seqname  # Store sequence name for plotting
        all_info.append(extra_info)

    # Evaluate the entire benchmark
    all_metrics = evaluate_benchmark(all_info)
    print_metrics('Summary Evaluation', all_metrics)

    # Generate evaluation plots
    generate_plots(all_info, all_metrics)

def generate_plots(all_info, summary_metrics):
    """
    Generate plots of evaluation metrics for each sequence and the overall benchmark.

    Parameters:
    - all_info: list of EasyDict, additional information from each sequence evaluation
    - summary_metrics: list, summarized evaluation metrics
    """
    sequences = [info.seq_name for info in all_info]
    MOTA_list = [info.metrics[14] for info in all_info]  # MOTA is at index 14
    MOTP_list = [info.metrics[15] for info in all_info]  # MOTP is at index 15
    IDF1_list = [info.metrics[0] for info in all_info]   # IDF1 is at index 0

    # Plot MOTA for each sequence
    plt.figure(figsize=(10, 6))
    plt.bar(sequences, MOTA_list, color='skyblue')
    plt.xlabel('Sequence')
    plt.ylabel('MOTA (%)')
    plt.title('Multiple Object Tracking Accuracy (MOTA) per Sequence')
    plt.ylim(0, 100)
    plt.grid(axis='y')
    plt.show()

    # Plot MOTP for each sequence
    plt.figure(figsize=(10, 6))
    plt.bar(sequences, MOTP_list, color='salmon')
    plt.xlabel('Sequence')
    plt.ylabel('MOTP (%)')
    plt.title('Multiple Object Tracking Precision (MOTP) per Sequence')
    plt.ylim(0, 100)
    plt.grid(axis='y')
    plt.show()

    # Plot IDF1 Score for each sequence
    plt.figure(figsize=(10, 6))
    plt.bar(sequences, IDF1_list, color='lightgreen')
    plt.xlabel('Sequence')
    plt.ylabel('IDF1 Score (%)')
    plt.title('IDF1 Score per Sequence')
    plt.ylim(0, 100)
    plt.grid(axis='y')
    plt.show()

    # Plot Summary Metrics
    metrics_names = ['IDF1', 'IDP', 'IDR', 'Recall', 'Precision', 'MOTA', 'MOTP']
    summary_values = [
        summary_metrics[0],  # IDF1
        summary_metrics[1],  # IDP
        summary_metrics[2],  # IDR
        summary_metrics[3],  # Recall
        summary_metrics[4],  # Precision
        summary_metrics[14], # MOTA
        summary_metrics[15]  # MOTP
    ]

    plt.figure(figsize=(12, 7))
    bars = plt.bar(metrics_names, summary_values, color='orchid')
    plt.xlabel('Metrics')
    plt.ylabel('Value (%)')
    plt.title('Summary of Evaluation Metrics')
    plt.ylim(0, 100)
    plt.grid(axis='y')

    # Annotate bars with values
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 1,
                 f'{height:.2f}%', ha='center', va='bottom')
    plt.show()

if __name__ == '__main__':
    # Define the sequences to evaluate
    sequences = ['MOT16-11', 'MOT16-13']  # Add your sequence names here

    # Define the directories for tracking results and ground truth data
    track_dir = 'MOT_Evaluation\data'  # Replace with your tracking results directory
    gt_dir = 'MOT_Evaluation\data'    # Replace with your ground truth data directory

    # Run evaluation
    evaluate_tracking(sequences, track_dir, track_dir)
