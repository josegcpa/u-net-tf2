import argparse
import numpy as np
import h5py
from scipy import integrate
from skimage import io,measure
from tqdm import tqdm

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--prediction_path',dest='prediction_path',required=True)
    parser.add_argument('--truth_path',dest='truth_path',required=True)
    parser.add_argument('--prediction_key',dest='prediction_key',default='prediction')
    parser.add_argument('--n_iou_threshold',dest='n_iou_threshold',default=10,type=int)
    args = parser.parse_args()

    all_iou = []
    all_fp = []
    with h5py.File(args.prediction_path,'r') as F:
        with h5py.File(args.truth_path,'r') as T:
            all_K = [k for k in F.keys()]
            for k in tqdm(all_K):
                if k in T:
                    mask = T[k]['mask'][()][:,:,0]
                    prediction = F[k][args.prediction_key][()]
                    if 'refine' not in args.prediction_key:
                        prediction = np.argmax(prediction,axis=-1)
                    mask_labels = measure.label(mask,background=0)
                    prediction_labels = measure.label(prediction,background=0)
                    all_js = []
                    correct_js = []
                    for i in filter(lambda x: x!=0,np.unique(mask_labels)):
                        detection,correct_j = 0,None
                        for j in filter(lambda x: x!=0,np.unique(prediction_labels)):
                            m = mask_labels == i
                            p = prediction_labels == j
                            intersection = np.logical_and(m,p).sum()
                            union = np.sum(m) + np.sum(p) - intersection
                            iou = intersection / union
                            if iou > detection:
                                detection = iou
                                correct_j = j
                            if j not in all_js:
                                all_js.append(j)
                        if correct_j is not None:
                            correct_js.append(correct_j)
                        all_iou.append(detection)
                    n_fp = len([x for x in all_js if x not in correct_js])
                    all_fp.append(n_fp)
    all_iou = np.array(all_iou)
    all_fp = np.array(all_fp)

    all_prec = []
    all_rec = []
    all_acc = []
    all_iou_t = np.linspace(0,1,num=args.n_iou_threshold)

    gt = np.ones_like(all_iou)
    for iou_t_val in all_iou_t:
        pred_pos = all_iou > iou_t_val
        rec = pred_pos.sum() / gt.sum()
        prec = pred_pos.sum() / (pred_pos.sum() + all_fp.sum())
        acc = pred_pos.sum() / (gt.sum() + all_fp.sum())
        all_prec.append(prec)
        all_rec.append(rec)
        all_acc.append(acc)

    all_prec = np.array(all_prec)
    all_rec = np.array(all_rec)
    all_acc = np.array(all_acc)
    #print(' '.join([str(x) for x in all_prec]))
    #print(' '.join([str(x) for x in all_rec]))
    print("{},precision,{}".format(args.prediction_path,integrate.trapz(all_prec,all_iou_t)))
    print("{},recall,{}".format(args.prediction_path,integrate.trapz(all_rec,all_iou_t)))
    print("{},accuracy,{}".format(args.prediction_path,integrate.trapz(all_acc,all_iou_t)))
