import numpy as np
import sklearn.metrics 
import matplotlib.pyplot as plt
import tqdm

def prcurve_from_similarity_matrix(GT_MAP, SIM_MAP, nintervals, visualize = False):
    GTP = np.sum(GT_MAP)
    alt_GTP = np.sum(np.sum(GT_MAP,axis=1) > 0)
    precisions = []
    recalls = []
    alt_precisions = []
    alt_recalls = []
    #min_dist = np.min(SIM_MAP)
    max_sim = np.max(SIM_MAP)
    min_sim = np.min(SIM_MAP)
    start = max_sim#2*min_dist
    step = (max_sim - min_sim) / nintervals#(1+start) / nintervals#(2-start)/nintervals

    def get_threshold(i):
        return start - step * (i+1)

    prev_alt_TP = np.zeros((GT_MAP.shape[0]))
    
    print("Generating PR curve ...")
    for i in tqdm.tqdm(range(nintervals+1)):
        threshold = get_threshold(i)
        MASK = SIM_MAP >= threshold
        TP = np.sum(GT_MAP * MASK)
        FP = np.sum((1-GT_MAP)*MASK)
        precision = TP/(TP+FP)
        recall = TP/GTP
        precisions.append(precision)
        recalls.append(recall)
        alt_FP = np.logical_and(np.sum((1-GT_MAP)*MASK,axis=1) > 0, np.logical_not(prev_alt_TP)) # nqueryx1, rows with at least one false positive
        alt_TP = np.logical_and(np.sum(GT_MAP * MASK,axis = 1) > 0,np.logical_not(alt_FP)) # nqueryx1, rows with at least one true positive and no false positives
        prev_alt_TP = np.logical_or(prev_alt_TP,alt_TP)
        alt_TP = np.sum(alt_TP)
        alt_FP = np.sum(alt_FP)
        alt_precision = alt_TP/(alt_TP+alt_FP)
        alt_recall = alt_TP/alt_GTP
        alt_precisions.append(alt_precision)
        alt_recalls.append(alt_recall)
    print("Done")
    nintervals = nintervals + 1

    recalls = np.array(recalls)
    precisions = np.array(precisions)
    auc = sklearn.metrics.auc(recalls, precisions)
    print("Area under curve: ",auc)

    alt_recalls = np.array(alt_recalls)
    alt_precisions = np.array(alt_precisions)
    alt_auc = sklearn.metrics.auc(alt_recalls, alt_precisions)
    print("Area under curve (alt): ",alt_auc)

    f1_scores = 2*(precisions * recalls)/(precisions+recalls+1e-6)
    optimal_threshold_index = np.argmax(f1_scores)
    optimal_threshold = get_threshold(optimal_threshold_index)
    print("optimal threshold: ",optimal_threshold)
    print("Precision at optimal threshold: ", precisions[optimal_threshold_index])
    print("Recall at optimal threshold: ", recalls[optimal_threshold_index])

    alt_f1_scores = 2*(alt_precisions * alt_recalls)/(alt_precisions+alt_recalls+1e-6)
    alt_optimal_threshold_index = np.argmax(alt_f1_scores)
    alt_optimal_threshold = get_threshold(alt_optimal_threshold_index)
    print("optimal threshold (alt): ",alt_optimal_threshold)
    print("Precision at optimal threshold (alt): ", alt_precisions[alt_optimal_threshold_index])
    print("Recall at optimal threshold (alt): ", alt_recalls[alt_optimal_threshold_index])

    max_recall_100 = 0
    max_recall_99 = 0
    max_recall_95 = 0

    for p,r in zip(precisions, recalls):
        if p >= 0.95:
            max_recall_95 = max(max_recall_95, r)
            if p >= 0.99:
                max_recall_99 = max(max_recall_99, r)
                if p >= 1.:
                    max_recall_100 = max(max_recall_100, r)

    print("R@100P:", max_recall_100)
    print("R@99P:", max_recall_99)
    print("R@95P:", max_recall_95)

    max_recall_100 = 0
    max_recall_99 = 0
    max_recall_95 = 0

    for p,r in zip(alt_precisions, alt_recalls):
        if p >= 0.95:
            max_recall_95 = max(max_recall_95, r)
            if p >= 0.99:
                max_recall_99 = max(max_recall_99, r)
                if p >= 1.:
                    max_recall_100 = max(max_recall_100, r)

    print("R@100P (alt):", max_recall_100)
    print("R@99P (alt):", max_recall_99)
    print("R@95P (alt):", max_recall_95)

    if visualize:
        _, axs = plt.subplots(1,2)
        axs[0].plot(recalls, precisions , marker='o')
        axs[0].axvline(recalls[optimal_threshold_index], color='r')
        axs[0].set(xlabel="recall", ylabel="precision")
        axs[0].title.set_text("PR Curve")
        axs[1].plot(list(range(nintervals)), precisions, color = 'b')
        axs[1].plot(list(range(nintervals)), recalls, color = 'g')
        axs[1].plot(list(range(nintervals)), f1_scores, color = 'k')
        axs[1].axvline(optimal_threshold_index, color='r')
        axs[1].set(xlabel="threshold", ylabel="precision - recall")
        axs[1].title.set_text("Precision (b) and recall (g) curves")
        plt.show()

        _, axs = plt.subplots(1,2)
        axs[0].plot(alt_recalls, alt_precisions , marker='o')
        axs[0].axvline(alt_recalls[alt_optimal_threshold_index], color='r')
        axs[0].set(xlabel="recall", ylabel="precision")
        axs[0].title.set_text("PR Curve")
        axs[1].plot(list(range(nintervals)), alt_precisions, color = 'b')
        axs[1].plot(list(range(nintervals)), alt_recalls, color = 'g')
        axs[1].plot(list(range(nintervals)), alt_f1_scores, color = 'k')
        axs[1].axvline(alt_optimal_threshold_index, color='r')
        axs[1].set(xlabel="threshold", ylabel="precision - recall")
        axs[1].title.set_text("Precision (b) and recall (g) curves")
        plt.show()

    return alt_precisions[alt_optimal_threshold_index], alt_auc, alt_optimal_threshold, [get_threshold(i) for i in range(nintervals)]
