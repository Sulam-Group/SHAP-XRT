import itertools
import os
import pickle

import matplotlib.patches as patches
import numpy as np
import pandas as pd
from tqdm import tqdm

root_dir = "../../"
data_dir = "data"
explanation_dir = "explanations"


def annotate(image_name, ax):
    df_train = pd.read_json(os.path.join(data_dir, "training.json"))
    df_test = pd.read_json(os.path.join(data_dir, "test_cropped.json"))
    frame = [df_train, df_test]
    gt_df = pd.concat(frame, ignore_index=True)

    _image_name = []
    for _, row in gt_df.iterrows():
        _image_name.append(os.path.basename(row["image"]["pathname"]).split(".")[0])
    gt_df["image_name"] = _image_name
    gt_df.set_index("image_name", inplace=True)

    cell = gt_df.at[image_name, "objects"]
    for c in cell:
        category = c["category"]
        if category == "trophozoite":
            bbox = c["bounding_box"]
            ul_r = bbox["minimum"]["r"]
            ul_c = bbox["minimum"]["c"]
            br_r = bbox["maximum"]["r"]
            br_c = bbox["maximum"]["c"]
            w = abs(br_c - ul_c)
            h = abs(br_r - ul_r)
            bbox = patches.Rectangle(
                (ul_c, ul_r),
                w,
                h,
                linewidth=2,
                edgecolor="g",
                facecolor="none",
            )
            ax.add_patch(bbox)


def get_J(gamma, h, w):
    gamma_per_dim = np.sqrt(gamma)
    return {
        (j + 1): (
            (j // 4 * h // gamma_per_dim, j % 4 * w // gamma_per_dim),
            ((j // 4 + 1) * h // 4, (j % 4 + 1) * w // 4),
        )
        for j in range(gamma)
    }


def get_C(gamma):
    C = []
    for c in range(gamma + 1):
        C.extend(itertools.combinations(range(1, gamma + 1), c))
    assert len(C) == 2**gamma
    return C


def p_threshold(gamma, output_threshold):
    with open(os.path.join("demo", f"F_{gamma}.pkl"), "rb") as f:
        F = pickle.load(f)

    p = {}
    for j in range(1, gamma + 1):
        p_j = {
            _C: (
                (F[tuple(sorted(_C + (j,)))] > output_threshold)
                - (F[tuple(sorted(_C))] > output_threshold)
            )
            <= 0
            for _C in get_C(gamma)
            if j not in _C
        }
        p[j] = p_j
    return p


def reject_fpr_tpr_threshold(gamma, true_feature_trophozoite, output_threshold):
    p = p_threshold(gamma, output_threshold)

    reject, fpr, tpr = {}, {}, {}
    for j in range(1, gamma + 1):
        is_j_true_trophozoite = j in true_feature_trophozoite

        h0_is_true_tests, h0_is_false_tests = 0, 0
        reject_tests, reject_when_h0_is_true_tests, reject_when_h0_is_false_tests = (
            0,
            0,
            0,
        )
        for C, p_C in p[j].items():
            is_true_trophozoite_in_C = any([jj in C for jj in true_feature_trophozoite])

            if p_C == 0:
                reject_tests += 1

            if is_true_trophozoite_in_C or (not is_j_true_trophozoite):
                h0_is_true_tests += 1
                if p_C == 0:
                    reject_when_h0_is_true_tests += 1
            else:
                h0_is_false_tests += 1
                if p_C == 0:
                    reject_when_h0_is_false_tests += 1

        total_tests = h0_is_true_tests + h0_is_false_tests
        assert total_tests == 2 ** (gamma - 1)

        reject[j] = reject_tests / total_tests
        fpr[j] = reject_when_h0_is_true_tests / h0_is_true_tests
        tpr[j] = reject_when_h0_is_false_tests / (h0_is_false_tests + 1e-08)
    return reject, fpr, tpr


def shapxrt_heatmap_threshold(j, gamma, output_threshold):
    p = p_threshold(gamma, output_threshold)

    reject = {jj: 0 for jj in range(1, gamma + 1) if jj != j}
    for C, p_C in p[j].items():
        if p_C == 0:
            for jj in C:
                reject[jj] += 1
    return reject


def reject_fpr_tpr(gamma, true_feature_trophozoite):
    reject = {j: [] for j in range(1, gamma + 1)}
    fpr = {j: [] for j in range(1, gamma + 1)}
    tpr = {j: [] for j in range(1, gamma + 1)}

    thresholds = np.linspace(0, 0.9, 100).tolist()
    for output_threshold in tqdm(thresholds):
        reject_t, fpr_t, tpr_t = reject_fpr_tpr_threshold(
            gamma, true_feature_trophozoite, output_threshold
        )

        for j in range(1, gamma + 1):
            reject[j].append(reject_t[j])
            fpr[j].append(fpr_t[j])
            tpr[j].append(tpr_t[j])

    return thresholds, reject, fpr, tpr


def shapxrt_heatmap(ax, h, w, j, gamma):
    J = get_J(gamma, h, w)

    thresholds = np.linspace(0, 0.9, 10).tolist()
    reject = [
        shapxrt_heatmap_threshold(j, gamma, output_threshold)
        for output_threshold in tqdm(thresholds)
    ]
    reject_mean = {
        jj: np.mean([v[jj] for v in reject]) for jj in range(1, gamma + 1) if jj != j
    }
    max_reject_mean = 450
    reject_alpha = {
        jj: 1 / np.exp(-4 * (v / max_reject_mean - 1)) for jj, v in reject_mean.items()
    }
    for jj in range(1, gamma + 1):
        ul_r = J[jj][0][0]
        ul_c = J[jj][0][1]
        br_r = J[jj][1][0]
        br_c = J[jj][1][1]
        w = abs(br_c - ul_c)
        h = abs(br_r - ul_r)
        if jj == j:
            bbox = patches.Rectangle(
                (ul_c, ul_r),
                w,
                h,
                linewidth=2,
                edgecolor="none",
                facecolor="#1f77b490",
            )
        else:
            facecolor = f"#d62728{int(100*reject_alpha[jj]):02d}"
            bbox = patches.Rectangle(
                (ul_c, ul_r),
                w,
                h,
                linewidth=2,
                edgecolor="none",
                facecolor=facecolor,
            )
        ax.add_patch(bbox)
