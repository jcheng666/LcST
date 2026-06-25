import os
import time

import numpy as np
import torch
from matplotlib import pyplot as plt


def draw_loss_line(train_loss_line, val_loss_line, save_path):
    plt.figure()
    plt.plot(train_loss_line["x"], train_loss_line["y"], label="train loss")
    plt.plot(val_loss_line["x"], val_loss_line["y"], label="val loss")
    plt.ylabel("loss")
    plt.xlabel("epoch")
    plt.legend()
    plt.savefig(save_path)
    plt.close()


def check_dir(path: str, mkdir=False):
    if os.path.exists(path):
        return True
    if mkdir:
        os.makedirs(path, exist_ok=True)
        return True
    return False


def get_time_str():
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))


def get_adjacency_matrix(distance_df_filename, num_of_vertices, id_filename=None):
    if "npy" in distance_df_filename:
        adj_mx = np.load(distance_df_filename)
        return adj_mx, adj_mx

    import csv

    A = np.zeros((int(num_of_vertices), int(num_of_vertices)), dtype=np.float32)
    distaneA = np.zeros((int(num_of_vertices), int(num_of_vertices)), dtype=np.float32)

    if id_filename:
        with open(id_filename, "r") as f:
            id_dict = {str(i): idx for idx, i in enumerate(f.read().strip().split("\n"))}

        with open(distance_df_filename, "r") as f:
            f.readline()
            reader = csv.reader(f)
            for row in reader:
                if len(row) != 3:
                    continue
                i, j, distance = str(row[0]), str(row[1]), float(row[2])
                A[id_dict[i], id_dict[j]] = 1
                distaneA[id_dict[i], id_dict[j]] = distance
        return A, distaneA

    with open(distance_df_filename, "r") as f:
        f.readline()
        reader = csv.reader(f)
        for row in reader:
            if len(row) != 3:
                continue
            i, j, distance = int(row[0]), int(row[1]), float(row[2])
            A[i, j] = 1
            distaneA[i, j] = distance
    return A, distaneA


def get_randmask(observed_mask, min_miss_ratio=0.0, max_miss_ratio=1.0):
    rand_for_mask = torch.rand_like(observed_mask) * observed_mask
    rand_for_mask = rand_for_mask.reshape(-1)
    sample_ratio = np.random.rand() * (max_miss_ratio - min_miss_ratio) + min_miss_ratio
    num_observed = observed_mask.sum().item()
    num_masked = round(num_observed * sample_ratio)
    rand_for_mask[rand_for_mask.topk(num_masked).indices] = -1

    context_mask = (rand_for_mask > 0).reshape(observed_mask.shape).float()
    return context_mask
