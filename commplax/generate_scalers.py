#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
generate_scalers.py

这个脚本会：
  1) 从训练集列表里读取所有样本；
  2) 对每个样本调用 get_efp_data + pad_efp_data——得到 X_1, X_2, X_3, X_4；
  3) 把 X_1, X_2, X_3, X_4 扁平化后分别 fit 四个 MaxAbsScaler；
  4) 把训练好的 scaler.dump 到 dftpy/scalers/ 目录下。
"""

import os
import sys
from pathlib import Path
import joblib
import numpy as np
from sklearn.preprocessing import MaxAbsScaler

# 把项目根目录（包含 dftpy/）加入 Python 路径，方便 import
repo_root = Path(__file__).parent.parent
sys.path.append(str(repo_root))

from dftpy.data_io import read_file_list, get_max_atom_count, get_efp_data, pad_efp_data

# ----------------------------------------------------------------------------------------------------------------------
# 你需要手动修改这两个路径：
#  1) train_list_csv: 你的训练集 CSV 文件路径，必须包含一列 “files” 指向结构文件夹
#  2) scalers_dir:    最终要把新的 .joblib 保存到这里
# ----------------------------------------------------------------------------------------------------------------------
train_list_csv = "/content/drive/MyDrive/DFT_CSVs/Train.csv"  # 例：Train.csv
scalers_dir    = repo_root / "dftpy" / "scalers"

# 如果目录不存在就先创建
os.makedirs(scalers_dir, exist_ok=True)

# 1) 读取 train_folders
folders = read_file_list(train_list_csv, col="files")  # 列名要和 CSV 里的列名一致

# 2) 找到 padding_size （和你后面那次训练时完全一样的计算方式）
max_atoms = get_max_atom_count(folders)
padding_size = int(max_atoms * 1.0)   # 如果你的 padding_multiplier=1.0，就直接乘 1

print(f">>> 共计 {len(folders)} 个训练样本，max_atoms = {max_atoms}，padding_size = {padding_size}")

# 3) 用 get_efp_data + pad_efp_data 生成 X_1..X_4
#    get_efp_data 会把所有样本的 “dset, basis, forces, etc.” 读出来，返回给你一大堆 numpy 列表
ener_ref, forces_pre, press_ref, X_pre_list, basis_pre_list, X_at_list, X_el_list, X_elem_list = get_efp_data(folders)

#    pad_efp_data 会把指纹和 basis、forces pad 到 padding_size，并且拆成 X_1..X_4
forces1, forces2, forces3, forces4, \
X_1, X_2, X_3, X_4, \
basis1, basis2, basis3, basis4, \
C_m, H_m, N_m, O_m = pad_efp_data(
    X_elem_list,      # 每个样本的 at_elem 信息
    X_pre_list,       # 每个样本的原始 fingerprint
    forces_pre,       # 每个样本的原始力
    basis_pre_list,   # 每个样本的原始 basis
    padding_size
)

# 此时 X_1, X_2, X_3, X_4 的形状分别都是 (n_samples, padding_size, feat_dim)
# feat_dim 一般是 360（假设 num_gamma=18，总共有 radial/dipole/quad 共 10 种，每类 18 → 180，再和下一块叠加→360）

print("X_1.shape =", X_1.shape)
print("X_2.shape =", X_2.shape)
print("X_3.shape =", X_3.shape)
print("X_4.shape =", X_4.shape)

# 4) 分别对 X_1..X_4 扁平化后 fit MaxAbsScaler
#    先 reshape (n_samples, padding_size, feat_dim) -> (n_samples*padding_size, feat_dim)
n_samples, P, feat_C = X_1.shape
assert feat_C == X_2.shape[-1] == X_3.shape[-1] == X_4.shape[-1], "四个 feat_dim 必须一致"

flat_C = X_1.reshape(-1, feat_C)  #  (n_samples*P, feat_C)
flat_H = X_2.reshape(-1, feat_C)
flat_N = X_3.reshape(-1, feat_C)
flat_O = X_4.reshape(-1, feat_C)

print(f">>> 开始 fit MaxAbsScaler（C 类）: flat_C.shape = {flat_C.shape}")
scaler_C = MaxAbsScaler()
scaler_C.fit(flat_C)
print("    完成 C 类")

print(f">>> 开始 fit MaxAbsScaler（H 类）: flat_H.shape = {flat_H.shape}")
scaler_H = MaxAbsScaler()
scaler_H.fit(flat_H)
print("    完成 H 类")

print(f">>> 开始 fit MaxAbsScaler（N 类）: flat_N.shape = {flat_N.shape}")
scaler_N = MaxAbsScaler()
scaler_N.fit(flat_N)
print("    完成 N 类")

print(f">>> 开始 fit MaxAbsScaler（O 类）: flat_O.shape = {flat_O.shape}")
scaler_O = MaxAbsScaler()
scaler_O.fit(flat_O)
print("    完成 O 类")

# 5) 把四个 scaler.dump 到 dftpy/scalers/ 目录
path_C = scalers_dir / "Scale_model_C.joblib"
path_H = scalers_dir / "Scale_model_H.joblib"
path_N = scalers_dir / "Scale_model_N.joblib"
path_O = scalers_dir / "Scale_model_O.joblib"

joblib.dump(scaler_C, path_C)
joblib.dump(scaler_H, path_H)
joblib.dump(scaler_N, path_N)
joblib.dump(scaler_O, path_O)

print(">>> 四个 MaxAbsScaler 已保存到：")
print("    ", path_C)
print("    ", path_H)
print("    ", path_N)
print("    ", path_O)
