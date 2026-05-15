#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从已构建的完整 S1 数据集（27 动态维：24 met + zenith + pm10 + pm2p5，例如 PMST_s1_data_pm10.ipynb
或含 pm2p5 的变体 → ifs_baseline/ml_dataset_pmst_v5_aligned_12h_pm10_pm25）派生「仅重合气象槽位」版本，
供与 IFS/Tianji overlap 槽位对齐；若 S1 已用 s2_data_monthtail_v2 全变量数据训练，可跳过本脚本。

做法：对每条样本的动态张量 (12,27)，只保留 T2M/PRECIP/MSLP/SW_RAD/U10/V10（及由 U/V 计算的 WSPD10），
其余 24 维气象槽置 0；天顶角与 PM10、PM2.5 保持原值；按新 dyn 重算 32 维雾 FE，
后 4 维时间周期特征从源行原样保留（与样本标签时刻一致）。静态+植被列不变。

注意：当前 overlap S1 训练脚本要求显式 X_train/y_train 与 X_val/y_val；
默认写出两份 split 以保留源数据的验证边界。论文实验不要使用 --merge_train_val。

用法:
  /path/to/python build_s1_pm10_overlap_from_full.py \\
    --source_dir .../ifs_baseline/ml_dataset_pmst_v5_aligned_12h_pm10_pm25 \\
    --out_dir .../ifs_baseline/ml_dataset_pmst_v5_aligned_12h_pm10_pm25_overlap
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np

from pmst_overlap_common import (
    OVERLAP_CANONICAL,
    OVERLAP_PMST_INDICES,
    TOTAL_DYN,
    apply_wspd10_from_uv,
    compute_fog_features_pmst,
)

WINDOW = 12
DYN_VARS = 27
BASE_DYN = WINDOW * DYN_VARS  # 324
STATIC_VEG = 6  # 5 + 1
FE_DIM = 36
EXPECTED_ROW = BASE_DYN + STATIC_VEG + FE_DIM


def _transform_chunk(dyn: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    dyn: (N, 12, 27) full dynamic from source (24 met + zenith + pm10 + pm2p5).
    Returns (dyn_new, fe32) — fe32 与 pmst_overlap_common.compute_fog_features_pmst 一致；
    全行 FE 为 36 = fe32 + 源数据中保留的 4 维周期时间。
    """
    met = dyn[:, :, :24].copy()
    zen_pm = dyn[:, :, 24:].copy()
    met_new = np.zeros_like(met)
    for name in OVERLAP_CANONICAL:
        j = OVERLAP_PMST_INDICES[name]
        met_new[:, :, j] = met[:, :, j]
    apply_wspd10_from_uv(met_new)
    dyn_new = np.concatenate([met_new, zen_pm], axis=-1).astype(np.float32)
    fe32 = compute_fog_features_pmst(dyn_new, WINDOW, TOTAL_DYN)
    return dyn_new, fe32


def transform_file(src_path: str, dst_path: str, chunk: int) -> int:
    X = np.load(src_path, mmap_mode="r")
    if len(X.shape) != 2 or X.shape[1] != EXPECTED_ROW:
        raise ValueError(f"{src_path}: expected shape [N,{EXPECTED_ROW}], got {X.shape}")
    n = X.shape[0]
    out = np.lib.format.open_memmap(dst_path, mode="w+", dtype=np.float32, shape=(n, EXPECTED_ROW))
    for i in range(0, n, chunk):
        sl = slice(i, min(i + chunk, n))
        block = np.array(X[sl], dtype=np.float32)
        dyn = block[:, :BASE_DYN].reshape(-1, WINDOW, DYN_VARS)
        dyn_new, fe32 = _transform_chunk(dyn)
        block[:, :BASE_DYN] = dyn_new.reshape(-1, BASE_DYN)
        c0 = BASE_DYN + STATIC_VEG
        block[:, c0 : c0 + 32] = fe32
        # c0+32:c0+36 周期时间自源行拷贝，已在 block 中
        out[sl] = block
    del out
    return n


def copy_if_exists(src_dir: str, name: str, dst_dir: str) -> bool:
    p = os.path.join(src_dir, name)
    if not os.path.isfile(p):
        return False
    import shutil

    shutil.copy2(p, os.path.join(dst_dir, name))
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--source_dir",
        default="/public/home/putianshu/vis_mlp/ifs_baseline/ml_dataset_pmst_v5_aligned_12h_pm10_pm25",
    )
    ap.add_argument(
        "--out_dir",
        default="/public/home/putianshu/vis_mlp/ifs_baseline/ml_dataset_pmst_v5_aligned_12h_pm10_pm25_overlap",
    )
    ap.add_argument("--chunk_rows", type=int, default=4096)
    ap.add_argument(
        "--merge_train_val",
        action="store_true",
        help="合并源目录中 X_train 与 X_val 为单一 X_train.npy（y同步合并），不写 X_val",
    )
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    if args.merge_train_val:
        xt = os.path.join(args.source_dir, "X_train.npy")
        xv = os.path.join(args.source_dir, "X_val.npy")
        yt = os.path.join(args.source_dir, "y_train.npy")
        yv = os.path.join(args.source_dir, "y_val.npy")
        if not os.path.isfile(xt) or not os.path.isfile(xv):
            raise FileNotFoundError("merge_train_val 需要源目录同时存在 X_train.npy 与 X_val.npy")
        Xtr = np.load(xt, mmap_mode="r")
        Xva = np.load(xv, mmap_mode="r")
        ytr = np.load(yt)
        yva = np.load(yv)
        n_total = Xtr.shape[0] + Xva.shape[0]
        out_x = os.path.join(args.out_dir, "X_train.npy")
        out_y = os.path.join(args.out_dir, "y_train.npy")
        out = np.lib.format.open_memmap(out_x, mode="w+", dtype=np.float32, shape=(n_total, EXPECTED_ROW))
        row_off = 0
        for src_path, n_src in [(xt, Xtr.shape[0]), (xv, Xva.shape[0])]:
            X = np.load(src_path, mmap_mode="r")
            for i in range(0, n_src, args.chunk_rows):
                sl = slice(i, min(i + args.chunk_rows, n_src))
                block = np.array(X[sl], dtype=np.float32)
                dyn = block[:, :BASE_DYN].reshape(-1, WINDOW, DYN_VARS)
                dyn_new, fe32 = _transform_chunk(dyn)
                block[:, :BASE_DYN] = dyn_new.reshape(-1, BASE_DYN)
                c0 = BASE_DYN + STATIC_VEG
                block[:, c0 : c0 + 32] = fe32
                out[row_off + sl.start : row_off + sl.stop] = block
            row_off += n_src
        del out
        np.save(out_y, np.concatenate([ytr, yva], axis=0))
        print(f"[OK] merged train+val -> {out_x} (N={n_total})", flush=True)
    else:
        for tag in ("train", "val"):
            x_name = f"X_{tag}.npy"
            src_x = os.path.join(args.source_dir, x_name)
            if not os.path.isfile(src_x):
                print(f"[SKIP] missing {src_x}", flush=True)
                continue
            dst_x = os.path.join(args.out_dir, x_name)
            n = transform_file(src_x, dst_x, args.chunk_rows)
            print(f"[OK] {x_name} -> {dst_x} (N={n})", flush=True)
            y_name = f"y_{tag}.npy"
            copy_if_exists(args.source_dir, y_name, args.out_dir)

    for meta in ("meta_train.csv", "meta_val.csv"):
        copy_if_exists(args.source_dir, meta, args.out_dir)

    cfg = {
        "dataset": "s1_pm10_overlap_derived_from_full",
        "source_dir": args.source_dir,
        "row_layout": f"{BASE_DYN} dyn + {STATIC_VEG} static/veg + {FE_DIM} FE",
        "source_row_requirement": "12*27 dyn + 5 static + 1 veg + 36 FE",
        "overlap_channels": OVERLAP_CANONICAL,
        "note": "FE: 32-d fog recompute from masked dyn +4-d cyclical time kept from source row.",
    }
    with open(os.path.join(args.out_dir, "dataset_build_config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)
    print(f"[OK] config -> {args.out_dir}/dataset_build_config.json", flush=True)


if __name__ == "__main__":
    main()
