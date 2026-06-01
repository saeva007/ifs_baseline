#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从已构建的完整 S1 数据集（27 动态维：24 met + zenith + pm10 + pm2p5，例如 PMST_s1_data_pm10.ipynb
或含 pm2p5 的变体 → /public/home/putianshu/vis_mlp/ml_dataset_pmst_v5_aligned_12h_pm10_pm25）
派生固定变量子集版本。
`overlap_full` 供与 IFS/Tianji overlap 槽位对齐；`common_core` 供 Pangu 兼容的五源公平比较。
`compact_common_core` 写出真实较小输入，并移除 Pangu 中由 1000 hPa 湿度近似的 RH2M。

做法：对每条样本的动态张量 (12,27)，只保留 Tianji/IFS 共同可填充的气象槽位（含 RH2M、
Q_1000、DP_1000、RH_925 等，并由 U/V 或温湿度派生风速、风向、露点差等），
普通 27-dyn 子集会将其余 24 维气象槽置 0；compact 子集会直接切成较小 dyn；按新 dyn 重算雾 FE，
后 4 维时间周期特征从源行原样保留（与样本标签时刻一致）。静态+植被列不变。

注意：当前 overlap S1 训练脚本要求显式 X_train/y_train 与 X_val/y_val；
默认写出两份 split 以保留源数据的验证边界。论文实验不要使用 --merge_train_val。

用法:
  /path/to/python build_s1_pm10_overlap_from_full.py \\
    --source_dir .../ml_dataset_pmst_v5_aligned_12h_pm10_pm25 \\
    --out_dir .../ifs_baseline/ml_dataset_pmst_v5_aligned_12h_pm10_pm25_overlap

common_core 通常从已构建的 overlap_full S1 数据继续派生：
  /path/to/python build_s1_pm10_overlap_from_full.py \\
    --feature_set common_core \\
    --source_dir .../ifs_baseline/ml_dataset_pmst_v5_aligned_12h_pm10_pm25_overlap \\
    --out_dir .../ifs_baseline/ml_dataset_pmst_v5_aligned_12h_pm10_pm25_common_core
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np

from pmst_overlap_common import (
    COMMON_CORE_PMST_FEATURES,
    FEATURE_SET_CHOICES,
    FINAL_FEATURE_ORDER,
    OVERLAP_CANONICAL,
    PMST_INDEX,
    PMST_SOURCE_FIELDS,
    TOTAL_DYN,
    compute_fog_features_pmst,
    dynamic_feature_order_for_feature_set,
    dynamic_layout_name,
    dyn_vars_for_feature_set,
    resolve_pmst_feature_set,
    scatter_overlap_fields,
    select_dynamic_layout,
)

WINDOW = 12
SOURCE_DYN_VARS = 27
SOURCE_BASE_DYN = WINDOW * SOURCE_DYN_VARS  # 324
STATIC_VEG = 6  # 5 + 1
SOURCE_FE_DIM = 36
SOURCE_EXPECTED_ROW = SOURCE_BASE_DYN + STATIC_VEG + SOURCE_FE_DIM


def _output_dims(feature_set: str) -> tuple[int, int, int, int]:
    feature_vars = resolve_pmst_feature_set(feature_set, PMST_SOURCE_FIELDS)
    dynamic_order = dynamic_feature_order_for_feature_set(feature_set, feature_vars)
    dyn_vars = dyn_vars_for_feature_set(feature_set, feature_vars)
    dyn_dim = WINDOW * dyn_vars
    fog_fe_dim = compute_fog_features_pmst(
        np.zeros((1, WINDOW, dyn_vars), dtype=np.float32), WINDOW, dyn_vars, dynamic_order
    ).shape[1]
    fe_dim = fog_fe_dim + 4
    row_dim = dyn_dim + STATIC_VEG + fe_dim
    return dyn_vars, dyn_dim, fog_fe_dim, row_dim


def _transform_chunk(dyn: np.ndarray, feature_vars: list[str], feature_set: str) -> tuple[np.ndarray, np.ndarray]:
    """
    dyn: (N, 12, 27) full dynamic from source (24 met + zenith + pm10 + pm2p5).
    Returns selected dynamic layout and fog FE based only on that layout.
    """
    met = dyn[:, :, :24].copy()
    zen_pm = dyn[:, :, 24:].copy()
    fields = {name: met[:, :, PMST_INDEX[name]] for name in FINAL_FEATURE_ORDER}
    met_new = scatter_overlap_fields(met.shape[0], met.shape[1], fields, feature_vars)
    dyn_27 = np.concatenate([met_new, zen_pm], axis=-1).astype(np.float32)
    dyn_new = select_dynamic_layout(dyn_27, feature_set, feature_vars)
    dynamic_order = dynamic_feature_order_for_feature_set(feature_set, feature_vars)
    fe_base = compute_fog_features_pmst(dyn_new, WINDOW, dyn_new.shape[-1], dynamic_order)
    return dyn_new, fe_base


def transform_file(src_path: str, dst_path: str, chunk: int, feature_vars: list[str], feature_set: str) -> int:
    X = np.load(src_path, mmap_mode="r")
    if len(X.shape) != 2 or X.shape[1] != SOURCE_EXPECTED_ROW:
        raise ValueError(f"{src_path}: expected shape [N,{SOURCE_EXPECTED_ROW}], got {X.shape}")
    n = X.shape[0]
    dyn_vars, dyn_dim, fog_fe_dim, row_dim = _output_dims(feature_set)
    out = np.lib.format.open_memmap(dst_path, mode="w+", dtype=np.float32, shape=(n, row_dim))
    for i in range(0, n, chunk):
        sl = slice(i, min(i + chunk, n))
        block = np.array(X[sl], dtype=np.float32)
        dyn = block[:, :SOURCE_BASE_DYN].reshape(-1, WINDOW, SOURCE_DYN_VARS)
        dyn_new, fe_base = _transform_chunk(dyn, feature_vars, feature_set)
        if dyn_new.shape[-1] != dyn_vars or fe_base.shape[1] != fog_fe_dim:
            raise RuntimeError("Unexpected compact output layout while transforming S1 data.")
        out_block = np.empty((block.shape[0], row_dim), dtype=np.float32)
        out_block[:, :dyn_dim] = dyn_new.reshape(-1, dyn_dim)
        out_block[:, dyn_dim : dyn_dim + STATIC_VEG] = block[
            :, SOURCE_BASE_DYN : SOURCE_BASE_DYN + STATIC_VEG
        ]
        src_fe0 = SOURCE_BASE_DYN + STATIC_VEG
        src_time = block[:, src_fe0 + 32 : src_fe0 + SOURCE_FE_DIM]
        out_block[:, dyn_dim + STATIC_VEG : dyn_dim + STATIC_VEG + fog_fe_dim] = fe_base
        out_block[:, dyn_dim + STATIC_VEG + fog_fe_dim :] = src_time
        out[sl] = out_block
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
        default="/public/home/putianshu/vis_mlp/ml_dataset_pmst_v5_aligned_12h_pm10_pm25",
    )
    ap.add_argument(
        "--out_dir",
        default="/public/home/putianshu/vis_mlp/ifs_baseline/ml_dataset_pmst_v5_aligned_12h_pm10_pm25_overlap",
    )
    ap.add_argument(
        "--feature_set",
        choices=FEATURE_SET_CHOICES,
        default="overlap_full",
        help=(
            "S1 PMST met slots to keep. overlap_full preserves Tianji/IFS overlap slots; "
            "common_core keeps only the Pangu-compatible fair-comparison slots."
        ),
    )
    ap.add_argument("--chunk_rows", type=int, default=4096)
    ap.add_argument(
        "--merge_train_val",
        action="store_true",
        help="合并源目录中 X_train 与 X_val 为单一 X_train.npy（y同步合并），不写 X_val",
    )
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    feature_vars = resolve_pmst_feature_set(args.feature_set, PMST_SOURCE_FIELDS)
    dyn_vars, dyn_dim, fog_fe_dim, row_dim = _output_dims(args.feature_set)

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
        out = np.lib.format.open_memmap(out_x, mode="w+", dtype=np.float32, shape=(n_total, row_dim))
        row_off = 0
        for src_path, n_src in [(xt, Xtr.shape[0]), (xv, Xva.shape[0])]:
            X = np.load(src_path, mmap_mode="r")
            for i in range(0, n_src, args.chunk_rows):
                sl = slice(i, min(i + args.chunk_rows, n_src))
                block = np.array(X[sl], dtype=np.float32)
                dyn = block[:, :SOURCE_BASE_DYN].reshape(-1, WINDOW, SOURCE_DYN_VARS)
                dyn_new, fe_base = _transform_chunk(dyn, feature_vars, args.feature_set)
                out_block = np.empty((block.shape[0], row_dim), dtype=np.float32)
                out_block[:, :dyn_dim] = dyn_new.reshape(-1, dyn_dim)
                out_block[:, dyn_dim : dyn_dim + STATIC_VEG] = block[
                    :, SOURCE_BASE_DYN : SOURCE_BASE_DYN + STATIC_VEG
                ]
                src_fe0 = SOURCE_BASE_DYN + STATIC_VEG
                src_time = block[:, src_fe0 + 32 : src_fe0 + SOURCE_FE_DIM]
                out_block[:, dyn_dim + STATIC_VEG : dyn_dim + STATIC_VEG + fog_fe_dim] = fe_base
                out_block[:, dyn_dim + STATIC_VEG + fog_fe_dim :] = src_time
                out[row_off + sl.start : row_off + sl.stop] = out_block
            row_off += n_src
        del out
        np.save(out_y, np.concatenate([ytr, yva], axis=0))
        print(f"[OK] merged train+val -> {out_x} (N={n_total})", flush=True)
    else:
        wrote_any = False
        for tag in ("train", "val"):
            x_name = f"X_{tag}.npy"
            src_x = os.path.join(args.source_dir, x_name)
            if not os.path.isfile(src_x):
                print(f"[SKIP] missing {src_x}", flush=True)
                continue
            dst_x = os.path.join(args.out_dir, x_name)
            n = transform_file(src_x, dst_x, args.chunk_rows, feature_vars, args.feature_set)
            print(f"[OK] {x_name} -> {dst_x} (N={n})", flush=True)
            y_name = f"y_{tag}.npy"
            copy_if_exists(args.source_dir, y_name, args.out_dir)
            wrote_any = True
        if not wrote_any:
            raise FileNotFoundError(
                f"No X_train.npy or X_val.npy found under source_dir={args.source_dir}. "
                "For FEATURE_SET=common_core, build overlap_full first or set "
                "SOURCE_DIR=/public/home/putianshu/vis_mlp/ifs_baseline/"
                "ml_dataset_pmst_v5_aligned_12h_pm10_pm25_overlap."
            )

    for meta in ("meta_train.csv", "meta_val.csv"):
        copy_if_exists(args.source_dir, meta, args.out_dir)

    cfg = {
        "dataset": f"s1_pm10_{args.feature_set}_derived_from_source",
        "source_dir": args.source_dir,
        "feature_set": args.feature_set,
        "row_layout": f"{dyn_dim} dyn + {STATIC_VEG} static/veg + {fog_fe_dim + 4} FE",
        "source_row_requirement": "12*27 dyn + 5 static + 1 veg + 36 FE",
        "dynamic_feature_order": dynamic_feature_order_for_feature_set(args.feature_set, feature_vars),
        "dyn_layout": dynamic_layout_name(args.feature_set, feature_vars),
        "dyn_vars": int(dyn_vars),
        "fog_fe_dim": int(fog_fe_dim),
        "fe_dim": int(fog_fe_dim + 4),
        "overlap_channels": OVERLAP_CANONICAL,
        "common_core_channels": COMMON_CORE_PMST_FEATURES,
        "populated_pmst_features": feature_vars,
        "zero_filled_pmst_features": [],
        "excluded_pmst_features": [name for name in FINAL_FEATURE_ORDER if name not in feature_vars],
        "note": "FE recomputed from the selected dynamic layout; 4-d cyclical time kept from source row.",
    }
    with open(os.path.join(args.out_dir, "dataset_build_config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)
    print(f"[OK] config -> {args.out_dir}/dataset_build_config.json", flush=True)


if __name__ == "__main__":
    main()
