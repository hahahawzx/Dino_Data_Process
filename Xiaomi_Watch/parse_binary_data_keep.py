#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
二进制传感器数据解析脚本
解析 eng/accel/gyroscope/PPG7 二进制数据文件，输出CSV（含FrameUTC），检测丢数，保留原始文件。

用法:
  python parse_binary_data_keep.py <目录路径> [--plot] [--no-frame-utc] [--delete]

参数说明:
  默认行为: 解析 + 计算FrameUTC + 保留原始文件
  --plot          生成数据概览图（保存到数据目录同级的 _plots/ 文件夹）
  --no-frame-utc  不计算FrameUTC
  --delete        解析后删除原始二进制文件及辅助文件

示例:
  # 解析目录（默认：生成FrameUTC + 保留原文件）
  python parse_binary_data_keep.py /path/to/20260401_191704_xxx-Fast_Run-outdoor-Offline

  # 解析 + 生成概览图
  python parse_binary_data_keep.py /path/to/数据目录 --plot

  # 解析但不计算FrameUTC
  python parse_binary_data_keep.py /path/to/数据目录 --no-frame-utc

  # 解析后删除原始文件（节省空间）
  python parse_binary_data_keep.py /path/to/数据目录 --delete

关于 FrameUTC:
  FrameUTC(ms) 是通过线性回归拟合计算出的每帧真实UTC时间（毫秒）。
  拟合时跳过前5秒的数据（设备启动阶段时钟不稳定），仅使用5秒之后的
  稳定数据建立 EventTimestamp → UTC 的映射关系，再反推所有帧的UTC。
  因此，建议实际使用时也以第5秒之后的数据为准，前5秒的数据可能存在
  时钟抖动或丢数，仅供参考。
"""
import struct
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

FILE_CONFIGS = {
    'eng': {
        'pattern': 'eng-800hz.csv',
        'format': '<QQffffI4x',
        'fields': ['utc', 'ts', 'voltage0', 'voltage1', 'voltage2', 'voltage3', 'stat'],
        'csv_columns': ['CurrentTimestamp(ms)', 'EventTimestamp(us)', 'voltage0', 'voltage1', 'voltage2', 'voltage3', 'stat0', 'stat1', 'stat2', 'stat3'],
        'plot_fields': ['voltage0', 'voltage1', 'voltage2'],
        'plot_title': 'EMG (eng)',
    },
    'accel': {
        'pattern': 'accel-*hz.csv',
        'format': '<QQfff4x',
        'fields': ['utc', 'ts', 'acc_x', 'acc_y', 'acc_z'],
        'csv_columns': ['CurrentTimestamp(ms)', 'EventTimestamp(us)', 'acc_x', 'acc_y', 'acc_z'],
        'plot_fields': ['acc_x', 'acc_y', 'acc_z'],
        'plot_title': 'Accelerometer',
    },
    'gyroscope': {
        'pattern': 'gyroscope-*hz.csv',
        'format': '<QQfff4x',
        'fields': ['utc', 'ts', 'gyro_x', 'gyro_y', 'gyro_z'],
        'csv_columns': ['CurrentTimestamp(ms)', 'EventTimestamp(us)', 'gyro_x', 'gyro_y', 'gyro_z'],
        'plot_fields': ['gyro_x', 'gyro_y', 'gyro_z'],
        'plot_title': 'Gyroscope',
    },
    'ppg': {
        'pattern': 'PPG7-*hz.csv',
        'format': '<QQLLLLLHHHH4x',
        'fields': ['utc', 'ts', 'ppg0', 'ppg1', 'ppg2', 'ppg3', 'current', 'gain0', 'gain1', 'gain2', 'gain3'],
        'csv_columns': ['CurrentTimestamp(ms)', 'EventTimestamp(us)', 'ppg0', 'ppg1', 'ppg2', 'ppg3', 'current', 'gain0', 'gain1', 'gain2', 'gain3'],
        'plot_fields': ['ppg0', 'ppg1', 'ppg2', 'ppg3'],
        'plot_title': 'PPG',
    },
}


def get_sample_rate(filename):
    parts = str(filename).split("-")
    return int(parts[-1].replace("hz", ""))


def parse_binary_file(file_path, config):
    fmt = config['format']
    record_size = struct.calcsize(fmt)
    fields = config['fields']
    records = {f: [] for f in fields}
    with open(file_path, 'rb') as f:
        while True:
            data = f.read(record_size)
            if len(data) < record_size:
                break
            values = struct.unpack(fmt, data)
            for field, val in zip(fields, values):
                records[field].append(val)
    return records


def check_data_loss(timestamps, sample_fs, file_path):
    if len(timestamps) <= 101:
        return False, 0.0, "数据量太少，跳过丢数检测"
    ts_check = np.array(timestamps[100:])
    diffs = np.diff(ts_check)
    threshold = 1000000 / sample_fs + 300
    if np.all(diffs <= threshold):
        return False, 0.0, "没有丢数"
    large_diffs = diffs[diffs > threshold]
    total_count = int((ts_check[-1] - ts_check[10]) / (1000000 / sample_fs))
    sum_lost = int(np.sum(large_diffs) / (1000000 / sample_fs))
    lost_pct = 100 * sum_lost / total_count if total_count > 0 else 0
    info = (f"丢数! 总计{total_count}个采样点, 丢失{sum_lost}个({lost_pct:.2f}%), "
            f"异常跳变{len(large_diffs)}次, 最大跳变{np.max(large_diffs):.0f}us")
    return True, lost_pct, info


def compute_frame_utc(utc_arr, ts_arr, sample_fs, skip_seconds=5):
    utc = np.array(utc_arr, dtype=float)
    ts = np.array(ts_arr, dtype=float)
    if len(ts) < 200:
        return utc, "数据量太少，直接使用打包UTC"
    t0 = ts[0]
    stable_mask = (ts - t0) >= skip_seconds * 1e6
    if np.sum(stable_mask) < 100:
        stable_mask = np.ones(len(ts), dtype=bool)
    s_utc = utc[stable_mask]
    s_ts = ts[stable_mask]
    utc_diff = np.diff(s_utc)
    change_idx = np.where(utc_diff != 0)[0] + 1
    if len(change_idx) < 2:
        return utc, "UTC锚点不足，直接使用打包UTC"
    anchor_utc = s_utc[change_idx]
    anchor_ts = s_ts[change_idx]
    coeffs = np.polyfit(anchor_ts, anchor_utc, 1)
    frame_utc = np.polyval(coeffs, ts)
    fitted = np.polyval(coeffs, anchor_ts)
    residual = anchor_utc - fitted
    abs_std = np.std(residual)
    slope_ppm = (coeffs[0] - 0.001) * 1e6
    info = f"拟合斜率偏差={slope_ppm:.3f}ppm, 绝对精度std={abs_std:.1f}ms, 帧间精度由EventTS决定(~{1000/sample_fs:.3f}ms间隔)"
    return frame_utc, info


def save_csv(records, config, file_path, sample_fs, calc_frame_utc=True):
    data = {}
    columns = config['csv_columns']
    fields = config['fields']
    if calc_frame_utc:
        frame_utc, utc_info = compute_frame_utc(records['utc'], records['ts'], sample_fs)
        print(f"    FrameUTC: {utc_info}")
        data['FrameUTC(ms)'] = [int(round(v)) for v in frame_utc]
    data[columns[0]] = records['utc']
    data[columns[1]] = records['ts']
    if 'stat' in fields:
        for i, f in enumerate(fields[2:], 2):
            if f == 'stat':
                data['stat0'] = [int(v & 0xFF) for v in records[f]]
                data['stat1'] = [None] * len(records[f])
                data['stat2'] = [None] * len(records[f])
                data['stat3'] = [None] * len(records[f])
            else:
                data[columns[i]] = records[f]
    else:
        for i, f in enumerate(fields[2:], 2):
            data[columns[i]] = records[f]
    df = pd.DataFrame(data)
    stem = file_path.stem
    if 'PPG' in stem:
        out_name = f'{stem.split("-")[0]}-{sample_fs}hz_parsed.csv'
    elif 'eng' in stem:
        out_name = 'eng-800hz_parsed.csv'
    elif 'accel' in stem:
        out_name = f'accel-{sample_fs}hz_parsed.csv'
    else:
        out_name = f'gyroscope-{sample_fs}hz_parsed.csv'
    result_file = file_path.parent / out_name
    info_row = f"Info,SampleRate:{sample_fs},Gender:Male,Age:28,Height:0,Weight:0.000000,MainScene:Normal_Walk,SubScene:outdoor"
    with open(result_file, "w", newline="") as f:
        f.write(info_row + "\n")
        df.to_csv(f, index=False)
    return result_file


def detect_file_type(file_path):
    name = file_path.name
    if name.startswith('eng'):
        return 'eng'
    elif 'accel' in name:
        return 'accel'
    elif 'gyroscope' in name or 'gyro' in name:
        return 'gyroscope'
    elif 'PPG' in name:
        return 'ppg'
    return None


def process_file(file_path, calc_frame_utc=True):
    ftype = detect_file_type(file_path)
    if ftype is None:
        return None
    config = FILE_CONFIGS[ftype]
    sample_fs = get_sample_rate(file_path.stem)
    print(f"  解析: {file_path.parent.name}/{file_path.name} ...", end=" ")
    records = parse_binary_file(file_path, config)
    n = len(records['utc'])
    print(f"{n}条记录, {sample_fs}Hz", end=" ")
    is_lost, lost_pct, info = check_data_loss(records['ts'], sample_fs, file_path)
    print(f"-> {info}")
    out = save_csv(records, config, file_path, sample_fs, calc_frame_utc)
    print(f"    -> {out.name}")
    return (file_path, ftype, is_lost, lost_pct, info, records)


def plot_subfolder(subfolder_name, folder_records, plot_dir):
    plot_order = ['eng', 'accel', 'gyroscope', 'ppg']
    available = [t for t in plot_order if t in folder_records]
    if not available:
        return
    fig, axes = plt.subplots(len(available), 1, figsize=(16, 3.5 * len(available)))
    if len(available) == 1:
        axes = [axes]
    for ax, ftype in zip(axes, available):
        config = FILE_CONFIGS[ftype]
        records = folder_records[ftype]
        for field in config['plot_fields']:
            ax.plot(records[field], label=field, linewidth=0.5)
        ax.set_title(config['plot_title'], fontsize=11)
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('Sample')
    fig.suptitle(subfolder_name, fontsize=12, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    plot_path = plot_dir / f"{subfolder_name}.png"
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"  图片已保存: {plot_path}")


def process_directory(src_dir, no_plot=True, calc_frame_utc=True):
    src_dir = Path(src_dir)
    results = {'eng': [], 'accel': [], 'gyroscope': [], 'ppg': []}
    folder_data = {}
    for ftype, cfg in FILE_CONFIGS.items():
        pattern = cfg['pattern']
        files = sorted(src_dir.rglob(pattern))
        files = [f for f in files if '_parsed' not in f.name]
        print(f"\n{'='*60}")
        print(f"[{ftype.upper()}] 找到 {len(files)} 个文件")
        print(f"{'='*60}")
        for fp in files:
            result = process_file(fp, calc_frame_utc)
            if result:
                file_path, ft, is_lost, lost_pct, info, records = result
                results[ftype].append((file_path, ft, is_lost, lost_pct, info))
                folder_name = file_path.parent.name
                if folder_name not in folder_data:
                    folder_data[folder_name] = {}
                folder_data[folder_name][ftype] = records
    if not no_plot:
        plot_dir = src_dir.parent / f'{src_dir.name}_plots'
        plot_dir.mkdir(exist_ok=True)
        print(f"\n{'='*60}")
        print(f"生成数据概览图 -> {plot_dir}")
        print(f"{'='*60}")
        for folder_name in sorted(folder_data.keys()):
            plot_subfolder(folder_name, folder_data[folder_name], plot_dir)
    print(f"\n{'='*80}")
    print("汇总报告")
    print(f"{'='*80}")
    total_files = 0
    total_lost = 0
    for ftype in ['eng', 'accel', 'gyroscope', 'ppg']:
        items = results[ftype]
        lost_items = [(fp, pct, info) for fp, _, is_lost, pct, info in items if is_lost]
        total_files += len(items)
        total_lost += len(lost_items)
        print(f"\n[{ftype.upper()}] 共 {len(items)} 个文件, 丢数 {len(lost_items)} 个:")
        if lost_items:
            for fp, pct, info in lost_items:
                status = "通过" if pct <= 3.0 else "不通过"
                print(f"  {fp.parent.name}/{fp.name}: {pct:.2f}% -- {status}")
        else:
            print("  全部正常!")
    print(f"\n总计: {total_files} 个文件, {total_lost} 个存在丢数")
    if not no_plot:
        print(f"图片: {len(folder_data)} 张已保存到 {plot_dir}")
    return results


def cleanup_raw_files(src_dir):
    src_dir = Path(src_dir)
    delete_patterns = ['Extra.json', 'falldown_input.csv', 'Compress.bin']
    binary_patterns = ['eng-800hz.csv', 'accel-*hz.csv', 'gyroscope-*hz.csv', 'PPG7-*hz.csv']
    count, size = 0, 0
    to_delete = []
    for pat in binary_patterns:
        for f in src_dir.rglob(pat):
            if '_parsed' not in f.name:
                to_delete.append(f)
    for pat in delete_patterns:
        for f in src_dir.rglob(pat):
            to_delete.append(f)
    for f in to_delete:
        size += f.stat().st_size
        f.unlink()
        count += 1
    print(f"\n清理完成: 删除 {count} 个文件, 释放 {size/1024/1024:.1f} MB")


def main():
    if len(sys.argv) < 2:
        print(f"用法: python {sys.argv[0]} <目录或文件路径> [--plot] [--no-frame-utc] [--delete]")
        print(f"  默认: 解析 + 计算FrameUTC + 保留原始文件")
        print(f"  --plot          生成数据概览图")
        print(f"  --no-frame-utc  不计算FrameUTC")
        print(f"  --delete        解析后删除原始二进制文件及辅助文件")
        sys.exit(1)

    args = sys.argv[1:]
    no_plot = '--plot' not in args
    calc_frame_utc = '--no-frame-utc' not in args
    delete = '--delete' in args
    target = Path([a for a in args if not a.startswith('--')][0])

    if target.is_dir():
        process_directory(target, no_plot=no_plot, calc_frame_utc=calc_frame_utc)
        if delete:
            cleanup_raw_files(target)
    elif target.is_file():
        result = process_file(target, calc_frame_utc)
        if result is None:
            print(f"无法识别文件类型: {target.name}")
        elif delete:
            target.unlink()
            print(f"已删除原文件: {target}")
    else:
        print(f"路径不存在: {target}")
        sys.exit(1)


if __name__ == '__main__':
    main()
