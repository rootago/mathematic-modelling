"""
Matlab/archive 数据集转换工具
将 MATR batchdata_updated_struct_errorcorrect.mat 转为 Excel
支持 MATLAB v7.3 (HDF5) 格式，使用 h5py 读取。
数据结构参考: Severson et al. Nature Energy 2019, Attia et al. Nature 2020
"""

import pandas as pd
import numpy as np
import os
from pathlib import Path

try:
    import scipy.io as sio
except ImportError:
    sio = None

try:
    import h5py
except ImportError:
    h5py = None


def _to_flat_array(arr):
    """将 MATLAB 数组转为可写入 Excel 的一维/二维数组"""
    if arr is None or (isinstance(arr, np.ndarray) and arr.size == 0):
        return []
    a = np.asarray(arr)
    a = np.atleast_1d(a.squeeze())
    return a.tolist()


def _extract_field(rec, name, default=None):
    """从 struct 或 record 中安全取字段"""
    try:
        if hasattr(rec, '_fieldnames') and name in rec._fieldnames:
            return getattr(rec, name)
        if isinstance(rec, np.ndarray) and rec.dtype.names and name in rec.dtype.names:
            return rec[name]
        if isinstance(rec, dict) and name in rec:
            return rec[name]
    except Exception:
        pass
    return default


def _struct_to_dict(obj, max_cycle_rows=50000):
    """
    将 MATLAB struct 递归转为可序列化的 dict/list。
    若为 cycle 时序数据且过长，则截断行数避免 Excel 过慢。
    """
    if obj is None:
        return None
    if isinstance(obj, (int, float, str, bool)):
        return obj
    if isinstance(obj, np.ndarray):
        if obj.dtype.names:
            # struct array -> list of dict
            out = []
            for i in range(np.size(obj)):
                elem = np.atleast_1d(obj).flat[i]
                out.append(_struct_to_dict(_struct_elem_to_dict(elem), max_cycle_rows))
            return out
        else:
            arr = np.atleast_1d(obj).squeeze()
            if arr.size > max_cycle_rows and arr.ndim == 1:
                arr = arr[:max_cycle_rows]
            return _to_flat_array(arr)
    if isinstance(obj, (list, tuple)):
        return [_struct_to_dict(x, max_cycle_rows) for x in obj]
    if hasattr(obj, '_fieldnames'):
        return _struct_to_dict(_struct_elem_to_dict(obj), max_cycle_rows)
    return obj


def _struct_elem_to_dict(elem):
    """单个 struct 元素转 dict（scipy loadmat 的 struct 或已是 dict）"""
    if elem is None:
        return {}
    if isinstance(elem, dict):
        return elem
    if hasattr(elem, '_fieldnames'):
        return {k: getattr(elem, k) for k in elem._fieldnames}
    if isinstance(elem, np.ndarray) and elem.dtype.names:
        return {k: elem[k] for k in elem.dtype.names}
    return {}


def _read_h5_ref(f, ref):
    """从 h5py 文件中解引用并读取为 numpy 或 dict/list"""
    if ref is None:
        return None
    try:
        obj = f[ref]
        if isinstance(obj, h5py.Dataset):
            return obj[()]
        if isinstance(obj, h5py.Group):
            return _read_h5_group(f, obj)
    except Exception:
        pass
    return None


def _read_h5_group(f, grp):
    """将 HDF5 Group 递归读为 dict（解引用所有 ref）"""
    out = {}
    for key in grp.keys():
        if key.startswith('#'):
            continue
        try:
            obj = grp[key]
            if isinstance(obj, h5py.Dataset):
                d = obj[()]
                if d.dtype == np.dtype('object') and d.size > 0:
                    # 可能是 ref 或 ref 数组
                    try:
                        if np.issubdtype(d.dtype, np.void) or d.dtype.kind == 'O':
                            flat = np.atleast_1d(d).flatten()
                            if len(flat) == 1:
                                out[key] = _read_h5_ref(f, flat[0])
                            else:
                                out[key] = [_read_h5_ref(f, flat[i]) for i in range(min(len(flat), 500))]
                        else:
                            out[key] = d
                    except Exception:
                        out[key] = d.tolist() if d.size < 10000 else str(d.shape)
                else:
                    out[key] = d
            elif isinstance(obj, h5py.Group):
                out[key] = _read_h5_group(f, obj)
        except Exception as e:
            out[key] = None
    return out


def _load_mat_v73(mat_path):
    """用 h5py 读取 MATLAB v7.3 (.mat) 文件，返回与 loadmat 兼容的 data 结构。"""
    if h5py is None:
        raise ImportError("需要安装 h5py 以读取 v7.3 格式: pip install h5py")
    data = {}
    with h5py.File(mat_path, 'r') as f:
        for key in f.keys():
            if key.startswith('#'):
                continue
            obj = f[key]
            if isinstance(obj, h5py.Dataset):
                d = obj[()]
                if d.dtype == np.dtype('object') or (hasattr(d.dtype, 'kind') and d.dtype.kind == 'O'):
                    try:
                        flat = np.atleast_1d(d).flatten()
                        if flat.size == 0:
                            data[key] = []
                        elif flat.size == 1:
                            data[key] = _read_h5_ref(f, flat[0])
                        else:
                            # batch 常为“引用数组”：每个元素指向一个 struct
                            data[key] = [_read_h5_ref(f, flat[i]) for i in range(flat.size)]
                    except Exception:
                        data[key] = d
                else:
                    data[key] = d
            elif isinstance(obj, h5py.Group):
                # struct array: 每个字段是一个 dataset（可能为 ref 数组）
                try:
                    n = None
                    batch_list = []
                    for field in obj.keys():
                        if field.startswith('#'):
                            continue
                        ds = obj[field]
                        if isinstance(ds, h5py.Dataset):
                            arr = ds[()]
                            arr_flat = np.atleast_1d(arr).flatten()
                            if n is None:
                                n = len(arr_flat)
                            for i in range(min(n, len(arr_flat))):
                                if i >= len(batch_list):
                                    batch_list.append({})
                                if arr.dtype == np.dtype('object') or (hasattr(arr.dtype, 'kind') and arr.dtype.kind == 'O'):
                                    val = _read_h5_ref(f, arr_flat[i])
                                    batch_list[i][field] = val if isinstance(val, dict) else val
                                else:
                                    batch_list[i][field] = arr_flat[i] if arr_flat.size > 1 else np.asarray(arr).item()
                    if batch_list:
                        data[key] = batch_list
                    else:
                        data[key] = _read_h5_group(f, obj)
                except Exception:
                    data[key] = _read_h5_group(f, obj)
    return data


def convert_batchdata_mat(mat_path, output_dir, max_cycle_rows=50000):
    """
    将单个 batchdata .mat 转为 Excel（多工作表）。
    
    参数:
        mat_path: .mat 文件路径
        output_dir: 输出目录
        max_cycle_rows: 每个 cycle 表最多保留行数，避免 Excel 过大
    """
    mat_path = Path(mat_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 加载 .mat：优先 scipy，v7.3 则用 h5py
    data = None
    try:
        if sio is not None:
            data = sio.loadmat(str(mat_path), struct_as_record=False, squeeze_me=True)
            keys = [k for k in data.keys() if not k.startswith('__')]
    except NotImplementedError as e:
        if "v7.3" in str(e) or "HDF" in str(e):
            data = _load_mat_v73(str(mat_path))
            keys = [k for k in data.keys() if not k.startswith('__')]
        else:
            raise
    if not data or not keys:
        print(f"  未找到有效变量: {mat_path.name}")
        return

    batch_key = 'batch' if 'batch' in data else keys[0]
    raw = data[batch_key]

    # 统一为 list：scipy 为 numpy struct array，h5py 为 list of dict
    if isinstance(raw, list):
        batches = raw
    else:
        batch_arr = np.atleast_1d(raw)
        if batch_arr.size == 0:
            print(f"  batch 为空: {mat_path.name}")
            return
        batches = [batch_arr.flat[i] for i in range(batch_arr.size)]

    # 收集 summary 和 cycle
    summary_rows = []
    cycle_sheets = []  # [(sheet_name, df), ...]

    for idx, b in enumerate(batches):
        b_dict = _struct_elem_to_dict(b)
        # Summary 表：每个 batch 一行
        summary_row = {'batch_index': idx + 1}
        if 'summary' in b_dict:
            s = b_dict['summary']
            s_dict = _struct_elem_to_dict(s) if hasattr(s, '_fieldnames') or (isinstance(s, np.ndarray) and getattr(s, 'dtype', None) and getattr(s.dtype, 'names', None)) else {}
            if not s_dict and hasattr(s, '__iter__') and not isinstance(s, (str, bytes)):
                try:
                    s_arr = np.atleast_1d(s)
                    if s_arr.dtype.names:
                        s_dict = {k: _to_flat_array(s_arr[k]) for k in s_arr.dtype.names}
                except Exception:
                    pass
            for k, v in s_dict.items():
                try:
                    if isinstance(v, dict):
                        summary_row[k] = str(v)[:200]
                        continue
                    vflat = _to_flat_array(v)
                    if len(vflat) == 1:
                        summary_row[k] = vflat[0]
                    elif len(vflat) <= 30:
                        # 用逗号分隔的字符串写入，避免 Excel 显示成 [51][95][67]
                        summary_row[k] = ", ".join(str(x) for x in vflat)
                    else:
                        summary_row[k + '_len'] = len(vflat)
                except Exception:
                    summary_row[k] = str(v)[:500] if not isinstance(v, (int, float, str)) else v
        if 'policy' in b_dict:
            summary_row['policy'] = str(b_dict['policy'])[:200]
        summary_rows.append(summary_row)

        # Cycle 数据：每个 batch 可做一个 sheet（数据量大时只取部分 cycle 或截断行）
        if 'cycles' in b_dict:
            cy = b_dict['cycles']
        elif 'cycle' in b_dict:
            cy = b_dict['cycle']
        else:
            cy = None

        if cy is not None:
            # h5py 返回 list of dict
            if isinstance(cy, list) and len(cy) > 0:
                try:
                    cycles_list = []
                    for j, c in enumerate(cy[:500]):
                        c_d = c if isinstance(c, dict) else _struct_elem_to_dict(c)
                        row = {'cycle': j + 1}
                        for k, v in c_d.items():
                            if isinstance(v, dict):
                                row[k] = str(v)[:100]
                                continue
                            vf = _to_flat_array(v)
                            if isinstance(vf, list):
                                if len(vf) == 1:
                                    row[k] = vf[0]
                                else:
                                    row[k] = ", ".join(str(x) for x in vf[:15])
                            else:
                                row[k] = vf
                        cycles_list.append(row)
                    if cycles_list:
                        cycle_sheets.append((f"batch_{idx+1}_cycles", pd.DataFrame(cycles_list)))
                except Exception as e:
                    cycle_sheets.append((f"batch_{idx+1}_cycles", pd.DataFrame([{'error': str(e)}])))
                cy = None  # 已处理，避免下面再走 numpy 分支
            if cy is not None:
                cy_arr = np.atleast_1d(cy)
                if getattr(cy_arr.dtype, 'names', None):
                    try:
                        cycles_list = []
                        for j in range(min(len(cy_arr), 500)):
                            c = cy_arr.flat[j]
                            c_d = _struct_elem_to_dict(c)
                            row = {'cycle': j + 1}
                            for k, v in c_d.items():
                                vf = _to_flat_array(v)
                                if len(vf) == 1:
                                    row[k] = vf[0]
                                else:
                                    row[k] = ", ".join(str(x) for x in vf[:15])
                            cycles_list.append(row)
                        if cycles_list:
                            cycle_sheets.append((f"batch_{idx+1}_cycles", pd.DataFrame(cycles_list)))
                    except Exception as e:
                        cycle_sheets.append((f"batch_{idx+1}_cycles", pd.DataFrame([{'error': str(e)}])))
                else:
                    try:
                        arr = np.asarray(cy).squeeze()
                        if arr.size > 0 and arr.size < max_cycle_rows * 10:
                            if arr.ndim == 1:
                                cycle_sheets.append((f"batch_{idx+1}_cycles", pd.DataFrame({'value': _to_flat_array(arr)})))
                            elif arr.ndim == 2:
                                df = pd.DataFrame(arr[:max_cycle_rows])
                                cycle_sheets.append((f"batch_{idx+1}_cycles", df))
                    except Exception:
                        pass

    # 写出 Excel（若文件被占用则写到 _new.xlsx）
    out_name = mat_path.stem + ".xlsx"
    out_path = output_dir / out_name

    def _write_excel(path):
        with pd.ExcelWriter(path, engine='openpyxl') as writer:
            if summary_rows:
                df_summary = pd.DataFrame(summary_rows)
                df_summary.to_excel(writer, sheet_name='summary', index=False)
            for sheet_name, df in cycle_sheets:
                safe_name = sheet_name[:31]
                df.to_excel(writer, sheet_name=safe_name, index=False)

    try:
        _write_excel(out_path)
        print(f"  已保存: {out_path}")
    except PermissionError:
        alt_path = output_dir / (mat_path.stem + "_new.xlsx")
        try:
            _write_excel(alt_path)
            print(f"  原文件被占用，已另存为: {alt_path}")
            print(f"  请关闭 Excel 中打开的 {out_name} 后重新运行以覆盖原文件。")
        except Exception as e2:
            print(f"  写入失败: {e2}")
    return out_path


def convert_archive_folder(input_dir, output_dir, max_cycle_rows=50000):
    """转换 archive 目录下所有 .mat 文件"""
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    mat_files = list(input_dir.glob("*.mat"))
    if not mat_files:
        print(f"未找到 .mat 文件: {input_dir}")
        return
    print(f"找到 {len(mat_files)} 个 .mat 文件")
    for m in mat_files:
        print(f"处理: {m.name}")
        try:
            convert_batchdata_mat(m, output_dir, max_cycle_rows=max_cycle_rows)
        except Exception as e:
            print(f"  错误: {e}")
            import traceback
            traceback.print_exc()


def main():
    base = Path(__file__).resolve().parent
    archive_dir = base / "Matlab" / "archive"
    output_dir = base / "Excel输出" / "archive"

    print("=" * 60)
    print("Matlab/archive (MATR batchdata) 转 Excel")
    print("=" * 60)
    print(f"输入: {archive_dir}")
    print(f"输出: {output_dir}")
    print()

    if not archive_dir.exists():
        print(f"目录不存在: {archive_dir}")
        return

    convert_archive_folder(archive_dir, output_dir)
    print()
    print("=" * 60)
    print("完成")
    print("=" * 60)


if __name__ == "__main__":
    main()
