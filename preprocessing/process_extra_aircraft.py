# -*- coding: utf-8 -*-
"""四机型之外的机型数据预处理

流程与 process_multi_aircraft_data 完全一致：解压到标准输入目录 → 按机型文件夹
读取原始 CSV → convert_units_and_process → compute_aircraft_specific_flight_phases，
输出列与列序同 all_merged_multi_aircraft_7phase.csv。

与主流程的差别是以下几处机型特有的处理：

  1. 列名映射。A350 的 QAR 参数命名与 A32x 不同，航迹角记在 FLT_PATH_ANG_CDS_CPT；
     B737 的航迹角记在 FLT_PATH_ANGL。已核实 B737 的 FLT_PATH_ANGL 为度、IVV 为
     ft/min、TAS 为 kt：IVV 对 TAS·sin(FPA) 的回归斜率 1.01–1.03，相关系数 ≥ 0.99，
     与 A320 相同（0.99）。
  2. 无效总重剔除。总重列有采集满量程的坏值（如 409.6 t），远超最大起飞重量，
     先置零交给主流程既有的航段内填补处理。B737 另有一个固定占位值 74.3 t：
     在不同飞机、两个机型上都一模一样，只出现在落地后的滑入与停车阶段（编码 14、1），
     低于 B737-800 的 MTOW，上下界拦不住，因此按占位值显式剔除。
  3. 非法相位帧。FLIGHT_PHASE = 0（ILLEGAL）是未定义的状态机取值。B737-800 中共
     2 帧，各自孤立地夹在同一相位的记录之间，且该帧 GWC 与 FPA 同时缺测，属采集坏帧，
     整帧剔除。主流程遇到未定义编码会直接报错，因此须在相位映射之前处理。
  4. 数据来源。A350、A319 为逐航段压缩包，按目录解压。B737 的航段分散在三类包中：
     厦门进出港的嵌套压缩包、新建文件夹/B737-800.zip、下载不完整的
     新建文件夹/B737-700.zip。由 audit_extra_qar 统一枚举，机型按机号查 CES 汇总表确定。
  5. 独立输出。写入单独的 CSV，不并入四机型主数据，既有实验不受影响。
     主流程的 convert_units_and_process 会把中间结果写到
     src/data/all_merged_multi_aircraft_converted.csv，这里在调用前后保存并恢复该文件，
     不让它被覆盖。
"""

from __future__ import annotations

import argparse
import contextlib
import re
import shutil
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence

import numpy as np
import pandas as pd

from process_multi_aircraft_data import (
    REQUIRED_COLUMNS, compute_aircraft_specific_flight_phases,
    convert_units_and_process, get_flight_id_from_filename,
)

INPUT_ROOT = Path('src/data/input/0512（内含7种机型）')
OUTPUT_DIR = Path('src/data')
MAIN_CSV = OUTPUT_DIR / 'all_merged_multi_aircraft_7phase.csv'
# 主流程 convert_units_and_process 的副产物，不属于本脚本的输出
CONVERTED_SIDE_FILE = OUTPUT_DIR / 'all_merged_multi_aircraft_converted.csv'

ILLEGAL_PHASE_CODE = 0


@dataclass
class AircraftSource:
    """一个机型的原始数据位置与机型特有的处理设置。"""
    aircraft_type: str
    # 逐航段压缩包所在目录；为 None 时由 audit_extra_qar 的清单按机号收集
    zip_dir: Optional[Path]
    # 主流程所需列 -> 该机型 QAR 中的实际列名，只列出不同名的
    column_source: Dict[str, str] = field(default_factory=dict)
    # 总重的物理上下界，单位吨。超界即判为采集坏值
    weight_bounds_tonne: Optional[tuple] = None
    # 总重的固定占位值，单位吨。落在上下界之内但不是实测值
    weight_placeholders_tonne: tuple = ()


SOURCES: Dict[str, AircraftSource] = {
    'A350-900': AircraftSource(
        aircraft_type='A350-900',
        zip_dir=Path('backup/A350（B-XXXX）部分数据/A350（B-XXXX）部分数据'),
        column_source={'FPA': 'FLT_PATH_ANG_CDS_CPT'},
        weight_bounds_tonne=(142.4, 280.0),      # OEW 与 MTOW，OpenAP
    ),
    'A319': AircraftSource(
        aircraft_type='A319',
        zip_dir=Path('backup/A319/A319'),
        column_source={},
        weight_bounds_tonne=(40.8, 75.5),        # OEW 与 MTOW，OpenAP
    ),
    # 剔除占位值后实测总重：B737-800 52.5–70.1 t，B737-700 49.8–64.2 t，均在界内
    'B737-800': AircraftSource(
        aircraft_type='B737-800',
        zip_dir=None,
        column_source={'FPA': 'FLT_PATH_ANGL'},
        weight_bounds_tonne=(41.4, 79.0),        # OEW 与 MTOW，OpenAP
        weight_placeholders_tonne=(74.3,),
    ),
    'B737-700': AircraftSource(
        aircraft_type='B737-700',
        zip_dir=None,
        column_source={'FPA': 'FLT_PATH_ANGL'},
        weight_bounds_tonne=(37.6, 70.0),        # OEW 与 MTOW，OpenAP
        weight_placeholders_tonne=(74.3,),
    ),
}


def extract_from_inventory(source: AircraftSource, target: Path, force: bool) -> None:
    """按机号收集分散在多类压缩包中的航段，原样写出原始 CSV。

    记录每个航段的来源压缩包到 _sources.csv，便于追溯。
    校验失败（CRC 错误）或末尾为零字节填充（下载不完整）的航段跳过并报告。
    """
    from audit_extra_qar import enumerate_flights

    flights = [f for f in enumerate_flights() if f.aircraft_type == source.aircraft_type]
    if not flights:
        raise FileNotFoundError(f'清单中没有 {source.aircraft_type} 的航段')
    print(f'从清单收集 {len(flights)} 个 {source.aircraft_type} 航段到 {target}')

    records = []
    for flight in flights:
        name = re.sub(r'\.csv.*$', '.csv', flight.name)
        out = target / name
        status = 'ok'
        if out.exists() and not force:
            status = 'exists'
        else:
            try:
                raw = flight.loader()
            except (zipfile.BadZipFile, OSError, EOFError) as exc:
                raw, status = None, f'skip: {exc!r}'
            if raw is not None and b'\x00' in raw:
                raw, status = None, 'skip: 末尾为零字节填充，航段不完整'
            if raw is not None:
                out.write_bytes(raw)
        if status.startswith('skip'):
            print(f'  跳过 {name}：{status[6:]}')
        records.append({'file': name, 'source': flight.source,
                        'declared_type': flight.declared_type, 'status': status})
    pd.DataFrame(records).to_csv(target / '_sources.csv', index=False, encoding='utf-8-sig')


def extract_flights(source: AircraftSource, force: bool = False) -> Path:
    """把压缩的 QAR 解到标准输入目录下的机型文件夹。"""
    target = INPUT_ROOT / source.aircraft_type
    if target.exists() and not force:
        existing = list(target.glob('*.csv'))
        if existing:
            print(f'已存在解压结果 {target}（{len(existing)} 个文件），跳过解压')
            return target
    target.mkdir(parents=True, exist_ok=True)

    if source.zip_dir is None:
        extract_from_inventory(source, target, force)
        return target

    archives = sorted(p for p in source.zip_dir.glob('*.zip'))
    if not archives:
        raise FileNotFoundError(f'{source.zip_dir} 下没有压缩的 QAR 文件')
    print(f'解压 {len(archives)} 个航班文件到 {target}')
    for archive_path in archives:
        with zipfile.ZipFile(archive_path) as archive:
            members = [n for n in archive.namelist() if n.lower().endswith('.csv')]
            if not members:
                print(f'  跳过 {archive_path.name}：压缩包内没有 CSV')
                continue
            member = members[0]
            out = target / Path(member).name
            if out.exists() and not force:
                continue
            with archive.open(member) as src, open(out, 'wb') as dst:
                shutil.copyfileobj(src, dst)
    print(f'  解压完成，共 {len(list(target.glob("*.csv")))} 个 CSV')
    return target


def read_flight(csv_path: Path, source: AircraftSource) -> pd.DataFrame:
    """按主流程的列清单读取一个航班，必要时做列名映射。"""
    header = pd.read_csv(csv_path, nrows=0, encoding='utf-8',
                         encoding_errors='replace', low_memory=False)
    columns = list(header.columns)

    position: Dict[str, int] = {}
    for target in REQUIRED_COLUMNS:
        name = source.column_source.get(target, target)
        # 同名列可能出现多次，pandas 会加后缀，取第一次出现
        found = [i for i, c in enumerate(columns)
                 if c == name or c.startswith(name + '.')]
        if not found:
            raise ValueError(f'{csv_path.name} 缺少列 {name}（对应 {target}）')
        position[target] = found[0]

    frame = pd.read_csv(csv_path, usecols=sorted(position.values()),
                        encoding='utf-8', encoding_errors='replace',
                        low_memory=False)
    frame.columns = [t for t, _ in sorted(position.items(), key=lambda kv: kv[1])]
    return frame[REQUIRED_COLUMNS]


def mark_invalid_weight(frame: pd.DataFrame, bounds: Optional[tuple],
                        placeholders: tuple = ()) -> int:
    """把物理上不可能的总重与固定占位值置零，交给主流程的航段内填补处理。"""
    if bounds is None and not placeholders:
        return 0
    weight = pd.to_numeric(frame['GWC'], errors='coerce')
    invalid = pd.Series(False, index=frame.index)
    if bounds is not None:
        low, high = bounds
        invalid |= ((weight > high) | ((weight > 0) & (weight < low))).fillna(False)
    if placeholders:
        invalid |= weight.isin(placeholders)
    frame.loc[invalid, 'GWC'] = 0.0
    return int(invalid.sum())


def drop_illegal_phase(frame: pd.DataFrame) -> int:
    """剔除 FLIGHT_PHASE = 0（ILLEGAL）的采集坏帧。"""
    phase = pd.to_numeric(frame['FLIGHT_PHASE'], errors='coerce')
    illegal = phase.eq(ILLEGAL_PHASE_CODE)
    if illegal.any():
        frame.drop(index=frame.index[illegal], inplace=True)
        frame.reset_index(drop=True, inplace=True)
    return int(illegal.sum())


@contextlib.contextmanager
def preserve_file(path: Path) -> Iterator[None]:
    """保存并恢复一个文件，使被调用的主流程函数不改变它。"""
    saved = path.read_bytes() if path.exists() else None
    try:
        yield
    finally:
        if saved is None:
            path.unlink(missing_ok=True)
        else:
            path.write_bytes(saved)


def build(source: AircraftSource, output_csv: Path, force: bool = False) -> pd.DataFrame:
    folder = extract_flights(source, force=force)
    # 下划线开头的是来源清单等说明文件，不是航段
    csv_files = sorted(p for p in folder.glob('*.csv') if not p.name.startswith('_'))
    print(f'\n处理机型：{source.aircraft_type}')
    print(f'  发现 {len(csv_files)} 个飞行数据文件')

    parts: List[pd.DataFrame] = []
    for csv_path in csv_files:
        frame = read_flight(csv_path, source)
        if frame.empty:
            print(f'    跳过空文件：{csv_path.name}')
            continue
        frame['flight_id'] = get_flight_id_from_filename(csv_path.name)
        frame['aircraft_type'] = source.aircraft_type
        n_illegal = drop_illegal_phase(frame)
        n_bad = mark_invalid_weight(frame, source.weight_bounds_tonne,
                                    source.weight_placeholders_tonne)
        print(f'    {csv_path.name[:44]:<44} {len(frame):>7,} 条   总重坏值 {n_bad:>6,}'
              + (f'   非法相位帧 {n_illegal}' if n_illegal else ''))
        parts.append(frame)

    merged = pd.concat(parts, ignore_index=True)
    print(f'\n合并完成，共 {len(merged):,} 条记录')

    with preserve_file(CONVERTED_SIDE_FILE):
        processed = convert_units_and_process(merged)
    processed = compute_aircraft_specific_flight_phases(processed)

    if source.weight_bounds_tonne is not None:
        low, high = source.weight_bounds_tonne
        keep = processed['GWC'].between(low * 1000, high * 1000)
        if (~keep).any():
            print(f'填补后总重仍越界 {int((~keep).sum()):,} 条，剔除')
            processed = processed.loc[keep].reset_index(drop=True)

    if MAIN_CSV.exists():
        order = list(pd.read_csv(MAIN_CSV, nrows=0).columns)
        missing = [c for c in order if c not in processed.columns]
        if missing:
            raise RuntimeError(f'输出缺少主数据的列 {missing}')
        processed = processed[order]

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    processed.to_csv(output_csv, index=False, encoding='utf-8-sig')

    modeled = processed[processed['phase_name'] != 'taxi']
    print(f'\n写出 {output_csv}')
    print(f'  {len(processed):,} 条   {processed["flight_id"].nunique()} 个航班   '
          f'建模相位 {len(modeled):,} 条')
    print('  总重 %.0f ~ %.0f kg   总油耗中位 %.3f kg/s'
          % (modeled['GWC'].min(), modeled['GWC'].max(),
             (modeled['FF1C'] + modeled['FF2C']).median()))
    print(modeled['phase_name'].value_counts().to_string())
    return processed


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description='四机型之外的机型数据预处理')
    parser.add_argument('--aircraft_type', default='A350-900', choices=sorted(SOURCES))
    parser.add_argument('--output_csv', type=Path, default=None)
    parser.add_argument('--force', action='store_true', help='重新解压并覆盖')
    args = parser.parse_args(argv)

    source = SOURCES[args.aircraft_type]
    output = args.output_csv or (
        OUTPUT_DIR / f'{source.aircraft_type.lower().replace("-", "")}_7phase.csv')
    build(source, output, force=args.force)


if __name__ == '__main__':
    main()
