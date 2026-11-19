# -*- coding: utf-8 -*-
"""
多機型飛行數據汇总處理腳本
作者: PINN Fuel Flow Prediction System
日期: 2025

功能：
1. 汇总src/data/input/0512文件夹下各机型的飞行数据
2. 添加flight_id和aircraft_type列
3. 进行数据单位转换
4. 计算各机型对应的飞行阶段
5. 生成合并的数据文件
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def get_flight_id_from_filename(filename):
    """从文件名提取航班ID"""
    # 移除.csv扩展名
    name = filename.replace('.csv', '')
    return name

# 從原始 QAR 逐航段 CSV 中抽取的列
#   N21 / N22 為兩台發動機的高壓轉子轉速（% 額定），僅用於派生發動機運轉標誌，
#   不得作為模型輸入特徵——N2 由 FADEC 依供油量調節，與 FF 近乎一一對應，
#   直接輸入等同於洩漏標籤，並會使反事實分析與 zero-shot 失去意義。
REQUIRED_COLUMNS = ['Time', 'FLIGHT_PHASE', 'ALT_QNH', 'TAS', 'SAT', 'GWC',
                    'FPA', 'IVV', 'MACH', 'FF1C', 'FF2C', 'N21', 'N22']

# 与论文、multi_aircraft_pinn.py 和 zero-shot 实验一致的研究机型范围。
STUDY_AIRCRAFT_TYPES = ('A320', 'A321', 'A320-NEO', 'A321-NEO')

# QAR FLIGHT_PHASE 的 14 種標準編碼
RECORDED_PHASE_CODE_NAMES = {
    0: 'ILLEGAL',        1: 'ENGINE_STOP',   2: 'ENGINE_START',
    3: 'TAKE_OFF',       4: 'REJECTED_TO',   5: 'SND_SEGMENT',
    6: 'INITIAL_CLIMB',  7: 'CLIMB',         8: 'CRUISE',
    9: 'DESCENT',       10: 'APPROACH',     11: 'FINAL_APPROACH',
    12: 'LANDING',      13: 'GO_AROUND',    14: 'TAXI_IN',
}

SEVEN_PHASE_NAMES = ['taxi', 'takeoff', 'climb', 'cruise',
                     'descent', 'approach', 'landing']

# 判定單台發動機是否在運轉的 N2（高壓轉子）轉速門檻，單位為 % 額定轉速。
#
# 閾值 30% 的依據（實測 9 個航段、155,160 個發動機樣本）：
#   N2 <20%   FF = 0 佔 100%     停車或起動機帶轉
#   N2 20-30% FF = 0 佔  85%     點火供油窗口
#   N2 30-55% FF = 0 佔  28%     點火後加速
#   N2 >55%   FF = 0 佔 0.16%    慢車及以上
# 取 30% 時與 FF>0 的判定分歧率最低（0.337%），且與空客起動程序中
# 「點火完成」的轉速點一致（供油約在 22-25% N2，慢車約 60%）。
N2_RUNNING_THRESHOLD = 30.0

# 秒。N2 偶有單點丟數（轉速歸零但發動機仍在工作），以航段內滾動中位數濾除，
# 避免運轉標誌出現單樣本跳變。
N2_FLAG_SMOOTH_WINDOW = 5

# QAR 編碼 -> 7 個分析相位
#
# 分相依據是「起落架是否承重」與「推力處於哪個區間」——跨越接地這條界線，
# 燃油流量的物理規律完全不同。
#
# 注意：本表必須與 multi_aircraft_pinn.RECORDED_PHASE_TO_SEVEN 保持一致。
# 模型在讀取原始 FLIGHT_PHASE 時使用該處的副本；此處用於預處理階段落盤。
RECORDED_PHASE_TO_SEVEN = {
    # taxi: 發動機停車、啟動、滑入 —— 地面慢車
    1: 0, 2: 0, 14: 0,
    # takeoff: 起飛滑跑、中斷起飛 —— 最大推力 + 地面
    3: 1, 4: 1,
    # climb: 二階段爬升、初始爬升、爬升、複飛 —— 大推力 + 正上升率
    5: 2, 6: 2, 7: 2, 13: 2,
    # cruise
    8: 3,
    # descent
    9: 4,
    # approach: 進近、最後進近 —— 均未接地，推力調制
    10: 5, 11: 5,
    # landing: 接地後著陸滑跑 —— 擾流板、反推、剎車
    12: 6,
}


def process_multi_aircraft_data(force_rebuild: bool = False):
    """處理多機型飛行數據 - 記憶體優化版本"""
    print("開始處理多機型飛行數據...")

    # 定義數據路徑
    input_dir = Path("src/data/input/0512（内含7种机型）")
    output_dir1 = Path("src/data/all_merged_multi_aircraft.csv")

    # 快取僅在包含全部所需列時可用；缺列（例如舊版沒有 N21/N22）必須重建，
    # 否則後續步驟會靜默地拿不到新增欄位
    if output_dir1.exists() and not force_rebuild:
        cached_cols = set(pd.read_csv(output_dir1, nrows=0).columns)
        missing = [c for c in REQUIRED_COLUMNS if c not in cached_cols]
        if not missing:
            cached = pd.read_csv(output_dir1)
            cached_types = set(cached['aircraft_type'].unique()) if 'aircraft_type' in cached else set()
            if cached_types <= set(STUDY_AIRCRAFT_TYPES):
                print(f"使用已合併的五机型快取：{output_dir1}")
                return cached
            print(f"快取包含非研究机型 {sorted(cached_types - set(STUDY_AIRCRAFT_TYPES))}，將重新合併")
        print(f"快取缺少列 {missing}，將從原始數據重新合併")

    if True:  # 保留原有縮排層級，避免大範圍改動

        if not input_dir.exists():
            print(f"錯誤：輸入目錄 {input_dir} 不存在")
            return None

        # 仅处理论文定义的五种 Airbus 机型；目录中任何 B 系列或其他机型均不进入数据文件。
        aircraft_dirs = [d for d in input_dir.iterdir()
                         if d.is_dir() and d.name in STUDY_AIRCRAFT_TYPES]
        found_dirs = {d.name for d in aircraft_dirs}
        missing_dirs = set(STUDY_AIRCRAFT_TYPES) - found_dirs
        if missing_dirs:
            raise ValueError(f"原始输入目录缺少研究机型文件夹: {sorted(missing_dirs)}")
        print(f"處理 {len(aircraft_dirs)} 個研究机型文件夹：{[d.name for d in aircraft_dirs]}")

        # 定義需要的列 - 只讀取必要的列來節省記憶體
        required_columns = REQUIRED_COLUMNS

        # 創建臨時CSV文件來分批保存數據
        temp_files = []
        aircraft_stats = {}
        
        # 處理每個機型
        for aircraft_dir in aircraft_dirs:
            aircraft_type = aircraft_dir.name
            print(f"\n處理機型：{aircraft_type}")
            
            # 獲取該機型下的所有CSV文件
            csv_files = list(aircraft_dir.glob("*.csv"))
            aircraft_stats[aircraft_type] = {'files': 0, 'records': 0, 'success_files': 0}
            
            print(f"  發現 {len(csv_files)} 個飛行數據文件")
            
            # 為該機型創建臨時文件
            temp_file = f"temp_{aircraft_type}.csv"
            temp_files.append(temp_file)
            first_write = True
            
            # 處理每個CSV文件
            for csv_file in csv_files:
                try:
                    # 顯示文件大小信息
                    file_size_mb = csv_file.stat().st_size / (1024 * 1024)
                    print(f"    處理文件：{csv_file.name} ({file_size_mb:.1f}MB)")
                    
                    # 嘗試讀取前幾行來檢查列名
                    # try:
                    #     sample_df = pd.read_csv(csv_file, nrows=1)
                    #     available_columns = sample_df.columns.tolist()
                        
                    #     # 選擇可用的列
                    #     cols_to_use = []
                    #     for col in required_columns:
                    #         if col in available_columns:
                    #             cols_to_use.append(col)
                    #         elif col == 'FF1C' and 'FF1' in available_columns:
                    #             cols_to_use.append('FF1')
                    #         elif col == 'FF2C' and 'FF2' in available_columns:
                    #             cols_to_use.append('FF2')
                        
                    #     if len(cols_to_use) < 8:  # 如果重要列太少，跳過
                    #         print(f"    跳過列不足文件：{csv_file.name} (只有{len(cols_to_use)}列可用)")
                    #         continue
                        
                    # except Exception:
                    #     print(f"    跳過無法解析的文件：{csv_file.name}")
                    #     continue
                    
                    # 分塊讀取大文件
                    chunk_size = 5000  # 每次讀取5000行
                    chunks_processed = 0
                    
                    for chunk in pd.read_csv(csv_file, chunksize=chunk_size, usecols=required_columns):
                        if len(chunk) == 0:
                            continue
                        
                        # 重命名列以標準化
                        # if 'FF1' in chunk.columns and 'FF1C' not in chunk.columns:
                        #     chunk = chunk.rename(columns={'FF1': 'FF1C'})
                        # if 'FF2' in chunk.columns and 'FF2C' not in chunk.columns:
                        #     chunk = chunk.rename(columns={'FF2': 'FF2C'})
                        
                        # 添加flight_id和aircraft_type列
                        flight_id = get_flight_id_from_filename(csv_file.name)
                        chunk['flight_id'] = flight_id
                        chunk['aircraft_type'] = aircraft_type
                        
                        # 基本數據清理
                        chunk = chunk.dropna(thresh=len(chunk.columns)//2)  # 至少一半列有數據
                        
                        if len(chunk) > 0:
                            # 保存到臨時文件
                            chunk.to_csv(temp_file, mode='w' if first_write else 'a', 
                                        header=first_write, index=False, encoding='utf-8')
                            first_write = False
                            aircraft_stats[aircraft_type]['records'] += len(chunk)
                            chunks_processed += 1
                        
                        # 釋放記憶體
                        del chunk
                    
                    if chunks_processed > 0:
                        aircraft_stats[aircraft_type]['success_files'] += 1
                        print(f"    已處理：{csv_file.name} ({chunks_processed}個塊)")
                    else:
                        print(f"    跳過空文件：{csv_file.name}")
                    
                    aircraft_stats[aircraft_type]['files'] += 1
                    
                except MemoryError:
                    print(f"    記憶體不足，跳過文件：{csv_file.name}")
                    continue
                except Exception as e:
                    print(f"    錯誤處理文件 {csv_file.name}: {e}")
                    continue
        
        # 檢查是否有成功處理的數據
        total_records = sum(stats['records'] for stats in aircraft_stats.values())
        if total_records == 0:
            print("錯誤：沒有成功讀取任何數據")
            # 清理臨時文件
            for temp_file in temp_files:
                if Path(temp_file).exists():
                    Path(temp_file).unlink()
            return None
        
        print(f"\n分批合併數據...")
        all_chunks = []
        
        # 讀取臨時文件並合併
        for temp_file in temp_files:
            if Path(temp_file).exists():
                try:
                    # 分塊讀取臨時文件
                    for chunk in pd.read_csv(temp_file, chunksize=10000):
                        all_chunks.append(chunk)
                    
                    # 刪除臨時文件
                    Path(temp_file).unlink()
                    
                except Exception as e:
                    print(f"    讀取臨時文件失敗 {temp_file}: {e}")
                    continue
        
        if not all_chunks:
            print("錯誤：沒有成功合併任何數據")
            return None
        
        # 最終合併
        print("执行最终合并...")
        merged_df = pd.concat(all_chunks, ignore_index=True, sort=False)
        print(f"合併完成，總共 {len(merged_df)} 條記錄")
        
        # 釋放記憶體
        del all_chunks
        
        # 顯示統計信息
        print("\n各模型数据统计：")
        for aircraft_type, stats in aircraft_stats.items():
            print(f"  {aircraft_type}: {stats['success_files']}/{stats['files']} 個文件成功, {stats['records']} 條記錄")

        merged_df.to_csv("src/data/all_merged_multi_aircraft.csv", index=False)
        
        return merged_df

def convert_units_and_process(df):
    """單位轉換和數據處理"""
    print("\n开始数据单位转换和处理...")
    
    # 檢查必要的列是否存在
    required_columns = ['Time', 'FLIGHT_PHASE', 'ALT_QNH', 'TAS', 'SAT', 'GWC', 'FPA', 'IVV', 'FF1C', 'FF2C', 'MACH']
    
    # missing_columns = [col for col in required_columns if col not in df.columns]
    # if missing_columns:
    #     print(f"警告：缺少必要的列：{missing_columns}")
    #     # 嘗試使用可能的替代列名
    #     column_mapping = {
    #         'FF1C': 'FF1',
    #         'FF2C': 'FF2'
    #     }
    #     for missing_col in missing_columns:
    #         if missing_col in column_mapping and column_mapping[missing_col] in df.columns:
    #             df[missing_col] = df[column_mapping[missing_col]]
    #             print(f"  使用 {column_mapping[missing_col]} 代替 {missing_col}")
    
    processed_df = df.copy()
    
    # 物理常數
    FT2M = 0.3048          # 英尺轉米
    KTS2MPS = 0.514444     # 節轉米/秒
    
    def to_numeric_safe(series, default_value=np.nan):
        """安全的數值轉換"""
        return pd.to_numeric(series, errors='coerce').fillna(default_value)
    
    print("执行单位转化...")
    
    # 過濾雙發全部停車的數據
    #
    # 判據由「兩台都在運轉」放寬為「至少一台在運轉」，以保留單發運行樣本：
    #   - 單發滑入（TAXI_IN）是落地後關閉一台發動機滑回機位的常規節油操作，
    #     屬真實運行狀態，其燃油消耗需要被建模，不應剔除；
    #   - 發動機啟動／停車瞬態同樣是真實的地面狀態，予以保留；
    #   - 雙發全停時 FF1C = FF2C = 0，沒有燃油流量可供預測，屬平凡樣本，仍予剔除。
    before_filter = len(processed_df)
    ff1 = to_numeric_safe(processed_df['FF1C'])
    ff2 = to_numeric_safe(processed_df['FF2C'])
    engine_running = (ff1 > 0) | (ff2 > 0)
    single_engine = engine_running & ((ff1 <= 0) | (ff2 <= 0))
    processed_df = processed_df[engine_running].reset_index(drop=True)
    after_filter = len(processed_df)
    print(f"剔除双发全停样本: {before_filter - after_filter} 条")
    print(f"保留单发运行样本: {int(single_engine.sum())} 条 "
          f"（占保留数据 {single_engine.sum() / max(after_filter, 1) * 100:.2f}%）")

    # ---------- 發動機運轉標誌 ----------
    #
    # 由高壓轉子轉速 N21 / N22 派生兩個離散標誌，用於告知模型哪一台發動機被關閉。
    # 這是機組選擇的「運行構型」（單發滑入時左右發動機通常輪換），屬外生條件，
    # 無法由飛行狀態推得，也不應由模型預測。
    #
    # 只取離散標誌、不取 N2 原始值的理由：
    #   N2 由 FADEC 依供油量調節，與燃油流量近乎一一對應。將連續 N2 作為輸入
    #   等同於用標籤的代理量回歸標籤，並會使反事實分析（掃馬赫數時 N2 取值未知）
    #   與 zero-shot 泛化失去意義。
    for eng, col in [('1', 'N21'), ('2', 'N22')]:
        n2 = to_numeric_safe(processed_df[col], default_value=0.0)
        processed_df[f'eng{eng}_running'] = (n2 > N2_RUNNING_THRESHOLD).astype(int)

    # N2 偶有單點丟數（轉速歸零但發動機仍在工作），以航段內滾動中位數濾除
    def _smooth_engine_flags(group):
        for eng in ('1', '2'):
            c = f'eng{eng}_running'
            group[c] = (group[c]
                        .rolling(N2_FLAG_SMOOTH_WINDOW, center=True, min_periods=1)
                        .median()
                        .round()
                        .astype(int))
        return group

    processed_df = (processed_df.groupby('flight_id', group_keys=False)
                                .apply(_smooth_engine_flags))

    n_run = processed_df['eng1_running'] + processed_df['eng2_running']
    print(f"发动机运转标志 (N2 > {N2_RUNNING_THRESHOLD:.0f}%): "
          f"双发 {int((n_run == 2).sum()):,} 条, "
          f"单发 {int((n_run == 1).sum()):,} 条, "
          f"标志为零 {int((n_run == 0).sum()):,} 条")

    # 時間轉換處理
    def parse_time_to_seconds(time_str):
        """將hh:mm:ss格式的時間轉換為秒數"""
        if pd.isna(time_str) or str(time_str).strip() == '':
            return 0.0
        
        try:
            # 處理不同的時間格式
            time_str = str(time_str).strip()
            
            # 如果已經是數值，直接返回
            try:
                return float(time_str)
            except ValueError:
                pass
            
            # 解析hh:mm:ss格式
            if ':' in time_str:
                parts = time_str.split(':')
                if len(parts) == 3:
                    hours = float(parts[0])
                    minutes = float(parts[1])
                    seconds = float(parts[2])
                    return hours * 3600 + minutes * 60 + seconds
                elif len(parts) == 2:
                    minutes = float(parts[0])
                    seconds = float(parts[1])
                    return minutes * 60 + seconds
            
            # 嘗試其他格式
            return float(time_str)
            
        except Exception as e:
            print(f"時間解析錯誤: {time_str}, 錯誤: {e}")
            return 0.0
    
    print("解析时间格式...")
    processed_df['Time'] = processed_df['Time'].apply(parse_time_to_seconds)
    processed_df.rename(columns={'Time': 'time'}, inplace=True)

    processed_df['ALT_QNH'] = to_numeric_safe(processed_df['ALT_QNH']) * FT2M   #英尺转米
    processed_df['TAS'] = to_numeric_safe(processed_df['TAS']) * KTS2MPS
    processed_df['SAT'] = to_numeric_safe(processed_df['SAT']) + 273.15
    processed_df['GWC'] = to_numeric_safe(processed_df['GWC']) * 1000.0
    processed_df['FPA'] = np.deg2rad(to_numeric_safe(processed_df['FPA']))
    processed_df['MACH'] = to_numeric_safe(processed_df['MACH'])
    processed_df['IVV'] = to_numeric_safe(processed_df['IVV']) * FT2M / 60.0
    processed_df['FF1C'] = to_numeric_safe(processed_df['FF1C']) / 3600.0
    processed_df['FF2C'] = to_numeric_safe(processed_df['FF2C']) / 3600.0

    # ---------- 量程守卫 ----------
    #
    # 逐列检查是否落在物理上可能的范围内，越界即剔除。上下界取得很松，目的是
    # 拦住传感器的无效标记，而不是修剪真实数据的尾部：
    #   TAS  <= 300 m/s   各机型巡航真空速最高实测 261.7 m/s。A350 在地面时空速管
    #                     数据无效，记为量程上限 768.00 kt = 395.09 m/s，正是此列要拦的。
    #   MACH <= 0.95      高于各机型的最大使用马赫数（A32x 为 0.82，A350 为 0.89）。
    #   SAT  180~330 K    即 -93 ~ +57 摄氏度，包住任何真实大气。
    #   ALT  <= 14000 m   高于各机型的使用升限（A350 为 43,100 ft = 13,137 m）。
    #
    # 对 A320 系列数据此步不删除任何样本（实测 874,081 条全部在界内），保留它是因为
    # 不同机型的 QAR 无效标记取值不同，静默混入比报错更难察觉。
    RANGE_LIMITS = {
        'TAS': (0.0, 300.0),
        'MACH': (0.0, 0.95),
        'SAT': (180.0, 330.0),
        'ALT_QNH': (-500.0, 14000.0),
    }
    out_of_range = pd.Series(False, index=processed_df.index)
    for col, (low, high) in RANGE_LIMITS.items():
        bad = (processed_df[col] < low) | (processed_df[col] > high)
        if bad.any():
            print(f"  {col} 越界 {int(bad.sum()):,} 条 "
                  f"（允许 {low} ~ {high}，实测 {processed_df[col].min():.3f} ~ "
                  f"{processed_df[col].max():.3f}）")
        out_of_range |= bad
    if out_of_range.any():
        affected = processed_df.loc[out_of_range, 'flight_id'].nunique()
        print(f"量程守卫剔除 {int(out_of_range.sum()):,} 条，涉及 {affected} 个航段")
        processed_df = processed_df[~out_of_range].reset_index(drop=True)
    else:
        print("量程守卫：各列均在物理可能范围内，无需剔除")

    # 时间标准化 - 每个航班内部解卷绕、归零并排序
    def normalize_time_per_flight(group):
        """將航班內的時鐘秒數還原為連續時間軸

        time 列是當天零點起的秒數，跨午夜航班會從 86399 回跳到 0。
        必須先按記錄順序識別回跳並累加 86400 秒，再排序歸零；
        否則午夜之後的數據會被排到起飛之前，航段時長被誤算為接近 24 小時。
        """
        if 'time' not in group.columns:
            return group

        # 此處行序仍為 QAR 原始記錄順序，可據此解卷繞
        t = group['time'].to_numpy(dtype=float)
        if len(t) > 1:
            # 相鄰時刻回跳超過 12 小時即判定為跨越午夜
            wraps = np.cumsum(np.diff(t) < -43200.0)
            t[1:] += wraps * 86400.0
        group = group.assign(time=t)

        # 解卷繞後時間已單調遞增，排序僅作保險
        group = group.sort_values('time').reset_index(drop=True)
        group['time'] = group['time'] - group['time'].min()
        return group

    print("标准化每个航班的时间...")
    processed_df = processed_df.groupby('flight_id').apply(normalize_time_per_flight).reset_index(drop=True)

    # ---------- 總重無效值填補 ----------
    #
    # FMS 只在特定階段輸出總重，其餘時段 GWC 記為 0。這不是「飛機零重量」，是缺測。
    # 實測缺測全部集中在航段兩端，中間沒有任何缺口：
    #   開頭  91 個航段，ENGINE_STOP 階段（推出前 FMS 尚未輸出），中位  29 秒
    #   末尾  90 個航段，TAXI_IN 階段（落地後停止輸出），        中位 336 秒
    #
    # 以航段內有效值向兩端填補：ffill 補末尾（落地重量），bfill 補開頭（起飛重量）。
    # 這兩段期間確實仍在耗油，但實測誤差極小：
    #   開頭 中位 1.6 kg（0.002%），末尾 中位 55.7 kg（0.080%），最大 152.8 kg（0.218%）
    #
    # 刻意不採用「有效重量減去積分耗油」的精確做法：GWC 是模型輸入而 FF 是預測
    # 目標，用 FF 積分反推 GWC 會使輸入成為目標的函數，構成標籤洩漏。
    #
    # 必須放在時間標準化之後——填補依賴正確的時序，而跨午夜航段在時間解卷繞前
    # 行序是錯亂的（會令末尾的缺測塊看起來位於航段中段）。
    zero_gw = processed_df['GWC'] <= 0
    n_zero = int(zero_gw.sum())
    if n_zero:
        processed_df.loc[zero_gw, 'GWC'] = np.nan
        processed_df['GWC'] = (processed_df.groupby('flight_id')['GWC']
                                           .transform(lambda s: s.ffill().bfill()))
        n_left = int(processed_df['GWC'].isna().sum())
        print(f"总重无效值填补: {n_zero} 条 (前向填补，航段内)"
              + (f"，仍无有效值 {n_left} 条将被剔除" if n_left else ""))

    # 數據清理
    #
    # 實測當前數據在此處不會刪除任何樣本：唯一的缺測是 GWC，已在上一步填補完畢。
    # 保留此步作為守衛，但改為顯式報告——若某條航段整段缺測（例如全程無有效總重），
    # 靜默刪除整條航段比報錯更難察覺。
    required_columns_final = ['time', 'ALT_QNH', 'TAS', 'SAT', 'GWC', 'FPA', 'IVV', 'MACH', 'FF1C', 'FF2C']

    na_counts = processed_df[required_columns_final].isna().sum()
    na_counts = na_counts[na_counts > 0]
    if len(na_counts):
        print("警告：以下列仍存在缺测，对应样本将被剔除")
        for col, n in na_counts.items():
            affected = processed_df.loc[processed_df[col].isna(), 'flight_id'].nunique()
            print(f"    {col}: {n} 条，涉及 {affected} 个航段")
        before_clean = len(processed_df)
        processed_df = processed_df.dropna(subset=required_columns_final).reset_index(drop=True)
        print(f"  共剔除 {before_clean - len(processed_df)} 条")
    else:
        print("关键列无缺测，无需剔除样本")


    print(f"单位转换完成，处理后数据形状: {processed_df.shape}")
    processed_df.to_csv("src/data/all_merged_multi_aircraft_converted.csv", index=False)
    
    return processed_df

def compute_aircraft_specific_flight_phases(df):
    """由 QAR 記錄的 FLIGHT_PHASE 映射出 7 個分析相位

    此前本函數以 ALT / TAS / IVV / FPA 的閾值規則逐點推斷相位。該做法存在若干
    問題：閾值按機型硬編碼、在高原機場失效、無法覆蓋 step climb 等真實機動，
    且與模型訓練時使用的相位來源不一致（實測兩者僅 86.8% 吻合）。

    現改為直接映射 QAR 記錄的 FLIGHT_PHASE。該欄位由機載 FDIU 依起落架承重
    開關、無線電高度表、襟縫翼位置與油門桿角度等訊號即時判定，是機載系統自身
    的狀態機輸出，而非事後推斷。

    映射依據為「起落架是否承重」與「推力處於哪個區間」：跨越接地這條界線，
    燃油流量的物理規律完全不同，因此 FINAL_APPROACH（空中、推力調制）歸入
    approach，而 LANDING（接地、反推、減速）單獨成相。

    輸出欄位沿用 computed_flight_phase / phase_name，使下游代碼無需改動；
    其取值語義由「規則推斷」變為「機載記錄」。
    """
    print("\n由 QAR 记录的 FLIGHT_PHASE 映射飞行阶段...")

    if 'FLIGHT_PHASE' not in df.columns:
        raise ValueError("数据中缺少 FLIGHT_PHASE 列，无法映射飞行阶段")

    recorded = pd.to_numeric(df['FLIGHT_PHASE'], errors='coerce')
    invalid = recorded.isna()
    if invalid.any():
        raise ValueError(f"FLIGHT_PHASE 存在无法解析的取值 {int(invalid.sum())} 条")

    recorded = recorded.astype(np.int64)
    unknown = sorted(set(recorded.unique()) - set(RECORDED_PHASE_TO_SEVEN))
    if unknown:
        raise ValueError(
            f"FLIGHT_PHASE 出现未定义的编码 {unknown}，"
            "请先确认其物理含义再加入 RECORDED_PHASE_TO_SEVEN"
        )

    df = df.copy()
    df['computed_flight_phase'] = recorded.map(RECORDED_PHASE_TO_SEVEN).astype(np.int64)
    df['phase_name'] = df['computed_flight_phase'].map(
        dict(enumerate(SEVEN_PHASE_NAMES))
    )

    print("  QAR 编码 -> 分析相位 的实际分布:")
    detail = (df.assign(code=recorded)
                .groupby(['computed_flight_phase', 'code'])
                .size().rename('n').reset_index()
                .sort_values(['computed_flight_phase', 'code']))
    for phase_idx, grp in detail.groupby('computed_flight_phase'):
        items = ', '.join(
            f"{int(r.code)}:{RECORDED_PHASE_CODE_NAMES.get(int(r.code), '?')}={int(r.n):,}"
            for r in grp.itertuples()
        )
        total = int(grp.n.sum())
        print(f"    {SEVEN_PHASE_NAMES[phase_idx]:9s} {total:>8,} 条  <- {items}")

    print("\n各机型飞行阶段分布:")
    for aircraft_type in sorted(df['aircraft_type'].unique()):
        sub = df[df['aircraft_type'] == aircraft_type]
        print(f"\n{aircraft_type}:")
        counts = sub['computed_flight_phase'].value_counts().sort_index()
        for phase_idx, count in counts.items():
            print(f"  {SEVEN_PHASE_NAMES[int(phase_idx)]}: {count} 条 "
                  f"({count / len(sub) * 100:.1f}%)")

    return df

def multi_records_unit_converted():
    print("多机型飞行数据处理开始...")
    print("=" * 60)

    # 1. 汇总多机型数据
    merged_df = process_multi_aircraft_data()
    if merged_df is None:
        print("数据汇总失敗，程序退出")
        return

    # 2. 单位转换和数据处理
    processed_df = convert_units_and_process(merged_df)

    # 3. 计算各机型特定的飞行阶段
    final_df = compute_aircraft_specific_flight_phases(processed_df)

    # 4. 保存处理后的数据
    output_file = "src/data/all_merged_multi_aircraft_7phase.csv"
    #补充高海拔地区数据
    output_dir = Path(output_file).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n保存处理后的数据到: {output_file}")
    final_df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"已保存 {len(final_df)} 条记录")

    # 5. 显示最终统计信息
    print(f"\n最终处理结果:")
    print(f"条记录数: {len(final_df)}")
    print(f"机型数量: {final_df['aircraft_type'].nunique()}")
    print(f"航班数量: {final_df['flight_id'].nunique()}")

    print("\n各机型记录统计:")
    aircraft_summary = final_df.groupby('aircraft_type').agg({
        'flight_id': 'nunique',
        'time': 'count'
    }).rename(columns={'flight_id': '航班数', 'time': '记录数'})
    print(aircraft_summary)

    # 显示数据列信息
    print(f"\n数据列信息:")
    print(f"列数: {len(final_df.columns)}")
    print(f"主要列: {list(final_df.columns[:15])}")

    if 'FF1C' in final_df.columns and 'FF2C' in final_df.columns:
        print(f"FF1C范围: {final_df['FF1C'].min():.4f} - {final_df['FF1C'].max():.4f} kg/s")
        print(f"FF2C范围: {final_df['FF2C'].min():.4f} - {final_df['FF2C'].max():.4f} kg/s")

    print("\n多机型飞行数据处理完成！")
    print("=" * 60)

    return final_df
import pandas as pd
import re
from datetime import datetime

# ==================== 第一步：定义机场信息数据库 ====================
airport_db = {
    # 中国大陆机场 (已包含你数据中出现的所有机场)
    "ZHHH":{"name": "武汉天河国际机场", "city": "武汉", "country": "中国大陆"},
    "ZPPP":{"name": "昆明长水国际机场", "city": "昆明", "country": "中国大陆"},
    "ZSPD": {"name": "上海浦东国际机场", "city": "上海", "country": "中国大陆"},
    "ZBAA": {"name": "北京首都国际机场", "city": "北京", "country": "中国大陆"},
    "ZBAD": {"name": "北京大兴国际机场", "city": "北京", "country": "中国大陆"},
    "ZGGG": {"name": "广州白云国际机场", "city": "广州", "country": "中国大陆"},
    "ZGSZ": {"name": "深圳宝安国际机场", "city": "深圳", "country": "中国大陆"},
    "ZUCK": {"name": "重庆江北国际机场", "city": "重庆", "country": "中国大陆"},
    "ZUUU": {"name": "成都双流国际机场", "city": "成都", "country": "中国大陆"},
    "ZUTF": {"name": "天府机场（成都）", "city": "成都", "country": "中国大陆"},
    "ZYTL": {"name": "大连周水子国际机场", "city": "大连", "country": "中国大陆"},
    "ZYTX": {"name": "沈阳桃仙国际机场", "city": "沈阳", "country": "中国大陆"},
    "ZYHB": {"name": "哈尔滨太平国际机场", "city": "哈尔滨", "country": "中国大陆"},
    "ZSQD": {"name": "青岛胶东国际机场", "city": "青岛", "country": "中国大陆"},
    "ZSAM": {"name": "厦门高崎国际机场", "city": "厦门", "country": "中国大陆"},
    "ZSFZ": {"name": "福州长乐国际机场", "city": "福州", "country": "中国大陆"},
    "ZSJN": {"name": "济南遥墙国际机场", "city": "济南", "country": "中国大陆"},
    "ZSYT": {"name": "烟台蓬莱国际机场", "city": "烟台", "country": "中国大陆"},
    "ZSCN": {"name": "南昌昌北国际机场", "city": "南昌", "country": "中国大陆"},
    "ZSOF": {"name": "合肥新桥国际机场", "city": "合肥", "country": "中国大陆"},
    "ZLXY": {"name": "西安咸阳国际机场", "city": "西安", "country": "中国大陆"},
    "ZLIC": {"name": "银川河东国际机场", "city": "银川", "country": "中国大陆"},
    "ZLLL": {"name": "兰州中川国际机场", "city": "兰州", "country": "中国大陆"},
    "ZGNN": {"name": "南宁吴圩国际机场", "city": "南宁", "country": "中国大陆"},
    "ZGKL": {"name": "桂林两江国际机场", "city": "桂林", "country": "中国大陆"},
    "ZGBH": {"name": "乌兰浩特依勒力特机场", "city": "乌兰浩特", "country": "中国大陆"},
    "ZGSD": {"name": "佛山沙堤机场", "city": "佛山", "country": "中国大陆"},
    "ZGZH": {"name": "柳州白莲机场", "city": "柳州", "country": "中国大陆"},
    "ZGZJ": {"name": "湛江吴川机场", "city": "湛江", "country": "中国大陆"},
    "ZHYC": {"name": "宜春明月山机场", "city": "宜春", "country": "中国大陆"},
    "ZJSY": {"name": "三亚凤凰国际机场", "city": "三亚", "country": "中国大陆"},
    "ZPLJ": {"name": "泸沽湖机场", "city": "宁蒗（泸沽湖）", "country": "中国大陆"},
    "ZBDS": {"name": "鄂尔多斯伊金霍洛机场", "city": "鄂尔多斯", "country": "中国大陆"},
    "ZBDT": {"name": "大同云冈机场", "city": "大同", "country": "中国大陆"},
    "ZBHD": {"name": "邯郸机场", "city": "邯郸", "country": "中国大陆"},
    "ZBHH": {"name": "呼和浩特白塔国际机场", "city": "呼和浩特", "country": "中国大陆"},
    "ZBYC": {"name": "运城张孝机场", "city": "运城", "country": "中国大陆"},
    "ZUXC": {"name": "西昌青山机场", "city": "西昌", "country": "中国大陆"},
    "ZUZY": {"name": "遵义新舟机场", "city": "遵义", "country": "中国大陆"},
    "ZWSH": {"name": "喀什徕宁国际机场", "city": "喀什", "country": "中国大陆"},
    "ZYYJ": {"name": "延吉朝阳川国际机场", "city": "延吉", "country": "中国大陆"},
    # 中国港澳台地区机场
    "VHHH": {"name": "香港国际机场", "city": "香港", "country": "中国香港"},
    "VMMC": {"name": "澳门国际机场", "city": "澳门", "country": "中国澳门"},
    "RCKH": {"name": "高雄国际机场", "city": "高雄", "country": "中国台湾"},
    # 日本机场
    "RJAA": {"name": "成田国际机场", "city": "东京（成田）", "country": "日本"},
    "RJBB": {"name": "关西国际机场", "city": "大阪", "country": "日本"},
    "RJCC": {"name": "新千岁机场", "city": "札幌", "country": "日本"},
    "RJFF": {"name": "福冈机场", "city": "福冈", "country": "日本"},
    "RJGG": {"name": "中部国际机场", "city": "名古屋", "country": "日本"},
    "RJNS": {"name": "静滨机场", "city": "静冈", "country": "日本"},
    "RJTT": {"name": "羽田国际机场", "city": "东京（羽田）", "country": "日本"},
    "ROAH": {"name": "那霸机场", "city": "那霸（冲绳）", "country": "日本"},
    # 其他国家机场
    "RKPK": {"name": "金海国际机场", "city": "釜山", "country": "韩国"},
    "RPLL": {"name": "尼诺·阿基诺国际机场", "city": "马尼拉", "country": "菲律宾"},
    "VVNB": {"name": "内排国际机场", "city": "河内", "country": "越南"},
    "WMKK": {"name": "吉隆坡国际机场", "city": "吉隆坡", "country": "马来西亚"},
    "WSSS": {"name": "新加坡樟宜机场", "city": "新加坡", "country": "新加坡"},
}


# ==================== 第二步：判断flight_id是否有效的函数 ====================
def is_valid_flight_id(flight_id):
    """
    判断flight_id是否有效。
    有效标准：必须同时包含起飞机场ICAO和降落机场ICAO。
    格式要求：{4位ICAO}_{4位ICAO} {注册号}_{时间戳}
    """
    if pd.isna(flight_id):  # 处理空值
        return False
    # 检查是否匹配标准格式
    pattern = r'^([A-Z]{4})_([A-Z]{4}) [A-Z0-9\-]+_\d+$'
    return bool(re.match(pattern, str(flight_id)))


# ==================== 第三步：解析函数（仅对有效数据解析） ====================
def parse_flight_id(flight_id):
    """
    解析单个flight_id字符串。
    对于无效数据，只保留flight_id，其他字段赋空值。
    """
    # result = {
    #     "flight_id": flight_id,
    #     "起飞机场ICAO": None, "起飞机场名称": None, "起飞机场城市/地区": None, "起飞机场国家/地区": None,
    #     "降落机场ICAO": None, "降落机场名称": None, "降落机场城市/地区": None, "降落机场国家/地区": None,
    #     "飞机注册号": None,
    #     "起飞时间": None
    # }

    # 检查有效性
    if not is_valid_flight_id(flight_id):
        return []  # 直接返回空值结果

    try:
        # 解析有效数据
        match = re.match(r'([A-Z]{4})_([A-Z]{4}) ([A-Z0-9\-]+)_(\d+)', str(flight_id))
        if not match:
            return []

        origin_icao, dest_icao, registration, timestamp = match.groups()

        # 获取机场信息
        origin_info = airport_db.get(origin_icao, {"name": origin_icao, "city": "未知", "country": "未知"})
        dest_info = airport_db.get(dest_icao, {"name": dest_icao, "city": "未知", "country": "未知"})

        # 解析时间戳
        time_str = timestamp[:14]  # 取前14位：YYYYMMDDHHMMSS
        try:
            takeoff_time = datetime.strptime(time_str, '%Y%m%d%H%M%S').strftime('%Y/%m/%d %H:%M:%S')
        except ValueError:
            takeoff_time = "时间格式错误"

        # 更新结果列表
        result = [
                  origin_icao,origin_info["name"], origin_info["city"],origin_info["country"],
                  dest_icao,dest_info["name"],dest_info["city"],dest_info["country"],
                  registration,takeoff_time]

        # # 更新结果字典
        # result.update({
        #     "起飞机场ICAO": origin_icao,
        #     "起飞机场名称": origin_info["name"],
        #     "起飞机场城市/地区": origin_info["city"],
        #     "起飞机场国家/地区": origin_info["country"],
        #     "降落机场ICAO": dest_icao,
        #     "降落机场名称": dest_info["name"],
        #     "降落机场城市/地区": dest_info["city"],
        #     "降落机场国家/地区": dest_info["country"],
        #     "飞机注册号": registration,
        #     "起飞时间": takeoff_time
        # })

    except Exception as e:
        print(f"解析时发生错误 {flight_id}: {e}")

    return result


# ==================== 第四步：主程序 - 读取文件并处理 ====================
def process_flight_data(input_file, flight_id_column='flight_id'):
    """
    主处理函数：读取文件，解析flight_id列，生成新表格。

    参数:
        input_file: 输入文件路径（支持.xlsx, .xls, .csv）
        flight_id_column: 包含flight_id的列名，默认为'flight_id'
    """
    # 1. 读取文件
    print(f"正在读取文件: {input_file}")
    try:
        if input_file.endswith('.csv'):
            df = pd.read_csv(input_file)
        else:  # 默认为Excel文件
            df = pd.read_excel(input_file)
    except Exception as e:
        print(f"读取文件失败: {e}")
        return None
    columns = list(df.columns)
    column_order = [
        "起飞机场ICAO", "起飞机场名称", "起飞机场城市/地区", "起飞机场国家/地区",
        "降落机场ICAO", "降落机场名称", "降落机场城市/地区", "降落机场国家/地区",
        "飞机注册号",
        "起飞时间"
    ]
    columns.extend(column_order)
    # 检查是否存在指定列
    if flight_id_column not in df.columns:
        print(f"错误：文件中没有名为 '{flight_id_column}' 的列。")
        print(f"可用列名: {list(df.columns)}")
        return None

    print(f"成功读取 {len(df)} 行数据。")

    # 2. 对每一行进行解析
    print("开始解析flight_id数据...")

    parsed_data = []


    for idx, row in df.iterrows():
        flight_id = row[flight_id_column]
        parsed_record = parse_flight_id(flight_id)
        row_list = list(row)

        row_list.extend(parsed_record)
        parsed_data.append(row_list)


        # 显示进度
        if (idx + 1) % 50000 == 0:
            print(f"  已处理 {idx + 1}/{len(df)} 行...")

    # print(f"解析完成。有效数据: {valid_count} 条，无效/空值: {len(df) - valid_count} 条。")

    # 3. 创建新的DataFrame
    # 指定列顺序

    result_df = pd.DataFrame(parsed_data,columns=columns)


    return result_df

def process_flight_id():
    print("多机型飞行数据flight_id处理开始...")
    print("=" * 60)

    # 1. 汇总多机型数据
    INPUT_FILE ="src/data/analysis/all_merged_multi_aircraft_7phase_A321-NEO.csv"
    OUTPUT_FILE1 = "src/data/analysis/all_merged_multi_aircraft_7phase_flight_id_A321-NEO.csv"
    # OUTPUT_FILE2 = "src/data/analysis/all_merged_multi_aircraft_7phase_flight_id2.csv"
    # INPUT_FILE = "your_input_file.csv"  # 如果是CSV文件，使用这行
    FLIGHT_ID_COLUMN = "flight_id"  # flight_id所在的列名

    # 执行处理
    result_data = process_flight_data(INPUT_FILE, FLIGHT_ID_COLUMN)
    result_data.to_csv(OUTPUT_FILE1,index=0,encoding='gb18030')
    # result_data.to_csv(OUTPUT_FILE2, index=0, encoding='utf-8')






def main():
    """主程序"""
    result = multi_records_unit_converted()
    process_flight_id()




if __name__ == "__main__":
    main()



