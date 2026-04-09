"""
main.py  –  Calibration Pose Combination Validation
=====================================================
목표: LOOCV / K-Fold 방식으로 최소 포즈 수가 몇 개이면
      포지션 오차 < 1 mm, 회전 오차 < 1 deg 를 달성하는지 검증

검증 흐름:
  1) 전체 데이터 로드 (HandEyeCalibration 재사용)
  2) 공간적 다양성(FPS) 기반 대표 포즈 선택
  3) LOOCV  – 포즈 1개씩 제거한 N 개 조합
  4) K-Fold – K-겹 교차검증 (train 부분집합으로 캘리브, test 로 평가)
  5) n=5 ~ N-1 에 대해 반복 실험, 조합당 캘리브 → 전체 데이터로 평가
  6) 결과 CSV / 시각화 저장
"""

import sys
import json
import itertools
import warnings
from pathlib import Path
from copy import deepcopy

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from sklearn.model_selection import KFold

# ── 프로젝트 내부 모듈 ────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent / 'src'))
from calibration import HandEyeCalibration

# ── 경로 상수 ────────────────────────────────────────────────────
CSV_PATH        = './dataset/calibration/calibration_data.csv'
RESULTS_DIR     = Path('./dataset/results')
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ── 실험 파라미터 ────────────────────────────────────────────────
POS_THRESHOLD   = 1.0          # mm
ROT_THRESHOLD   = 1.0          # deg
N_KFOLD         = 5            # K-Fold의 K 값
MIN_TRAIN_POSES = 5            # 최소 훈련 포즈 수
# 포즈가 많으면 조합 수가 폭발적으로 증가하므로 cap 을 설정
MAX_COMBINATIONS_PER_N = 100  # 특정 n 에 대해 최대 조합 수 (랜덤 샘플링)
RANDOM_SEED     = 42

np.random.seed(RANDOM_SEED)

# ─────────────────────────────────────────────────────────────────
# 헬퍼: 공간 다양성 기반 포즈 선택 (Farthest Point Sampling)
# ─────────────────────────────────────────────────────────────────
def fps_pose_indices(data: pd.DataFrame, n: int) -> list[int]:
    """
    로봇 xyz 공간에서 Farthest Point Sampling 으로
    공간적으로 고르게 분포된 n 개 포즈의 인덱스를 반환.
    """
    xyz = data[['x', 'y', 'z']].to_numpy(dtype=float)
    N   = len(xyz)
    if n >= N:
        return list(range(N))

    selected = [np.random.randint(N)]
    dists    = np.full(N, np.inf)
    for _ in range(n - 1):
        last  = xyz[selected[-1]]
        d     = np.linalg.norm(xyz - last, axis=1)
        dists = np.minimum(dists, d)
        selected.append(int(np.argmax(dists)))
    return selected


# ─────────────────────────────────────────────────────────────────
# 헬퍼: 부분 데이터로 캘리브레이션 수행 (full pipeline)
# ─────────────────────────────────────────────────────────────────
def calibrate_subset(base_calib: HandEyeCalibration,
                     subset_data: pd.DataFrame):
    """
    base_calib 의 averaged_data 에서 subset 만 뽑아 캘리브레이션.
    T_ndi_base, T_ee_marker, ndi_bias, ndi_scale 반환.
    """
    import cv2
    from scipy.linalg import logm, expm
    from scipy.optimize import least_squares

    # 임시 객체 생성
    cal = deepcopy(base_calib)
    cal.all_data   = subset_data.reset_index(drop=True)
    cal.pose_cache = cal._build_pose_cache(cal.all_data)

    # ① PARK hand-eye
    T_em = cal.solve_handeye_for_T_ee_marker(cal.all_data)
    # ② Point registration
    T_nb, _ = cal.solve_point_registration(cal.all_data, T_em)
    # ③ Nonlinear refinement
    T_nb_opt, T_em_opt, bias_opt, scale_opt, _ = \
        cal.refine_nonlinear_with_ndi_axis_scale(
            cal.all_data, T_nb, T_em,
            ndi_position_bias_init=np.zeros(3),
            ndi_axis_scale_init=np.ones(3),
            pos_weight=1.0, rot_weight=20.0,
            bias_reg_weight=0.02, scale_reg_weight=10.0)

    return T_nb_opt, T_em_opt, bias_opt, scale_opt


# ─────────────────────────────────────────────────────────────────
# 헬퍼: 전체 데이터에 대해 캘리브 결과 평가
# ─────────────────────────────────────────────────────────────────
def evaluate_on_full(base_calib: HandEyeCalibration,
                     T_ndi_base, T_ee_marker,
                     ndi_bias, ndi_scale):
    """전체 averaged_data 에 대한 (pos_errors, rot_errors) 반환."""
    pe, re = base_calib.evaluate_absolute_position(
        base_calib.all_data,
        T_ndi_base, T_ee_marker,
        ndi_position_bias=ndi_bias,
        ndi_axis_scale=ndi_scale)
    return pe, re


# ─────────────────────────────────────────────────────────────────
# 1. LOOCV  (Leave-One-Out Cross-Validation)
# ─────────────────────────────────────────────────────────────────
def run_loocv(base_calib: HandEyeCalibration) -> pd.DataFrame:
    """
    포즈 1개씩 제거 → N-1 개로 캘리브 → 전체 포즈로 평가.
    각 조합의 mean/max pos/rot 오차를 기록.
    """
    print("\n" + "=" * 60)
    print("LOOCV (Leave-One-Out Cross-Validation)")
    print("=" * 60)

    data     = base_calib.all_data
    N        = len(data)
    records  = []

    for leave_out in range(N):
        subset_idx  = [i for i in range(N) if i != leave_out]
        subset_data = data.iloc[subset_idx].copy()

        try:
            T_nb, T_em, bias, scale = calibrate_subset(base_calib, subset_data)
            pe, re = evaluate_on_full(base_calib, T_nb, T_em, bias, scale)
            records.append({
                'method':        'LOOCV',
                'n_train':       N - 1,
                'left_out_pose': int(data.iloc[leave_out]['pose_id']),
                'pos_mean':      float(np.mean(pe)),
                'pos_max':       float(np.max(pe)),
                'rot_mean':      float(np.mean(re)),
                'rot_max':       float(np.max(re)),
                'pass':          bool(np.mean(pe) < POS_THRESHOLD and
                                      np.mean(re) < ROT_THRESHOLD),
            })
            print(f"  Leave out pose {int(data.iloc[leave_out]['pose_id']):>3d} | "
                  f"pos_mean={np.mean(pe):.4f}mm  rot_mean={np.mean(re):.4f}deg  "
                  f"{'PASS' if records[-1]['pass'] else 'FAIL'}")
        except Exception as e:
            print(f"  [WARN] leave_out={leave_out}: {e}")

    df = pd.DataFrame(records)
    print(f"\nLOOCV summary: {df['pass'].sum()}/{len(df)} pass")
    return df


# ─────────────────────────────────────────────────────────────────
# 2. K-Fold (다양한 n 에 대해)
# ─────────────────────────────────────────────────────────────────
def run_kfold_by_n(base_calib: HandEyeCalibration) -> pd.DataFrame:
    """
    n = MIN_TRAIN_POSES ~ total-1 에 대해
      - FPS 로 공간 대표 포즈 선택
      - 추가로 랜덤 조합 MAX_COMBINATIONS_PER_N 개 포함
      - 각 조합으로 캘리브 → 전체 데이터로 평가
    """
    print("\n" + "=" * 60)
    print("K-Fold / Subset Sweep  (n = train poses)")
    print("=" * 60)

    data    = base_calib.all_data
    N       = len(data)
    indices = list(range(N))
    records = []

    for n in range(MIN_TRAIN_POSES, N):
        combos = set()

        # ① FPS 대표 포즈
        fps_idx = fps_pose_indices(data, n)
        combos.add(tuple(sorted(fps_idx)))

        # ② K-Fold 방식 (fold 마다 train 인덱스 사용)
        if N >= N_KFOLD:
            kf = KFold(n_splits=N_KFOLD, shuffle=True, random_state=RANDOM_SEED)
            for train_idx, _ in kf.split(indices):
                # K-Fold train 크기가 n 보다 크면 FPS 로 n 개 부분선택
                if len(train_idx) >= n:
                    sub = data.iloc[train_idx]
                    fps_sub = fps_pose_indices(sub, n)
                    real_idx = [int(sub.index[i]) for i in fps_sub]
                    combos.add(tuple(sorted(real_idx)))

        # ③ 랜덤 조합 보충
        rng = np.random.default_rng(RANDOM_SEED + n)
        while len(combos) < min(MAX_COMBINATIONS_PER_N,
                                _ncr(N, n)):
            sample = tuple(sorted(
                rng.choice(N, size=n, replace=False).tolist()))
            combos.add(sample)

        print(f"\n  n={n:>3d}  조합 수={len(combos)}")

        for combo in combos:
            subset_data = data.iloc[list(combo)].copy()
            try:
                T_nb, T_em, bias, scale = calibrate_subset(base_calib, subset_data)
                pe, re = evaluate_on_full(base_calib, T_nb, T_em, bias, scale)
                passed = bool(np.mean(pe) < POS_THRESHOLD and
                              np.mean(re) < ROT_THRESHOLD)
                records.append({
                    'method':   'KFold_Sweep',
                    'n_train':  n,
                    'combo':    str(combo),
                    'pos_mean': float(np.mean(pe)),
                    'pos_max':  float(np.max(pe)),
                    'rot_mean': float(np.mean(re)),
                    'rot_max':  float(np.max(re)),
                    'pass':     passed,
                })
            except Exception as e:
                print(f"    [WARN] combo={combo}: {e}")

        # n 별 소요 통계 출력
        n_records = [r for r in records if r['n_train'] == n]
        if n_records:
            pass_cnt = sum(r['pass'] for r in n_records)
            pos_vals = [r['pos_mean'] for r in n_records]
            print(f"    pass={pass_cnt}/{len(n_records)} | "
                  f"pos_mean min={min(pos_vals):.4f} max={max(pos_vals):.4f}mm")

    return pd.DataFrame(records)


def _ncr(n, r):
    """조합 수 C(n,r) – 폭발 방지용 상한 계산."""
    from math import comb
    try:
        return comb(n, r)
    except Exception:
        return MAX_COMBINATIONS_PER_N + 1


# ─────────────────────────────────────────────────────────────────
# 3. 결과 분석 및 최소 포즈 수 판정
# ─────────────────────────────────────────────────────────────────
def analyze_results(loocv_df: pd.DataFrame,
                    kfold_df: pd.DataFrame) -> dict:
    """
    n 별 통과율, 최소 포즈 수 판정.
    통과 기준: 해당 n 의 모든 조합이 기준을 만족해야 '안정적 n'.
    """
    print("\n" + "=" * 60)
    print("분석 결과")
    print("=" * 60)

    summary = {}

    # LOOCV 분석
    if not loocv_df.empty:
        loocv_pass_rate = loocv_df['pass'].mean()
        print(f"\n[LOOCV] 통과율: {loocv_pass_rate*100:.1f}%  "
              f"(n_train={loocv_df['n_train'].iloc[0]})")
        summary['loocv_pass_rate'] = float(loocv_pass_rate)
        summary['loocv_n_train']   = int(loocv_df['n_train'].iloc[0])

    # KFold Sweep 분석
    if not kfold_df.empty:
        group = kfold_df.groupby('n_train')
        stats = group.agg(
            total    =('pass', 'count'),
            n_pass   =('pass', 'sum'),
            pos_min  =('pos_mean', 'min'),
            pos_med  =('pos_mean', 'median'),
            pos_max  =('pos_mean', 'max'),
            rot_min  =('rot_mean', 'min'),
            rot_med  =('rot_mean', 'median'),
            rot_max  =('rot_mean', 'max'),
        ).reset_index()
        stats['pass_rate'] = stats['n_pass'] / stats['total']

        print("\n[K-Fold Sweep] n_train 별 통계:")
        print(f"{'n':>5} {'pass%':>7} {'pos_med':>9} {'pos_max':>9} "
              f"{'rot_med':>9} {'rot_max':>9}")
        print("-" * 52)
        for _, row in stats.iterrows():
            flag = " ✓" if row['pass_rate'] == 1.0 else ""
            print(f"{int(row['n_train']):>5} "
                  f"{row['pass_rate']*100:>6.1f}% "
                  f"{row['pos_med']:>9.4f} "
                  f"{row['pos_max']:>9.4f} "
                  f"{row['rot_med']:>9.4f} "
                  f"{row['rot_max']:>9.4f}{flag}")

        # 최소 안정 n: 해당 n 부터 모든 조합이 100% 통과
        min_stable_n = None
        sorted_n = sorted(stats['n_train'].unique())
        for n in sorted_n:
            row = stats[stats['n_train'] == n].iloc[0]
            if row['pass_rate'] == 1.0:
                # 이후 n 도 모두 100% 인지 확인
                tail = stats[stats['n_train'] >= n]
                if (tail['pass_rate'] == 1.0).all():
                    min_stable_n = int(n)
                    break

        # 80% 이상 통과하는 첫 n
        min_80pct_n = None
        for n in sorted_n:
            row = stats[stats['n_train'] == n].iloc[0]
            if row['pass_rate'] >= 0.8:
                min_80pct_n = int(n)
                break

        summary['kfold_stats']    = stats.to_dict('records')
        summary['min_stable_n']   = min_stable_n
        summary['min_80pct_n']    = min_80pct_n

        print(f"\n★ 최소 안정 포즈 수 (100% pass): "
              f"{min_stable_n if min_stable_n else '미달성'}")
        print(f"★ 최초 80% 이상 pass 포즈 수   : "
              f"{min_80pct_n  if min_80pct_n  else '미달성'}")

    return summary


# ─────────────────────────────────────────────────────────────────
# 4. 시각화
# ─────────────────────────────────────────────────────────────────
def visualize(loocv_df: pd.DataFrame,
              kfold_df: pd.DataFrame,
              summary: dict):
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    fig.suptitle("Calibration Pose Combination Validation\n"
                 f"(pos < {POS_THRESHOLD}mm, rot < {ROT_THRESHOLD}deg)",
                 fontsize=14, fontweight='bold')

    # ── (0,0) LOOCV pos error bar ─────────────────────────────
    ax = axes[0, 0]
    if not loocv_df.empty:
        colors = ['steelblue' if p else 'tomato' for p in loocv_df['pass']]
        ax.bar(range(len(loocv_df)), loocv_df['pos_mean'], color=colors, alpha=0.8)
        ax.axhline(POS_THRESHOLD, color='orange', linestyle='--',
                   label=f'Target {POS_THRESHOLD}mm')
        ax.set_xlabel('Left-out pose index')
        ax.set_ylabel('pos_mean (mm)')
        ax.set_title(f'LOOCV – Position Error (n={loocv_df["n_train"].iloc[0]})')
        ax.legend(); ax.grid(True, alpha=0.3)

    # ── (0,1) LOOCV rot error bar ─────────────────────────────
    ax = axes[0, 1]
    if not loocv_df.empty:
        colors = ['seagreen' if p else 'tomato' for p in loocv_df['pass']]
        ax.bar(range(len(loocv_df)), loocv_df['rot_mean'], color=colors, alpha=0.8)
        ax.axhline(ROT_THRESHOLD, color='orange', linestyle='--',
                   label=f'Target {ROT_THRESHOLD}deg')
        ax.set_xlabel('Left-out pose index')
        ax.set_ylabel('rot_mean (deg)')
        ax.set_title('LOOCV – Rotation Error')
        ax.legend(); ax.grid(True, alpha=0.3)

    # ── (0,2) LOOCV pass/fail scatter ────────────────────────
    ax = axes[0, 2]
    if not loocv_df.empty:
        pass_mask = loocv_df['pass']
        ax.scatter(loocv_df.loc[pass_mask,  'pos_mean'],
                   loocv_df.loc[pass_mask,  'rot_mean'],
                   c='steelblue', label='PASS', alpha=0.8, s=60)
        ax.scatter(loocv_df.loc[~pass_mask, 'pos_mean'],
                   loocv_df.loc[~pass_mask, 'rot_mean'],
                   c='tomato', label='FAIL', alpha=0.8, s=60, marker='x')
        ax.axvline(POS_THRESHOLD, color='orange', linestyle='--')
        ax.axhline(ROT_THRESHOLD, color='orange', linestyle='--')
        ax.set_xlabel('pos_mean (mm)'); ax.set_ylabel('rot_mean (deg)')
        ax.set_title('LOOCV – Pass/Fail Scatter')
        ax.legend(); ax.grid(True, alpha=0.3)

    # ── (1,0) KFold pos_mean vs n_train (box-like range plot) ─
    ax = axes[1, 0]
    if not kfold_df.empty:
        ns = sorted(kfold_df['n_train'].unique())
        pos_medians, pos_maxs, pos_mins = [], [], []
        for n in ns:
            sub = kfold_df[kfold_df['n_train'] == n]['pos_mean']
            pos_medians.append(sub.median())
            pos_maxs.append(sub.max())
            pos_mins.append(sub.min())
        ax.fill_between(ns, pos_mins, pos_maxs, alpha=0.25, color='steelblue',
                        label='min–max range')
        ax.plot(ns, pos_medians, 'o-', color='steelblue', label='median')
        ax.axhline(POS_THRESHOLD, color='orange', linestyle='--',
                   label=f'Target {POS_THRESHOLD}mm')
        if summary.get('min_stable_n'):
            ax.axvline(summary['min_stable_n'], color='red', linestyle=':',
                       label=f"min stable n={summary['min_stable_n']}")
        ax.set_xlabel('n_train'); ax.set_ylabel('pos_mean (mm)')
        ax.set_title('K-Fold Sweep – Position Error vs n')
        ax.legend(); ax.grid(True, alpha=0.3)

    # ── (1,1) KFold rot_mean vs n_train ──────────────────────
    ax = axes[1, 1]
    if not kfold_df.empty:
        rot_medians, rot_maxs, rot_mins = [], [], []
        for n in ns:
            sub = kfold_df[kfold_df['n_train'] == n]['rot_mean']
            rot_medians.append(sub.median())
            rot_maxs.append(sub.max())
            rot_mins.append(sub.min())
        ax.fill_between(ns, rot_mins, rot_maxs, alpha=0.25, color='seagreen',
                        label='min–max range')
        ax.plot(ns, rot_medians, 'o-', color='seagreen', label='median')
        ax.axhline(ROT_THRESHOLD, color='orange', linestyle='--',
                   label=f'Target {ROT_THRESHOLD}deg')
        if summary.get('min_stable_n'):
            ax.axvline(summary['min_stable_n'], color='red', linestyle=':',
                       label=f"min stable n={summary['min_stable_n']}")
        ax.set_xlabel('n_train'); ax.set_ylabel('rot_mean (deg)')
        ax.set_title('K-Fold Sweep – Rotation Error vs n')
        ax.legend(); ax.grid(True, alpha=0.3)

    # ── (1,2) KFold pass rate vs n_train ─────────────────────
    ax = axes[1, 2]
    if not kfold_df.empty:
        pass_rates = []
        for n in ns:
            sub = kfold_df[kfold_df['n_train'] == n]
            pass_rates.append(sub['pass'].mean() * 100)
        ax.bar(ns, pass_rates,
               color=['green' if p == 100 else 'steelblue' if p >= 80 else 'tomato'
                      for p in pass_rates],
               alpha=0.8)
        ax.axhline(100, color='green',  linestyle='--', alpha=0.5)
        ax.axhline(80,  color='orange', linestyle='--', alpha=0.5, label='80% target')
        ax.set_xlabel('n_train'); ax.set_ylabel('Pass rate (%)')
        ax.set_title('K-Fold Sweep – Pass Rate vs n')
        ax.set_ylim(0, 110)
        ax.legend(); ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    out = RESULTS_DIR / 'validation_results.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nVisualization saved → {out}")


# ─────────────────────────────────────────────────────────────────
# 5. 결과 저장
# ─────────────────────────────────────────────────────────────────
def save_results(loocv_df: pd.DataFrame,
                 kfold_df: pd.DataFrame,
                 summary: dict):
    loocv_df.to_csv(RESULTS_DIR / 'loocv_results.csv', index=False)
    kfold_df.to_csv(RESULTS_DIR / 'kfold_results.csv', index=False)

    # JSON summary (non-serializable 타입 처리)
    def _to_serializable(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    clean_summary = json.loads(
        json.dumps(summary, default=_to_serializable))
    with open(RESULTS_DIR / 'validation_summary.json', 'w') as f:
        json.dump(clean_summary, f, indent=2, ensure_ascii=False)

    print(f"CSV/JSON saved → {RESULTS_DIR}/")


# ─────────────────────────────────────────────────────────────────
# main
# ─────────────────────────────────────────────────────────────────
def main():
    csv_path = sys.argv[1] if len(sys.argv) > 1 else CSV_PATH
    print(f"Input CSV: {csv_path}")
    print(f"Thresholds: pos < {POS_THRESHOLD}mm, rot < {ROT_THRESHOLD}deg")
    print(f"K-Fold K={N_KFOLD}, max_combinations_per_n={MAX_COMBINATIONS_PER_N}")

    # ── 기본 캘리브레이션 객체 준비 ─────────────────────────
    base_calib = HandEyeCalibration(csv_path=csv_path)
    base_calib.load_and_preprocess_data()

    N = len(base_calib.all_data)
    print(f"\n총 포즈 수: {N}")

    if N < MIN_TRAIN_POSES + 1:
        print(f"[ERROR] 포즈 수({N})가 최소 실험 요구({MIN_TRAIN_POSES + 1})보다 적습니다.")
        return

    # ── LOOCV ────────────────────────────────────────────────
    loocv_df = run_loocv(base_calib)

    # ── K-Fold Sweep ─────────────────────────────────────────
    kfold_df = run_kfold_by_n(base_calib)

    # ── 분석 ─────────────────────────────────────────────────
    summary = analyze_results(loocv_df, kfold_df)

    # ── 시각화 & 저장 ────────────────────────────────────────
    visualize(loocv_df, kfold_df, summary)
    save_results(loocv_df, kfold_df, summary)

    print("\n" + "=" * 60)
    print("검증 완료.")
    print("=" * 60)


if __name__ == '__main__':
    main()