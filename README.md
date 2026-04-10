# Calibration Pose Combination Validation

이 프로젝트는 Hand-Eye Calibration에서 **최소 몇 개의 포즈가 있어야 안정적인 성능을 얻는지**를 검증하는 실험 코드입니다.

---

## 1. 목표

다음 조건을 만족하는 최소 포즈 개수(n)를 찾습니다:

- Position error < 1 mm
- Rotation error < 1 deg

---

## 2. 핵심 질문

- 포즈를 몇 개 사용해야 calibration이 안정적인가?
- 어떤 포즈 조합이 좋은 결과를 만드는가?
- 포즈 선택이 성능에 얼마나 영향을 주는가?

---

## 3. 실험 방법

### 3.1 LOOCV (Leave-One-Out Cross Validation)

- 전체 포즈에서 1개씩 제거
- 나머지로 calibration 수행
- 제거된 포즈 포함 전체 데이터로 평가

👉 목적: 데이터 1개 변화에 대한 민감도 분석

---

### 3.2 K-Fold / Subset Sweep

- 전체 포즈에서 n개씩 선택
- 다양한 조합으로 calibration 수행
- FPS (Farthest Point Sampling) + random sampling 사용

👉 목적: 포즈 조합에 따른 성능 비교

---

## 4. 평가 지표

각 실험에서 다음 값을 계산합니다:

- pos_mean: 평균 위치 오차 (mm)
- pos_max: 최대 위치 오차 (mm)
- rot_mean: 평균 회전 오차 (deg)
- rot_max: 최대 회전 오차 (deg)

---

## 5. PASS 기준

다음 조건을 만족하면 PASS:
pos_mean < 1.0 mm AND rot_mean < 1.0 deg

---

## 6. 실행 방법

```bash
python main.py
```
또는 CSV 파일 직접 지정:
```bash
python main.py path/to/calibration_data.csv
```

---

## 7. 입력 데이터
기본 경로:
```bash
./dataset/calibration/calibration_data.csv
```
포함 데이터:
- robot pose (x, y, z)
- calibration pose 정보
- pose_id

---

## 8. 출력 결과
결과는 아래 경로에 저장됩니다:
```bash
./dataset/results/
```
생성 파일:
```
loocv_results.csv
kfold_results.csv
validation_results.png
validation_summary.json
```

## 9. 결과 해석
- pass = True → 해당 pose 조합으로 calibration 성공
- pass = False → 해당 조합은 기준 실패