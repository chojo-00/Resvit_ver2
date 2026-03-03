import os
import glob
import numpy as np
import pydicom
import pandas as pd
import math
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

# =========================================================
# [설정] 경로 및 파라미터 설정
# =========================================================

# 1. 생성된(Fake) DICOM 파일이 있는 최상위 경로
FAKE_B_ROOT = "/workspace/bc_cho/2_Code/ResViT_ver2/results/ct_contrast_ver3/test_latest/dcm"

# 2. 정답(Real, GT) DICOM 파일 경로 (70keV 원본 데이터가 있는 곳)
REAL_B_ROOT = "/workspace/bc_cho/2_Code/ResViT_ver2/datasets/CCY_PE_DECT/journal_data/internal/test/70keV"

# 3. 결과 엑셀 파일 저장 이름
OUTPUT_EXCEL_NAME = "evaluation_test_metrics_result.xlsx"

# 4. CT HU 범위 (PSNR/SSIM 계산용)
MIN_HU = -1024.0
MAX_HU = 3071.0
DATA_RANGE = MAX_HU - MIN_HU  # 4095.0

# =========================================================

def read_dicom_to_hu(path):
    """DICOM 파일을 읽어 HU(Hounsfield Unit) 값의 Numpy 배열로 변환"""
    try:
        dcm = pydicom.dcmread(path, force=True)
        img = dcm.pixel_array.astype(np.float32)
        
        # Rescale Slope/Intercept 적용
        slope = getattr(dcm, 'RescaleSlope', 1)
        intercept = getattr(dcm, 'RescaleIntercept', 0)
        img = img * slope + intercept
        return img
    except Exception as e:
        print(f"Error reading {path}: {e}")
        return None

def calculate_rmse(img1, img2):
    """Calculate Root Mean Square Error"""
    mse = np.mean((img1 - img2) ** 2)
    return math.sqrt(mse)

def evaluate():
    print(f"🔍 Searching for generated DICOM files in: {FAKE_B_ROOT}")
    
    # 1. 모든 DICOM 파일 탐색
    all_files = glob.glob(os.path.join(FAKE_B_ROOT, "**", "*.dcm"), recursive=True)
    fake_files = [f for f in all_files if "fake_B" in f]

    if not fake_files:
        print("❌ 'fake_B' 폴더 내의 DICOM 파일을 찾을 수 없습니다.")
        print("   경로를 확인해주세요:", FAKE_B_ROOT)
        return

    print(f"   Total generated files found: {len(fake_files)}")
    print(f"📊 Starting evaluation with Metrics (PSNR, SSIM, MAE, RMSE)...")
    
    results_data = []
    missing_gt_count = 0

    for idx, fake_path in enumerate(fake_files):
        # -----------------------------------------------------------
        # 경로 파싱 및 정보 추출
        # -----------------------------------------------------------
        filename = os.path.basename(fake_path)
        filename_no_ext = os.path.splitext(filename)[0]
        
        # Patient ID 추출 (파일명 규칙: PE275_0001 -> PE275)
        patient_id = filename_no_ext.split('_')[0]
        
        # Source keV 추출
        path_parts = fake_path.split(os.sep)
        source_kev = "Unknown"
        for part in reversed(path_parts):
            if "keV" in part and part != "70keV":
                source_kev = part
                break
        
        # -----------------------------------------------------------
        # 정답(GT) 파일 매칭
        # -----------------------------------------------------------
        gt_name_1 = f"{filename_no_ext}_70keV.dcm"
        gt_name_2 = f"{filename_no_ext}_70 keV.dcm"
        gt_name_3 = filename

        real_path = None
        candidate_paths = [
            os.path.join(REAL_B_ROOT, patient_id, gt_name_1),
            os.path.join(REAL_B_ROOT, patient_id, gt_name_2),
            os.path.join(REAL_B_ROOT, patient_id, gt_name_3)
        ]

        for p_cand in candidate_paths:
            if os.path.exists(p_cand):
                real_path = p_cand
                break
        
        if real_path is None:
            missing_gt_count += 1
            if missing_gt_count <= 5:
                print(f"⚠️ GT File missing for: {filename} (Patient: {patient_id})")
            continue

        # -----------------------------------------------------------
        # 이미지 로드 및 평가
        # -----------------------------------------------------------
        fake_img = read_dicom_to_hu(fake_path)
        real_img = read_dicom_to_hu(real_path)

        if fake_img is None or real_img is None:
            continue
        if fake_img.shape != real_img.shape:
            # shape이 다르면 skip
            continue

        # 지표 계산
        val_psnr = psnr(real_img, fake_img, data_range=DATA_RANGE)
        val_ssim = ssim(real_img, fake_img, data_range=DATA_RANGE)
        val_mae = np.mean(np.abs(real_img - fake_img))
        val_rmse = calculate_rmse(real_img, fake_img)

        # 결과 리스트에 추가
        results_data.append({
            "Source_keV": source_kev,
            "Patient_ID": patient_id,
            "Filename": filename,
            "PSNR": val_psnr,
            "SSIM": val_ssim,
            "MAE": val_mae,
            "RMSE": val_rmse,
            "Fake_Path": fake_path,
            "Real_Path": real_path
        })
        
        if (idx + 1) % 100 == 0:
            print(f"   Processed {idx + 1}/{len(fake_files)} files...")

    # =========================================================
    # [데이터 처리 및 엑셀 저장]
    # =========================================================
    if len(results_data) > 0:
        print("\n📈 Calculating Statistics...")
        
        # 1. 기본 DataFrame 생성 (Detail Data)
        df_detail = pd.DataFrame(results_data)
        
        # --- [Sheet 2] Detail_All_Files: 소수점 3자리 정리 ---
        metric_cols = ["PSNR", "SSIM", "MAE", "RMSE"]
        df_detail_rounded = df_detail.copy()
        df_detail_rounded[metric_cols] = df_detail_rounded[metric_cols].round(3)

        # --- [Sheet 3] Patient_Average: 환자별 평균 ---
        # (Source_keV, Patient_ID) 기준으로 그룹화하여 평균 계산
        df_patient = df_detail.groupby(["Source_keV", "Patient_ID"])[metric_cols].mean().reset_index()
        # 소수점 3자리 반올림
        df_patient = df_patient.round(3)

        # --- [Sheet 1] Summary: keV별 Mean & Std ---
        # 평균(mean)과 표준편차(std) 집계
        summary_agg = df_detail.groupby("Source_keV")[metric_cols].agg(['mean', 'std'])
        
        # 컬럼 이름 평탄화 (예: ('PSNR', 'mean') -> 'PSNR_Mean')
        summary_agg.columns = [f"{col}_{stat.capitalize()}" for col, stat in summary_agg.columns]
        summary_agg = summary_agg.reset_index()
        
        # 데이터 개수(Count) 추가
        summary_agg["Count"] = df_detail.groupby("Source_keV")["Filename"].count().values

        # 전체 평균(Total Average) 행 계산
        total_stats = {}
        total_stats["Source_keV"] = "TOTAL_AVERAGE"
        total_stats["Count"] = len(df_detail)
        for col in metric_cols:
            total_stats[f"{col}_Mean"] = df_detail[col].mean()
            total_stats[f"{col}_Std"] = df_detail[col].std()
            
        total_df = pd.DataFrame([total_stats])
        
        # Summary 테이블 합치기
        df_summary = pd.concat([summary_agg, total_df], ignore_index=True)
        
        # 소수점 3자리 반올림
        df_summary = df_summary.round(3)
        
        # 컬럼 순서 보기 좋게 정렬 (Source_keV, Count, 그 뒤로 지표들)
        cols = ['Source_keV', 'Count'] + [c for c in df_summary.columns if c not in ['Source_keV', 'Count']]
        df_summary = df_summary[cols]

        # -----------------------------------------------------
        # 콘솔 출력 (Summary)
        # -----------------------------------------------------
        print("-" * 80)
        print("   [Summary by keV (Mean / Std)]")
        print("-" * 80)
        print(df_summary.to_string(index=False))
        print("-" * 80)

        # -----------------------------------------------------
        # 엑셀 저장
        # -----------------------------------------------------
        with pd.ExcelWriter(OUTPUT_EXCEL_NAME, engine='openpyxl') as writer:
            # Sheet 1: Summary
            df_summary.to_excel(writer, sheet_name='Summary', index=False)
            
            # Sheet 2: Detail (Original)
            df_detail_rounded.to_excel(writer, sheet_name='Detail_All_Files', index=False)
            
            # Sheet 3: Patient Average (New!)
            df_patient.to_excel(writer, sheet_name='Patient_Average', index=False)
            
        print(f"\n✅ Excel saved successfully: {os.path.abspath(OUTPUT_EXCEL_NAME)}")
        if missing_gt_count > 0:
            print(f"⚠️ Warning: {missing_gt_count} files were skipped because GT file was not found.")
        
    else:
        print("\n❌ No data processed. Please check file paths.")

if __name__ == "__main__":
    evaluate()