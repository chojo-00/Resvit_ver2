import os
import glob
import numpy as np
import pydicom
import pandas as pd
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error as mse
import time

# 실행 확인용 프린트 (맨 위에 추가)
print("--------------------------------------------------")
print("✅ Script Started...")
print("--------------------------------------------------")

# =========================================================
# [설정] 경로 및 타겟 설정
# =========================================================

# 1. 데이터가 있는 최상위 네트워크 경로 (리눅스 마운트 경로로 수정)
# 만약 경로가 다르다면 서버의 마운트 위치를 확인해주세요 (예: /mnt/nas206/...)
BASE_DIR = "/mnt/nas206/ANO_DET/GAN_body/Pulmonary_Embolism/sampled_data/CCY_PE_DECT/journal_data/internal"

# 2. 평가할 데이터셋 목록
DATA_SPLITS = ["train", "test", "val"]
GT_KEV = "70keV"
TARGET_KEVS = ["80keV", "84keV", "90keV", "100keV", "110keV", "125keV"]
OUTPUT_EXCEL_NAME = "comprehensive_quality_report.xlsx"
DATA_RANGE = 4095.0 

# =========================================================

def read_dicom_to_hu(path):
    try:
        dcm = pydicom.dcmread(path, force=True)
        img = dcm.pixel_array.astype(np.float32)
        slope = getattr(dcm, 'RescaleSlope', 1)
        intercept = getattr(dcm, 'RescaleIntercept', 0)
        img = img * slope + intercept
        return img
    except Exception:
        return None

def get_target_filename(gt_filename, gt_kev, target_kev):
    if "70 keV" in gt_filename:
        target_val = target_kev.replace("keV", "")
        return gt_filename.replace("70 keV", f"{target_val} keV")
    if gt_kev in gt_filename:
        return gt_filename.replace(gt_kev, target_kev)
    return gt_filename

def evaluate_all_splits():
    print(f"🚀 Starting Evaluation")
    print(f"📂 Base Path: {BASE_DIR}")
    
    if not os.path.exists(BASE_DIR):
        print(f"\n❌ [ERROR] Base path not found: {BASE_DIR}")
        return

    print(f"Splits to process: {DATA_SPLITS}\n")
    
    all_results = []

    for split in DATA_SPLITS:
        current_root = os.path.join(BASE_DIR, split)
        gt_kev_dir = os.path.join(current_root, GT_KEV)
        
        if not os.path.exists(gt_kev_dir):
            print(f"⚠️ Warning: Directory not found for split '{split}': {gt_kev_dir}")
            continue
            
        print(f"🔹 Processing Split: [{split}] ...")
        
        try:
            patient_dirs = [d for d in os.listdir(gt_kev_dir) if os.path.isdir(os.path.join(gt_kev_dir, d))]
            patient_dirs.sort()
            print(f"   Total Patients to process: {len(patient_dirs)}")
        except Exception as e:
            print(f"   Error accessing directory: {e}")
            continue
        
        processed_count = 0
        start_time = time.time() # 시작 시간 기록
        
        # 환자 루프
        for i, patient_id in enumerate(patient_dirs):
            # [진행 상황 출력] 환자 이름 출력 (줄바꿈 없이)
            print(f"   [{i+1}/{len(patient_dirs)}] Patient: {patient_id} ... ", end="", flush=True)
            
            gt_patient_path = os.path.join(gt_kev_dir, patient_id)
            gt_files = glob.glob(os.path.join(gt_patient_path, "*.dcm"))
            gt_files += glob.glob(os.path.join(gt_patient_path, "*.DCM"))
            
            file_count = 0
            
            for gt_path in gt_files:
                gt_filename = os.path.basename(gt_path)
                gt_img = read_dicom_to_hu(gt_path)
                if gt_img is None: continue

                for target_kev in TARGET_KEVS:
                    target_patient_path = os.path.join(current_root, target_kev, patient_id)
                    if not os.path.exists(target_patient_path): continue

                    target_filename = get_target_filename(gt_filename, GT_KEV, target_kev)
                    target_path = os.path.join(target_patient_path, target_filename)
                    
                    if not os.path.exists(target_path):
                        target_path_alt = os.path.join(target_patient_path, gt_filename)
                        if os.path.exists(target_path_alt):
                            target_path = target_path_alt
                        else:
                            continue 
                    
                    target_img = read_dicom_to_hu(target_path)
                    if target_img is None: continue
                    if gt_img.shape != target_img.shape: continue

                    val_psnr = psnr(gt_img, target_img, data_range=DATA_RANGE)
                    val_ssim = ssim(gt_img, target_img, data_range=DATA_RANGE)
                    val_mae = np.mean(np.abs(gt_img - target_img))
                    val_mse = mse(gt_img, target_img)

                    all_results.append({
                        "Split": split,
                        "Target_keV": target_kev,
                        "Patient_ID": patient_id,
                        "Filename": gt_filename,
                        "PSNR": val_psnr,
                        "SSIM": val_ssim,
                        "MAE": val_mae,
                        "MSE": val_mse
                    })
                    file_count += 1
            
            # [완료 출력] 한 환자가 끝나면 Done 메시지와 처리한 파일 수 출력
            print(f"Done ({file_count} comparisons)")
            
        print(f"   ✅ Finished split: {split} (Time: {time.time() - start_time:.2f}s)\n")

    if len(all_results) > 0:
        print("📊 Calculating Statistics...")
        df = pd.DataFrame(all_results)
        
        overall_summary = df.groupby("Target_keV")[["PSNR", "SSIM", "MAE", "MSE"]].agg(['mean', 'std', 'count'])
        split_summary = df.groupby(["Split", "Target_keV"])[["PSNR", "SSIM", "MAE", "MSE"]].agg(['mean', 'std', 'count'])
        
        print("\n--- Overall Summary ---")
        print(overall_summary)

        with pd.ExcelWriter(OUTPUT_EXCEL_NAME) as writer:
            overall_summary.to_excel(writer, sheet_name="Overall_Summary")
            split_summary.to_excel(writer, sheet_name="Split_Summary")
            df.to_excel(writer, sheet_name="Detail_Data", index=False)
            
        print(f"\n✅ All Done! Report saved to: {os.path.abspath(OUTPUT_EXCEL_NAME)}")
    else:
        print("\n❌ No results found. Please check your paths.")

if __name__ == "__main__":
    evaluate_all_splits()