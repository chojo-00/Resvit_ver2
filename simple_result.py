import pandas as pd
import os

# =========================================================
# [설정] 파일 이름
# =========================================================
INPUT_EXCEL = "/mnt/nas100/forGPU/bc_cho/2_Code/ResViT/evaluation_metrics_result.xlsx"         # 원본 파일
OUTPUT_EXCEL = "/mnt/nas100/forGPU/bc_cho/2_Code/ResViT/evaluation_metrics_new_result.xlsx" # 결과 파일
# =========================================================

def update_summary_only():
    if not os.path.exists(INPUT_EXCEL):
        print(f"❌ 파일을 찾을 수 없습니다: {INPUT_EXCEL}")
        return

    print(f"📂 Loading data from: {INPUT_EXCEL} ...")
    
    try:
        # 1. 엑셀 파일의 모든 시트 읽기
        sheets_dict = pd.read_excel(INPUT_EXCEL, sheet_name=None)
        
        # 상세 데이터 시트 찾기
        target_sheet_name = 'Detail_All_Files'
        if target_sheet_name not in sheets_dict:
            target_sheet_name = list(sheets_dict.keys())[-1] # 없을 경우 마지막 시트 사용
        
        df_detail = sheets_dict[target_sheet_name]
        print(f"   Data loaded! ({len(df_detail)} rows)")

        # ---------------------------------------------------------
        # [Summary 계산] Mean & Std
        # ---------------------------------------------------------
        print("⚡ Calculating Summary Statistics (Mean & Std)...")
        
        metric_cols = ["PSNR", "SSIM", "MAE", "RMSE"]
        
        # 1. keV별 평균(mean)과 표준편차(std) 구하기
        summary_agg = df_detail.groupby("Source_keV")[metric_cols].agg(['mean', 'std'])
        
        # 컬럼 이름 정리 (예: PSNR_Mean, PSNR_Std)
        summary_agg.columns = [f"{col}_{stat.capitalize()}" for col, stat in summary_agg.columns]
        summary_agg = summary_agg.reset_index()
        
        # 개수(Count) 추가
        summary_agg["Count"] = df_detail.groupby("Source_keV")[df_detail.columns[0]].count().values

        # 2. 전체 평균(TOTAL_AVERAGE) 행 추가
        total_stats = {"Source_keV": "TOTAL_AVERAGE", "Count": len(df_detail)}
        for col in metric_cols:
            total_stats[f"{col}_Mean"] = df_detail[col].mean()
            total_stats[f"{col}_Std"] = df_detail[col].std()
            
        summary_df = pd.concat([summary_agg, pd.DataFrame([total_stats])], ignore_index=True)
        
        # 3. 소수점 3자리 반올림
        summary_df = summary_df.round(3)
        
        # 컬럼 순서 정리 (Source_keV, Count, 나머지...)
        cols = ['Source_keV', 'Count'] + [c for c in summary_df.columns if c not in ['Source_keV', 'Count']]
        summary_df = summary_df[cols]

        # ---------------------------------------------------------
        # [저장] Summary 업데이트 + Detail 유지
        # ---------------------------------------------------------
        print(f"💾 Saving to: {OUTPUT_EXCEL}")
        
        with pd.ExcelWriter(OUTPUT_EXCEL, engine='openpyxl') as writer:
            # 1. 새로 만든 Summary 저장
            summary_df.to_excel(writer, sheet_name='Summary', index=False)
            
            # 2. 기존 상세 데이터 저장 (소수점 3자리 적용)
            df_detail_rounded = df_detail.round(3)
            df_detail_rounded.to_excel(writer, sheet_name='Detail_All_Files', index=False)

        print("\n✅ 작업 완료! Summary가 업데이트되었습니다.")
        print(f"   결과 파일: {os.path.abspath(OUTPUT_EXCEL)}")
        
        # 콘솔에 결과 미리보기
        print("\n[Updated Summary Table]")
        print(summary_df.to_string(index=False))

    except Exception as e:
        print(f"\n❌ 에러 발생: {e}")

if __name__ == "__main__":
    update_summary_only()