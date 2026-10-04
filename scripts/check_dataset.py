"""
Dataset Integrity and Environment Verification Script.

Checks for required processed data (for running experiments) and optional raw datasets
(for raw data regeneration or biological reference validation).
"""

import os
import sys
from pathlib import Path

def check_dataset_status():
    project_root = Path(__file__).resolve().parents[1]
    dataset_dir = project_root / "dataset"
    
    processed_csv = dataset_dir / "processed_faers_warfarin.csv"
    faers_dir = dataset_dir / "faers"
    pharmgkb_dir = dataset_dir / "PharmGKB"
    sider_dir = dataset_dir / "sider"
    
    print("=" * 60)
    print("ADR PREDICTION FRAMEWORK — DATASET INTEGRITY CHECK")
    print("=" * 60)
    
    all_ok = True
    
    # 1. Processed Dataset Check (REQUIRED FOR EXPERIMENTS)
    print("\n[1] Checking Processed Dataset (Required for Mode A Experiments)...")
    if processed_csv.exists() and processed_csv.stat().st_size > 0:
        size_mb = processed_csv.stat().st_size / (1024 * 1024)
        print(f"  [OK] Processed dataset found: {processed_csv.name} ({size_mb:.2f} MB)")
    else:
        print(f"  [ERROR] Processed dataset missing or empty: {processed_csv}")
        print("          Run raw data parsing first or supply processed_faers_warfarin.csv.")
        all_ok = False
        
    # 2. Raw FAERS Check (OPTIONAL FOR EXPERIMENTS / REQUIRED FOR REGENERATION)
    print("\n[2] Checking Raw FAERS XML Archives (Required only for Mode B Raw Regeneration)...")
    if faers_dir.exists() and faers_dir.is_dir():
        years = [d.name for d in faers_dir.iterdir() if d.is_dir() and d.name.isdigit()]
        if years:
            print(f"  [OK] Raw FAERS directory present with years: {sorted(years)}")
        else:
            print(f"  [WARNING] Raw FAERS directory present at {faers_dir}, but no year subdirectories found.")
            print("            Raw data is required ONLY if you intend to re-parse XML files.")
    else:
        print("  [WARNING] Raw FAERS directory not found.")
        print("            Raw FAERS data (approx. 40 GB) is required ONLY for regenerating processed data.")
        
    # 3. PharmGKB Check (OPTIONAL / REFERENCE VALIDATION)
    print("\n[3] Checking PharmGKB Reference Database...")
    if pharmgkb_dir.exists() and pharmgkb_dir.is_dir():
        print(f"  [OK] PharmGKB directory present: {pharmgkb_dir.name}")
    else:
        print("  [WARNING] PharmGKB reference directory not found.")
        print("            Required ONLY for pharmacogenomics reference validation.")
        
    # 4. SIDER Check (OPTIONAL / REFERENCE VALIDATION)
    print("\n[4] Checking SIDER Reference Database...")
    if sider_dir.exists() and sider_dir.is_dir():
        print(f"  [OK] SIDER directory present: {sider_dir.name}")
    else:
        print("  [WARNING] SIDER reference directory not found.")
        print("            Required ONLY for side effect reference validation.")
        
    print("\n" + "=" * 60)
    if all_ok:
        print("STATUS: READY TO RUN EXPERIMENTS (Mode A Enabled)")
    else:
        print("STATUS: ACTION REQUIRED (Processed dataset missing)")
    print("=" * 60 + "\n")
    
    return all_ok

if __name__ == "__main__":
    success = check_dataset_status()
    sys.exit(0 if success else 1)
