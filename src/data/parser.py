import xml.etree.ElementTree as ET
import os
import glob
import pandas as pd
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

# Repository-relative root path resolution
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FAERS_DIR = PROJECT_ROOT / "dataset" / "faers"
DEFAULT_OUTPUT_CSV = PROJECT_ROOT / "dataset" / "processed_faers_warfarin.csv"

def parse_single_xml(file_path):
    """
    Parses a single FAERS XML file and extracts reports for patients taking Warfarin.
    Returns a list of dictionaries with clinical features and outcomes.
    
    Phenotype Extraction Methodology:
    - Drug Exposures: Uses substring keyword matching against medicinalproduct and activesubstancename.
    - Outcome Phenotype: Uses string keyword matching against MedDRA Preferred Terms (PTs).
      Note: This implementation uses direct keyword matching on reported MedDRA PT text rather than
      formal MedDRA hierarchy (HLT/HLGT/SOC) traversal.
    """
    file_path = str(file_path)
    print(f"Parsing {os.path.basename(file_path)}...")
    t0 = time.time()
    records = []
    
    # Non-aspirin NSAID keywords (ASPIRIN removed to prevent overlapping definition with aspirin)
    nsaid_keywords = ['IBUPROFEN', 'NAPROXEN', 'CELECOXIB', 'DICLOFENAC', 'MELOXICAM', 'KETOPROFEN', 'INDOMETHACIN']
    bleed_keywords = ['BLEED', 'HAEMORRHAGE', 'HEMORRHAGE', 'HEMATOMA', 'HAEMATOMA', 'MELENA', 'MELAENA', 'COAGULOPATHY', 'HEMOPTYSIS', 'HAEMOPTYSIS', 'EPISTAXIS', 'HAEMATURIA', 'HEMATURIA']
    
    try:
        for event, elem in ET.iterparse(file_path, events=('end',)):
            if elem.tag == 'safetyreport':
                patient = elem.find('patient')
                if patient is None:
                    elem.clear()
                    continue
                
                # Scan drugs to check for Warfarin and other exposures
                has_warfarin = False
                has_amiodarone = 0
                has_metformin = 0
                has_aspirin = 0
                has_nsaid = 0
                
                drugs_list = patient.findall('drug')
                for drug in drugs_list:
                    drug_names = []
                    med_prod = drug.find('medicinalproduct')
                    if med_prod is not None and med_prod.text:
                        drug_names.append(med_prod.text.upper())
                    
                    active_sub = drug.find('activesubstance')
                    if active_sub is not None:
                        sub_name_elem = active_sub.find('activesubstancename')
                        if sub_name_elem is not None and sub_name_elem.text:
                            drug_names.append(sub_name_elem.text.upper())
                    
                    for dname in drug_names:
                        if 'WARFARIN' in dname or 'COUMADIN' in dname:
                            has_warfarin = True
                        if 'AMIODARONE' in dname or 'CORDARONE' in dname:
                            has_amiodarone = 1
                        if 'METFORMIN' in dname or 'GLUCOPHAGE' in dname:
                            has_metformin = 1
                        if 'ASPIRIN' in dname or 'ACETYLSALICYLIC' in dname:
                            has_aspirin = 1
                        for nsaid in nsaid_keywords:
                            if nsaid in dname:
                                has_nsaid = 1
                
                if has_warfarin:
                    report_id = elem.findtext('safetyreportid', '')
                    
                    # Age parsing (handling age units 801: years, 802: months, 803: weeks, 804: days)
                    age_str = patient.findtext('patientonsetage', None)
                    age_unit = patient.findtext('patientonsetageunit', '801')
                    
                    age = None
                    if age_str is not None:
                        try:
                            age_val = float(age_str)
                            if age_unit == '801':
                                age = age_val
                            elif age_unit == '802':
                                age = age_val / 12.0
                            elif age_unit == '803':
                                age = age_val / 52.14
                            elif age_unit == '804':
                                age = age_val / 365.25
                        except ValueError:
                            pass
                    
                    # Gender parsing (1: Male, 2: Female)
                    sex_str = patient.findtext('patientsex', None)
                    gender = None
                    if sex_str == '1':
                        gender = 1.0
                    elif sex_str == '2':
                        gender = 0.0
                    
                    # Reaction parsing (bleeding events via PT keyword matching)
                    has_bleeding = 0
                    reactions = patient.findall('reaction')
                    for react in reactions:
                        react_pt = react.findtext('reactionmeddrapt', '')
                        if react_pt:
                            react_pt_up = react_pt.upper()
                            for kw in bleed_keywords:
                                if kw in react_pt_up:
                                    has_bleeding = 1
                                    break
                        if has_bleeding:
                            break
                    
                    records.append({
                        'subject_id': report_id,
                        'age': age,
                        'gender': gender,
                        'drug_a': 1,  # Warfarin cohort inclusion indicator
                        'drug_b': has_amiodarone, # Primary treatment/exposure variable
                        'drug_c': has_metformin,  # Baseline control medication covariate
                        'aspirin': has_aspirin,   # Independent Aspirin covariate
                        'nsaid': has_nsaid,       # Independent non-aspirin NSAID covariate
                        'num_drugs': len(drugs_list), # Polypharmacy indicator
                        'adr_bleeding': has_bleeding  # Bleeding outcome variable
                    })
                
                elem.clear() # Clear nodes to free memory
    except Exception as e:
        print(f"Error parsing {file_path}: {e}")
        
    print(f"Finished {os.path.basename(file_path)} in {time.time() - t0:.2f}s (Extracted {len(records)} Warfarin reports)")
    return records

def build_processed_dataset(faers_dir=None, output_path=None):
    """
    Scans the FAERS directories, parses all XML files in parallel,
    aggregates results, cleans missing values, and saves the final CSV.
    """
    if faers_dir is None:
        faers_dir = DEFAULT_FAERS_DIR
    if output_path is None:
        output_path = DEFAULT_OUTPUT_CSV
        
    faers_dir = Path(faers_dir)
    output_path = Path(output_path)
    
    t_start = time.time()
    
    xml_files = [str(p) for p in faers_dir.glob("**/*.xml")]
    print(f"Found {len(xml_files)} FAERS XML files to parse.")
    if not xml_files:
        raise FileNotFoundError(f"No FAERS XML files found in {faers_dir}!")
        
    all_records = []
    
    # Parse files in parallel using multiprocessing
    max_workers = min(4, os.cpu_count() or 1)
    print(f"Starting parallel parsing using {max_workers} processes...")
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(parse_single_xml, xml_files))
        
    for rlist in results:
        all_records.extend(rlist)
        
    df = pd.DataFrame(all_records)
    print(f"Total raw Warfarin reports extracted: {len(df)}")
    
    # Impute missing demographics (Age and Gender) using median/mode representation
    if 'age' in df.columns:
        median_age = df['age'].median()
        df['age'] = df['age'].fillna(median_age if not pd.isna(median_age) else 65.0)
    if 'gender' in df.columns:
        mode_gender = df['gender'].mode()
        df['gender'] = df['gender'].fillna(mode_gender[0] if len(mode_gender) > 0 else 1.0)
        
    # Save cache
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Saved processed dataset to {output_path} (Shape: {df.shape}) in {time.time() - t_start:.2f}s")
    return df

def validate_references():
    """
    Validates Warfarin and Amiodarone interactions and side effects
    using SIDER and PharmGKB reference datasets if available locally.
    """
    validations = {}
    
    sider_path = PROJECT_ROOT / "dataset" / "sider" / "meddra_all_se.tsv" / "meddra_all_se.tsv"
    if not sider_path.exists():
        # Fallback if meddra_all_se.tsv is a direct file
        sider_path = PROJECT_ROOT / "dataset" / "sider" / "meddra_all_se.tsv"
        
    sider_names_path = PROJECT_ROOT / "dataset" / "sider" / "drug_names.tsv"
    
    if sider_path.exists() and sider_names_path.exists():
        try:
            names_df = pd.read_csv(sider_names_path, sep="\t", header=None, names=['cid', 'name'])
            warfarin_cids = names_df[names_df['name'].str.contains('warfarin', case=False, na=False)]['cid'].tolist()
            
            if warfarin_cids:
                se_df = pd.read_csv(sider_path, sep="\t", header=None, 
                                    names=['flat_cid', 'stereo_cid', 'umls_c', 'type', 'umls_t', 'term_name'])
                warfarin_se = se_df[(se_df['flat_cid'].isin(warfarin_cids)) & (se_df['type'] == 'PT')]
                bleeding_se = warfarin_se[warfarin_se['term_name'].str.contains('bleed|haemorrhage|hemorrhage|haematoma|hematoma', case=False, na=False)]
                
                validations['sider_warfarin_bleeding_terms'] = bleeding_se['term_name'].unique().tolist()
                validations['sider_warfarin_has_bleeding'] = len(bleeding_se) > 0
        except Exception as e:
            print("SIDER reference validation error:", e)
            validations['sider_error'] = str(e)
            
    pkgb_path = PROJECT_ROOT / "dataset" / "PharmGKB" / "relationships" / "relationships.tsv"
    if pkgb_path.exists():
        try:
            rel_df = pd.read_csv(pkgb_path, sep="\t")
            warfarin_genes = rel_df[
                ((rel_df['Entity1_name'].str.lower() == 'warfarin') & 
                 (rel_df['Entity2_name'].str.upper().isin(['CYP2C9', 'CYP2C19', 'VKORC1', 'CYP3A4']))) |
                ((rel_df['Entity2_name'].str.lower() == 'warfarin') & 
                 (rel_df['Entity1_name'].str.upper().isin(['CYP2C9', 'CYP2C19', 'VKORC1', 'CYP3A4'])))
            ]
            
            amio_genes = rel_df[
                ((rel_df['Entity1_name'].str.lower() == 'amiodarone') & 
                 (rel_df['Entity2_name'].str.upper().isin(['CYP2C9', 'CYP2D6', 'CYP3A4']))) |
                ((rel_df['Entity2_name'].str.lower() == 'amiodarone') & 
                 (rel_df['Entity1_name'].str.upper().isin(['CYP2C9', 'CYP2D6', 'CYP3A4'])))
            ]
            
            validations['pkgb_warfarin_gene_relationships'] = warfarin_genes[['Entity1_name', 'Entity2_name', 'Association', 'Evidence']].to_dict('records')
            validations['pkgb_amio_gene_relationships'] = amio_genes[['Entity1_name', 'Entity2_name', 'Association', 'Evidence']].to_dict('records')
        except Exception as e:
            print("PharmGKB reference validation error:", e)
            validations['pkgb_error'] = str(e)
            
    return validations

if __name__ == '__main__':
    vals = validate_references()
    print("SIDER validation finds bleeding side effects for Warfarin:", vals.get('sider_warfarin_has_bleeding', False))
    print("SIDER Warfarin bleeding terms:", vals.get('sider_warfarin_bleeding_terms', [])[:5])
    print("PharmGKB Warfarin gene relationships found:", len(vals.get('pkgb_warfarin_gene_relationships', [])))
    print("PharmGKB Amiodarone gene relationships found:", len(vals.get('pkgb_amio_gene_relationships', [])))
