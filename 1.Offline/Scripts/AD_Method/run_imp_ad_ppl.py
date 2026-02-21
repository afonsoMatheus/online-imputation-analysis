
import os
import subprocess
import pandas as pd
import shutil
from tqdm import tqdm
import threading
import argparse
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import ThreadPoolExecutor, as_completed

# Utilizar Python 3.6.15
MECHANISMS = [["MCAR"], ["MAR_l"], ["MAR_m"], ["MAR_h"], ["MNAR_l"], ["MNAR_m"], ["MNAR_h"]]
#MECHANISMS = [["MAR_l"]]
EXCLUDED_PATIENTS = ['AJ7TSV9','AS2MVDL']
N = 1
MRS = ["05", "10", "15", "20", "25"]
IMP_MODELS = [
    'mean',
    'reg_opt_00001',
    'tree_gp_10000',
    'tree-ad_gp_5000-dw_30000',
    'mlp_opt_0001'
]

MAX_WORKERS_IMP = min(len(IMP_MODELS), os.cpu_count())
MAX_MR_WORKERS = min(len(MRS), os.cpu_count())

if __name__ == "__main__":

    folder_path_c = os.path.join(os.path.dirname(__file__), "../../Data/COVID-19-Wearables")
    files_st = [f for f in os.listdir(folder_path_c) if f.endswith("_steps.csv")]
    threshold_contamination_path = os.path.join(os.path.dirname(__file__),"../../Data/thresh_cont.csv")
    thresholds = pd.read_csv(threshold_contamination_path)

    def process_imputer(pat, mr, imp, i, mech, metric,
                     files_st, folder_path_c, folder_end,
                     contaminations, failed_patients):

        print(f"Processing imputer {imp} for MR {mr} and mechanism {mech}")

        folder_path_m = f"{pat}/{i}/{imp}"
        files_hr = [
            f for f in os.listdir(folder_path_m)
            if f.endswith(f"_hr_{mech}_{i}_{mr}_{imp}.csv")
        ]

        for file_hr in files_hr:
            myphd_id = file_hr.split('_')[0]
            matching_steps = [f for f in files_st if f.startswith(myphd_id)]

            if not matching_steps:
                print(f"❌ Corresponding steps file not found for HR: {file_hr}")
                return

            hr_path = os.path.join(folder_path_m, file_hr)
            steps_path = os.path.join(folder_path_c, matching_steps[0])
            myphd_folder = os.path.join(folder_end, myphd_id)

            anomalies_folder = os.path.join(myphd_folder, "Anomalies", mech, str(i), imp)
            os.makedirs(anomalies_folder, exist_ok=True)

            anomalies_path = os.path.join(
                anomalies_folder,
                f"{myphd_id}_anomalies_{mech}_{i}_{mr}_{imp}.csv"
            )

            if os.path.exists(anomalies_path):
                os.remove(anomalies_path)

            CONTAMINATION = contaminations.get(myphd_id, 0.1)

            if metric == "RHR":
                command = [
                    "python", "Metrics/rhrad_offline.py",
                    "--heart_rate", hr_path,
                    "--steps", steps_path,
                    "--myphd_id", myphd_id,
                    "--anomalies", anomalies_path,
                    "--random_seed", "1",
                    "--outliers_fraction", str(CONTAMINATION)
                ]
            else:  # HROS
                command = [
                    "python", "Metrics/hrosad_offline.py",
                    "--heart_rate", hr_path,
                    "--steps", steps_path,
                    "--myphd_id", myphd_id,
                    "--anomalies", anomalies_path,
                    "--random_seed", "1",
                    "--outliers_fraction", str(CONTAMINATION)
                ]

            try:
                subprocess.run(command, check=True)
            except subprocess.CalledProcessError as e:
                failed_patients.setdefault(myphd_id, []).append((mr, i))
                print(f"Error processing {myphd_id} with imputer {imp} for MR {mr} and mechanism {mech} at level {i}")

    def process_mr(pat, mr, i, mech, metric,
               IMP_MODELS, files_st, folder_path_c, folder_end,
               contaminations, failed_patients):

        max_workers_imp = min(len(IMP_MODELS), os.cpu_count())

        with ThreadPoolExecutor(max_workers=max_workers_imp) as executor:
            futures = [
                executor.submit(
                    process_imputer,
                    pat, mr, imp, i, mech, metric,
                    files_st, folder_path_c, folder_end,
                    contaminations, failed_patients
                )
                for imp in IMP_MODELS
            ]

            for future in as_completed(futures):
                future.result()

    def process_mechanism(metric, mech, n):

        folder_end = os.path.join(os.path.dirname(__file__), f"../../Results/Patients/{metric}")

        num_subfolders = n

        contaminations = dict(zip(thresholds['ParticipantID'], thresholds[metric]))

        failed_patients = {}
        for i in range(1, num_subfolders + 1):
            folder_path_m_base = os.path.join(os.path.dirname(__file__), f"../../Data/COVID-19-Wearables-Input/{mech}/")
            patients = [os.path.join(folder_path_m_base, name) 
                        for name in os.listdir(folder_path_m_base) 
                        if os.path.isdir(os.path.join(folder_path_m_base, name))
                        and name not in EXCLUDED_PATIENTS]
            
            for pat in tqdm(patients, desc=f"Processing mechanism {mech}"):
                with ThreadPoolExecutor(max_workers=MAX_MR_WORKERS) as mr_executor:
                    mr_futures = [
                        mr_executor.submit(
                            process_mr,
                            pat, mr, i, mech, metric,
                            IMP_MODELS, files_st, folder_path_c, folder_end,
                            contaminations, failed_patients
                        )
                        for mr in MRS
                    ]

                    for future in as_completed(mr_futures):
                        future.result()

   
    parser = argparse.ArgumentParser(description="Run AD metric processing")
    parser.add_argument("-m", "--metric", choices=["RHR", "HROS"], default="RHR", help="Metric to process (default: RHR)")
    args = parser.parse_args()
    METRIC = args.metric
    print(f"Processing metric: {METRIC}")
    print(f"MECHANISMS: {MECHANISMS}, MRS: {MRS}, IMP_MODELS: {IMP_MODELS}, N: {N}")

    threads = []
    for mech in MECHANISMS:
        for m in mech:
            t = threading.Thread(target=process_mechanism, args=(METRIC,m,N,))
            t.start()
            threads.append(t)
        for t in threads:
            t.join()
    