"""
This script automates the offline calculation of Resting Heart Rate (RHR) metrics for datasets with missing values, using different missing data mechanisms and rates.
For RHR calculation, it uses rhrad_offline.py file, available in https://github.com/gireeshkbogu/AnomalyDetect/blob/master/scripts/rhrad_offline.py

Command-line Arguments:
    -m: Missing values mechanism (choices: MCAR, MAR, MNAR)
    -p: Percentage of missing values (integer)
    -n: Number of subfolders to process (integer)

Outputs:
    - PDF figures and CSV files with detected anomalies for each patient, mechanism, and missing rate.
    - Console messages indicating missing steps files and failed subprocesses.
"""

import os
import subprocess
import argparse
import pandas as pd
import shutil
from tqdm import tqdm
import threading

if __name__ == "__main__":

    mechanisms = ["MCAR", "MAR_l", "MAR_h", "MNAR_l", "MNAR_h"]
    num_subfolders = 5

    # sym_dates_path = os.path.join(os.path.dirname(__file__), "../../Data/sym-dates.csv")
    # symptom_dates = {}
    # with open(sym_dates_path, 'r') as f:
    #     next(f)  # Skip header
    #     for line in f:
    #         parts = line.strip().split(',')
    #         if len(parts) >= 2:
    #             symptom_dates[parts[0]] = parts[1]

    folder_path_c = os.path.join(os.path.dirname(__file__), "../../Data/COVID-19-Wearables")
    files_st = [f for f in os.listdir(folder_path_c) if f.endswith("_steps.csv")]
    folder_end = os.path.join(os.path.dirname(__file__), "../../Results/Patients/RHR")

    # threshold_contamination_path = os.path.join(
    #     os.path.dirname(__file__),
    #     "../../Data/threshold_contamination.csv"
    # )
    # thresholds = pd.read_csv(threshold_contamination_path, delimiter=';', header=None)
    # threshold_p = dict(zip(thresholds.iloc[:, 0], thresholds.iloc[:, 1]))
    # print(threshold_p)

    
    def process_mechanism(mech):
        failed_patients = {}
        for i in range(1, num_subfolders + 1):
            folder_path_m_base = os.path.join(os.path.dirname(__file__), f"../../Data/COVID-19-Wearables-Missing/{mech}/")
            patients = [os.path.join(folder_path_m_base, name) for name in os.listdir(folder_path_m_base) if os.path.isdir(os.path.join(folder_path_m_base, name))]
            for pat in tqdm(patients, desc=f"Processing mechanism {mech}"):
                for mr in tqdm(["05", "10", "15", "20", "25"], desc = f"Processing {pat.split('/')[-1]}"):
                    folder_path_m = f"{pat}/{i}"
                    files_hr = [f for f in os.listdir(folder_path_m) if f.endswith(f"_hr_{mech}_{i}_{mr}.csv")]
                    for file_hr in files_hr:
                        myphd_id = file_hr.split('_')[0]
                        matching_steps = [f for f in files_st if f.startswith(myphd_id)]

                        if matching_steps:
                            hr_path = os.path.join(folder_path_m, file_hr)
                            steps_path = os.path.join(folder_path_c, matching_steps[0])
                            myphd_folder = os.path.join(folder_end, myphd_id)

                            figure_folder = os.path.join(myphd_folder, "Figures")
                            anomalies_folder = os.path.join(myphd_folder, "Anomalies")
                            os.makedirs(figure_folder, exist_ok=True)
                            os.makedirs(anomalies_folder, exist_ok=True)

                            mechanism_figure_folder = os.path.join(figure_folder, mech)
                            mechanism_anomalies_folder = os.path.join(anomalies_folder, mech)
                            os.makedirs(mechanism_figure_folder, exist_ok=True)
                            os.makedirs(mechanism_anomalies_folder, exist_ok=True)

                            ite_figure_folder = os.path.join(mechanism_figure_folder, str(i))
                            ite_anomalies_folder = os.path.join(mechanism_anomalies_folder, str(i))
                            os.makedirs(ite_figure_folder, exist_ok=True)
                            os.makedirs(ite_anomalies_folder, exist_ok=True)

                            figure_path = os.path.join(ite_figure_folder, f"{myphd_id}_{mech}_{i}_{mr}.pdf")
                            anomalies_path = os.path.join(ite_anomalies_folder, f"{myphd_id}_anomalies_{mech}_{i}_{mr}.csv") 

                            if os.path.exists(figure_path):
                                os.remove(figure_path)
                            if os.path.exists(anomalies_path):
                                os.remove(anomalies_path)

                            # symptom_date = symptom_dates.get(myphd_id)

                            command = [
                                "python", "Metrics/rhrad_offline.py",
                                "--heart_rate", hr_path,
                                "--steps", steps_path,
                                "--myphd_id", myphd_id,
                                "--figure", figure_path,
                                "--anomalies", anomalies_path,
                                "--random_seed", "1",
                                #"--symptom_date", symptom_date
                            ]

                            try:
                                subprocess.run(command, check=True)
                            except subprocess.CalledProcessError:
                                if myphd_id not in failed_patients:
                                    failed_patients[myphd_id] = []
                                failed_patients[myphd_id].append((mr, i))
                        else:
                            print(f"❌ Corresponding steps file not found for HR: {file_hr}")

            for patient_id, values in failed_patients.items():
                print(f"❌ Patient {patient_id} failed for:")
                for mr, i in values:
                    print(f"  - Missing Rate: {mr}, Subfolder: {i}")

    def process_original():
        failed_patients = []
        
        files_hr = [f for f in os.listdir(folder_path_c) if f.endswith(f"_hr.csv")]
        for file_hr in files_hr:
            myphd_id = file_hr.split('_')[0]
            matching_steps = [f for f in files_st if f.startswith(myphd_id)]
            files_hr = [f for f in os.listdir(folder_path_c) if f.endswith(f"_hr.csv")]

            if matching_steps:
                hr_path = os.path.join(folder_path_c, file_hr)
                steps_path = os.path.join(folder_path_c, matching_steps[0])
                myphd_folder = os.path.join(folder_end, myphd_id)

                figure_folder = os.path.join(myphd_folder, "Figures")
                anomalies_folder = os.path.join(myphd_folder, "Anomalies")
                os.makedirs(figure_folder, exist_ok=True)
                os.makedirs(anomalies_folder, exist_ok=True)

                figure_path = os.path.join(figure_folder, f"{myphd_id}_original.pdf")
                anomalies_path = os.path.join(anomalies_folder, f"{myphd_id}_original_anomalies.csv") 

                if os.path.exists(figure_path):
                    os.remove(figure_path)
                if os.path.exists(anomalies_path):
                    os.remove(anomalies_path)

                # symptom_date = symptom_dates.get(myphd_id)

                command = [
                    "python", "Metrics/rhrad_offline.py",
                    "--heart_rate", hr_path,
                    "--steps", steps_path,
                    "--myphd_id", myphd_id,
                    "--figure", figure_path,
                    "--anomalies", anomalies_path,
                    "--random_seed", "1",
                    #"--symptom_date", symptom_date
                ]

                try:
                    subprocess.run(command, check=True)
                except subprocess.CalledProcessError:
                    if myphd_id not in failed_patients:
                        failed_patients.append(myphd_id)
            else:
                print(f"❌ Corresponding steps file not found for HR: {file_hr}")

        for patient_id in failed_patients:
            print(f"❌ Patient {patient_id} failed for:")

    # threads = []
    # for mech in mechanisms:
    #     t = threading.Thread(target=process_mechanism, args=(mech,))
    #     t.start()
    #     threads.append(t)
    # for t in threads:
    #     t.join()

    process_original()