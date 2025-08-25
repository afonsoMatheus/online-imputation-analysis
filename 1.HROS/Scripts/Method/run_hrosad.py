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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Runs the script for RHR calculation.")
    parser.add_argument(
        "-m", 
        type=str, 
        required=True, 
        choices=["MCAR", "MAR", "MNAR"], 
        help="Folder of the missing values mechanism (choose between: MCAR, MAR, MNAR)."
    )
    parser.add_argument("-p", type=int, required=True, help="Folder corresponding to the percentage of missing values (e.g., 25, 50, ...).")
    parser.add_argument("-n", type=int, required=True, help="Number of subfolders to process (e.g., 1, 2, ...).")

    args = parser.parse_args()
    mechanism = args.m
    mr = args.p
    num_subfolders = args.n

    folder_path_c = os.path.join(os.path.dirname(__file__), "../../Data/COVID-19-Wearables")
    folder_path_m_base = os.path.join(os.path.dirname(__file__), f"../../Data/COVID-19-Wearables-Missing/{mechanism}/{mr}")
    folder_end = os.path.join(os.path.dirname(__file__), "../../Results/Patients")
    threshold_contamination_path = os.path.join(
        os.path.dirname(__file__),
        "../../Data/threshold_contamination.csv"
    )
    thresholds = pd.read_csv(threshold_contamination_path, delimiter=';', header=None)
    threshold_p = dict(zip(thresholds.iloc[:, 0], thresholds.iloc[:, 1]))
    print(threshold_p)

    failed_patients = {}

    for i in range(1, num_subfolders + 1):
        folder_path_m = f"{folder_path_m_base}/{i}"
        try:
            files_hr = [f for f in os.listdir(folder_path_c) if f.endswith(f"_hr.csv")]
            #files_hr = [f for f in os.listdir(folder_path_m) if f.endswith(f"_hr_{mechanism}_{mr}_{i}.csv")]
            files_st = [f for f in os.listdir(folder_path_c) if f.endswith("_steps.csv")]
        except FileNotFoundError:
            print(f"❌ Folder {folder_path_m} not found.")
            continue

        for file_hr in files_hr:
            myphd_id = file_hr.split('_')[0]
            matching_steps = [f for f in files_st if f.startswith(myphd_id)]

            if matching_steps:
                #hr_path = os.path.join(folder_path_m, file_hr)
                hr_path = os.path.join(folder_path_c, file_hr)
                steps_path = os.path.join(folder_path_c, matching_steps[0])
                myphd_folder = os.path.join(folder_end, myphd_id)

                figure_folder = os.path.join(myphd_folder, "Figures")
                anomalies_folder = os.path.join(myphd_folder, "Anomalies")
                os.makedirs(figure_folder, exist_ok=True)
                os.makedirs(anomalies_folder, exist_ok=True)

                mechanism_figure_folder = os.path.join(figure_folder, mechanism)
                mechanism_anomalies_folder = os.path.join(anomalies_folder, mechanism)
                mr_figure_folder = os.path.join(mechanism_figure_folder, str(mr))
                mr_anomalies_folder = os.path.join(mechanism_anomalies_folder, str(mr))
                os.makedirs(mr_figure_folder, exist_ok=True)
                os.makedirs(mr_anomalies_folder, exist_ok=True)

                #figure_path = os.path.join(mr_figure_folder, f"{myphd_id}_offline_{mechanism}_{mr}_{i}.pdf")
                figure_path = os.path.join(figure_folder, f"{myphd_id}_original.pdf")
                #anomalies_path = os.path.join(mr_anomalies_folder, f"{myphd_id}_offline_anomalies_{mechanism}_{mr}_{i}.csv") 
                anomalies_path = os.path.join(anomalies_folder, f"{myphd_id}_original_anomalies.csv")

                if os.path.exists(figure_path):
                    os.remove(figure_path)
                if os.path.exists(anomalies_path):
                    os.remove(anomalies_path)

                command = [
                    "python", "Metrics/hrosad_offline.py",
                    "--heart_rate", hr_path,
                    "--steps", steps_path,
                    "--myphd_id", myphd_id,
                    "--figure", figure_path,
                    "--anomalies", anomalies_path,
                    #"--outliers_fraction", str(threshold_p.get(myphd_id)),
                    "--random_seed", "1"
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
