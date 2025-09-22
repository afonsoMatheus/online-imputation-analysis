
import os
import subprocess
import pandas as pd
import shutil
from tqdm import tqdm
import threading

if __name__ == "__main__":

    # sym_dates_path = os.path.join(os.path.dirname(__file__), "../../Results/sym-dates.csv")
    # symptom_dates = {}
    # with open(sym_dates_path, 'r') as f:
    #     next(f)  # Skip header
    #     for line in f:
    #         parts = line.strip().split(',')
    #         if len(parts) >= 2:
    #             symptom_dates[parts[0]] = parts[1]

    folder_path_c = os.path.join(os.path.dirname(__file__), "../../Data/COVID-19-Wearables")
    files_st = [f for f in os.listdir(folder_path_c) if f.endswith("_steps.csv")]
    threshold_contamination_path = os.path.join(os.path.dirname(__file__),"../../Data/thresh_cont.csv")
    thresholds = pd.read_csv(threshold_contamination_path)
    

    def process_mechanism(metric, mech, n):

        folder_end = os.path.join(os.path.dirname(__file__), f"../../Results/Patients/{metric}")

        num_subfolders = n

        contaminations = dict(zip(thresholds['ParticipantID'], thresholds[metric]))

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
                            CONTAMINATION = contaminations.get(myphd_id, 0.1)

                            if metric == "RHR":
                                command = [
                                    "python", "Metrics/rhrad_offline.py",
                                    "--heart_rate", hr_path,
                                    "--steps", steps_path,
                                    "--myphd_id", myphd_id,
                                    "--figure", figure_path,
                                    "--anomalies", anomalies_path,
                                    "--random_seed", "1",
                                    #"--symptom_date", symptom_date,
                                    "--outliers_fraction", str(CONTAMINATION)
                                ]
                            elif metric == "HROS":
                                command = [
                                    "python", "Metrics/hrosad_offline.py",
                                    "--heart_rate", hr_path,
                                    "--steps", steps_path,
                                    "--myphd_id", myphd_id,
                                    "--figure", figure_path,
                                    "--anomalies", anomalies_path,
                                    "--random_seed", "1",
                                    #"--symptom_date", symptom_date
                                    "--outliers_fraction", str(CONTAMINATION)
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

    def process_original(metric):

        folder_end = os.path.join(os.path.dirname(__file__), f"../../Results/Patients/{metric}")

        contaminations = dict(zip(thresholds['ParticipantID'], thresholds[metric]))

        failed_patients = []
        
        files_hr = [f for f in os.listdir(folder_path_c) if f.endswith(f"_hr.csv")]
        for file_hr in tqdm(files_hr, desc="Processing patients"):
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

                CONTAMINATION = contaminations.get(myphd_id, 0.1)
                # symptom_date = symptom_dates.get(myphd_id)

                if metric == "RHR":
                    command = [
                        "python", "Metrics/rhrad_offline.py",
                        "--heart_rate", hr_path,
                        "--steps", steps_path,
                        "--myphd_id", myphd_id,
                        "--figure", figure_path,
                        "--anomalies", anomalies_path,
                        "--random_seed", "1",
                        # "--symptom_date", symptom_date,
                        "--outliers_fraction", str(CONTAMINATION)
                    ]
                elif metric == "HROS":
                    command = [
                        "python", "Metrics/hrosad_offline.py",
                        "--heart_rate", hr_path,
                        "--steps", steps_path,
                        "--myphd_id", myphd_id,
                        "--figure", figure_path,
                        "--anomalies", anomalies_path,
                        "--random_seed", "1",
                        #"--symptom_date", symptom_date,
                        "--outliers_fraction", str(CONTAMINATION)
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
   
    metric = "RHR"
    # process_original(metric)

    mechanisms = ["MAR_m", "MNAR_m"]
    threads = []
    for mech in mechanisms:
        t = threading.Thread(target=process_mechanism, args=(metric,mech,5,))
        t.start()
        threads.append(t)
    for t in threads:
        t.join()
    