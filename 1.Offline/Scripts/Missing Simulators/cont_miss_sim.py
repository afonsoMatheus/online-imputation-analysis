
import os
import pandas as pd
import numpy as np
from tqdm import tqdm

from mdatagen.univariate.uMCAR import uMCAR
from mdatagen.univariate.uMAR import uMAR
from mdatagen.univariate.uMNAR import uMNAR

if __name__ == "__main__":

    mr_f = 5
    mr_n = 5
    mechanisms = ["MAR_l", "MAR_h"]
    num_datasets = 5

    folder_path_o = os.path.join(
        os.path.dirname(__file__), '..', '..', 'Data', 'COVID-19-Wearables'
    )
    files = [file for file in os.listdir(folder_path_o) if file.endswith('_hr.csv')]

    for mechanism in mechanisms:

        for file_name in tqdm(files, desc=f"Processing files for {mechanism}"):

            for i in range(1, num_datasets + 1):

                for mr in tqdm(range(1, mr_n + 1), desc=f"  Missing rate iterations for {file_name}", leave=False):

                    if mr == 1:
                        file_path = os.path.join(folder_path_o, file_name)
                    else:
                        folder_path_m = os.path.join(
                        os.path.dirname(__file__), '..', '..', 'Data', 'COVID-19-Wearables-Missing',
                            mechanism, file_name.split('_')[0], str(i)
                        )
                        file_path = os.path.join(
                            folder_path_m,
                            file_name.replace('.csv', f'_{mechanism}_{i}_{((mr-1)*mr_f):02d}.csv')
                        )

                    try:
                        data = pd.read_csv(file_path)
                        if mr == 1:
                            data["target"] = data["heartrate"].astype(float)
                    except Exception as e:
                        print(f"Error loading file {file_name}: {e}")
                        continue

                    # cria identificador único para cada linha
                    data = data.reset_index(drop=True)
                    data["row_id"] = data.index  

                    match mechanism:
                        case "MCAR":
                            X = data[["datetime", "heartrate", "row_id"]][~data["heartrate"].isna()].copy()
                            X_ucmar = X.set_index("datetime")
                            generator = uMCAR(
                                X=X_ucmar,
                                y=X_ucmar.heartrate.to_numpy(),
                                missing_rate=mr_f,
                                x_miss="heartrate",
                                seed=i
                            )
                            generate_data = generator.random().reset_index()
                            # adiciona o row_id de volta para garantir alinhamento
                            generate_data["row_id"] = X["row_id"].to_numpy()
                        case "MAR_l":
                            X = data[["datetime", "heartrate", "row_id"]][~data["heartrate"].isna()].copy()
                            X["time"] = pd.to_datetime(X["datetime"]).dt.time
                            generator = uMAR(
                                X=X,
                                y=X.heartrate.to_numpy(),
                                missing_rate=mr_f,
                                x_miss='heartrate',
                                x_obs='time',
                                seed=i
                            )
                            generate_data = generator.lowest().reset_index()
                            generate_data["row_id"] = X["row_id"].to_numpy()
                        case "MAR_mix":
                            X = data[["datetime", "heartrate", "row_id"]][~data["heartrate"].isna()].copy()
                            X["time"] = pd.to_datetime(X["datetime"]).dt.time
                            generator = uMAR(
                                X=X,
                                y=X.heartrate.to_numpy(),
                                missing_rate=mr_f,
                                x_miss='heartrate',
                                x_obs='time',
                                seed=i
                            )
                            generate_data = generator.mix().reset_index()
                            generate_data["row_id"] = X["row_id"].to_numpy()
                        case "MAR_h":
                            X = data[["datetime", "heartrate", "row_id"]][~data["heartrate"].isna()].copy()
                            X["time"] = pd.to_datetime(X["datetime"]).dt.time
                            generator = uMAR(
                                X=X,
                                y=X.heartrate.to_numpy(),
                                missing_rate=mr_f,
                                x_miss='heartrate',
                                x_obs='time',
                                seed=i
                            )
                            generate_data = generator.highest().reset_index()
                            generate_data["row_id"] = X["row_id"].to_numpy()
                        case "MNAR_l":
                            X = data[["datetime", "heartrate", "row_id"]][~data["heartrate"].isna()].reset_index(drop=True).copy()
                            generator = uMNAR(
                                X=X,
                                y=X.heartrate.to_numpy(),
                                threshold=0,
                                missing_rate=mr_f,
                                x_miss='heartrate',
                                seed = i
                            )
                            generate_data = generator.run().reset_index()
                            generate_data["row_id"] = X["row_id"].to_numpy()
                        case "MNAR_h":
                            X = data[["datetime", "heartrate", "row_id"]][~data["heartrate"].isna()].reset_index(drop=True).copy()
                            generator = uMNAR(
                                X=X,
                                y=X.heartrate.to_numpy(),
                                threshold=1,
                                missing_rate=mr_f,
                                x_miss='heartrate',
                                seed = i
                            )
                            generate_data = generator.run().reset_index()
                            generate_data["row_id"] = X["row_id"].to_numpy()
                        case _:
                            print(f"Invalid mechanism {mechanism}.")
                            continue

                    # substitui valores 1-1 usando row_id
                    data.loc[
                        data["row_id"].isin(generate_data["row_id"]),
                        "heartrate"
                    ] = generate_data["heartrate"].to_numpy()

                    # remove chave auxiliar antes de salvar
                    data.drop(columns="row_id", inplace=True)

                    base_folder = os.path.join(
                        os.path.dirname(__file__), '..', '..', 'Data', 'COVID-19-Wearables-Missing'
                    )

                    mechanism_folder = os.path.join(base_folder, mechanism)
                    patient_folder = os.path.join(mechanism_folder, f'{file_name.split("_")[0]}')
                    iteration_folder = os.path.join(patient_folder, f'{i}')

                    os.makedirs(mechanism_folder, exist_ok=True)
                    os.makedirs(patient_folder, exist_ok=True)
                    os.makedirs(iteration_folder, exist_ok=True)

                    save_path = os.path.join(
                        iteration_folder,
                        file_name.replace('.csv', f'_{mechanism}_{i}_{mr*mr_f:02d}.csv')
                    )
                    data = data[["datetime", "heartrate", "target"]]
                    data.to_csv(save_path, index=False)
