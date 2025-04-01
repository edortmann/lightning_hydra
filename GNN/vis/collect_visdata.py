import pandas as pd
import os


if __name__ == "__main__":

    directory = '/home/dortmann/PycharmProjects/lightning_hydra_git/lightning_hydra/GNN/results/'

    index = ['train_mae', 'test_mae', 'frobenius_norm', 'margin', 'train-test mae', 'weight_decay']
    df = pd.DataFrame(columns=index)

    file_names = sorted([f.name for f in os.scandir(directory) if f.is_file()])

    for file in file_names:

        df_tmp = pd.read_csv(directory + file)

        results = df_tmp.iloc[:1]

        df = pd.concat([df, results], ignore_index=True)

    df.to_csv('gnn_regression_vis_data.csv')
