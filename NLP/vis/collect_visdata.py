import pandas as pd
import os


if __name__ == "__main__":

    #directory = '/scratch/dortmann/HiwiJob/CNNBilly/20240929_181617/'
    directory = '/scratch/dortmann/HiwiJob/BERT_lightning/20241023_014053/'

    index = ['margin', 'train-test acc', 'weight_decay']
    df = pd.DataFrame(columns=index)

    folder_names = sorted([f.name for f in os.scandir(directory) if f.is_dir()])

    for _, folder_name in enumerate(folder_names):

        weight_decay = float(folder_name.strip('wd_'))

        # traverse root directory, and list directories as dirs and files as files
        for root, dirs, files in os.walk(directory + folder_name):
            path = root.split(os.sep)
            print((len(path) - 1) * '---', os.path.basename(root))
            for file in files:
                print(len(path) * '---', file)
                if file.endswith('.csv'):
                    df_tmp = pd.read_csv(root + '/' + file)

                    train_results = df_tmp.iloc[4]
                    test_results = df_tmp.iloc[5]

                    row = [train_results['margin'], train_results['train_acc'] - test_results['test_acc'], weight_decay]
                    a = pd.Series(row, index=index)

                    df = pd.concat([df, a.to_frame().T], ignore_index=True)

    df.to_csv('vis_data.csv')
