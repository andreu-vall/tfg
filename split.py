from utils.andreu import MySplitDataset
import pandas as pd
import argparse


# Només es necessita el path i la length de la data. Igual que el PETER però ordenant els índexs
def create_split(data_path, new_index_dir, seed):
    data_length = pd.read_csv(f'{data_path}/reviews.csv').shape[0]
    return MySplitDataset(data_path, data_length, new_index_dir, load_split=False, seed=seed)


# Encara he de posar això net a train i test...
def parse_arguments():
    parser = argparse.ArgumentParser(description='Create a new split for the data.')
    parser.add_argument('data_path', type=str, help='Path to the data')
    parser.add_argument('split_id', type=int, help='New index directory.')
    parser.add_argument('seed', type=int, help='Seed for random number generator.')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    new_split = create_split(args.data_path, args.split_id, seed=args.seed)
