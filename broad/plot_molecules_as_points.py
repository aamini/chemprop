from argparse import ArgumentParser
import csv

from matplotlib import offsetbox
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import Draw
from tqdm import tqdm

from chemprop.utils import makedirs


def main(data_path: str,
         smiles_col_name: str,
         x_col_name: str,
         y_col_name: str,
         title: str,
         save_path: str):
    print('Loading data')
    with open(data_path) as f:
        data = list(csv.DictReader(f))

    all_smiles = [row[smiles_col_name] for row in data]
    all_x = [float(row[x_col_name]) for row in data]
    all_y = [float(row[y_col_name]) for row in data]

    print('Plotting')
    plt.rcParams.update({'font.size': 300})
    plt.figure(figsize=(6.4 * 50, 4.8 * 50))
    ax = plt.subplot(111)
    for smiles, x, y in tqdm(zip(all_smiles, all_x, all_y), total=len(data)):
        img = Draw.MolsToGridImage([Chem.MolFromSmiles(smiles)], molsPerRow=1, subImgSize=(400, 400))
        imagebox = offsetbox.AnnotationBbox(offsetbox.OffsetImage(img), (x, y))
        ax.add_artist(imagebox)

    min_x, max_x = min(all_x), max(all_x)
    x_range = max_x - min_x
    min_y, max_y = min(all_y), max(all_y)
    y_range = max_y - min_y

    plt.xlim(min_x - 0.05 * x_range, max_x + 0.05 * x_range)
    plt.ylim(min_y - 0.05 * y_range, max_y + 0.05 * y_range)

    plt.title(title)

    print('Saving figure')
    makedirs(save_path, isfile=True)
    plt.savefig(save_path)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to a .csv file with smiles and (x,y) coordinates')
    parser.add_argument('--smiles_col_name', type=str, default='\ufeffsmiles',
                        help='Name of column to use for smiles')
    parser.add_argument('--x_col_name', type=str, required=True,
                        help='Name of column to use for x coordinate')
    parser.add_argument('--y_col_name', type=str, required=True,
                        help='Name of column to use for y coordinate')
    parser.add_argument('--title', type=str, required=True,
                        help='Title of the plot')
    parser.add_argument('--save_path', type=str, required=True,
                        help='Path to .png file where plot will be saved')
    args = parser.parse_args()

    main(
        data_path=args.data_path,
        smiles_col_name=args.smiles_col_name,
        x_col_name=args.x_col_name,
        y_col_name=args.y_col_name,
        title=args.title,
        save_path=args.save_path
    )
