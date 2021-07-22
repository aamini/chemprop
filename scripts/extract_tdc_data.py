""" 
Helper script to extract data automatically from the Therapeutics data commons
"""

import os
from tdc.single_pred import ADME, Tox

data_dir = "data"
adme_datasets = ['Clearance_Hepatocyte_AZ', 
                 'PPBR_AZ']
tox_datasets = ['LD50_Zhu']

for dataset_names, data_importer in zip([adme_datasets, tox_datasets], 
                                        [ADME, Tox]): 
    for dataset_name in dataset_names: 
        data = data_importer(name =  dataset_name)
        split = data.get_split()
        df = data.get_data()
        df = df[['Drug', 'Y']].rename({"Y" : dataset_name.lower(),
                                       "Drug" : "smiles" }, axis= 1)
        data_out = os.path.join(data_dir, 
                                f"{dataset_name.lower()}.csv")
        print(f"Exporting dataset {dataset_name} into {data_out}")
        df.to_csv(data_out, index=False)





