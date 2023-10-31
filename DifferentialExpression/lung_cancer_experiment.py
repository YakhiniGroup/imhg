import math
import os.path

import GEOparse
from tqdm import tqdm
import json

from differentially_expressed_genes import DifferentiallyExpressedGenes


def preprocess(file_path):
    with open(file_path) as f:
        data = json.load(f)
    return data


def remove_duplicates(lst):
    seen = {}
    result = []

    for sublist in lst:
        name, value = sublist[0], sublist[1]
        if name not in seen:
            seen[name] = value
        elif value < seen[name]:
            seen[name] = value

    for name, value in seen.items():
        result.append((name, value))

    return result


def preprocess_dataset(dataset, experiment_name, geo_path):
    """
    Assuming experiment name is NCBI GSE dataset
    """
    keys = list(dataset.keys())
    gse = GEOparse.get_GEO(filepath=geo_path)
    if len(gse.gpls) > 1:
        raise RuntimeError("Don't know how to parse")
    gpl_name = list(gse.gpls.keys())[0]
    metadata = gse.gpls[gpl_name].table
    res = []
    for g in keys:
        symbol = metadata[metadata['ID'] == g]['Gene Symbol'].item()
        if isinstance(symbol, str) or not math.isnan(symbol):
            if '///' not in symbol:
                res.append([symbol, float(dataset[g])])

    res = remove_duplicates(res)

    with open(experiment_name + "-genes.txt", 'w') as f:
        for r in res:
            f.write(r[0] + " " + str(r[1]) + "\n")


def start_experiment(data_path, local_geos_path, output_directory):
    """
    This function is used to run the experiment on the lung cancer dataset
    :param data_path: The path to the data file. the data file should be a json file with the following format:
    {
        "experiment_name": {
            "gene_name": "differential_expression_value"
        }
    }
    :param local_geos_path: The path to the local GEO files. In this experiment, we used the GEO files from
    https://www.ncbi.nlm.nih.gov. For every experiment we downloaded its <experiment_name>_family.soft.gz file.
    The local_geos_path should be a mapping between experiment_name to its corresponding GEO file.
    :param output_directory: The directory to save the results
    """
    data = preprocess(data_path)

    experiment_range = list(range(400, 1200, 400))

    for experiment_name in data.keys():
        print("Experiment: {}".format(experiment_name))

        genes_file = experiment_name + "-genes.txt"
        if not os.path.exists(experiment_name + "-genes.txt"):
            preprocess_dataset(data[experiment_name], experiment_name, local_geos_path[experiment_name])
        deg = DifferentiallyExpressedGenes(experiment_name,
                                           output_directory=os.path.join(output_directory, experiment_name))
        deg.load_genes_from_spaced_text_file(genes_file)
        deg.generate_reduced_chromosomes()

        for i in tqdm(experiment_range):
            _ = deg.find_enriched_intervals(i, verbose=False, ignore_errors=False)


def main():
    data_path = ""  # The user should supply the data path
    local_geo_paths = {}  # The user should supply the local GEO paths
    output_directory = "enriched_genes/lung_cancer"
    start_experiment(data_path, local_geo_paths, output_directory)


if __name__ == '__main__':
    main()
