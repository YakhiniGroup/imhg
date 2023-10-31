import json
import os.path

from tqdm import tqdm

from differentially_expressed_genes import DifferentiallyExpressedGenes


def preprocess(file_path):
    with open(file_path) as f:
        data = json.load(f)
    return data


def start_experiment(data_path, output_directory):
    """
    This function is used to run the experiment on the breast cancer dataset.
    Any results will be saved automatically to the output directory.
    :param data_path: The path to the data file. the data file should be a json file with the following format:
    {
        "experiment_name": {
            "gene_name": "differential_expression_value"
        }
    }
    :param output_directory: The directory where the results will be saved.
    """
    data = preprocess(data_path)

    experiment_range = list(range(400, 1200, 400))

    for experiment_name in data.keys():
        print("Experiment: {}".format(experiment_name))

        deg = DifferentiallyExpressedGenes(experiment_name,
                                           output_directory=os.path.join(output_directory, experiment_name))
        deg.load_precomputed_genes_from_dict(data[experiment_name])
        deg.generate_reduced_chromosomes()

        for i in tqdm(experiment_range):
            _ = deg.find_enriched_intervals(i, verbose=False, ignore_errors=False)


def main():
    data_path = ""  # The user should provide the path to the data
    output_directory = "enriched_genes/breast_cancer"
    start_experiment(data_path, output_directory)


if __name__ == '__main__':
    main()
