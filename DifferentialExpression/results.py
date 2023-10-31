import datetime
import os

import pandas as pd


class ChromosomeResult(object):
    def __init__(self, experiment_name, chromosome, p_value, interval_length, interval_bs,
                 B, N, set_Bs, indexes):
        self.experiment_name = experiment_name
        self.chromosome = chromosome
        self.p_value = p_value
        self.interval_length = interval_length
        self.interval_bs = interval_bs
        self.B = B
        self.N = N
        self.set_Bs = set_Bs
        self.indexes = indexes

    def __str__(self):
        return "Experiment: {}, Chromosome: {}, p-value: {} Interval length: {} Interval B: {} B: {} N: {} Set B: {} Indexes: {}".format(
            self.experiment_name, self.chromosome, self.p_value, self.interval_length, self.interval_bs, self.B, self.N,
            self.set_Bs, self.indexes)

    def as_dataframe(self):
        data = {
            'experiment_name': [self.experiment_name],
            'chromosome': [self.chromosome],
            'p_value': [self.p_value],
            'interval_length': [self.interval_length],
            'interval_bs': [self.interval_bs],
            'B': [self.B],
            'N': [self.N],
            'set_Bs': [self.set_Bs],
            'indexes': [self.indexes]
        }
        df = pd.DataFrame(data)
        return df


class ExperimentResults(object):
    def __init__(self, experiment_name, output_directory):
        self.results = []
        self.experiment_name = experiment_name
        self.output_directory = output_directory

    def add_result(self, result):
        self.results.append(result)

    def __len__(self):
        return len(self.results)

    def __str__(self):
        output = ""
        for result in self.results:
            output += str(result) + "\n"

        if output == "":
            output = "No results found"
        return output

    def save_to_file(self):
        dfs = [res.as_dataframe() for res in self.results]
        df = pd.concat(dfs, axis=0)

        csv_name = os.path.join(self.output_directory, self.experiment_name + str(datetime.datetime.now()))
        df.to_csv(csv_name)
