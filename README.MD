# Statistically dense intervals in binary sequences with applications to assessing local enrichment in the human genome
This program is as described in the paper by Shahar mor et al. Link: TBD.    
With the following program you can find and report statistically dense intervals in genomic intervals with any given list of genes of interest.  

## Installation
pip install -r requirements.txt

## Usage
There are two main usages for this program as described in the paper.  
The first example shows how to find enriched intervals on GSEA human dataset.  
For your convenience, we have included the GSEA human dataset json bundle in the 'assets' folder. However, any other GSEA json bundle will be accepted as well.  
All of the below code is available in the 'GSEA/main.py' file.  
you can run the following command to find enriched intervals on the GSEA human dataset:  
First define the path to the GSEA human dataset json bundle:  
```python
gsea_json_path = "assets/msigdb.v2023.1.Hs.json"
```
Then, start the analysis by running the following command:  
```python
from GSEA.main import run_gsea_human_analysis
results = run_gsea_human_analysis(gsea_json_path)
```
Note, this process may take a whole day to complete due to the large number of experiments in the GSEA human dataset.  
To parse the results into 'pandas' dataframe, run the following command:  
```python
from GSEA.main import raw_results_to_df
df = raw_results_to_df(results)
```
To export the results to a csv file, run the following command:  
```python
from GSEA.main import dump_results_to_excel
dump_results_to_excel(results, excel_name="results.xlsx", start=0, end=None)
```
Note, the 'start' and 'end' parameters are used to limit the number of experiments to be exported.  
Set end to None to export all experiments.  

The second example shows how to find enriched intervals on a custom differential expression data.  
The full code for the following examples are visible in the 'DifferentialExpression/lung_cancer_experiment.py' and 'DifferentialExpression/breast_cancer_experiment.py' files.  
To test the program on a manual collected data, you will have to define your results file according to the following format:
```python
{
    "experiment_name_1": {
            "gene_name_1": "differential_expression_value",
            "gene_name_2": "differential_expression_value",
            "gene_name_3": "differential_expression_value",
            ...
        },
    "experiment_name_2": {
            "gene_name_1": "differential_expression_value",
            "gene_name_2": "differential_expression_value",
            "gene_name_3": "differential_expression_value",
            ...
        }
}
```
Note, full implementation details can be found on the 'start_experiment' method.  
Then, after defining the number of genes to be included in the analysis (see paper for more details), run the following code:  
```python
start_experiment(data_path, output_directory, interest_gene_sizes)
```
Note, the `breast_cancer_experiment.py` and the `lung_cancer_experiment.py` files are examples of how to use the program on a custom differential expression data.  
However, despite the common goal, these two files includes slightly different implementations due to different data sources.  
These files are not meant to be used as a general purpose tool, but rather as a reference for how to use the program on a custom differential expression data.  
Any other formatted data source will require minor adjustments to the code, but, after initial preprocessing, one can use the object `DifferentiallyExpressedGenes`  
to generate the reduced chromosome and run the analysis.  
The results will be saved to a csv file in the output directory. 
## Reference
TBD
