# Statistically dense intervals in binary sequences with applications to assessing local enrichment in the human genome
This program is as described in the paper by Shahar mor et al. Link: TBD.    
With the following program you can find and report statistically dense intervals in genomic intervals with any given list of genes of interest.  

## Installation
pip install -r requirements.txt

## Usage
There are two main usages for this program as described in the paper.  
The first example shows how to find enriched intervals on GSEA human dataset.  
For your convenience, we have included the GSEA human dataset json bundle in the 'assets' folder. However, any other GSEA json bundle will be accepted as well.  
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

## Reference
TBD

## Notes
