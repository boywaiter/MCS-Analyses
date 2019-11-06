# MCS-Analyses
This repo contains scripts and datasets we use in our data analyses for MCS. Run `pip install -r requirements.txt` 
to install required packages. Make sure to download the original Conceptnet csv separately into the datasets folder.

## Scripts: 
To filter out english relations in Conceptnet in a readable form:

`python categorize_conceptnet.py`   

Add categories to Conceptnet data using Wordnet (e.g "physical"):

`python categorize_conceptnet.py`   

Calculate statistics for relations in Conceptnet:
 
`python categorize_conceptnet.py`   
 
Generate LM fine-tuning datasets from the categorized Conceptnet data:
 
`python generate_datasets.py`   