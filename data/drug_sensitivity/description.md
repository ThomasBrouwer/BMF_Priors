Folder containing the drug sensitivity datasets used for the data integration model for predicting drug sensitivity values.

We consider the following datasets:
- Sanger GDSC (*Genomics of Drug Sensitivity in Cancer*)

  IC50 values. Number drugs: 139. Number cell lines: 707. Number of observed entries: 79262. Fraction observed: 0.806549103009
  
  http://www.cancerrxgene.org/downloads/
  
- CCLE (*Cancer Cell Line Encyclopedia*)
  IC50 and EC50 values. Number drugs: 24. Number cell lines: 504. Number of observed entries IC50 / EC50: 11670 / 7626. Fraction observed IC50 / EC50: 0.964781746032 / 0.630456349206.
  
  http://www.broadinstitute.org/ccle
  
  (if website is down, use https://cghub.ucsc.edu/datasets/ccle.html)
  
- CTRP (Cancer Therapeutics Response Portal)

  EC50 values. Number drugs: 545. Number cell lines: 887. Number of observed entries: 387130. Fraction observed: 0.800823309165. 
  
  http://www.broadinstitute.org/ctrp/?page=#ctd2BodyHome
  
  Download from https://ctd2.nci.nih.gov/dataPortal/, Cancer Therapeutics Response Portal (CTRP v2, 2015) dataset, file CTRPv2.0_2015_ctd2_ExpandedDataset.zip.

Difference IC50 and EC50 (from http://www.fda.gov/ohrms/dockets/ac/00/slides/3621s1d/sld036.htm):
"The IC50 represents the concentration of a drug that is required for 50% inhibition of viral replication in vitro (can be corrected for protein binding etc.).
The EC50 represents the plasma concentration/AUC required for obtaining 50% of the maximum effect in vivo."
(In vitro = in controlled environment outside living organism, in vivo = experimentation using a whole, living organism)

The datasets are stored in **/GDSC/**, **/CTRP/**, and **/CCLE/**. Each folder contains a **/raw/** folder containing the raw datasets, as downloaded, and a Python/numpy-friendly version in **/processed_all/**, created by the Python script **process_all_gdsc/ctrp/ccle.py**.

#### /GDSC/
- **/raw/** - Original Sanger "Genomics of Drug Sensitivity in Cancer" datasets (nothing filtered). Contains 140 drugs and 707 cell lines.
  - **gdsc_manova_input_w5.csv** - The complete raw dataset from Sanger.
- **/processed_all/** - Extracted the data from the raw data files and put it into a big matrix. There is one duplicate drug (AZD6482_IC_50) so we filter it.
  - **ic50.txt** - The IC50 values only for all drugs and cell lines. Rows are cell lines, columns are drugs.
  - **drugs.txt** - List of drugs (in order of ic50.txt) - normalised name, name.
  - **cell_lines.txt** - List of cell lines (in order of ic50.txt) - normalised name, name, COSMIC id, cancer type, tissue.

#### /CTRP/
- **/raw/** - Original Cancer Therapeutic Response Portal datasets. Contains 481 compounds (70 DFA approved, 100 clinical candidates, 311 small-molecule probes) and 860 cancer cell lines. Most of the following files not actually included due to Github size limit.
  - **CTRPv2.0._COLUMNS.xlsx** - Descriptions of the columns in CTRPv2.0_2015_ctd2_ExpandedDataset.
  - **CTRPv2.0._INFORMER_SET.xlsx** - Descriptions of the drugs in CTRPv2.0_2015_ctd2_ExpandedDataset (including SMILES codes).
  - **CTRPv2.0._README.docx** - Descriptions of files in CTRPv2.0_2015_ctd2_ExpandedDataset.zip
  - **CTRPv2.0_2015_ctd2_ExpandedDataset.zip** - Zipped raw data.
  - **CTRPv2.0_2015_ctd2_ExpandedDataset** Unzipped. Interesting files are:
    - **v20.meta.per_experiment.txt** - Data about experiments (each experiment corresponds to one cell line)
    - **v20.meta.per_compound.txt** - Data about drugs (id's, SMILES).
    - **v20.meta.per_cell_line.txt** - Data about cell lines (id's, tissue and cancer types).
    - **v20.data.curves_post_qc.txt** - Area-under-concentration-response curve (AUC) sensitivity scores, curve-fit parameters, and confidence intervals for each cancer cell line and each compound. experiment_id gives the experiment id, which can give the cell line name in v20.meta.per_experiment.txt. apparent_ec50_umol gives the EC50 value. master_cpd_id gives the drug id. So need to extract the apparent_ec50_umol values, link up the experiment_id to the cell line id, and then make one big matrix.
- /processed_all/ - Extracted the data from the raw data files and put it into a big matrix. There are some VERY large values that we may want to filter out.
  - **ec50.txt** - The EC50 values only for all drugs and cell lines. Rows are cell lines, columns are drugs.
  - **drugs.txt** - List of drugs (in order of ic50.txt) - normalised name, name, id, SMILES code.
  - **cell_lines.txt** - List of cell lines (in order of ic50.txt) - normalised name, name, id, cancer type, tissue.

####/CCLE/
- **/raw/** - Raw Cancer Cell Line Encyclopedia datasets. Contains 24 cancer drugs and 504 cell lines.
  - **CCLE_NP24.2009_Drug_data_2015.02.24.csv** - Drug sensitivity values as list of experiments - drug dose on a specific cell line, IC50, EC50.
  - **CCLE_NP24.2009_profiling_2012.02.20.csv** - Data about drugs - name, primary targets, manifacturer.
  - **CCLE_GNF_data_090613.xls** - Values of the 24 drugs on 504 cell lines. 
- **/processed_all/** - Extracted the data from the raw data files and put it into a big matrix.
  - **ec50.txt** - The EC50 values only for all drugs and cell lines. Rows are cell lines, columns are drugs.
  - **ic50.txt** - The IC50 values only for all drugs and cell lines. Rows are cell lines, columns are drugs.
  - **drugs.txt** - List of drugs (in order of ic50.txt) - normalised name, name, id, SMILES code.
  - **cell_lines.txt** - List of cell lines (in order of ic50.txt) - normalised name, name, id, cancer type, tissue.
