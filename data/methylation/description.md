Folder containing the methylation datasets.

There are 254 rows (samples / patients) and 160 columns (cancer driver genes).
There are more genes but we focus on 160 cancer driver genes.

load_data.py provides methods for loading in the data, either as the raw values,
or multiplying each value by 10 and casting it as an integer.

Files:
- matched_methylation_geneBody - Gene body methylation values. Tab-delimited, first row is sample names, first column is gene ids.
- matched_methylation_genePromoter - Promoter region methylation values.  Tab-delimited, first row is sample names, first column is gene ids.
- intogen-BRCA-drivers-data.geneid - Cancer driver gene ids (one on each line).

Source of data: see paper.