'''
Methods for loading in the methylation datasets. 

Rows are genes, columns are samples (patients). We filter out the 160 cancer 
driver genes.

There are no unobserved values. For the integer version, we multiply all values
by then and then cast values as an int.

SUMMARY: n_genes, n_samples, n_entries, fraction_obs
PM:      160,     254,       40640,     1.0
GM:      160,     254,       40640,     1.0
'''
import numpy

folder_data = '/home/tab43/Documents/Projects/libraries/BMF_Priors/data/methylation/' # '/Users/thomasbrouwer/Documents/Projects/libraries/BMF_Priors/data/methylation/' # 
file_promoter_methylation = folder_data+'matched_methylation_genePromoter'
file_gene_body_methylation = folder_data+'matched_methylation_geneBody'
file_driver_genes = folder_data+'intogen-BRCA-drivers-data.geneid'

DELIM = '\t'

def load_dataset_raw(filename):
    ''' Return a tuple (values,genes,samples) - numpy array, row names, column names. '''
    data = numpy.array([line.split('\n')[0].split(DELIM) for line in open(filename,'r').readlines()],dtype=str)
    sample_names = data[0,1:]
    gene_names = data[1:,0]
    values = numpy.array(data[1:,1:],dtype=float)
    return (values, gene_names, sample_names)
    
def load_dataset_filter_genes(filename_data, filename_genes):
    ''' Load the dataset, selecting only the appropriate genes. Return (R, M). ''' 
    (R, gene_names, sample_names) = load_dataset_raw(filename_data)
    driver_gene_names = [line.split("\n")[0] for line in open(file_driver_genes,'r').readlines()]
    gene_names = list(gene_names)
    
    genes_in_overlap = [gene for gene in driver_gene_names if gene in gene_names]
    genes_not_in_overlap = [gene for gene in driver_gene_names if not gene in gene_names]
    print "Selecting %s driver genes. %s driver genes are not in the methylation data." % \
        (len(genes_in_overlap),len(genes_not_in_overlap))
    
    driver_gene_indices = [gene_names.index(gene) for gene in genes_in_overlap]
    R = R[driver_gene_indices,:]
    M = numpy.ones(R.shape)
    return (R, M)


def load_promoter_methylation():
    ''' Return (R_pm, M_pm). '''
    return load_dataset_filter_genes(
        filename_data=file_promoter_methylation, filename_genes=file_driver_genes)

def load_promoter_methylation_integer():
    ''' Return (R_pm, M_pm), with all values multiplied by 20 and cast to int. '''
    R, M = load_promoter_methylation()
    R = numpy.array(20*R, dtype=int)
    return (R, M)
    
def load_gene_body_methylation():
    ''' Return (R_gm, M_gm). '''
    return load_dataset_filter_genes(
        filename_data=file_gene_body_methylation, filename_genes=file_driver_genes)

def load_gene_body_methylation_integer():
    ''' Return (R_gm, M_gm), with all values multiplied by 20 and cast to int. '''
    R, M = load_gene_body_methylation()
    R = numpy.array(20*R, dtype=int)
    return (R, M)


'''
R_pm, M_pm = load_promoter_methylation()
R_gm, M_gm = load_gene_body_methylation()
R_pm_int, M_pm_int = load_promoter_methylation_integer()
R_gm_int, M_gm_int = load_gene_body_methylation_integer()
'''