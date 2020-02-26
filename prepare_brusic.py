import pandas as pd
import os

samples_data = pd.read_csv('/x/SCT-10x-Metadata_readylist_merged-PBMC-tasks-short-Bgd.csv')
common_mouse_list_data = pd.read_csv('/x/common_mouse_list.csv')

"""**Load raw data**"""

data14  = pd.read_csv('/x/GSM3308814/GSM3308814_expr_readcount_BL5.csv')
data15  = pd.read_csv('/x/GSM3308815/GSM3308815_expr_readcount_BL6.csv')

data206 = pd.read_csv('/x/GSM3316206/GSM3316206_10X_whole_aorta_filtered_gene_bc_matrices_h5.csv')
data207 = pd.read_csv('/x/GSM3316207/GSM3316207_10X_lineage_positive_filtered_gene_bc_matrices_h5.csv')

def prepare_data(input_data, data_prefix):
    # path - absolute path
    
    filter_ = samples_data['SAMPLE'] == data_prefix
    target_genome = samples_data[filter_]['GENOME'].values[0]
    
    print('Using genome ' + target_genome + ' as a source.')

    # count number of nans
    print('Number of NA values: ')
    print(input_data.isnull().sum())
    
    common_mouse_list_reduced = common_mouse_list_data.loc[:, ['ENSMUSG_ID', target_genome]]
    # index =   'Index' GSM3316206 & GSM3316207
    #					  'cell'  GSM3308814 & GSM3308815
    index_ = 'cell' if '330' in data_prefix else 'Index'
    input_data_filtered = common_mouse_list_reduced.set_index('ENSMUSG_ID').join(input_data.set_index(index_), how='inner')
    
    unified_data = pd.concat([input_data_filtered, common_mouse_list_reduced.set_index('ENSMUSG_ID')], join='inner', axis=1)
    new_columns = [ data_prefix + '_' + str(num) for num in range(1,len(unified_data.columns)-1)]
    unified_data = unified_data.iloc[:, :-1]

    new_columns.insert(0,'cell_no')
    unified_data.columns = new_columns
    
    unified_data.set_index('cell_no', inplace = True)
    
    unified_data_transposed = unified_data.T

    # Filter cells
    # Sum of transcripts is at least 1000
    unified_data_transposed = unified_data_transposed.loc[unified_data_transposed.sum(axis=1) >= 1000, :]
    # Number of non-zero genes is at least 500
    unified_data_transposed = unified_data_transposed.loc[unified_data_transposed.astype(bool).sum(axis=1) >= 500, :]

    return unified_data_transposed

prepared14 = prepare_data(data14, 'GSM3308814')
prepared15 = prepare_data(data15, 'GSM3308815')
prepared206 = prepare_data(data206, 'GSM3316206')
prepared207 = prepare_data(data207, 'GSM3316207')

"""### Detect genes not present in any of the files"""

data_whole = pd.concat([prepared14, prepared15, prepared206, prepared207])

def filter_genes(min_percentage = 0.1):
  genes_before = data_whole.shape[1]
  gene_expr = data_whole.astype(bool).sum() 
  total_count = data_whole.shape[0]
  gene_expr_percentage = gene_expr / total_count
  filtered_data = data_whole.loc[:, gene_expr_percentage > min_percentage]
  print("Filtered %d genes that are present in less than %.2f percent of samples" % ((genes_before - filtered_data.shape[1]), min_percentage*100))
  return filtered_data

filt = filter_genes()

filt.to_csv('/content/drive/My Drive/x/filtered_whole')

filtered14 = filt.loc[['GSM3308814' == x.split('_')[0] for x in filt.index],:]
filtered15 = filt.loc[['GSM3308815' == x.split('_')[0] for x in filt.index],:]
filtered206 = filt.loc[['GSM3316206' == x.split('_')[0] for x in filt.index],:]
filtered207 = filt.loc[['GSM3316207' == x.split('_')[0] for x in filt.index],:]

print(filtered14.shape)
print(filtered15.shape)
print(filtered206.shape)
print(filtered207.shape)

group1 = pd.concat([filtered14, filtered15])
group2 = pd.concat([filtered206, filtered207])

group1.to_csv('/x/filteredGroup1')
group2.to_csv('/x/filteredGroup2')

filtered14.to_csv('/x/filtered14')
filtered15.to_csv('/x/filtered15')
filtered206.to_csv('/x/filtered206')
filtered207.to_csv('/x/filtered207')

"""### Detect genes that are specific to each file"""

set(prepared14.columns).difference(set(prepared206.columns))

"""All of the genes are present in all of the files."""
