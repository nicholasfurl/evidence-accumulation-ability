
#Based on cpc_poster_analysis_risk_beads.py. Runs a factor analysis on Justin's N=~140 sample, participants with CMT and gets factor scores for HDDM correlation with CMT drift rates and decision thresholds

import pandas as pd
#from sklearn.decomposition import FactorAnalysis
from factor_analyzer.factor_analyzer import FactorAnalyzer
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


##################
def calculate_pvalues(df):
    dfcols = pd.DataFrame(columns=df.columns)
    pvalues = dfcols.transpose().join(dfcols, how='outer')
    for r in df.columns:
        for c in df.columns:
            tmp = df[df[r].notnull() & df[c].notnull()]
            pvalues[r][c] = round(pearsonr(tmp[r], tmp[c])[1], 4)
    return pvalues



#####################
def read_in_data(filename,old_cols,new_cols):   
    data_path = r'C:\matlab_files\fiance\risk_beads\datafiles'
    temp = pd.read_csv(data_path+filename)
    data = temp[old_cols]
    data.columns = new_cols
    data = data.apply(pd.to_numeric, errors='coerce') #For some reason there are empty rows which were converted as objects instread of floats
    data = data.dropna()
    return data




#########################
def my_corr_mat(data):
    
    plt.figure(dpi=dpi)
    c= data.corr()
    
    annot = True
    if c.shape[0] > 10:
        annot = False

    #make triangular mask, revealing only lower half
    cp = (calculate_pvalues(data)<1).to_numpy()
    cp[np.triu_indices_from(cp)] = False
    sns.heatmap(c, annot = annot, mask = np.invert(cp),vmin=-1,vmax=1,cmap='seismic')
    
    if explore != 1:
    
        plt.savefig("datafile"+str(idx+1)+"_corrMat.png",dpi=dpi, bbox_inches="tight", set_size_inches = (inches,inches))
    


def my_dendos(data):
    
    annot = False
    if data.shape[0] < 40: annot = True
    
    sns.set(font_scale= 2)
    
    #plt.figure(dpi=300)
    plt.figure(dpi=dpi)
    
    cg = sns.clustermap(data, metric="correlation", method = "single", standard_scale = 1, cmap = "Blues", row_cluster=False, fmt = ".1g",  annot = annot, annot_kws={"size": 8}, yticklabels = False, tree_kws={"linewidths": 1.5}, xticklabels=1)
    
    plt.setp(cg.ax_heatmap.xaxis.get_majorticklabels(), rotation=90)
 
    sns.set(font_scale= 1)
    cg.cax.set_visible(False)
    ax = cg.ax_heatmap
    ax.set(ylabel="Participants")
    
    if explore != 1:
    
        plt.savefig("datafile"+str(idx+1)+"_clustermap.png",dpi=dpi, bbox_inches="tight", set_size_inches = (inches,inches))
    


def eigenvalues(data):
    
    #Makes no difference if you use this
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    
    #Run first pass to get number of features to extract
    fa_first = FactorAnalyzer(n_factors = data_scaled.shape[1], rotation=None)
    fa_first.fit(data_scaled) 

      
    #How many factors to keep?
    ev, v = fa_first.get_eigenvalues()
    num_factors = sum(ev>1)
    
    #Run second pass, extracting the features using Kaiser criterion and varimax rotating them
    
    fa_second = FactorAnalyzer(n_factors = num_factors, rotation='Varimax',svd_method='lapack')
    fa_second.fit(data_scaled) 
       
       # #get loadings and convert to datafram
    df_loadings = pd.DataFrame( fa_second.loadings_[:,:num_factors], index = data.columns,
                      columns = [i for i in range(1, num_factors+1)])
       
   

    #loadings of kept factors
    plt.figure(dpi=dpi)
    cg = sns.heatmap(df_loadings, annot = True, vmin=-1,vmax=1,cmap='PiYG',xticklabels=True,yticklabels=True,fmt='.1g',annot_kws={"size": 10})
    plt.xlabel('Factor')
    
    if explore != 1:
    
        plt.savefig("datafile"+str(idx+1)+"_loadingsPlot.png",dpi=dpi, bbox_inches="tight", set_size_inches = (inches,inches))





    
    
    
#controls whether loadings plots use pre-interpreted labels or are numbered and also controls whether figures afre written out in high res for use in poster
explore = 0    
dpi = 1000  
inches = 10

data_files = [
    '\study4_justinBestChoiceAndCMT_Napprox140.csv'
    ]

old_cols = [
    ["subj_idx","subnum_J","turkId","PrivateID","rt","response","samples", "ranks", "beadsDraws", "la_bead", "ipr", "pr", "overAdjust", "evidenceInt", "bace", "posResp", "la_bade", "PDI","cape", "CAPS", "ASI", "dospert", "crt", "verbalReasoning", "wordSum"]
    ]

new_cols = [
    ["subj_idx","subnum_J","turkId","PrivateID","Motion choice RT", "Motion choice accuracy", "Motion choice drift rate","Motion choice threshold","Best choice samples", "Best choice accuracy", "Beads samples","Beads liberal","Beads first rating","Beads prior rating", "Beads overadjustment","BADE bias", "BADE BACE", "BADE response", "BADE liberal", "PDI","PDI-CAPE", "CAPS", "ASI", "DOSPERT","Cognitive ability - crt", "Cognitive Ability verbal","Cognitive ability wordsum"]
    ]


for idx, file in enumerate(data_files):
    
    data = read_in_data(file,old_cols[idx],new_cols[idx]);

    my_dendos(data)
    my_corr_mat(data)
    eigenvalues(data)


