


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
    sns.heatmap(c, annot = annot, mask = np.invert(cp),vmin=-1,vmax=1,cmap='seismic',xticklabels=1,yticklabels=1, annot_kws={"size": annot_size-4}, fmt='.1g')
    plt.xticks(rotation=xaxis_rot) 

    
    if explore != 1:
    
        plt.savefig("datafile"+str(idx+1)+"_corrMat.png",dpi=dpi, bbox_inches="tight", set_size_inches = (inches,inches))
    


def my_dendos(data):
    
    annot = False
    if data.shape[0] < 40: annot = True
    
    sns.set(font_scale= 2)
    
    #plt.figure(dpi=300)
    plt.figure(dpi=dpi)
    
    cg = sns.clustermap(data, metric="correlation", method = "single", standard_scale = 1, cmap = "Blues", row_cluster=False, fmt = ".1g",  annot = annot, annot_kws={"size": 8}, yticklabels = False, tree_kws={"linewidths": 1.5}, xticklabels=1)
    
    plt.setp(cg.ax_heatmap.xaxis.get_majorticklabels(), rotation=xaxis_rot)
 
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
    
    if explore == 1:
       
       # #get loadings and convert to datafram
        df_loadings = pd.DataFrame( fa_second.loadings_[:,:num_factors], index = data.columns,
                      columns = [i for i in range(1, num_factors+1)])
       
    else:
        
       #get loadings and convert to datafram
       df_loadings = pd.DataFrame( fa_second.loadings_[:,:num_factors], index = data.columns,
                  columns = factor_names[idx])
   

    #loadings of kept factors
    plt.figure(dpi=dpi)
    cg = sns.heatmap(df_loadings, annot = True, vmin=-1,vmax=1,cmap='PiYG',xticklabels=True,yticklabels=True,fmt='.1g',annot_kws={"size": annot_size})
    plt.xlabel('Factor')
    plt.xticks(rotation=xaxis_rot) 

    
    if explore != 1:
    
        plt.savefig("datafile"+str(idx+1)+"_loadingsPlot.png",dpi=dpi, bbox_inches="tight", set_size_inches = (inches,inches))
    
#    plt.ylabel('Measure')

    # cg.set_xticks(factor_names[idx])



# # study1_fa = FactorAnalyzer()
# # study1_fa.analyze(study1_BCbeadsPDI[study1_cols_to_include], 25, rotation=None)
# # # Check Eigenvalues
# # ev, v = study1_fa.get_eigenvalues()
# # ev

# # study1_fa.analyze(study1_BCbeadsPDI, 25, rotation=None)







# def eigenvalues_PCA(data):
    
#     from advanced_pca import CustomPCA
    
#     scaler = StandardScaler()
#     data_scaled = scaler.fit_transform(data)
    
#     # varimax_pca = CustomPCA(n_components=data_scaled.shape[1], rotation='varimax',feature_selection='significant').fit(data_scaled)
    
#     varimax_pca = CustomPCA(n_components=2).fit(data_scaled)
    
#     num_factors = 2
    
#     #get loadings and convert to datafram
#     df_loadings = pd.DataFrame( varimax_pca.components_[:,:num_factors], index = data.columns,
#                     columns = [i for i in range(1, num_factors+1)])
      
#     #loadings of kept factors
#     plt.figure(figsize=(7, 5))
#     sns.heatmap(df_loadings, annot = True, vmin=-1,vmax=1,cmap='PiYG',xticklabels=True,yticklabels=True,fmt='.1g')
#     plt.xlabel('Factor')
    
    
    
    
    
#controls whether loadings plots use pre-interpreted labels or are numbered and also controls whether figures afre written out in high res for use in poster
explore = 0    
dpi = 1000  
inches = 10
xaxis_rot = 65
annot_size = 16

data_files = [
    '\study1_delusionData_pdiBeadsBestChoice.csv',
    '\study2_riskData_dospertHandLBestChoice.csv',
    '\study3_N136.csv',
    #'\study4_justinAndBestChoice_Nalmost300.csv',
    #'\study4_justinBestChoiceAndCMT_Napprox140.csv'
    ]

old_cols = [
    ["views_subject_human","rank_subjects","average_draws","average_correct", "pdi_total"],
    ["Views_Subjects","Rank_Subjects","ztree","NonSocial_Risk_Taking","Social_Risk_Taking","dospert_sum"],
    ["human_samples", "human_rank", "beads_draws", "beads_accuracy", "PDI", "holt_laurie","Total_dospert"],
    #["samples", "ranks", "beadsDraws", "la_bead", "ipr", "pr", "overAdjust", "evidenceInt", "bace", "posResp", "la_bade", "pdiTotal","cape", "dospert", "crt", "verbalReasoning", "wordSum"],
    #["rt","response","drift_rate","decision_trheshold","samples", "ranks", "beadsDraws", "la_bead", "ipr", "pr", "overAdjust", "evidenceInt", "bace", "posResp", "la_bade", "PDI","cape", "CAPS", "ASI", "dospert", "crt", "verbalReasoning", "wordSum"]
    ]

new_cols = [
    ["Best choice samples", "Best choice accuracy", "Beads samples","Beads accuracy","PDI (delusion)"],
    ["Best choice samples", "Best choice accuracy", "Risk seeking task","DOSPERT nonsocial risk","DOSPERT social risk", "DOSPERT total risk"],
    ["Best choice samples", "Best choice accuracy", "Beads samples","Beads accuracy","PDI (delusion)","Risk seeking task","DOSPERT (risk)"],
    #["Best choice samples", "Best choice accuracy", "Beads samples","Beads liberal","Beads first rating","Beads prior rating", "Beads overadjustment","BADE bias", "BADE BACE", "BADE response", "BADE liberal", "PDI","PDI-CAPE", "DOSPERT","Cognitive ability - crt", "Cognitive Ability verbal","Cognitive ability wordsum"],
   # ["Motion choice RT", "Motion choice accuracy", "Motion choice drift rate","Motion choice threshold","Best choice samples", "Best choice accuracy", "Beads samples","Beads liberal","Beads first rating","Beads prior rating", "Beads overadjustment","BADE bias", "BADE BACE", "BADE response", "BADE liberal", "PDI","PDI-CAPE", "CAPS", "ASI", "DOSPERT","Cognitive ability - crt", "Cognitive Ability verbal","Cognitive ability wordsum"]
    ]

#CAREFUL: Only applicable for certain factor analysis configuration
factor_names = [["Accuracy","Beads samples & delusion"],
                ["Risk seeking","Best Choice & risk seeking"],
                ["Beads samples & Delusion","Risk seeking","Best Choice & risk seeking"],
                #["Delusion", "BADE", "Beads & Cog ability", "BADE", "Best Choice", "Beads", "Beads"],
                #["Delusion", "Motion choice", "Motion choice", "BADE", "BADE", "Psychosis", "Beads & Cog ability", "Cog ability", "Beads"]
                ]

for idx, file in enumerate(data_files):
    
    data = read_in_data(file,old_cols[idx],new_cols[idx]);

    #my_dendos(data)
    my_corr_mat(data)
    eigenvalues(data)





# study3_N136 = pd.read_csv(data_path+)

# study4_bigN = pd.read_csv(data_path+)

# study4_cmt = pd.read_csv(data_path+)





# #correlation matrix



# #####study 2

# = pd.read_csv(data_path+'\study2_riskData_dospertHandLBestChoice.csv')
# study2_BCrisk  = temp[["views_subject_human","rank_subjects","average_draws","average_correct", "pdi_total"]]
# study2_BCrisk.columns = ["Best choice samples", "Best choice accuracy", "Beads samples","Beads accuracy","PDI"]
# study2_BCrisk = study2_BCrisk.apply(pd.to_numeric, errors='coerce') #For some reason there are empty rows which were converted as objects instread of floats
# study2_BCrisk = data.dropna()

# #correlation matrix
# plt.figure()
# c= data.corr()

# #make triangular mask, revealing only lower half
# cp = (calculate_pvalues(data)<1).to_numpy()
# cp[np.triu_indices_from(cp)] = False
# sns.heatmap(c, annot = True, mask = np.invert(cp),vmin=-1,vmax=1,cmap='seismic')


# # # Create factor analysis object and perform factor analysis


# # #Do initial pass for eignevalues
# # study1_fa = FactorAnalysis(n_components=data.shape[1], random_state=0)
# # study1_fa.fit(data)

# # #Look at results

