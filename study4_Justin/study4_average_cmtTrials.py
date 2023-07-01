
#When I received the datafile joining the new ddm data with the rest of Justin's, Francesco's and my data, there were rows for each trial in the coherent motion task (cmt) that (except
#for rt and response/correct) contain redundant information about each participant. 
#This code collapses over these extraneous rows by taking the first row for each participant
#keep that example row, then adding to it the average over every subject's rows for 
#rt and response. Then each subject will be one row.


data_path = r'C:\matlab_files\fiance\risk_beads'

import sys
sys.path.append(data_path)

import pandas as pd

#Read in data
raw_data = pd.read_csv(data_path+'\justin_data_risk_rdm_ddm_onetrial.csv')

#subset the first trial of each subject
raw_data_first = raw_data[raw_data["trial"]==1]

#Get rid of old rt and response columns
raw_data_first.drop(['rt','response'], axis=1, inplace=True)

#Now get the participant-wise averages for rt and response
new_cols = raw_data.groupby(" Private ID")[["rt", "response"]].agg("mean")

#join new cols to reduced single trial dataset
raw_data_first = raw_data_first.join(new_cols,on=" Private ID")

#Just checking join is correct
raw_data_first.sort_values(" Private ID",inplace=True)

#write out new data file
raw_data_first.to_csv(data_path+"\justin_data_risk_rdm_ddm_onetrial_edit.csv")