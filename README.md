# nanopore_project_with_metadata
Nanopore cancer detection project, using both time series signal and tabular metadata

Start by create a data folder by running in your terminal:

mkdir mixed_datasets

Then run this command with the fold of your choice (this will the create a dataframe later used for training the model):

python created_mixed_df.py --fold {fold}

After this is done, run the following command:

python run.py --fold {fold} --multimodel_flag {multimodel_flag}

multimodel_flag can be: "TS_only", "multimodel", or "tab_only". 

fold is any fold on which you previously ran the "create_mixed_df.py" script
