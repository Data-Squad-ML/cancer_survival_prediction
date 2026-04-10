import pandas as pd
train_pdf = pd.read_parquet("/Volumes/workspace/default/cancer_data/03_train.parquet")
nans = train_pdf.isna().sum()
print(nans[nans > 0])
