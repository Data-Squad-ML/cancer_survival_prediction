import pandas as pd
train_pdf = pd.read_parquet("/Volumes/workspace/default/cancer_data/03_train.parquet")
print("Total NaNs:", train_pdf.isna().sum().sum())
print(train_pdf.isna().sum()[train_pdf.isna().sum() > 0])
