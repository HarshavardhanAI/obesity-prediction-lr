# %% [markdown]
# ### preprocessing of datasets 

# %% [markdown]
# ### 1 importing modules and loading of dataset 
# 

# %%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder


# %%
df = pd.read_csv("../Data/dataset.csv")
print(df.shape)
print(df.dtypes)
print(df.describe())
print(df.isnull().sum())
print(df.nunique())
num_cols = ["Age","Height","FCVC","NCP","CH2O","FAF","TUE"]
non_numeric_col = ["Gender","family_history_with_overweight","FAVC","CAEC",
"SMOKE","SCC","CALC","MTRANS","NObeyesdad"]

# %% [markdown]
# ### 2 handling missing values sincve there is no missing values we are skiping these step 

# %% [markdown]
# ### 3 . outlier detection 
# 

# %%
def detect_outliers(df,col):
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    IQR = q3 - q1
    lower = q1 - 1.5 * IQR
    upper = q3 + 1.5 * IQR
    print(f"outlier at col {col} is {len( df[(df[col]<lower)| (df[col]>upper)])}")
    # transform outliers
    df[col] = np.where(df[col] > upper, upper, df[col])
    df[col] = np.where(df[col] < lower, lower, df[col])

for col in num_cols:
    outliers = detect_outliers(df,col)


# %% [markdown]
# ### 4. encoding data 
# 

# %%
encoder = LabelEncoder()

for col in non_numeric_col:
    print(f"{col} has {df[col].nunique()} values in it ")
non_bin_cols = ["CAEC","CALC","MTRANS","NObeyesdad"]

# %% [markdown]
# for the above we can identify the some binary col are gender , FAVC, CAEC , SMOKE , SCC
# 

# %%
bin_cols = ["Gender" , "FAVC" , "SMOKE" , "SCC","family_history_with_overweight"]
for col in bin_cols:
    df[col] = encoder.fit_transform(df[col])


# %% [markdown]
# ### ordinal encoding

# %%
for col in non_bin_cols:
    print(f"column {col} has these value {df[col].unique()}")

target_encoding_cols =["CAEC","CALC",]
map_guided = {"no": 0, "Sometimes": 1, "Frequently": 2, "Always": 3}

for col in target_encoding_cols:
    df[col] = df[col].map(map_guided)

map_guide_for_NObeyesdad = {
      'Normal_Weight':2,  'Overweight_Level_I':3, 'Overweight_Level_II':4,
      'Obesity_Type_I':5, 'Insufficient_Weight':1,     'Obesity_Type_II':6,
    'Obesity_Type_III':7    
}

df["NObeyesdad"] = df["NObeyesdad"].map(map_guide_for_NObeyesdad)



# %% [markdown]
# ### one hot enconding 

# %%
encoder = OneHotEncoder()
encoded = encoder.fit_transform(df[["MTRANS"]]).toarray()
encoder_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out())
df = df.drop(columns=["MTRANS"])
df = pd.concat([df, encoder_df], axis=1)

# %% [markdown]
# ### DOMAIN SPECIFIC TERMS 
# 

# %%

# introducing squared terms to capture non linear relationship 
df['Height_squared'] = df['Height'] ** 2
df['Age_squared'] = df['Age'] ** 2
df['CH2O_squared'] = df['CH2O'] ** 2  
## DOMAIN SPECIFIC TERMS
df['Height_x_FAF'] = df['Height'] * df['FAF'] 
df['Age_x_CH2O'] = df['Age'] * df['CH2O']
df['Genetics_x_FAVC'] = df['family_history_with_overweight'] * df['FAVC']

# %%
df.info()
df["Age"] = df["Age"].astype(int)
df['NCP']= df['NCP'].round()
df.drop("NObeyesdad",axis=1,inplace=True)

# %%
int_columns = df.select_dtypes(include=['int64']).columns
df[int_columns] = df[int_columns].astype(float)
df.to_csv('processed.csv', index=False)

# %%



