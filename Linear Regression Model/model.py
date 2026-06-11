# %%
import numpy as np 
import pandas as pd 
import seaborn as sns 
import matplotlib.pyplot as plt 
import math 

# %% [markdown]
# ### model prediction using vectorisation 

# %%
def predict(inputs,weights,bias):
    y = np.dot(inputs,weights) +bias
    return y 

# %% [markdown]
# ### computing cost of parameters

# %%
def compute_cost(inputs,output,weights,bias):
    cost =0 
    m = inputs.shape[0]
    for i in range(m):
        cost += (predict(inputs[i],weights,bias)-output[i])**2
    cost /= 2 * m
    return cost 

# %%
def compute_gradients(inputs,outputs,weights,bias):
    m,n = inputs.shape
    dj_dw = np.zeros((n,))
    dj_db = 0 
    for i in range(m):
        err = (predict(inputs[i],weights,bias) - outputs[i])
        for j in range(n):
            dj_dw[j] += err * inputs[i,j]
        dj_db += err
    dj_dw /= m
    dj_db /= m
    return dj_dw,dj_db

# %%
def gradient_descent(inputs,outputs,weights_in,bias_in,alpha,iterations,tolerance=1e-4):
    j_hist = []
    weights = weights_in
    bias = bias_in
    for i in range(iterations):
        dj_dw,dj_db = compute_gradients(inputs, outputs, weights, bias)
        weights = weights - alpha* dj_dw
        bias = bias - alpha* dj_db
        current_cost =compute_cost(inputs, outputs, weights, bias)
        
        if i<100000:
            j_hist.append(current_cost)

        if i > 0:
            cost_decrease = j_hist[-2] - current_cost
            if cost_decrease < tolerance:
                print(f"Converged early at iteration {i}! Cost decrease ({cost_decrease:.6f}) is below tolerance.")
                break
        if i% math.ceil(iterations / 10) == 0:
            print(f"Iteration {i:4d}: Cost {j_hist[-1]:8.2f}   ")
        
    return weights, bias, j_hist

# %%
def linear_regression(inputs,outputs,iterations=10000,alpha=0.1):
    m,n = inputs.shape
    intial_weights = np.zeros(n)
    intial_bias = 0.
    w_final, b_final, J_hist = gradient_descent(inputs, outputs, intial_weights, intial_bias, alpha, iterations)
    print(f"b,w found by gradient descent: {b_final:0.2f},{w_final} ")
    return w_final,b_final,J_hist
    

# %%
def train_test_split(x, y, test_size=0.25):
    
    # Shuffle indices
    shuffled_indices = x.sample(frac=1, random_state=42).index

    x = x.loc[shuffled_indices].reset_index(drop=True)
    y = y.loc[shuffled_indices].reset_index(drop=True)

    train_size = int((1 - test_size) * len(x))

    x_train = x.iloc[:train_size]
    x_test = x.iloc[train_size:]

    y_train = y.iloc[:train_size]
    y_test = y.iloc[train_size:]

    return x_train, x_test, y_train, y_test

# %%
def r_squared(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / ss_tot)

def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

# %%


# %%
df = pd.read_csv("../Preprocessing/processed.csv")
# no correlation with target variable 
df = df.drop(columns=['SMOKE', 'TUE', 'NCP', 'Gender'])

x = df.drop(columns=['Weight'])
y = df['Weight']
# feature scaling 
x_scaled = (x - x.mean()) / x.std()
x_train,x_test,y_train,y_test = train_test_split(x_scaled,y)

weight,bias,j_hist = linear_regression(x_train.values,y_train.values)
train_pred = predict(x_train,weight,bias)
test_pred = predict(x_test,weight,bias)

print(f"\nTrain R²  : {r_squared(y_train, train_pred):.4f}")
print(f"Test  R²  : {r_squared(y_test,  test_pred):.4f}")
print(f"Train RMSE: {rmse(y_train, train_pred):.4f} kg")
print(f"Test  RMSE: {rmse(y_test,  test_pred):.4f} kg")


# %%
print(train_pred,test_pred)

# %%
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].hist(df['Weight'], bins=30, edgecolor='black')
axes[0].set_title("Weight Distribution")
sns.boxplot(y=df['Weight'], ax=axes[1])
axes[1].set_title("Weight Boxplot — outliers visible here")
plt.tight_layout()
plt.show()

# %%



