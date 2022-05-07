#%%
import warnings
warnings.filterwarnings('ignore')

# %%
#importing the libraries 
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 
cars = pd.read_csv('Case_study_CarPrice_Assignment.csv') 
cars.head()

# %%
cars.info()
#%%
CompanyName = cars['CarName'].apply(lambda x : x.split(' ')[0]) 
cars.insert(3,"CompanyName",CompanyName)
cars.drop(['CarName'],axis=1,inplace=True)
cars.head()

# %%
cars.CompanyName.unique()

# %%
def replace_name(a,b):     
    cars.CompanyName.replace(a,b,inplace=True)

# %%
replace_name('maxda','mazda') 
replace_name('porcshce','porsche') 
replace_name('toyouta','toyota') 
replace_name('vokswagen','volkswagen') 
replace_name('vw','volkswagen') 
cars.CompanyName.unique()

# %%
cars.loc[cars.duplicated()]

# %%
cars.columns

# %%
plt.figure(figsize=(20,8))
plt.subplot(1,2,1)
plt.title('Car Price Distribution Plot')
sns.distplot(cars.price)
plt.subplot(1,2,2) 
plt.title('Car Price Spread') 
sns.boxplot(y=cars.price)
plt.show()

# %%
print(cars.price.describe(percentiles = [0.25,0.50,0.75,0.85,0.90,1]))

# %%
plt.figure(figsize=(25, 6))
plt.subplot(1,3,1) 
car_count = cars.CompanyName.value_counts()
car_count.plot.bar() 
plt.title('Companies Histogram')
plt.subplot(1,3,2) 
car_fueltype_count = cars.fueltype.value_counts() 
car_fueltype_count.plot.bar() 
plt.title('Fuel Type Histogram')
plt.subplot(1,3,3) 
carbody_count = cars.carbody.value_counts() 
carbody_count.plot.bar() 
plt.title('Car Type Histogram')
plt.show()

# %%
plt.figure(figsize=(20,8))
plt.subplot(1,2,1) 
plt.title('Symboling Histogram') 
sns.countplot(cars.symboling, palette=("cubehelix"))
plt.subplot(1,2,2) 
plt.title('Symboling vs Price') 
sns.boxplot(x=cars.symboling, y=cars.price, palette=("cubehelix"))
plt.show()

# %%
plt.figure(figsize=(20,8))
plt.subplot(1,2,1) 
plt.title('Engine Type Histogram') 
sns.countplot(cars.enginetype, palette=("Blues_d"))
plt.subplot(1,2,2) 
plt.title('Engine Type vs Price') 
sns.boxplot(x=cars.enginetype, y=cars.price, palette=("PuBuGn"))
plt.show()
df = pd.DataFrame(cars.groupby(['enginetype'])['price'].mean().sort_values(ascending = False)) 
df.plot.bar(figsize=(8,6)) 
plt.title('Engine Type vs Average Price') 
plt.show()

# %%
plt.figure(figsize=(25, 6))
df = pd.DataFrame(cars.groupby(['CompanyName'])['price'].mean().sort_values(ascending = False)) 
df.plot.bar() 
plt.title('Company Name vs Average Price') 
plt.show()
df = pd.DataFrame(cars.groupby(['fueltype'])['price'].mean().sort_values(ascending = False)) 
df.plot.bar() 
plt.title('Fuel Type vs Average Price') 
plt.show()
df = pd.DataFrame(cars.groupby(['carbody'])['price'].mean().sort_values(ascending = False)) 
df.plot.bar() 
plt.title('Car Type vs Average Price') 
plt.show()

# %%
plt.figure(figsize=(15,5))
plt.subplot(1,2,1) 
plt.title('Door Number Histogram') 
sns.countplot(cars.doornumber, palette=("plasma"))
plt.subplot(1,2,2) 
plt.title('Door Number vs Price') 
sns.boxplot(x=cars.doornumber, y=cars.price, palette=("plasma"))
plt.show()
plt.figure(figsize=(15,5))
plt.subplot(1,2,1) 
plt.title('Aspiration Histogram') 
sns.countplot(cars.aspiration, palette=("plasma"))
plt.subplot(1,2,2) 
plt.title('Aspiration vs Price') 
sns.boxplot(x=cars.aspiration, y=cars.price, palette=("plasma"))
plt.show()

# %%
def plot_count(x,fig):     
    plt.subplot(4,2,fig)     
    plt.title(x+' Histogram')     
    sns.countplot(cars[x],palette=("magma"))     
    plt.subplot(4,2,(fig+1))     
    plt.title(x+' vs Price')     
    sns.boxplot(x=cars[x], y=cars.price, palette=("magma"))

plt.figure(figsize=(15,20))
plot_count('enginelocation', 1) 
plot_count('cylindernumber', 3) 
plot_count('fuelsystem', 5) 
plot_count('drivewheel', 7)
plt.tight_layout()

# %%
def scatter(x,fig):
    plt.subplot(5,2,fig)
    plt.scatter(cars[x],cars['price'])
    plt.title(x+' vs Price')
plt.figure(figsize=(10,20))
scatter('carlength', 1) 
scatter('carwidth', 2) 
scatter('carheight', 3) 
scatter('curbweight', 4)
plt.tight_layout()
# %%
def pp(x,y,z):
    sns.pairplot(cars, x_vars=[x,y,z], y_vars='price',size=4, aspect=1, kind=' scatter')     
    plt.show()
pp('enginesize', 'boreratio', 'stroke')
pp('compressionratio', 'horsepower', 'peakrpm')
pp('wheelbase', 'citympg', 'highwaympg')

# %%
from scipy import stats 
print ("correlation between price and numeric variables") 
print ("enginesize") 
print (stats.pearsonr(cars.enginesize, cars.price)) 
print ("boreratio") 
print (stats.pearsonr(cars.boreratio, cars.price)) 
print ("stroke") 
print (stats.pearsonr(cars.stroke, cars.price)) 
print ("compressionratio") 
print (stats.pearsonr(cars.compressionratio, cars.price)) 
print ("horsepower") 
print (stats.pearsonr(cars.horsepower, cars.price)) 
print ("peakrpm") 
print (stats.pearsonr(cars.peakrpm, cars.price)) 
print ("wheelbase") 
print (stats.pearsonr(cars.wheelbase, cars.price)) 
print ("citympg") 
print (stats.pearsonr(cars.citympg, cars.price)) 
print ("highwaympg") 
print (stats.pearsonr(cars.highwaympg, cars.price))

# %%
np.corrcoef(cars['carlength'], cars['carwidth'])[0, 1] 
cars['fueleconomy'] = (0.55 * cars['citympg']) + (0.45 * cars['highwaympg']) 
cars['price'] = cars['price'].astype('int') 
temp = cars.copy() 
table = temp.groupby(['CompanyName'])['price'].mean() 
temp = temp.merge(table.reset_index(), how='left',on='CompanyName') 
bins = [0,10000,20000,40000] 
cars_bin=['Budget','Medium','Highend'] 
cars['carsrange'] = pd.cut(temp['price_y'],bins,right=False,labels=cars_bin) 
cars.head()

# %%
plt.figure(figsize=(8,6))
plt.title('Fuel economy vs Price') 
sns.scatterplot(x=cars['fueleconomy'],y=cars['price'],hue=cars['drivewheel']) 
plt.xlabel='Fuel Economy' 
plt.ylabel='Price'
plt.show() 
plt.tight_layout()

# %%
plt.figure(figsize=(25, 6))
df = pd.DataFrame(cars.groupby(['fuelsystem','drivewheel','carsrange'])['price'].mean().unstack(fill_value=0)) 
df.plot.bar() 
plt.title('Car Range vs Average Price') 
plt.show()

# %%
cars_lr = cars[['price', 'fueltype', 'aspiration','carbody', 'drivewheel','wheelbase','curbweight', 'enginetype', 'cylindernumber', 'enginesize', 'boreratio','horsepower','fueleconomy', 'carlength','carwidth', 'carsrange']] 
cars_lr.head()

# %%
sns.pairplot(cars_lr) 
plt.show()

# %%
# Defining the map function
def dummies(x,df):     
    temp = pd.get_dummies(df[x], drop_first = True)     
    df = pd.concat([df, temp], axis = 1)     
    df.drop([x], axis = 1, inplace = True) 
    return df # Applying the function to the cars_lr
cars_lr = dummies('fueltype',cars_lr) 
cars_lr = dummies('aspiration',cars_lr) 
cars_lr = dummies('carbody',cars_lr) 
cars_lr = dummies('drivewheel',cars_lr) 
cars_lr = dummies('enginetype',cars_lr) 
cars_lr = dummies('cylindernumber',cars_lr) 
cars_lr = dummies('carsrange',cars_lr)
print(cars_lr.head()) 
print(cars_lr.shape)

# %%
from sklearn.model_selection import train_test_split
np.random.seed(0) 
df_train, df_test = train_test_split(cars_lr, train_size = 0.7, test_size = 0.3, random_state = 100)

# %%
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler() 
num_vars = ['wheelbase', 'curbweight', 'enginesize', 'boreratio', 'horsepower' ,'fueleconomy','carlength','carwidth','price'] 
df_train[num_vars] = scaler.fit_transform(df_train[num_vars])

# %%
df_train.head()

# %%
df_train.describe()


# %%
plt.figure(figsize = (30, 25)) 
sns.heatmap(df_train.corr(), annot = True, cmap="YlGnBu") 
plt.show()

# %%
y_train = df_train.pop('price') 
X_train = df_train
#%%
#RFE 
from sklearn.feature_selection import RFE 
from sklearn.linear_model import LinearRegression 
import statsmodels.api as sm 
from statsmodels.stats.outliers_influence import variance_inflation_factor
model1 = LinearRegression() 
model1.fit(X_train,y_train) 
rfe = RFE(model1, 10) 
rfe = rfe.fit(X_train, y_train)
# %%
list(zip(X_train.columns,rfe.support_,rfe.ranking_))
#%%
X_train_rfe = X_train[X_train.columns[rfe.support_]] 
X_train_rfe.head()

# %%
model2 = LinearRegression() 
model2.fit(X_train_rfe, y_train)

# %%
from sklearn.metrics import r2_score 
y_pred_model1 = model1.predict(X_train) 
print ("r2 model1: ", r2_score(y_pred_model1, y_train))
y_pred_model2 = model2.predict(X_train_rfe) 
print ("r2 model2: ", r2_score(y_pred_model2, y_train))

# %%
