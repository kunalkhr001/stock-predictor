import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import csv
from datetime import date
import tkinter
from tkinter import Listbox, Label, messagebox

# Importing the dataset
dataset = pd.read_csv('BSE_30.csv')
input_file = open("BSE_30.csv","r+")
reader_file = csv.reader(input_file)
#rows = len(list(reader_file)) - 1

df=dataset
df[['M','D','Y']] = df['Date'].str.split('/',expand=True)
del df['Date']
df = df.convert_objects(convert_numeric=True)
df = df.dropna() 
df = df.reset_index(drop=True)

rows = len(df)

from sklearn.preprocessing import LabelEncoder
#
labelencoder_X = LabelEncoder()
df['Symbol']= labelencoder_X.fit_transform(df['Symbol'])
X = df.loc[1:, ['Symbol','D','M','Y']].values
y = np.zeros((rows,2), dtype='float64')

for i in range (1,rows):
    roc=(df['Close'][i]-df['Close'][i-1])/df['Close'][i-1]
    y[i][0]=roc*100
    n=1
    if df['M'][i]==df['M'][i-1]:
        n=df['D'][i]-df['D'][i-1]
    else :
        n=df['D'][i]
    day_gain=df['High'][i]-df['Close'][i-1]
    day_loss=df['Low'][i]-df['Close'][i-1]
    if day_loss==0:
        day_loss=0.0000001
    if day_gain==0:
        day_gain=0.0000001    
    avg_gain=day_gain/n
    avg_loss=day_loss/n
    first_rs=abs(avg_gain)/abs(avg_loss)
    rsi=100-(100/(1+first_rs))
    y[i][1]=rsi
    
    upmove=df['High'][i]-df['High'][i-1]
    downmove=df['Low'][i]-df['Low'][i-1]
    plusdm=0
    minusdm=0
    if upmove > downmove and upmove >0:
        plusdm=upmove
    if upmove < downmove and downmove >0:
        minusdm=downmove
        
    #plusdi=100*em
    #minusdi=100*em
    #adx = 100*em
    
y = y[1:,]

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 10, random_state = 0)

# Feature Scaling
"""
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)
"""

# Fitting the Regression Model to the dataset

from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 25, random_state = 0)
regressor.fit(X,y)

# Listing stocks

def onselect(evt):
    w = evt.widget
    global index
    index = int(w.curselection()[0])
    value = w.get(index)
    print ('You selected item %d: "%s"' % (index, value))
    

uniq_len= (len(dataset.Symbol.unique()))
top = tkinter.Tk()
top.geometry('500x500')

l=Label(text="Select your Stock")
l.pack()

Lb1 = Listbox(top,height=30)
for i in range(0,20):
    Lb1.insert(i+1, (dataset.Symbol.unique())[i])

Lb1.insert(i+1, (dataset.Symbol.unique())[uniq_len-1])

for i in range(20,uniq_len-1):
    Lb1.insert(i+1, (dataset.Symbol.unique())[i])

Lb1.pack()

index=0
Lb1.bind('<<ListboxSelect>>', onselect)    

cs = Lb1.curselection()
top.mainloop()



# Predicting a new result
xpred = np.zeros((1,4), dtype='int64')
print(index)
xpred[0][0]=index
xpred[0][1]=date.today().day
xpred[0][2]=date.today().month
xpred[0][3]=date.today().year
ypred= regressor.predict(xpred)
y_pred = regressor.predict(X_test)

tm=tkinter.Tk()
if ypred[0][0]<0 and ypred[0][1]<30:
    messagebox.showinfo(title="Result",message="Buy")
elif ypred[0][0]>0 and ypred[0][1]>70:
    messagebox.showinfo(title="Result",message="Sell")
else:
    messagebox.showinfo(title="Result",message="Wait")    
tm.mainloop()

# Visualising the Regression results 
yc_t=y_test[:,1:]
yc_p=y_pred[:,1:]
x_base = np.zeros((10,1), dtype='int64')
index=0
for i in range(10, 110, 10):
    x_base[index][0]=i
    index +=1
plt.scatter(x_base, yc_t, color= 'red')
plt.scatter(x_base, yc_p , color= 'blue')
plt.xlabel('Range')
plt.ylabel('Values(RSI)')
plt.show()
