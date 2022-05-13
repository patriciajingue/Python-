# -*- coding: utf-8 -*-
"""
Created on Mon Apr 25 22:34:10 2022

@author: 18329
"""
import pandas as pd

from sklearn.linear_model import LogisticRegression

import numpy as np
import streamlit as st

df= pd.read_csv("C:\\Users\\18329\\Downloads\\Pre_term.csv")
x= df.iloc[:,:-1]
print(x.shape)
y = df.iloc[:,-1:]
print(y.shape)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x, y, test_size= 0.1, random_state=10)
model = LogisticRegression()

from imblearn.under_sampling import RandomUnderSampler
rus= RandomUnderSampler(random_state=1)
x_train_resampled,y_train_resampled = rus.fit_resample(x_train,y_train)


clf= LogisticRegression(max_iter=1200000).fit(x_train_resampled,np.ravel(y_train_resampled, order='c'))

y_pred=clf.predict_proba(x_test)[:,1]
print(y_pred)
dataset= st.container()
header= st.container()
with dataset:
    st.title('Pre-Term Predictions')
    st.header('20 years of responses from the National Health Interview Survey')
    st.text('Investigated how maternal characteristics during pregnancy affected birth outcomes')
def premie_pred(input_data):
    input_np= np.array(input_data)
    input_data_reshape=input_np.reshape(1,-1)
    
    prediction= clf.predict_proba(input_data_reshape)
    print(prediction)
    
    
    
def main():

    
    #input data from user
    CHEBMOM= st.text_input('Number of children  mother has')
    PGMECLAMP= st.text_input('Does mother have Eclampsia')
    PGMHYPER= st.text_input('Does mother have Hpertension')
    PGMEMBOL= st.text_input('Does mother have Embolism')
    PGMUTI= st.text_input('Does mother have Urinary tract infection')
    PGMSUGUR= st.text_input('Does mother have Sugar in urine')
    PGMSUGBL = st.text_input("Does mother have high sugar in blood")
    PGMDIAB =  st.text_input('Does mother have Diates')
    PGMABNCORD= st.text_input('Does mother have abnormal position of cord')
    PGMABNPLAC= st.text_input('Does mother have abnormal position of placenta')
    PGMAVAGBLED= st.text_input('Does mother have vaginal bleeding')
    PGMTRANQFREQ= st.text_input('does mother take tranquilizers')
    PGMDRECBED= st.text_input('Did doctore recommend bed rest')
    PGMWTCHANUP= st.text_input('Pounds mother gained  pregnancy')
    BORNCONGEN= st.text_input('Any congenital problems for child')
    MOMAGEATBORN= st.text_input('Age of Mother')
   # code for prediction
    diagnosis= ''

   #creating a button for pres
    if st.button('premie Prediction') :
        diagnosis= premie_pred([CHEBMOM,PGMECLAMP,PGMHYPER,PGMEMBOL,PGMUTI,
        PGMSUGUR,PGMSUGBL,PGMDIAB,PGMABNCORD,PGMABNPLAC,PGMAVAGBLED,PGMTRANQFREQ,
        PGMDRECBED,PGMWTCHANUP,BORNCONGEN,MOMAGEATBORN])
       
        
    st.success('The probability of you having a preemie is {}'.format(diagnosis))
    
if __name__ =='__main__':
    main()
       
    

