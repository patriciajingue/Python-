# -*- coding: utf-8 -*-
"""
Created on Wed Apr 27 10:00:44 2022 

@author: 18329
"""

import numpy as np
import pickle
import streamlit as st
loaded_model= pickle.load(open("C:\\Users\\18329\\Downloads\\train.sav",'rb'))
# creating function
def premie_pred(input_data):
    input_np= np.array(input_data)
    input_data_reshape=input_np.reshape(1,-1)
    
    prediction= loaded_model.predict_proba(input_data_reshape)
    print(prediction)
    
    
    
def main():
    st.title('Premies Prediction web app')
    
    
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
       
    