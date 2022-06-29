#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 18:41:56 2020

@author: Zahra Alimadadi
"""
from configurations import featureset, flowthreshold, flowduration, clusters, k_windows, classificationtask, processing, Processing, outputdir, dsdir, csvdir, settingsname,justRunTraces, sessionthreshold,sp 
# from neTLang_Kfold import neTLangKFold
#from neTLang_VSet import neTLangValidationSet
from neTLang_FastTest import neTLangFastTest
from tracegeneratorLib import dataSetPreparation
import datetime
import os

def main():
        
    date = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    
    date = settingsname+"_"+date
    
    outdir = outputdir + "/" + date
    
    os.makedirs (outdir,exist_ok=True)
    
    frameworknetconfigs = []
    
    for sth in sessionthreshold:   
        for stats in featureset:
            for fth in flowthreshold:
                for fd in flowduration:     
                    for s in sp:
                        for cl in clusters:
                            frameworknetconfigs.append((sth, fth, fd, stats,cl,s))
        
    ## convert traffic to csv and merge the similars 
    if (not justRunTraces) and (not os.listdir(csvdir)):
        dataSetPreparation(dsdir)
        print("Csv is extracted --> check it out in /Preprocess/Csv")     
              
    for classifierTask in classificationtask:
#        if not classifierTask == "Class":
#            continue
        print("classifierTask: "+classifierTask+"\n")   
    
        excelfile = outdir+"//nosplit-"+classifierTask+"-Results-"+date+".xlsx" 
        
        if processing == Processing.kfold.name:
            neTLangKFold(classifierTask, frameworknetconfigs, k_windows, excelfile, date) #??? I should modify this func because of adding sth to frameonfig 
            
        elif processing == Processing.validation.name:
            print("This Part is not Ready :D")
#            neTLangValidationSet(classifierTask, frameworknetconfigs, k_windows, excelfile, date)
            
        elif processing == Processing.fasttest.name:
            neTLangFastTest(classifierTask, frameworknetconfigs, k_windows, excelfile, date)
    
if __name__ == "__main__":
    main()
