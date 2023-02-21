#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 20 14:40:36 2020

@author: zahra
"""

from configurations import statsname, tracesdir, outputdir, preproccesedir, DSName, csvAddHeader, fvdir, figdir, numberoffold, removefiles, cuttingLengthCoe
from tracegeneratorLib import networkUnitExtraction, networkUnitClustering, networkUnitClustering_Assignment, traceConvertor, kfold_split, removemodels
from ktssLib import ktssTrain, ktssTest
#from multiprocessing.dummy import Pool as ThreadPool
import pandas as pd
import time, os

def neTLangKFold(classifierTask, frameworknetconfigs, k_windows, excelfile, date):

    accuracies = {}
      
    with pd.ExcelWriter(excelfile) as writer, open(outputdir + "/" + date+"/nosplit-Test-Results-"+date+".txt", 'a') as resultfile:  
        
        resultfile.write("Setting: nosplit \n")
        
        resultfile.write("classifierTask: "+classifierTask+"\n") 
    
        for sth, fTh, fD, stats, cl, sp in frameworknetconfigs: 
            
            for k_window in k_windows:
                
                colnames = statsname[stats].copy()
                colnames.extend(csvAddHeader)            
                
                networksettings = "F%s_D%s_Stats%s_Cl%s"%(fTh, fD, stats, cl)
                
                k_splitsettings = "%sk_K%s"%(cuttingLengthCoe, k_window)
                
                with open(outputdir + "/" + date+"/nosplit-Time-Report-"+date+".txt", "a") as rfile:
                                    
                    rfile.write("-------------------------------------------------------------------------------------------------\n")
                    rfile.write("Start: Running on %s with flowthreshold = %s, flowduration = %s, & stats #%s, & cluster #%s, & k = %s: %s \n" %(DSName, fTh, fD, stats, cl, k_window, time.time()))
                    print("Running on %s with flowthreshold = %s, flowduration = %s, & stats #%s, & cluster #%s, & k = %s " %(DSName, fTh, fD, stats, cl, k_window))
                    
                    currentfv = fvdir+"/FV_F%s_D%s.csv"%(fTh, fD)
                    
                    ## Train and Test Extracting Feature Vector
                    rfile.write("DS: Splitting and Feature Vector Extraction: "+str(int(time.time()))+"\n")
                        
                    if not os.path.exists(currentfv): 
                        ds = networkUnitExtraction(sth,fTh, fD)                         
                        ds.to_csv(currentfv) 
#                        ds = ds[colnames]
                        ds = kfold_split(ds, numberoffold, classifierTask, networksettings, date)
                    # load fv    
                    else:
                        if not len(ds) == numberoffold:
                            ds = pd.read_csv(currentfv)#, usecols=colnames)    
                            ds = kfold_split(ds, numberoffold, classifierTask, networksettings, date)
                    
                    rfile.write("DS: Feature Vector Loaded/Extracted: "+str(int(time.time()))+"\n")
                    
                    
                    
                    accuracy = 0
                        
                    for k_fold, (trainDS, testDS) in enumerate(ds):  
                        
                        modeldir = preproccesedir + "/Model/%s/%s/%s/fold%s/joblib"%(classifierTask, k_splitsettings, networksettings, k_fold)
                        featureVectorstandarddir = preproccesedir + "/FVStandardaized/%s/%s/%s/fold%s/joblib"%(classifierTask, k_splitsettings, networksettings, k_fold)
                        currentfigdir = figdir + "/%s/%s/%s/fold%s"%(classifierTask, k_splitsettings, networksettings, k_fold)
                        
                        dirs= [modeldir, currentfigdir, featureVectorstandarddir]
                        
                        for d in dirs:
                            os.makedirs (d,exist_ok=True)
                        
                        ## Train Phase   
                        traintracedir = tracesdir+"/%s/%s/%s/fold%s/Train"%(classifierTask, k_splitsettings, networksettings, k_fold)
                        
                        if (os.path.exists(traintracedir) and not os.listdir(traintracedir)) or not os.path.exists(traintracedir):
                            
                            print("Creating Train DS")
                           
                            rfile.write("Train DS: networkUnitClustering: "+str(int(time.time()))+"\n")                
                            
                            trainDS = networkUnitClustering(trainDS, stats, modeldir, featureVectorstandarddir, currentfigdir, cl == 1)
                   
                            rfile.write("Train DS: traceConvertor: "+str(int(time.time()))+"\n")
                            
                            traceConvertor(trainDS, traintracedir, classifierTask, cl == 1)         
                                                 
                        # Language Learner Train Part
                            
                        rfile.write("Train DS: Language Learner: K = "+str(k_window)+": " +str(int(time.time()))+"\n")
                        ktssArr,filenameArr,predatasetLengths,datasetLengths = ktssTrain(traintracedir, k_window)            
                        
                        ## Test Phase         
                         
                        rfile.write("Test DS: Splitting and Feature Vector Extraction: "+str(int(time.time()))+"\n")
                                        
                        ## Test Extracting Feature Vector
                        if "Cluster" in colnames:
                            colnames.remove("Cluster")    
                            
                        testtracedir = tracesdir+"/%s/%s/%s/fold%s/Test"%(classifierTask, k_splitsettings, networksettings, k_fold)
                        
                        if (os.path.exists(testtracedir) and not os.listdir(testtracedir)) or not os.path.exists(testtracedir):
                            colnames.extend(['Cluster'])     
                            rfile.write("Test DS: networkUnitClustering_Assignment: "+str(int(time.time()))+"\n")
                            
                            testDS = networkUnitClustering_Assignment(testDS, stats, modeldir, featureVectorstandarddir, cl == 1)
 
                            rfile.write("Test DS: traceConvertor: "+str(int(time.time()))+"\n") 
                               
                            traceConvertor(testDS, testtracedir, classifierTask, cl == 1)
                                                       
                        # Language Learner Test Part                    
                        rfile.write("Test DS: Language Learner: K = "+str(k_window)+": " +str(int(time.time()))+"\n")
    
                        efficeincy, acc = ktssTest(ktssArr,filenameArr, predatasetLengths, datasetLengths, testtracedir, k_window, outputdir + "/" + date+"/"+ networksettings, date)
                        
                        accuracy += acc
                                           
                        efficeincy.to_excel(writer, sheet_name="%s_fold%s_k%s"%(networksettings, k_fold, k_window), index=False)
                                
                    aveacc = accuracy/numberoffold
                    
                    resultfile.write("Parameters: %sk %s_k%s : Ave Accuracy = %s \n"%(cuttingLengthCoe, networksettings, k_window, aveacc)) 
                    
                    accuracies["%sk %s_k%s"%(cuttingLengthCoe, networksettings, k_window)] = aveacc
                    
                    rfile.write("End:" +str(int(time.time()))+"\n")
                                
            if removefiles:
                removemodels(preproccesedir + "/FVStandardaized/%s/%s/%s/"%(classifierTask, k_splitsettings, networksettings), 
                                 preproccesedir + "/Model/%s/%s/%s/"%(classifierTask, k_splitsettings, networksettings))

                
        best =max(accuracies, key=lambda params: accuracies[params])
        
        print("Max Acc = %s for %s \n"%(accuracies[best], best))
        resultfile.write("Max Acc = %s for %s \n"%(accuracies[best], best))
        resultfile.write("--------------------------------------------------------------------------------------------------------\n")  