#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 00:32:32 2020

@author: zahra
"""
from configurations import numberoffold,statsname, tracesdir, outputdir, preproccesedir, DSName, csvAddHeader, fvdir, figdir, removefiles, trainPercentage, partitionbasedon,justRunTraces,sp #cuttingLengthCoe #,dldir, traindatathreshold, testdatathreshold
from tracegeneratorLib import networkUnitExtraction, networkUnitClustering, networkUnitClustering_Assignment, traceConvertor, removemodels, half_split#, fast_split, independent_split
#*** Union
# from ktssLib_Union import ktssTrain, ktssTest
#**** to disabel KFOLD uncomment the above line
# from ktssLib_Union_DL import KTSSFeatureVectorGenerator
from ktssLib_Union_DL_KFold import KTSSFeatureVectorGenerator,KTSSFVGFold

from MLP import MLPNetwork
from statistics import mean

import pandas as pd
import time, os


def neTLangFastTestKFold(classifierTask, frameworknetconfigs, k_windows, excelfile, date):
              
    with pd.ExcelWriter(excelfile) as writer, open(outputdir + "/" + date+"/nosplit-Test-Results-"+date+".txt", 'a') as resultfile:  
        
        # resultfile.write("Setting: nosplit \n")
        
        # resultfile.write("classifierTask: "+classifierTask+"\n") 
        
        all_acc = []
    
        # for sth, fTh, fD, stats, cl in frameworknetconfigs:  # I add sth to folder name
        for sth, fTh, fD, stats, cl, sp in frameworknetconfigs:  # I add sth to folder name
            
            for k_window in k_windows:
                
                colnames = statsname[stats].copy()
                colnames.extend(csvAddHeader)            
                
                # networksettings = "S%s_F%s_D%s_Stats%s_Cl%s"%(sth,fTh, fD, stats, cl)
                # networksettings = "S%s_F%s_D%s_Sp%s_k%s"%(sth,fTh, fD, sp,k_window)
                networksettings = "S%s_F%s_D%s"%(sth,fTh, fD)
                
                #k_splitsettings = "%sk_K%s"%(cuttingLengthCoe, k_window)
                
                with open(outputdir + "/" + date+"/nosplit-Time-Report-"+date+".txt", "a") as rfile:
                                    
                    # rfile.write("-------------------------------------------------------------------------------------------------\n")
                    # rfile.write("Start: Running on %s with sessionThreshold = %s & flowthreshold = %s, flowduration = %s, & stats #%s, & cluster #%s, & k = %s: %s \n" %(DSName, sth, fTh, fD, stats, cl, k_window, time.time()))
                    # print("Running on %s with sessionThreshold = %s & flowthreshold = %s, flowduration = %s, & stats #%s, & cluster #%s, & k = %s " %(DSName, sth, fTh, fD, stats, cl, k_window))
                    print("Running on %s with sessionThreshold = %s & flowthreshold = %s, flowduration = %s, & sp #%s, & k = %s " %(DSName, sth, fTh, fD, sp, k_window))
        
                    percentfolder = int(trainPercentage*100)

                    traintracedir = tracesdir+"/%s/%s/%s/Train"%(classifierTask, networksettings, percentfolder)

                    if (not justRunTraces):
                        currentfv = fvdir+"/FV_S%s_F%s_D%s.csv"%(sth, fTh, fD)
                        
                        ## Train and Test Extracting Feature Vector
                        # rfile.write("DS: Splitting and Feature Vector Extraction: "+str(int(time.time()))+"\n")
                            
                        if not os.path.exists(currentfv): 
                           
                            start_time = time.time()
                            ds = networkUnitExtraction(sth, fTh, fD)   
                            elapsed_time = time.time() - start_time
                            print("time for net unit extracion: "+time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))
                            
                            ds.to_csv(currentfv) 
                            #ds = ds[colnames]
                        # load fv    
                        else:  
                            ds = pd.read_csv(currentfv)                
                        
                        # rfile.write("DS: Feature Vector Loaded/Extracted: "+str(int(time.time()))+"\n")
                        
                        ## the floowing naming is for independent split 
                        
                        fvtrain = fvdir+"/%s/Train/FV_S%s_F%s_D%s.csv"%(percentfolder,sth, fTh, fD)
                        
                        fvtest = fvdir+"/%s/Test/FV_S%s_F%s_D%s.csv"%(percentfolder, sth,fTh, fD)   
                        
                        ## for saving clusters
                        fvtraincluster = fvdir+"/%s/Train/FV_S%s_F%s_D%s_stast%s.csv"%(percentfolder, sth, fTh, fD, stats)
    #                    fvvalcluster = fvdir+"/Validation/FV_%sk_K%s_F%s_D%s_stast%s.csv"%(cuttingLengthCoe, k_window, fTh, fD, stats)
                        fvtestcluster = fvdir+"/%s/Test/FV_S%s_F%s_D%s_stast%s.csv"%(percentfolder, sth, fTh, fD, stats)
                        
                        dirs= []
                        for dstype in ["Train", "Test"]:
                            dirs.append(fvdir+"/%s/"%percentfolder+dstype)
                        
                        for d in dirs:
                            os.makedirs (d,exist_ok=True)
                        
                        ## partitioning to test and train. The first time it save the partitions then in next try with the same setting it only loads it
                        if not os.path.exists(fvtrain):  
                            #trainDS , testDS = fast_split(ds, trainPercentage, partitionbasedon, networksettings, date)
                            #trainDS , testDS = independent_split(ds, k_window, cuttingLengthCoe, trainPercentage) # for 4k seperation
                            trainDS , testDS = half_split(ds, trainPercentage, partitionbasedon, networksettings, date)
                            trainDS.to_csv(fvtrain)
                            testDS.to_csv(fvtest)
                            trainDS = trainDS[colnames]
                            testDS = testDS[colnames]
                        else:
                            trainDS = pd.read_csv(fvtrain, usecols=colnames)
                            testDS = pd.read_csv(fvtest, usecols=colnames)
         
                        # availableLabels = list(testDS[classifierTask].unique())
                        availableLabels = list(trainDS[classifierTask].unique())
                        
    #                    print ("Test Label: ", availableLabels)
                        
                        trainDS = trainDS[trainDS[classifierTask].isin(availableLabels)]
                        trainDS = trainDS.reset_index(drop=True) 
                        
                        availableLabels = list(trainDS[classifierTask].unique())
                        
    #                    print ("Train Label: ", availableLabels)
                        
                        ## To ommit labels having limitted number of samples                  
                        
    #                    sample_length = cuttingLengthCoe*k_window                    
    #                    
    #                    removedlables = set()
    #                    
    #                    traintracesbyfile = trainDS.groupby ( [classifierTask] )
    #                    for label, data in traintracesbyfile:
    #                        numberofsamples = len(data) // sample_length
    #                        if numberofsamples < traindatathreshold:
    #                            removedlables.add(label)
    #                            
    #                    testtracesbyfile = testDS.groupby ( [classifierTask] )
    #                    for label, data in testtracesbyfile:
    #                        numberofsamples = len(data) // sample_length
    #                        if numberofsamples < testdatathreshold:
    #                            removedlables.add(label)
    #
    #                    availableLabels = set(testDS[classifierTask].unique())
    #                    availableLabels = availableLabels - removedlables                    
    #                    
    #                    print("Labels having limitted number of samples: ", removedlables)
    #                    
    #                    print("")
    #                    
    #                    print("Remaining Labels: ", availableLabels)
    #                    
    #                    removedlables = list(removedlables)
    #                    
    #                    trainDS = trainDS[~trainDS[classifierTask].isin(removedlables)]
    #                    trainDS = trainDS.reset_index(drop=True)
    #                    
    #                    testDS = testDS[~testDS[classifierTask].isin(removedlables)]
    #                    testDS = testDS.reset_index(drop=True)
                        
                        
                        modeldir = preproccesedir + "/Model/%s/%s/%s/joblib"%(classifierTask, networksettings,  percentfolder)
                        featureVectorstandarddir = preproccesedir + "/FVStandardaized/%s/%s/%s/joblib"%(classifierTask, networksettings,  percentfolder)
                        currentfigdir = figdir + "/%s/%s/%s"%(classifierTask, networksettings,  percentfolder)
                        
                        dirs= [modeldir, currentfigdir, featureVectorstandarddir]
                        
                        for d in dirs:
                            os.makedirs (d,exist_ok=True)
                        
                        ## Train Phase   
                        
                        if (os.path.exists(traintracedir) and not os.listdir(traintracedir)) or not os.path.exists(traintracedir):
                            
                            # print("Creating Train DS")
                           
                            # rfile.write("Train DS: networkUnitClustering: "+str(int(time.time()))+"\n")                
                            
                            if not os.path.exists(fvtraincluster):
                                start_time = time.time()
                           
                                trainDS = networkUnitClustering(trainDS, stats, modeldir, featureVectorstandarddir, currentfigdir, cl == 1)
                                elapsed_time = time.time() - start_time
                                print("time for net unit clustering: "+time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))
                            
                                trainDS.to_csv(fvtraincluster)
                                
                            else:
    #                            print("loaded")
                                colnames = statsname[stats].copy()
                                colnames.extend(csvAddHeader)
                                colnames.extend(['Cluster'])#,'Task'])#, 'Trace']) #Task required if extractUserAction=true in config,  trace required for 4k version
                                trainDS = pd.read_csv(fvtraincluster, usecols=colnames)
                                
                        # rfile.write("Train DS: traceConvertor: "+str(int(time.time()))+"\n")
                            start_time = time.time()

                            traceConvertor(trainDS, traintracedir, classifierTask, cl == 1)         
                            elapsed_time = time.time() - start_time
                            print("time for train  trace converter: "+time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))
                                           
                    # Language Learner Train Part
                    # rfile.write("Train DS: Language Learner: K = "+str(k_window)+": " +str(int(time.time()))+"\n")
                    # ktssArr,filenameArr,predatasetLengths,datasetLengths,glossary,WArr = ktssTrain(traintracedir, k_window)            
           
                    testtracedir = tracesdir+"/%s/%s/%s/Test"%(classifierTask, networksettings,  percentfolder)

                    if (not justRunTraces):
                        ## Test Phase                                 
                        # rfile.write("Test DS: Splitting and Feature Vector Extraction: "+str(int(time.time()))+"\n")
                                        
                        ## Test Extracting Feature Vector
                        if "Cluster" in colnames:
                            colnames.remove("Cluster")    
                            
                        
                        if (os.path.exists(testtracedir) and not os.listdir(testtracedir)) or not os.path.exists(testtracedir):
                            colnames.extend(['Cluster'])     
                            # rfile.write("Test DS: networkUnitClustering_Assignment: "+str(int(time.time()))+"\n")
                            
                            if not os.path.exists(fvtestcluster):
                                
                                start_time = time.time()

                                testDS = networkUnitClustering_Assignment(testDS, stats, modeldir, featureVectorstandarddir, cl == 1)
                                elapsed_time = time.time() - start_time
                                print("time for test unit assignment  : "+time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))
                                testDS.to_csv(fvtestcluster)
                                
                            else:
    #                            print("loaded")
                                colnames = statsname[stats].copy()
                                colnames.extend(csvAddHeader)
                                colnames.extend(['Cluster'])#,'Task'])#, 'Trace']) #Task required if extractUserAction=true in config,  trace required for 4k version
                                testDS = pd.read_csv(fvtestcluster, usecols=colnames)
                           
                                   
                        # rfile.write("Test DS: traceConvertor: "+str(int(time.time()))+"\n") 
                        start_time = time.time()
   
                        traceConvertor(testDS, testtracedir, classifierTask, cl == 1)
                        elapsed_time = time.time() - start_time
                        print("time for test  trace converter: "+time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))                         
                        
                    # Language Learner Test Part                    
                    # rfile.write("Test DS: Language Learner: K = "+str(k_window)+": " +str(int(time.time()))+"\n")
                    # efficeincy, acc = ktssTest(ktssArr,filenameArr, predatasetLengths, datasetLengths, testtracedir, k_window, outputdir + "/" + date, networksettings, date,glossary,WArr)
                    
                    # rfile.write("Deep Learning phase: K = "+str(k_window)+": " +str(int(time.time()))+"\n")


                    #NEW NEW 
                    
                    indir = tracesdir + "/%s/%s/%s/"%(classifierTask, networksettings, percentfolder)+"Train"  #"/Application/S500_F5_D5_Stats1_Cl1/100/Train"
                    filenameArr,trainHash,testHash = KTSSFeatureVectorGenerator(indir,k_window,sp)
                    
                    i = 0

                    while i < numberoffold :
                        print("starting fold %s"%(i))
                        csvdir= tracesdir + "/%s/%s/%s/"%(classifierTask, networksettings, percentfolder)+"DL_sp%s_k%s_fold%s"%(sp,k_window,i)  #"/Application/S500_F5_D5_Stats1_Cl1/100/DL"

                       
                        KTSSFVGFold(filenameArr,trainHash,testHash,i,csvdir,k_window)
                        
                        
                        print("Deep Learning phase: K = "+str(k_window)+": " +str(int(time.time()))+"\n")
    
                        efficeincy, acc,nClasses = MLPNetwork(csvdir,filenameArr)
                        efficeincy.to_excel(writer, sheet_name="R_%s_sp%s_k%s_fold%s"%(networksettings, sp,k_window,i), index=False)
    
                        all_acc.append(["R_%s_sp%s_k%s_fold%s"%(networksettings, sp,k_window,i),acc,nClasses])
                        print("Test Acc = %s for %s_sp%s_k%s_fold%s for %s Applications\n"%(acc, networksettings, sp, k_window,nClasses,i))
                        i+=1
                    
                    print("The Avg acc of kfold is: "+mean(all_acc))
                    # resultfile.write("Test Acc = %s for %s_K%s \n"%(acc, networksettings,  k_window))
                    # resultfile.write("--------------------------------------------------------------------------------------------------------\n")  


            if removefiles:
                removemodels(preproccesedir + "/FVStandardaized/%s/%s/%s/"%(classifierTask, networksettings,  percentfolder), 
                     preproccesedir + "/Model/%s/%s/%s/"%(classifierTask, networksettings,  percentfolder))
                
        df = pd.DataFrame(data=all_acc,columns=['SheetName','Acc','App No.'])
        df.to_excel(writer, sheet_name='all_accs', index=False)
        
    return all_acc
    