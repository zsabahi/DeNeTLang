#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  3 18:54:42 2020
@author: Zeynab Sabahi
"""

import sys, os, time #,getopt
import pandas as pd

from learning_union_ktss_master.HierarchicalClusteringKTSS_Weighted import learn_unions_ktss
from learning_union_ktss_master.KTestable_Weighted import calculateEIFTC
from configurations import testPercentage, splitTrain, splitTest, trainLenCoeThreshold, testLenCoeThreshold, tracesdir # validationFromTrainPercentage,

from sklearn.model_selection import train_test_split


      
def KTSSFeatureVectorGenerator(tracesdir,k_window,outputdir,sp):
    
    # datasetArr = {} predatasetLengths = {} datasetLengths = {}  dfaArr={}
    filenameArr = [] 
    
    os.makedirs (outputdir,exist_ok=True)

    glossary={} #with [TF, IDF, weight]
   
    tr_ktssArr={}
    tr_TArr={}
    tr_WArr={}
    
    v_ktssArr={}
    v_TArr={}
    v_WArr={}
    
    ts_ktssArr={}
    ts_TArr={}
    ts_WArr={}  
    
    statisticsfile = open(outputdir+"/statistics.csv", 'w')
    # statisticsfile.write("filename,train_No,validation_No,test_No\n")
    statisticsfile.write("index,filename,train_No,test_No\n")

    testArr = {}
    
    for file in os.listdir(tracesdir):
        filename = file.split(".txt")[0] 
        #print(file,"\n")
        
        #traces exactly includes two line train and test
        traces = [line.rstrip('\n') for line in open(tracesdir + '/' + file)] 
        
        # predatasetLengths[filename]=len(predataset)
        # dataset = preprocess(tracesdir,predataset,filename,k_window,True)
        # datasetLengths[filename]=[len(dataset)]
        
        # confirmed, train, validation , test = prepareTraces(traces,k_window)
        confirmed, train , test = prepareTracesFromOneTraces(traces,k_window,sp)
        
        if not confirmed:  
            print(filename + "*** Error: Inputs are not sufficient!")
            confirmed = confirmed

        else:   
           filenameArr.append(filename)
           testArr[filename] = test
           # printNewTraces(train,validation,test,outputdir,filename)
           # statisticsfile.write(filename+","+str(len(train))+","+str(len(validation))+","+str(len(test))+"\n")
           printNewTraces(train,test,outputdir,filename)
           statisticsfile.write(str(filenameArr.index(filename))+","+filename+","+str(len(train))+","+str(len(test))+"\n")
           
           v_ktssArr[filename]=[]
           ts_ktssArr[filename]=[]

           # start_time = time.time()
           glossary,tr_ktssArr,tr_WArr,tr_TArr = buildKtssAndWords(train,k_window,filename,glossary,tr_ktssArr,tr_WArr,tr_TArr)
           # elapsed_time = time.time() - start_time
           # print("time for train ktss learner : "+time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))
                        
    
           # start_time = time.time()
           
           # glossary,v_ktssArr, v_WArr, v_TArr  = buildKtssAndWords(validation,k_window,filename,glossary,v_ktssArr,v_WArr,v_TArr)
           glossary,ts_ktssArr,ts_WArr,ts_TArr = buildKtssAndWords(test,k_window,filename,glossary,ts_ktssArr,ts_WArr,ts_TArr)
     
        
           # elapsed_time = time.time() - start_time
           # print("time for test ktss learner : "+time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))
                        
    
    # start_time = time.time()
   
    buildFeatureVectorFile(filenameArr,outputdir,"train", glossary,tr_WArr,tr_TArr)    
    # elapsed_time = time.time() - start_time
    # print("time for train feature vector generator : "+time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))
                               
    # buildFeatureVectorFile(outputdir,"validation", glossary,v_WArr,v_TArr)    
    # start_time = time.time()
    buildFeatureVectorFile(filenameArr,outputdir,"test", glossary,ts_WArr,ts_TArr)    
    # elapsed_time = time.time() - start_time
    # print("time for test feature vector generator : "+time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))

    # from utils2 import genMixtraffic

    # genMixtraffic(glossary,tracesdir,testArr,filenameArr)
                                
    statisticsfile.close()
    return filenameArr
    
    

#Type = "train" or "Validation" or "test"
def buildFeatureVectorFile(filenameArr,outputdir, trainOrtest, glossary,WArr,TArr):
    
    TermFreqPerLang=[]
    TermFreqPerApp=[]

    perAppTable,perLangTable = buildWordTable(outputdir,trainOrtest, glossary,WArr,TArr)
    TermFreqPerLang.append(perAppTable)
    TermFreqPerApp.append(perLangTable)

    buildCSVOut(filenameArr,outputdir,trainOrtest,perLangTable,TArr)
    # glossary = weightGlossary(glossary,TermFreqPerApp)             
    

def buildKtssAndWords(dataset,k_window,filename,glossary,ktssArr,WArr,TArr):
    
    ktssArr[filename]=[]
    union_number = len(dataset)
    # 
    # print("1")
    labels, dict_ktss, W  = learn_unions_ktss(dataset,union_number,k_window,0,0,"output\plot.pdf")
    # print("2")

    glossary=updateGlossary(glossary,W)
    # print("3")

    WArr[filename]=W

    counter=0
    for ktss in dict_ktss:
        # print("4")

        counter+=1
        TArr[filename+str(counter)]=[]
        TArr[filename+str(counter)].append(ktss.T)
        TArr[filename+str(counter)].append(filename)
        ktssArr[filename].append(ktss)    
#       print(ktss)
        #dfa = learnKtestable_fromEIFTC(ktss.E,ktss.I,ktss.F,ktss.T,ktss.C,k_window)
        # dfaArr[filename] = dfa
                                
        #*** Union
        #ktssArr[filename]=ktss
#       print(ktss.T)
#       print("train: "+str(len(ktss.T))+" "+str(len(set(ktss.T).intersection(set(glossary)))))
#       for t in ktss.T:
#       if t not in glossary.keys():
#       print("*** not found: "+t)

    # print("5")

    return glossary, ktssArr, WArr, TArr


def setTrainAndTest(longertrace,shortertrace, longersize, shortersize,k_window):

    confirmed = True
    
    if longersize < (trainLenCoeThreshold * k_window):
        confirmed = False

    if shortersize < (testLenCoeThreshold * k_window):
        confirmed = False
       
    train = []
    test = []
    train.append(longertrace)
    test.append(shortertrace)
    
    return confirmed, train, test

def SplitTraceOverlap(dataset,cuttingLength,k):
#    print(dataset,cuttingLength)
    newdataset = []
    j = 0
    
    while j < len(dataset) :
        anythingIsAdded=False
        splitted = dataset[j].split(' ')     
       
        cnt = 1
        while cnt <= (len(splitted)/cuttingLength) :
            # newdataset.append(' '.join(splitted[(cnt-1)*cuttingLength:cnt*cuttingLength+k-2]))  #because of k-2 there is some overlap between traces!
            newdataset.append(' '.join(splitted[(cnt-1)*cuttingLength:cnt*cuttingLength]))  #there is not nay overlap between traces!
            anythingIsAdded=True

            cnt += 1
        if len(splitted)%cuttingLength > 0:
             if anythingIsAdded==False: #len(newdataset) == 0 :
                 newdataset.append(dataset[j])
             else:
                if( len(splitted)%cuttingLength < k):
                    newdataset[-1]= newdataset[-1]+" "+(' '.join(splitted[(cnt-1)*cuttingLength:]))
                       
                else:
                    newdataset.append(' '.join(splitted[(cnt-1)*cuttingLength:]))
                    
        j += 1
    return newdataset


def prepareTracesFromOneTraces(traces,k_window,sp):

    confirmed = False

    train = []
    test  = []
    chunkSize = sp*k_window
    
    traces = removeExtraSpaces(traces)
    # newdataset = attachShortTraceToAfter(newdataset,k_window) 
      
    #in moshkel darae baraye chand tracei bayad baraye hame len ro check konam va inke hame append beshan alan engar akhari faghat mishe
    j=0   
    while j < len(traces):
        splitted0 = traces[0].split(' ')
        confirmed = True
        
        if len(splitted0) < ( (trainLenCoeThreshold+testLenCoeThreshold) * k_window):
            confirmed = False
        
        if confirmed:
            if splitTrain:
                train = SplitTraceOverlap(traces, chunkSize, k_window)
                train = removeExtraSpaces(train)
                # checkTwoSuccessiveSpaces(newdataset,k_window)
                # print("len: " +str(len(train))+" len splited: "+str(len(splitted0)))
                if len(train) < 2:
                    # print ("hhhhhhhhhhhhhhhhhhhhhhhhh")
                    confirmed = False
                else:
                    train , test = train_test_split(train,test_size=testPercentage)
                    if len(train) < 2:
                        # print ("hhhhhhhhhhhhhhhhhhhhhhhhh")
                        confirmed = False
        j+=1

    # return confirmed, train, validation , test 
    return confirmed, train , test 


def prepareFlodsFromOneTraces(traces,k_window,sp):

    confirmed = False

    train = []
    test  = []
    chunkSize = sp*k_window
    
    traces = removeExtraSpaces(traces)
    # newdataset = attachShortTraceToAfter(newdataset,k_window) 
      
    #in moshkel darae baraye chand tracei bayad baraye hame len ro check konam va inke hame append beshan alan engar akhari faghat mishe
    j=0   
    while j < len(traces):
        splitted0 = traces[0].split(' ')
        confirmed = True
        
        if len(splitted0) < ( (trainLenCoeThreshold+testLenCoeThreshold) * k_window):
            confirmed = False
        
        if confirmed:
            if splitTrain:
                train = SplitTraceOverlap(traces, chunkSize, k_window)
                train = removeExtraSpaces(train)
                # checkTwoSuccessiveSpaces(newdataset,k_window)
                # print("len: " +str(len(train))+" len splited: "+str(len(splitted0)))
                if len(train) < 2:
                    # print ("hhhhhhhhhhhhhhhhhhhhhhhhh")
                    confirmed = False
                else:
                    train , test = train_test_split(train,test_size=testPercentage)
                    if len(train) < 2:
                        # print ("hhhhhhhhhhhhhhhhhhhhhhhhh")
                        confirmed = False
        j+=1

    # return confirmed, train, validation , test 
    return confirmed, train , test 



# def prepareTracesFromOneTraces(traces,k_window,sp):

#     confirmed = False

#     train = []
#     test  = []
#     chunkSize = sp*k_window
    
#     traces = removeExtraSpaces(traces)
#     # newdataset = attachShortTraceToAfter(newdataset,k_window) 
    
#     train = []
#     test = []
#     #in moshkel darae baraye chand tracei bayad baraye hame len ro check konam va inke hame append beshan alan engar akhari faghat mishe
#     j=0   
#     print(traces)
#     while j < len(traces):
#         print(j)
#         splitted0 = traces[j].split(' ')
#         confirmed = True
        
#         # if len(splitted0) < ( (trainLenCoeThreshold+testLenCoeThreshold) * k_window):
#         #     confirmed = False
        
#         if confirmed:
#             if splitTrain:
#                 temp_train = SplitTraceOverlap(traces, chunkSize, k_window)
#                 temp_train = removeExtraSpaces(train)
#                 # checkTwoSuccessiveSpaces(newdataset,k_window)
#                 temp_train , temp_test = train_test_split(train,test_size=testPercentage)
#                 train.append(temp_train)
#                 test.append(temp_test)
                
#         j+=1

#     if len(train) < 2:
#         # print ("hhhhhhhhhhhhhhhhhhhhhhhhh")
#         confirmed = False
            
#     # return confirmed, train, validation , test 
#     return confirmed, train , test 




def prepareTracesFromTwoTraces(traces,k_window,sp):

    chunkSize = sp*k_window
    
    traces = removeExtraSpaces(traces)
    # newdataset = attachShortTraceToAfter(newdataset,k_window) 
    
    splitted0 = traces[0].split(' ')
    splitted1 = traces[1].split(' ')
    
    if len(splitted0) >= len(splitted1):
        confirmed, train, test = setTrainAndTest(traces[0],traces[1], len(splitted0), len(splitted1),k_window)
        
    else:
        confirmed, train, test = setTrainAndTest(traces[1],traces[0], len(splitted1), len(splitted0),k_window)
    
    # validation = []

    if confirmed:
        
        if splitTrain:
            train = SplitTraceOverlap(train, chunkSize, k_window)
            train = removeExtraSpaces(train)
            # checkTwoSuccessiveSpaces(newdataset,k_window)
            # train , validation = train_test_split(train,test_size=validationFromTrainPercentage)
    
        if splitTest:            
            test = SplitTraceOverlap(test, chunkSize, k_window)
            test = removeExtraSpaces(test)
            # checkTwoSuccessiveSpaces(newdataset,k_window)             

    # return confirmed, train, validation , test 
    return confirmed, train , test 


# def printNewTraces(train,validation,test,outputdir,filename):
def printNewTraces(train,test,outputdir,filename):

    
    PATH = outputdir+"/chunked/"+filename
    os.makedirs (PATH,exist_ok=True)

 
    outfile = open(PATH+"/train.txt", 'w')
    for trace in train:
        outfile.write(str(trace)+"\n")
    outfile.close()

    # outfile = open(PATH+"/validation.txt", 'w')
    # for trace in validation:
    #     outfile.write(str(trace)+"\n")
    # outfile.close()

    outfile = open(PATH+"/test.txt", 'w')
    for trace in test:
        outfile.write(str(trace)+"\n")
    outfile.close()
            
      
def checkTwoSuccessiveSpaces(dataset,k_window):
     
    for cluster in dataset:
        E,I,F,T,C = calculateEIFTC([cluster],k_window)
        if E.__contains__(''):
            print("yes: "+cluster)
#            print(E)
       

def removeExtraSpaces(dataset):
    j = 0
    while j < len(dataset) :
        dataset[j] = dataset[j].rstrip()
        dataset[j] = dataset[j].lstrip()
        dataset[j] = ' '.join(filter(None,dataset[j].split(' ')))        
        j += 1
    return dataset

    
def updateGlossary(glossary,W):

    for wordList in W:
        for word in wordList:
            if word in glossary:
                prev = glossary[word]
                glossary[word]=[prev[0]+wordList[word],prev[1]+1]
            else:
                glossary[word]=[wordList[word],1]
    return glossary


def buildWordTable(outputdir,trainOrtest, glossary,WArr,TArr):
   
    perAppTable={}
    perAppTable["terms"]=list(glossary.keys())
    
    perLangTable={}
    perLangTable["terms"]=list(glossary.keys())

    for filename in WArr:
        perAppTable[filename]=[]
        for term in glossary:
            if term in WArr[filename]:
                perAppTable[filename].append(WArr[filename][term])  
                # print("OOOOOOOOOOOOOO")
            else:
                perAppTable[filename].append(0)
#        print(filename,len(perAppTable[filename]))
        

    for tracename in TArr:
#        print("tracename",str(tracename))
        perLangTable[tracename]=[]
        for term in glossary:
            if term in TArr[tracename][0]:
#                print("terml",str(TArr[tracename][0][term]))
                perLangTable[tracename].append(TArr[tracename][0][term])          
            else:
                perLangTable[tracename].append(0)
    
    
    print("number of apps: "+str(len(perAppTable)-1),"number of terms: "+str(len(perAppTable["terms"])))        
    df1 = pd.DataFrame(data=perAppTable)
    df2 = pd.DataFrame(data=perLangTable)
    
    # writer = pd.ExcelWriter(outputdir+'/glossay_'+type+'.xlsx', engine='xlsxwriter')
    # df1.to_excel(writer,sheet_name='App')  
    # df2.to_excel(writer,sheet_name='Lang')  
    # writer.save()
    
    return  perAppTable, perLangTable


def buildCSVOut(filenameArr,outputdir,trainOrtest,perLangTable,TArr) :
    
    resultfile = open(outputdir+"/"+trainOrtest+".csv", 'w')
    labelfile = open(outputdir+"/"+trainOrtest+"Label.csv", 'w')

    max = 0
    
    flag=0
    # for term in perLangTable["terms"]:
    #     if flag == 0:    
    #         resultfile.write(''.join(term.split(' ')))
    #         flag = 1
    #     else:
    #         resultfile.write(","+''.join(term.split(' ')))
    # resultfile.write(",class\n")
  
    for tracename in TArr:
        # if not (trainOrtest == "test" and TArr[tracename][1] == "remotedesktop"):
        flag=0
#        print("perLangTable[tracename]",str(perLangTable[tracename]))
        for cell in perLangTable[tracename]:
            if flag == 0:
                resultfile.write(str(cell))
                flag=1
            else:
                resultfile.write(","+str(cell))
            if cell > max:
                max = cell
        # resultfile.write(","+TArr[tracename][1]+"\n")
        resultfile.write("\n")
        if TArr[tracename][1] in filenameArr:
            labelfile.write(str(filenameArr.index(TArr[tracename][1]))+"\n")
        else:
            print("Error in buildCSVOut function, type: "+trainOrtest+"=>  "+TArr[tracename][1]+" is not in filenameArr")

           
    # print("max cell of "+type+": "+str(max))

    resultfile.close()
    labelfile.close()

def main(argv):
    
    indir = tracesdir + "/Application/S500_F5_D5_Stats1_Cl1/100/Train"
    csvdir = tracesdir + "/Application/S500_F5_D5_Stats1_Cl1/100/DL"
    k_window = 3

    KTSSFeatureVectorGenerator(indir,k_window,csvdir)

if __name__ == "__main__":
    main(sys.argv[1:])


