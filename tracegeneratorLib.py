#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  3 18:50:41 2020

@author: Zahra Alimadadi
"""
from flowLib import pcap2csv, extractFeatures_stats, splitTraffic, psml2csv, csvPacketFiltering
from mlLib import elbow, clustering_Kmeans, assignCluster_Kmeans
import pandas as pd
import numpy as np
from configurations import statsname, numberofclusterdir, csvdir, filterdir,  outputdir, rawDataType, preproccesedir, classificationtask, threshold,  extractUserAction, dsSplitdir# filenameToClassLabel,dsSplitdir,
from sklearn.model_selection import KFold
from shutil import move, rmtree#, copyfile
import os, random
import simplejson as json
import math


pd.options.mode.chained_assignment = None 



##
# Input:
# ds: dataset to be partitionned
# trainPercentage: percentage for train data
# fvtrain/test: where to store the train/test dataset
# classifierTask: the classitication task
# settings: the settings for which we are doing the partitionng, it is only for giving us report of the sampls which are not enough for the current task
# outputdir: where the report is save
# data: the date on which the script is started   
# Output:
# train and test data 
##    
def fast_split(ds, trainPercentage, partitionbasedon, settings, date):
    
    with open(outputdir + "/" + date+"/nosplit-Kfold-Report-"+date+".txt", "a") as rfile:
        trainDS = pd.DataFrame(columns=ds.columns)
        testDS =  pd.DataFrame(columns=ds.columns)
        tracesbyclassifierTask = ds.groupby ( [partitionbasedon] )    # classifierTask
        # rfile.write("Settings: "+settings+"\n")
        for task, flows in tracesbyclassifierTask:            
            sessions = list(flows['Session'].unique())
            numberofsamples = len(sessions)
            if numberofsamples < threshold: 
                # rfile.write("App/Task: "+ str(task)+": Number of sessions = "+str(len(sessions))+"\n")
                continue
            else:                
                rang = list(range(0,numberofsamples))
                traincount = int(trainPercentage * numberofsamples)   
                trainindex = random.sample(list(rang), traincount)                
                testindex = list(set(rang)-set(trainindex))
                
                trainDS = trainDS.append(ds[ds['Session'].isin([sessions[i] for i in trainindex])], ignore_index=True).sort_values(by=['TimeStampStart'])
                testDS = testDS.append(ds[ds['Session'].isin([sessions[i] for i in testindex])], ignore_index=True).sort_values(by=['TimeStampStart'])
        # rfile.write("-------------------------------------------------------------------------------------------------------------------------\n")

    return trainDS , testDS

##
# Input:
# ds: dataset to be partitionned
# trainPercentage: percentage for train data
# fvtrain/test: where to store the train/test dataset
# classifierTask: the classitication task
# settings: the settings for which we are doing the partitionng, it is only for giving us report of the sampls which are not enough for the current task
# outputdir: where the report is save
# data: the date on which the script is started   
# Output:
# train and test data 
##    
def half_split(ds, trainPercentage, partitionbasedon, settings, date):
    with open(outputdir + "/" + date+"/nosplit-Kfold-Report-"+date+".txt", "a") as rfile:
        trainDS = pd.DataFrame(columns=ds.columns)
        testDS =  pd.DataFrame(columns=ds.columns)
        tracesbyclassifierTask = ds.groupby ( [partitionbasedon] )    # classifierTask
        # rfile.write("Settings: "+settings+"\n")
        for task, flows in tracesbyclassifierTask:   
            temp = flows.groupby(['Session']).size().reset_index(name='counts')
            
            sessions = list(flows['Session'].unique())            
#            numberofsamples = len(sessions)
            numberofsamples = len(temp)
            if numberofsamples < threshold: 
                # rfile.write("App/Task: "+ str(task)+": Number of sessions = "+str(numberofsamples)+"\n")
                continue
            else:                
#                rang = list(range(0,numberofsamples))
                traincount = int(trainPercentage * numberofsamples)   
#                trainindex = random.sample(list(rang), traincount)                
#                testindex = list(set(rang)-set(trainindex))
                testcount = numberofsamples - traincount
                if traincount < testcount:
                    traincount, testcount = testcount, traincount #swapping
                # print("train count: %s, test count: %s"%(traincount,testcount))
                train = temp.nlargest(traincount, 'counts')
                train = train["Session"].to_numpy()
                test = list (set (sessions) - set(train))
                
                # print("task ",task, "train ",  len(ds[ds['Session'].isin(train)]),"test ", len(ds[ds['Session'].isin(test)]))    

                trainDS = trainDS.append(ds[ds['Session'].isin(train)], ignore_index=True).sort_values(by=['TimeStampStart'])
                testDS = testDS.append(ds[ds['Session'].isin(test)], ignore_index=True).sort_values(by=['TimeStampStart'])
        # rfile.write("-------------------------------------------------------------------------------------------------------------------------\n")
    return trainDS , testDS
##
# Input:
# ds: dataset to be partitionned
# Output:
# train and test data 
##  
def independent_split(ds, k_window, cuttingLengthCoe, trainPercentage):
    print("if you get error running with this split, you should add trace in header configuration, I have commented it")
    sample_length = cuttingLengthCoe*k_window
    colname = list(ds.columns)
    colname.append('Trace')
    trainDS = pd.DataFrame(columns=colname)
    testDS =  pd.DataFrame(columns=colname)
    tracesbyfile = ds.groupby ( ['Session'] )  
    
    for file, networkalphabets in tracesbyfile: 
        networkalphabets = networkalphabets.sort_values(by=['TimeStampStart'])
        networkalphabets = networkalphabets.reset_index(drop=True)
        
        numberofalphabets = len(networkalphabets)
    
        numberofsamples =  numberofalphabets // sample_length
        
        rang = list(range(0,numberofsamples))
        traincount = int(trainPercentage * numberofsamples)
       # print(traincount)
        if traincount == 0 and numberofsamples > 0:
            traincount = 1
#        traincount = math.ceil(trainPercentage * numberofsamples)   
        trainindex = random.sample(list(rang), traincount)                
        
        trainindex = [x * sample_length for x in trainindex]
        
        samples = [False] * numberofalphabets
        for i in trainindex:
            for j in range(0,sample_length):
                samples[i+j] = True
    
        if numberofalphabets != numberofsamples * sample_length:
            dif = numberofalphabets - numberofsamples * sample_length    
            value = samples[numberofalphabets-dif-1] 
            samples[numberofalphabets-dif:] = [value] * dif
        
#        print(file)
#        print(samples)
        traintracesindex = []
        testtracesindex = []
        
        startind = 0
        swap = False
        currentval = samples[0]
        for ind, val in enumerate(samples):
            if currentval != val:
                swap = True
            else:
                swap = False
            if swap and currentval:
                traintracesindex.extend([list(range(startind,ind))])
            elif swap and not currentval:
                testtracesindex.extend([list(range(startind,ind))])
            if swap:
                startind = ind
            currentval = val
            
        if startind < len(samples) - 1:
            if samples[-1]: #train
                traintracesindex.extend([list(range(startind,len(samples)))])
            else: #test
                testtracesindex.extend([list(range(startind,len(samples)))])
           
        networkalphabets['Trace'] = file
        
        for traceid, trace in enumerate(traintracesindex):
#            print( 'Train  : ' , traceid, trace)
            networkalphabets.loc[trace, 'Trace'] = file + '_Train_%s'%traceid
                
        for traceid, trace in enumerate(testtracesindex):
#            print( 'Test  : ' , traceid, trace)
            networkalphabets.loc[trace, 'Trace'] = file + '_Test_%s'%traceid
            
#        print("---------------------------------------------------------------------------")
        trainDS = trainDS.append(networkalphabets[networkalphabets['Trace'].str.contains('Train')], ignore_index=True).sort_values(by=['TimeStampStart'])
        testDS = testDS.append(networkalphabets[networkalphabets['Trace'].str.contains('Test')], ignore_index=True).sort_values(by=['TimeStampStart'])

    return trainDS , testDS 


##
# Input:
# ds: dataset to be partitionned
# numberoffold: the K in k-fold
# classifierTask: the classitication task
# settings: the settings for which we are doing the partitionng, it is only for giving us report of the sampls which are not enough for the current task
# outputdir: where the report is save
# data: the date on which the script is started   
# Output:
# list of partittions of each fold    
##    
def kfold_split(ds, numberoffold, classifierTask, settings, date):
    
    with open(outputdir + "/" + date+"/nosplit-Kfold-Report-"+date+".txt", "a") as rfile:
        train = [pd.DataFrame(columns=ds.columns) for i in np.arange(numberoffold)] 
        test =  [pd.DataFrame(columns=ds.columns) for i in np.arange(numberoffold)]
        tracesbyclassifierTask = ds.groupby ( [classifierTask] )   
        # rfile.write("Settings: "+settings+"\n")
        for task, flows in tracesbyclassifierTask:
            sessions = list(flows['Session'].unique())
            if len(sessions) < numberoffold:
                # rfile.write("Task: "+ str(task)+": Number of sessions = "+str(len(sessions))+"\n")
                continue
            else:
                kfold = KFold(numberoffold, True, 1)
                for i, (trainindex, testindex) in enumerate(kfold.split(sessions)):
                    train[i] = train[i].append(ds[ds['Session'].isin([sessions[i] for i in trainindex])], ignore_index=True).sort_values(by=['TimeStampStart'])
                    test[i] = test[i].append(ds[ds['Session'].isin([sessions[i] for i in testindex])], ignore_index=True).sort_values(by=['TimeStampStart'])
        # rfile.write("-------------------------------------------------------------------------------------------------------------------------\n")
        partitions = [(train[i], test[i]) for i in np.arange(numberoffold)]
    
    return partitions


def kfold_split_CeoK(ds, numberoffold, classifierTask, settings, date):
    
    with open(outputdir + "/" + date+"/nosplit-Kfold-Report-"+date+".txt", "a") as rfile:
        train = [pd.DataFrame(columns=ds.columns) for i in np.arange(numberoffold)] 
        test =  [pd.DataFrame(columns=ds.columns) for i in np.arange(numberoffold)]
        tracesbyclassifierTask = ds.groupby ( [classifierTask] )   
        # rfile.write("Settings: "+settings+"\n")
        for task, flows in tracesbyclassifierTask:
            sessions = list(flows['Session'].unique())
            if len(sessions) < numberoffold:
                # rfile.write("Task: "+ str(task)+": Number of sessions = "+str(len(sessions))+"\n")
                continue
            else:
                kfold = KFold(numberoffold, True, 1)
                for i, (trainindex, testindex) in enumerate(kfold.split(sessions)):
                    train[i] = train[i].append(ds[ds['Session'].isin([sessions[i] for i in trainindex])], ignore_index=True).sort_values(by=['TimeStampStart'])
                    test[i] = test[i].append(ds[ds['Session'].isin([sessions[i] for i in testindex])], ignore_index=True).sort_values(by=['TimeStampStart'])
        # rfile.write("-------------------------------------------------------------------------------------------------------------------------\n")
        partitions = [(train[i], test[i]) for i in np.arange(numberoffold)]
    
    return partitions

##
# This method converts the traffic into csv format
##    
def Convert2Csv(filedir, dirname, filename):                    
    
    csvdir = preproccesedir+"/PreCsv/"+dirname
    csvfile = csvdir+"/"+filename+".csv"
    os.makedirs (csvdir,exist_ok=True)
    
    if rawDataType == "PSML":
         ## Convert PSML To CSV
         psml2csv(filedir, csvfile)
            
    elif rawDataType == "Pcap":
        ## Convert Pcap To CSV  
        pcap2csv (filedir, csvfile)  
    else:
        print("The file type is not supported at the moment!")
        exit
            
##
# This method, first, filterout the unnecessary packets, 
# afterward split the traffic based on the sessionthreshold
# Input: 
# filedir: where the file is located, filename: what it should be called, 
# sessionthreshold is the framework parameter for spliting traffic into sessions 
#
# Output: dataframe containing the filtered data #directory where splited data is stored
##             
def DSPreprocess(filedir, filename,sessionthreshold):#, sessionthreshold):
    
    # print("    filterout packet : ", filename)         
    filteredfile = filterdir+"/"+filename+"_filtered.csv"

    ## Filtering Packets and correting the name of protos 
    if not os.path.exists(filteredfile):  
        traffic = csvPacketFiltering(filedir, filteredfile)
    else:
        traffic = pd.read_csv ( filteredfile, skipinitialspace=True,
                             usecols=['frame.time_epoch', '_ws.col.Source', '_ws.col.Destination', '_ws.col.SrcPort', 
                                     '_ws.col.DstPort', '_ws.col.Protocol', 'frame.len', 'FileSource'], na_filter=False,
                             encoding="iso-8859-1",low_memory=False )
        
        
    splitdir = dsSplitdir+"/"+filename+"/%s/"%sessionthreshold
    os.makedirs (splitdir,exist_ok=True)
    
    ## Split to Sessions
    if not os.listdir(splitdir):
        traffic = pd.read_csv ( filteredfile, skipinitialspace=True,
                            usecols=['frame.time_epoch', '_ws.col.Source', '_ws.col.Destination', '_ws.col.SrcPort', 
                                    '_ws.col.DstPort', '_ws.col.Protocol', 'frame.len', 'FileSource'], 
                                      na_filter=False, dtype=str, encoding="iso-8859-1",low_memory=False )

        trafficbyfilesource = traffic.groupby ( ['FileSource'])
        for filesource, traffic in trafficbyfilesource:
            splitTraffic(traffic, splitdir, filename, sessionthreshold) 
            # splitTraffic(traffic, splitdir, filename+"_"+filesource, sessionthreshold) 
            
    return splitdir #traffic

###
# get class name from filename based on filenameToClassLabel in configuration class, 
# filename is lowered when this fucntion is called
##
def getClass(filename):    
    
    keys = filenameToClassLabel.keys()
    
    for key in keys:
        if key in filename:
            return filenameToClassLabel[key]
    
##
# It first call for feature extraction, by which packets are grouped into network unit and their statistical features extracted
# Then, it assign different label for the dataset depends on the classification tasks to be done 
# It should be eddited based on tasks and how files are named
##    
def featureVectorCaller(splitdir, filename, flowthreshold, flowduration):
# def featureVectorCaller(traffic, filename, flowthreshold, flowduration):

    # print("    Feature Extraction: ", filename)
    #features are the extracted units from a file
    # features = extractFeatures_stats(flowthreshold, flowduration, traffic)# splitdir+"/"+filename)
    features = extractFeatures_stats(flowthreshold, flowduration, splitdir+filename) #splitdir+"/"+filename

    features['Session'] =  session = filename.lower() # filename[:filename.rfind ( '.' )].lower()

    if 'Application' in classificationtask:
        features['Application'] =  session[:session.find ( '_' )]
        
    if 'Class' in classificationtask:
        features['Class'] = getClass(session)
        
    ## still based on naming
    if extractUserAction:
        app_session = session[:session.rfind ( '_' )]
        if '-' in app_session:
            features['Task'] = app_session[:app_session.rfind ( '-' )]
        else:
            features['Task'] = app_session[:app_session.rfind ( '_' )]
        
    if 'VPNnonVPN' in classificationtask:
        if "vpn" in session:
            features['VPNnonVPN'] = "VPN"  
        elif "tor" in session: 
            features['VPNnonVPN'] = "UnKnown"  
        else:
            features['VPNnonVPN'] = "nonVPN"  
            
    if 'Tor' in classificationtask:
        if "tor" in session:
            features['Tor'] = "Tor"  
        else: 
            features['Tor'] = "nonTor"  
    
    return features
##
# It is for transfering splited file into directory per our discution. 
# I should have done modifications in the code as I loop through file for csvfile in os.listdir(splitdir) per psml.
# Ex. convert :  DataSetSplited/com.gau.go.launcherex_1/10/com.gau.go.launcherex_1_0.csv  --> DataSetSplited/com.gau.go.launcherex/10/com.gau.go.launcherex_1_0.csv    
##    
#def transferSplitedFile(sessionthreshold):
#    for root, dirs, files in os.walk(dsSplitdir): 
#        for dirname in dirs: # 10 / 15
#            if dirname == str(sessionthreshold):
#                parentdir = root[:root.rfind ( '/' )]
#                parentfolder = root[root.rfind ( '/' )+1:]
#                
#                newdir = "%s/%s/%s"%(parentdir, parentfolder[:parentfolder.rfind ( '_' )], dirname)
#                os.makedirs (newdir,exist_ok=True)
#                
#                for filename in os.listdir(root+"/"+dirname): 
#                    move("%s/%s/%s"%(root,dirname,filename), "%s/%s"%(newdir, filename))
#                os.rmdir("%s/%s"%(root,dirname))
#                os.rmdir("%s"%(root))
#
#    return

##
# Given a file directory and file name this method read file and append the file source and user action (Task) as a column
##    
def preMerging(csvfile, dirname, filename):
    
    traffic = pd.read_csv ( csvfile, skipinitialspace=True,
                             usecols=['frame.time_epoch', '_ws.col.Source', '_ws.col.Destination', '_ws.col.SrcPort', 
                                     '_ws.col.DstPort', '_ws.col.Protocol', 'frame.len', '_ws.col.Info'], 
                                      na_filter=False, dtype=str, encoding="iso-8859-1", low_memory=False )  
    
    traffic['FileSource'] = filename  
    
#    traffic['Task'] = ""
    
    if extractUserAction:
        traffic['FileSource'] = filename[filename.rfind ( '_' )+1 :]
        traffic['Task'] = filename[:filename.rfind ( '_' )]
    
    return traffic

## 
# Convert traffic to csv format and merge csv related to same category into one. Result is stored in preproccesedir+"/Csv/" 
##    
def dataSetPreparation(dsdir):
    
    if rawDataType == "PSML":
        traffictype = [".xml"]
    elif rawDataType == "Pcap":
        traffictype = [".pcap",".pcapng"]
    
    for root, dirs, files in os.walk(dsdir): 
        for dirname in dirs:
            for filename in os.listdir(root+"/"+dirname): 
                ## Just making sure the traffic files are in correct format
                if filename[filename.rfind ( '.' ):] in traffictype:                    
#                    print(dirname+"_"+filename, "Started") 
                    Convert2Csv(root+"/"+dirname+"/"+filename, dirname, filename[:filename.rfind ( '.' )])
    
    
    for root, dirs, files in os.walk(preproccesedir+"/PreCsv"): 
        
        for dirname in dirs:
            traffics = []
            
            innercsvdir = csvdir+"/"+dirname
            os.makedirs (innercsvdir,exist_ok=True)
            
            for filename in os.listdir(root+"/"+dirname): 
                # Just making sure notting else is in this folder
                if filename.endswith(".csv"):
                    traffics.extend([preMerging(root+"/"+dirname+"/"+filename, dirname, filename[:filename.rfind ( '.' )])])    
                            
            # print("........ "+dirname)
            traffics = pd.concat(traffics)
            traffics = traffics.reset_index(drop=True)
            
            traffics = traffics.sort_values(by=['frame.time_epoch'])  
            
            if extractUserAction:
                trafficbytask = traffics.groupby ( ['Task'] )
                for task, traffic in trafficbytask:
                    traffic.to_csv(innercsvdir+"/"+dirname+"_"+task+".csv")
            else:
                traffics.to_csv(innercsvdir+"/"+dirname+".csv")

    #To remove pre csv file uncomment following
    #rmtree(preproccesedir+"/PreCsv", ignore_errors=True) 
    


##
# input is parameters to split network traffic into network units
# output is a csv containing network units and their labels
##    
def networkUnitExtraction(sessionthreshold, flowThresold, flowDuration):#, sessionthreshold):
    featurelist=[]

    for root, dirs, files in os.walk(csvdir): 
        for dirname in dirs:
            # print(dirname)
            for filename in os.listdir(root+"/"+dirname): 
                ## Just making sure the traffic files are in correct format
                if filename.endswith(".csv"): 
                    
                    print("  "+filename, "Started") 
                    splitdir = DSPreprocess(root+"/"+dirname+"/"+filename, filename[:filename.rfind ( '.' )], sessionthreshold)
                    # traffic = DSPreprocess(root+"/"+dirname+"/"+filename, filename[:filename.rfind ( '.' )])
                    for csvfile in os.listdir(splitdir):                        
#                        # Just making sure notting else is in this folder
                        if csvfile.endswith(".csv"):
                            # print("    "+csvfile+" VS "+filename[:filename.rfind ( '.' )]+"_%s"%csvfile)
                            # featurelist.extend([featureVectorCaller(splitdir, filename[:filename.rfind ( '.' )]+"_%s"%csvfile, flowThresold, flowDuration)])    
                            featurelist.extend([featureVectorCaller(splitdir, csvfile, flowThresold, flowDuration)])    

                    # trafficbyfilesource = traffic.groupby ( ['FileSource'])
                    # for filesource, traffic in trafficbyfilesource:
                    #     featurelist.extend([featureVectorCaller(traffic, filename[:filename.rfind ( '.' )]+"_%s"%filesource, flowThresold, flowDuration)])    
                            
    fv = pd.concat(featurelist)

    fv = fv.reset_index(drop=True)   
    
    fv = fv.sort_values(by=['TimeStampStart'])     
    ## transfer files
#    transferSplitedFile(sessionthreshold)

    return fv
 
#def nameproto(ds):    
#    ds['Protocol'] = ds['Protocol'].map ( 
#            {'R3':'RR', 'ICAP':'IC', 'PKIX-CRL':'PC', 'WireGuard':'W', 'giFT':'GI', '104asdu':'A', 'DIAMETER':'DI', 
#             'PPTP':'P', 'Socks':'K', 'TPKT':'Q',
#             'KNXnet_IP': 'KI', 'BJNP': 'J', 'BROWSER':'B', 'BitTorrent':'I', 'Chargen':'CH', 'DTLS':'D', 
#             'DTLSv1.0':'DV', 'EAPOL':'E', 'ENIP':'EN', 'Elasticsearch':'EL', 'FTP':'F', 'GPRS-NS':'GP', 'GQUIC':'G', 
#             'HTTP':'H', 'HTTP_XML':'HX', 'IEEE802.15.4':'IE', 'IPv6':'IP', 'LSD':'L', 'MiNT':'M', 'NXP802.15.4SNIFFER':'N', 
#             'OCSP':'O', 'RTCP':'R', 'RTMP':'RT', 'SIP':'SI', 'SMB2':'SN', 'SRVLOC':'SR', 'SSH':'S', 'SSHv2':'SV', 'SSL':'SL', 
#             'SSLv2':'SS', 'SSLv3':'SO', 'STUN':'ST', 'TCP':'T', 'TFPoverTCP':'TT', 'TLSv1':'TL', 'TLSv1.1':'TS', 'TLSv1.2':'TV', 
#             'UDP':'U', 'VNC':'V', 'X11':'X', 'XMPP_XML':'XX','CAT-TP':'CT', 'DB-LSP-DISC':'DB', 'IPXSAP':'IS', 'SMB':'SM'})    
#    return ds

##
# Input: 
# trainDS: dataset in the csv format
# stats: the key for statistical feature to be used 
# modeldir: where kmeans++ model is stored + number of clusted chosen for each protocol/ all
# featureVectorstandarddir: where standardized trainDS is stored
# figdir: where elbow diagrams are stored
# protobase: whether or not we are doing the clustering protocol-aware
# output: the trainDS with additional column 'Cluster' containing cluster number each row belong to
##    
def networkUnitClustering(trainDS, stats, modeldir, featureVectorstandarddir, figdir, protobase): 
    
    trainDS['Cluster'] = -1
       
    ## the first if is added for speading up to ignore elbow method    
    if os.path.isfile(numberofclusterdir) and os.stat(numberofclusterdir).st_size != 0:
        
        proto_cluster = loadjson(numberofclusterdir) 
        create = False 
     
    elif os.path.isfile(modeldir+'/../#ofCluster.json') and os.stat(modeldir+'/../#ofCluster.json').st_size != 0:
        
        proto_cluster = loadjson(modeldir+'/../#ofCluster.json') 
        create = False
    else:
        create = True

    if protobase:        
        tracesbyproto = trainDS.groupby ( ['Protocol'] )
        if create:
            # print(".... ",trainDS.Protocol.unique())
            proto_cluster = { proto : 1 for proto in list(trainDS.Protocol.unique())}
#            with ProcessPoolExecutor() as executor:
#                results = [executor.submit(elbow, flows[statsname[stats]], proto, figdir) for proto, flows in tracesbyproto]
#            for i, (proto, flows) in enumerate(tracesbyproto):
#                proto_cluster[proto] = results[i].get()
#                print(results[i])
#                print(proto, results[i].get())
            for proto, trainDSproto in tracesbyproto:
                proto_cluster[proto] = elbow(trainDSproto[statsname[stats]], proto, figdir) # Deside on number of cluster upon this  
                
            savejson(proto_cluster, modeldir+'/../#ofCluster.json') 
                     
#        with ProcessPoolExecutor() as executor:            
#            results = [executor.submit(clustering_Kmeans, 
#                                       proto_cluster[proto], flows[statsname[stats]], modeldir+'/Model_%s.joblib'%proto, 
#                                       featureVectorstandard+'/Standard_%s.joblib'%proto) for proto, flows in tracesbyproto]  
#        for i, (proto, flows) in enumerate(tracesbyproto):
#            trainDS.loc[trainDS['Protocol'] == proto, ['Cluster']] = list(results[i].get())
        for proto, trainDSproto in tracesbyproto:   
            protocluster = clustering_Kmeans(proto_cluster[proto], trainDSproto[statsname[stats]], modeldir+'/Model_%s.joblib'%proto, 
                                             featureVectorstandarddir+'/Standard_%s.joblib'%proto)
            trainDS.loc[trainDS['Protocol'] == proto, ['Cluster']] = list(protocluster)
        
#        trainDS = nameproto(trainDS)
    else:
        proto = "all"
        if create:
            proto_cluster = { proto : elbow(trainDS[statsname[stats]], proto, figdir)} # Decide on number of cluster upon this
            savejson(proto_cluster, modeldir+'/../#ofCluster.json') 
        protocluster = clustering_Kmeans(proto_cluster[proto], trainDS[statsname[stats]], modeldir+'/Model.joblib', 
                                         featureVectorstandarddir+'/Standard.joblib')
        trainDS['Cluster'] = list(protocluster)
        
    return trainDS

##
# Input: 
# testDS: dataset in the csv format
# stats: the key for statistical feature to be used 
# modeldir: where kmeans++ model is stored + number of clusted chosen for each protocol/ all
# featureVectorstandarddir: where standardized trainDS is stored
# protobase: whether or not we are doing the clustering protocol-aware
# output: the testDS with additional column 'Cluster' containing cluster number each row belong
## 
def networkUnitClustering_Assignment(testDS, stats, modeldir, featureVectorstandarddir, protobase): 
    testDS['Cluster'] = -1 
    if protobase:
        tracesbyproto = testDS.groupby ( ['Protocol'] )       
        for proto, testDSptoto in tracesbyproto:
            if os.path.exists(modeldir+'/Model_%s.joblib'%proto):
                assigned = assignCluster_Kmeans(testDSptoto[statsname[stats]], modeldir+'/Model_%s.joblib'%proto, 
                                                featureVectorstandarddir+'/Standard_%s.joblib'%proto)
                testDS.loc[testDS['Protocol'] == proto, ['Cluster']] = list(assigned)
            else:
                print(proto, "does not exists")
                testDS.loc[testDS['Protocol'] == proto, ['Cluster']] = 0
    else:
        assigned = assignCluster_Kmeans(testDS[statsname[stats]], modeldir+'/Model.joblib', featureVectorstandarddir+'/Standard.joblib')
        testDS['Cluster'] = list(assigned)
       
    return testDS

##
# Based on the algotrithm to be protobase or not this method handle how the trace of alphabet generete
# Input: 
# ds: train/test dataset in the csv format
# tracedir: where the trace will be stored    
# classifierTask: classification task
# protobase: whether or not the clustering is done protocol-aware
## 
def traceConvertor(ds, tracedir, classifierTask, protobase):
    ds['Alphabet'] = ""     
    os.makedirs (tracedir,exist_ok=True)   
    tracesbyclass = ds.groupby ( [classifierTask] )   
    for traceClass, dataset in tracesbyclass:
        if protobase:
            saveTrace_ProtoBase(tracedir+"/"+traceClass+".txt", dataset)            
        else:
            saveTrace(tracedir+"/"+traceClass+".txt", dataset)  
    return

##
# The alphabet is created here by concating Protocol and Cluster 
# Input: 
# filename: directory for storing traces of given class
# dataset: dataset    
##    
def saveTrace_ProtoBase(filename, dataset):    
    with open(filename, 'w', newline='') as myfile:
        dataset['Alphabet'] = dataset['Protocol'].astype(str) + "-" + dataset['Cluster'].astype(str)
        dataset = dataset.sort_values(by=['TimeStampStart']) 
       # trace.to_csv(filename.replace(".txt",".csv"))
        tracesbyclass = dataset.groupby ( ['Session'] ) ## changed for 4k split  into 'Trace'# previous version : 'Trace'
        for traceClass, trace in tracesbyclass:
            myfile.write(" ".join(s for s in trace['Alphabet'])) 
            myfile.write("\n")         
    return

##
# The alphabet is created here by concating a universal name (a) and Cluster 
# Input: 
# filename: directory for storing traces of given class
# traces: dataset  
## 
def saveTrace(filename, dataset):    
    with open(filename, 'w', newline='') as myfile:
        dataset['Alphabet'] = "a-" + dataset['Cluster'].astype(str)
        dataset = dataset.sort_values(by=['TimeStampStart']) 
      #  trace.to_csv(filename.replace(".txt",".csv"))
        tracesbyclass = dataset.groupby ( ['Session'] ) ## changed for 4k split into 'Trace' # previous version : 'Trace'
        for traceClass, trace in tracesbyclass:
            myfile.write(" ".join(s for s in trace['Alphabet'])) 
            myfile.write("\n")         
    return

##
# Dummping the number of clusters determined for each protocol/ all
##
def savejson(protocluster, file):
    f = open(file,"w+")   
    f.seek(0)
    f.write( json.dumps(protocluster) )
    f.close()
    return

##
# Loading the number of clusters determined for each protocol/ all
##
def loadjson(file):
    f = open(file,"r+")  
    protocluster = json.loads(f.read())
    f.close()
    return protocluster

##
# Delete featureVectorstandarddir directory recursively and kmeans++ models from modeldir
##    
def removemodels(featureVectorstandarddir, modeldir):
    rmtree(featureVectorstandarddir+"/", ignore_errors=True) 
    
    ## For not removing the number of clusterس file        
    for root, dirs, files in os.walk(modeldir): 
        for dirname in dirs: 
            if dirname == "joblib":
                rmtree(root+"/"+dirname+"/", ignore_errors=True) 
    return 
    
