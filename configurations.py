#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  3 18:54:42 2020
@author: Zahra Alimadadi & Zeynab Sabahi
"""
import os
import enum

class Processing(enum.Enum): 
    kfold = "KFold"
    validation = "Validation"
    fasttest = "FastTest"
    

#header = ['Protocol', 'Srcip','SrcPort', 'Dstip', 'DstPort', 'Duration',
#          'TotalPktf','TotalLf','MinLf','MeanLf','MaxLf','StdLf', 'MeanIntervalf', "PktCountRatiof", 'DataSizeRatiof', 
#          'TotalPktb','TotalLb','MinLb','MeanLb','MaxLb','StdLb', 'MeanIntervalb', "PktCountRatiob", 'DataSizeRatiob']

## library to be installed:
# 1- kneed: pip3 install kneed
# 2- simplejson: pip3 install simplejson

## framework hyperparameters:

sessionthreshold = [100001]#5000, 15]
flowthreshold = [4.6]#0.2[0.1,0.2,0.3,0.4,0.5,0.6]#[0.5]#[0.1,0.3,0.5]#, 10, 15]
flowduration = [5.5]#0.2[0.1,0.2,0.3,0.4,0.5]#[0.1,0.3,0.5]#,8]#, 10, 15]
featureset = [1]#, 2]#, 2] # include keys of statsname selected to be used as statistical features
k_windows = [3]#[3,4]#, 4, 5]#, 6, 7, 8]

# 1 is coresponds to protocol-aware clustering, otherwise it cluster all together
clusters = [1] 

## statistical features in case of network traffic
statsname = { 
        1: ['TotalPktf', 'TotalLf', 'MinLf', 'MeanLf', 'MaxLf', 'StdLf',
             'TotalPktb', 'TotalLb', 'MinLb', 'MeanLb', 'MaxLb', 'StdLb'],        
        2: ['MinLf', 'MeanLf', 'MaxLf', 'StdLf', 'MeanIntervalf', 'PktCountRatiof', 'DataSizeRatiof',
            'MinLb', 'MeanLb', 'MaxLb', 'StdLb', 'MeanIntervalb', 'PktCountRatiob', 'DataSizeRatiob'],
        3: ['MeanIntervalf', 'PktCountRatiof', 'DataSizeRatiof',
            'MeanIntervalb', 'PktCountRatiob', 'DataSizeRatiob'],
        4: ['TotalPkt', 'TotalL', 'MinL',
            'MeanL', 'MaxL', 'StdL', 'TotalPktf', 'TotalLf', 'MinLf', 'MeanLf',
            'MaxLf', 'StdLf', 'TotalPktb', 'TotalLb', 'MinLb',
            'MeanLb', 'MaxLb', 'StdLb']}  # Packet Size Classification 
        
## list the classification task here        
classificationtask = ['Application']#, 'Class']#, 'Application', 'VPNnonVPN', 'Tor']#   

## set the following to True if user action should be extracted from the file name ( cases like unb)
extractUserAction = False # True #  

MapFileName = False

#filenameToClassLabel = {"vpnchat":"VPNChat", "vpnstreaming":"VPNStreaming", "vpnfiletransfer":"VPNFiletransfer", 
#                      "vpnemail":"VPNEmail", "vpnvoip":"VPNVoIP", "vpnp2p":"VPNTorrent", "chat":"Chat",
#                      "streaming":"Streaming", "filetransfer":"Filetransfer", "email":"Email", "voip":"VoIP",
#                      "torrent":"Torrent", "tor":"Tor"}

# this used if te classification task contains 'Class'
#If we want to map some apps to a specific application class
#  filenameToClassLabel = {"GIT":"VERSION", "SVN":"VERSION", "Joinme":"SHARING", 
#                      "Vsee":"CHAT", "Skype":"CHAT", "Team":"SHARING", "Psiphone":"VPN",
#                      "Ultra":"VPN"} 

# newly commented
#filenameToClassLabel = {"git":"GIT", "svn":"SVN", "joinme":"Joinme", 
                      # "vsee":"Vsee", "skype":"Skype", "team":"Team", "psiphone":"Psiphone",
                      # "ultra":"Ultra"}

#if extractUserAction =  True, partitionbasedon should be 'Task'
## for better sampling data into groups
partitionbasedon = 'Application'# 'Task'# for the Mobile dataset set it to Application  #

## It is just for naming the output folder with the name we are testing the result
settingsname =  'Simple_WiseHalfSplit' #  'Union_WiseHalfSplit' #

## evaluation method: kfold or validaion set or fasttest
processing = Processing.fasttest.name

## percentage for train set in the case of fasttest, rest is used for test
trainPercentage = 1 # 0.999999 #1 #0.8

## the least number of traces (traces extracted from input files) needed for each application to be considered in the classification task
threshold = 1 #2 #3

#This config was commented in neTLang_FastTest.py file.
#traindatathreshold = 20
#testdatathreshold = 5

## number of fold in case of Kfold
numberoffold = 10

#The following three configs are used in ktsslib
## trace cutting length coeficient for decision making during test 
# cuttingLengthCoe = 3
## whether or not shorten trace lenght for test
# cutLenght = False
## whether or not split the test trace by a fixed size
# splitTest = False

## CSV headers should be read from the dataset aditional to statistical features, it should contian classification tasks as well as 
## Session (those packet related to each other could be find by their Session value, for independent partitionning of the dataset),
## TimeStampStart (we need it for keep the packet sorted), and Protocol (to be used for protocol-aware clustering)
#if sth is disabled, then 'Session' columns equlas with 'fileSource'
csvAddHeader = ['Protocol', 'Session', 'TimeStampStart', 'TotalPkt'] # Trace',  add Trace for 4k split
csvAddHeader.extend(classificationtask)

## set operating system: linux = true, windows = false
linux = False

## in case traffic is collected on a public IP address we should set it here. It is used for defining flow direction
static_ip="131.202.240."

protocol2filterout = ['ARP', 'DNS', '? KNXnet/IP', 'AMS', 'ANSI C12.22', 'BAT_VIS', 'BJNP', 'CAT-TP', 'Chargen', 'DB-LSP-DISC', 'DCERPC', 'DCP-AF',
                      'DCP-PFT', 'DHCP', 'DHCPv6', 'DIAMETER', 'EAP', 'EAPOL', 'ENIP', 'ESP', 'Elasticsearch', 'GPRS-NS', 'H.225.0', 'H1', 'HIP',
                      'ICAP', 'ICMP', 'ICMPv6', 'IEEE 802.15.4', 'IGMPv2', 'IGMPv3', 'IPX SAP', 'IPv6', 'KRB5', 'LANMAN', 'LLC', 'LLDP', 'LLMNR',
                      'LSD', 'MDNS', 'MiNT', 'NAT-PMP', 'NBIPX', 'NBNS', 'NBSS', 'NTP', 'NXP 802.15.4 SNIFFER', 'OCSP', 'PKIX-CRL', 'PKTC', 
                      'Pathport', 'Q.931', 'R3', 'RTPproxy', 'SMPP', 'SNMP', 'SPOOLSS', 'SSDP', 'STP', 'TC-NV', 'TFP over TCP', 'TPKT', 
                      'TURN CHANNEL', 'WOW', 'giFT', 'OmniPath'] 

## Row data type: in case of network traffic it could be PSML or Pcap 
rawDataType = "Pcap" #"Pcap" #"Pcap" #"PSML"# "Pcap" #

## Remove created file in the middle
removefiles = False

## cwd is the code directory, it is assumed that the dataset is located here: cwd+/../DataSets/DSName/Pcap|PSML depending on the format
## It is important that the dataset files be in PSML(Pcap) folder as the rowdata
cwd = os.getcwd()

## Parent folder of Pcap/PSML folder, located in cwd+/../DataSets/, and only contains Pcap/PSML folder at the begining 
DSName = "4-UT"#"Tor"#7-UT_Class"#"UNB2_class"#"4-UT"#"UNB2_class_t"#"7-UT_Class"#"4-UT"#""UNB3"#""""7-UT_Class"#            "UNB2_class"#"5-UNB"#"UNB3"#"3-cross-android-ios"#"UNB3"#"4-UT"#"5-UNB"#"1-android-all-Copy"#"4-UT-copy"#"0-recon50-clusters"#"4-UT"#"0-recon99-50" #"2-ios-all"#"6-UNB_Class"#"5-UNB"#"0-recon99-50"#"3-cross-android-ios"#"1-android-all"#"2-ios-all"#"UNB3"#"Marzani2"#"UNB2"#"6-UNBClass"#"6-UNBClass"#"recon99"#"3-cross-android-ios"#"2-ios-all"#"1-android-all"#"0-recon99-50"#"6-UNB_Class"# "7-UT_Class"#"5-UNB" #"4-UT"#"3-cross-android-ios"#"1-android-all-100"#"2-ios-all" #"recon99-50"#"ios-all" #"recon99" #"recon99-30" "android-all-100" "android-all" "india-android-all" "us-android-all" "unb-1gig" "UT" "recon99-30" "UNB_Final" 

basedir = cwd+"/../DataSets/%s"%DSName

## following are the directories requred for saving different steps in the processing
dsdir = basedir + "/%s"% rawDataType
preproccesedir = basedir+"/Pre_Proccesed"
csvdir = preproccesedir+"/Csv"
filterdir = preproccesedir+"/Filtered"
dsSplitdir = preproccesedir +"/DataSetSplited"
fvdir = basedir +"/FeatureVectors"
figdir = basedir + "/Diagram"
tracesdir = basedir+"/Traces"
outputdir = basedir+"/Output"


## set number of cluster dir here, if this file exist, in clustering the elbow method not executed
numberofclusterdir = basedir+'/#ofCluster.json'

dirs= [tracesdir, outputdir, csvdir, filterdir, fvdir, figdir,  dsSplitdir]
for d in dirs:
    os.makedirs (d,exist_ok=True)
    

justRunTraces = False    


#***** configuration added for Deep Learning method used in ktssLib_Union_DL.py

## whether or not split test trace by a fixed size
splitTrain = True
splitTest = True

# validationFromTrainPercentage = 0.3  #I can add six variables for percentage of train,validation and test in each trace
testPercentage = 0.2 #0.4

# trace cutting length coeficient for decision making during test 
# cuttingLengthCoe = [3]
sp = [7]#[3,4]
#These configs check the min acceptable length of the train and test traces

trainLenCoeThreshold = 3
testLenCoeThreshold = 3

# dldir = basedir+"/DeepLearning"


