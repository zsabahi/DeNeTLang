# -*- coding: utf-8 -*-
"""
Created on Sun Oct 27 11:34:07 2019

@author: Zahra Alimadadi 
"""
import os.path

import pandas as pd
from subprocess import call
import xml.etree.cElementTree as ET
import re
from configurations import protocol2filterout, linux, static_ip

# filter just ack not syn ack : http://rapid.web.unc.edu/resources/tcp-flag-key/
def packetFiltering(pcap,output):
    
    cmd = 'tshark -2 -r %s -R "!tcp.flags == 0x00000010 and !tcp.len == 0 and !c1222 and !arp and !cattp and !db-lsp-disc and !dcerpc and !dcp-af and !dcp-pft and !bootp and !dhcpv6 and !dns and !esp and !h225 and !hip and !icap and !icmp and !icmpv6 and !igmp and !ipxsap and !lanman and !llc and !lldp and !llmnr and !mdns and !nat-pmp and !nbipx and !nbns and !nbss and !ntp and !ocsp and !pktc and !pathport and !smpp and !snmp and !spoolss and !ssdp and !stp and !tpkt " -w %s'%(pcap,output)
    if linux:        
        call (cmd)
    else:
        cmd = 'C:/"Program Files"/Wireshark/'+cmd
        call(cmd, shell=True) #in windows
    return 

##
# pcap: pcap file 
# csv: output file
## 
def pcap2csv(pcap,csv):
    
    cmd = 'tshark -r %s -T fields -E separator=, -e frame.number -e frame.time_epoch -e _ws.col.Source -e _ws.col.Destination -e tcp.flags -e _ws.col.SrcPort -e _ws.col.DstPort -e _ws.col.Protocol -e ip.proto -e frame.len -e ip.len -e _ws.col.Info -E header=y -E quote=d -E occurrence=f  > %s'%(pcap, csv)
    
    if linux:        
        call (cmd)
    else:
        cmd = 'C:/"Program Files"/Wireshark/'+cmd
        # print(cmd)
        call(cmd, shell=True) #in windows
        
    return 

##
# psml: psml file 
# csv: output file
##    
def psml2csv(psml, csvfile):
    tree = ET.parse(psml)
    rebase = tree.getroot()
    
    pdata = []
    pcolumns = []

    structure = rebase.find('structure')
    for section in structure.findall('section'):
        pcolumns.append(section.text)
           
    for packet in rebase.findall('packet'):
        newRow = []
        for section in packet.findall('section'):
            #print(section.text)
            newRow.append(section.text)
        pdata.append(newRow)
        
    df = pd.DataFrame(data=pdata,columns=pcolumns)
    
    ## renaming the colum to be same as the csv generated from the pcap file
    df.rename(columns = {'No.':'No.', 'Time':'frame.time_epoch', 'Source':'_ws.col.Source', 
                         'Destination':'_ws.col.Destination', 'src port':'_ws.col.SrcPort', 
                         'dst port':'_ws.col.DstPort', 'Protocol':'_ws.col.Protocol', 
                         'Length':'frame.len', 'Info':'_ws.col.Info'}, inplace = True) 
    
    df.to_csv(csvfile, index=False)
    
    return    
##
# Filtering Packets and correting the name of protocols     
##
def csvPacketFiltering(csvfile, filtered):
    traffic = pd.read_csv ( csvfile, skipinitialspace=True,
                             usecols=['frame.time_epoch', '_ws.col.Source', '_ws.col.Destination', '_ws.col.SrcPort', 
                                     '_ws.col.DstPort', '_ws.col.Protocol', 'frame.len', '_ws.col.Info', 'FileSource'], 
                                      na_filter=False, encoding="iso-8859-1", low_memory=False ) # dtype=str, 
    
#    nulls = []
#    for i, item in enumerate(traffic['frame.time_epoch']):
#       try:
#          float(item)
#       except ValueError:
#           nulls.append(i)
#           print('ERROR at index {}: {!r}'.format(i, item))
#      
#    print(len(traffic))
#    print(nulls)
#    if nulls:
#        for index in nulls:
#            print(traffic.loc[index-1, '_ws.col.Info'])
#            print(traffic.loc[index, 'frame.time_epoch'])
#            traffic.loc[index-1, '_ws.col.Info'] = traffic.loc[index-1, '_ws.col.Info'] + traffic.loc[index, 'frame.time_epoch']
#            print(traffic.loc[index-1, '_ws.col.Info'])
#            
#    traffic = traffic.drop(nulls)
#    
#    print(len(traffic))
    traffic['_ws.col.Info'] = traffic['_ws.col.Info'].apply(str)
    traffic = traffic[~traffic['_ws.col.Info'].str.contains("TCP Retransmission")]
    
    traffic = traffic[~(traffic['_ws.col.Info'].str.contains("ACK") & traffic['_ws.col.Info'].str.contains("Len=0"))]
    
    traffic = traffic[~traffic['_ws.col.Protocol'].isin(protocol2filterout)]
    traffic = traffic.reset_index(drop=True)      
    
    traffic['_ws.col.Protocol'] = traffic['_ws.col.Protocol'].str.replace('/','_')
    traffic['_ws.col.Protocol'] = traffic['_ws.col.Protocol'].str.replace('?','')
    traffic['_ws.col.Protocol'] = traffic['_ws.col.Protocol'].str.replace(' ','')
    traffic['_ws.col.Protocol'] = traffic['_ws.col.Protocol'].str.replace('-','_')
    traffic['_ws.col.Protocol'] = traffic['_ws.col.Protocol'].str.replace('.','_')
    
    ## we should put protocol version emission here.
    
    traffic.to_csv(filtered, index=False)
    
    return traffic
##
# Inputs:
# traffic: is the traffic 
# basedir: is the directory in which the splited traffic is stored
# fname: is the csv file name, so that the splited traffic be named upon it
# sessionthreshold is the framework parameter for spliting traffic into sessions    
##  
def splitTraffic(traffic, basedir, fname, sessionthreshold):
#    complete = pd.read_csv ( traffic, skipinitialspace=True,
#                             usecols=['frame.time_epoch', '_ws.col.Source', '_ws.col.Destination', '_ws.col.SrcPort', 
#                                     '_ws.col.DstPort', '_ws.col.Protocol', 'frame.len'], 
#                                      na_filter=False, dtype=str, encoding="iso-8859-1",low_memory=False )
            
#    nulls = []
#    for i, item in enumerate(traffic['frame.time_epoch']):
#       try:
#          float(item)
#       except ValueError:
#           nulls.append(i)
#           print('ERROR at index {}: {!r}'.format(i, item))
#        
#    traffic = traffic.drop(nulls)
        
    traffic["frame.time_epoch"] = traffic["frame.time_epoch"].astype(float) 
            
    splitID = (traffic["frame.time_epoch"] > (traffic["frame.time_epoch"].shift() + sessionthreshold)).cumsum()
    
    splitedtraffic = traffic.groupby(splitID)
    
    for k, split in splitedtraffic:
        # print(split)
        if not os.path.exists(basedir + "%s_%s.csv" % (fname,k)):
            # print("in if")
            split.to_csv(basedir + "%s_%s.csv" % (fname,k), index=False, mode='a')
        else:
            # print("in else")
            split.to_csv(basedir + "%s_%s.csv" % (fname,k), index=False, mode='a',header=False)
        
    return
##
# extract network flows, the output is dictionary of the extracted flows
##    
def extractflow(traffic):
    complete = pd.read_csv ( traffic, skipinitialspace=True,
                            usecols=['frame.time_epoch', '_ws.col.Source', '_ws.col.Destination', '_ws.col.SrcPort', 
                                    '_ws.col.DstPort', '_ws.col.Protocol', 'frame.len'], na_filter=False,
                            encoding="iso-8859-1",low_memory=False )
#    
#    complete.loc[complete['ip.len'].isin(['']).values, 'ip.len'] = -1

    traffic = complete
    # print(traffic)    
    pd.to_numeric(traffic["frame.len"])
    pd.to_numeric(traffic["_ws.col.SrcPort"])
    pd.to_numeric(traffic["_ws.col.DstPort"])
    pd.to_numeric(traffic["frame.time_epoch"])
    
    
#    traffic['DeltaTime'] = 0
    
    # Update flag mapping if any other flags were needed
#    traffic['tcp.flags'] = traffic['tcp.flags'].map ( {"0x00000002":'S', "0x00000004":'R', "0x00000008":'P', "0x00000010":'A',
#                                             "0x00000011":'FA', "0x00000012":'SA', "0x00000014":'RA', "0x00000018":'PA'})
#     I should solve the problem with IPv6
#    index = traffic['ip.len'].isin([-1])
#    traffic.loc[index,'header.len'] = 40    
#    traffic.loc[~index,'header.len'] = traffic.loc[~index,'frame.len'] - traffic.loc[~index,'ip.len'] 
    
        
    df = traffic.copy ()
    
    traffic.loc[(traffic['_ws.col.Destination'].apply(isprivate)),'frame.len'] = traffic['frame.len'] * -1
    traffic.loc[(traffic['frame.len'] < 0),'_ws.col.Source'] = df['_ws.col.Destination']
    traffic.loc[(traffic['frame.len'] < 0),'_ws.col.Destination'] = df['_ws.col.Source']
    traffic.loc[(traffic['frame.len'] < 0),'_ws.col.SrcPort'] = df['_ws.col.DstPort']
    traffic.loc[(traffic['frame.len'] < 0),'_ws.col.DstPort'] = df['_ws.col.SrcPort']
    complete_flow = traffic.groupby ( ['_ws.col.Source','_ws.col.SrcPort','_ws.col.Destination','_ws.col.DstPort','_ws.col.Protocol'])#, 'FileSource'] )

    return dict ( list ( complete_flow ) )

##
# Split traffic into network unit based on the two given criteria
##      
def extractNetworkUnits(traffic, flowthreshold, flowduration):
    
    flowdf = extractflow(traffic)    

    keys = flowdf.keys ()
    complete_flows = {}

    # print("$$$$$$$$$$$$$$$    "+str(len(keys)))
    
    for flow in keys:
        flow_ids = (flowdf[flow]['frame.time_epoch'] > (flowdf[flow]['frame.time_epoch'].shift () + flowthreshold)).cumsum ()

        flow_key = flowdf[flow].groupby (flow_ids)
        flowdf2 = dict ( list ( flow_key ) )

        
        ## Added for handling one packet lenght unit
        onelenght = set()
        prependto = set()
        for key in flowdf2.keys ():
            if len(flowdf2[key]) == 1:
                onelenght.add(key)
                prependto.add(key+1)
        i = 0  
        flaged = False
        for key in flowdf2.keys ():
            if key in onelenght:
                flaged = True
                continue
            elif key in prependto:
                flaged = False
                complete_flows[flow + (i,)] = pd.concat([flowdf2[key-1], flowdf2[key]])
                onelenght.remove(key-1)
            else:            
                complete_flows[flow + (i,)] = flowdf2[key]
            i += 1
            
        if flaged:
            for key in onelenght:
                complete_flows[flow + (i,)] = flowdf2[key]
                i+= 1
#            print("Warning - Main Lenght: This traffic contians one length flow without splitting ", flow)
        #del flowdf2
        
    splitedFlow = {}     
    
    for flow in complete_flows.keys():
        start = 0
        i = 0
        base = complete_flows[flow].iloc[0]['frame.time_epoch']
        if len(complete_flows[flow]) == 1:
#            print("Warning: This traffic contians one length flow without splitting ", flow)
            splitedFlow[flow + (i,)] = complete_flows[flow]               
        elif complete_flows[flow].iloc[-1,0] - complete_flows[flow].iloc[0,0] < flowduration:
            splitedFlow[flow + (i,)] = complete_flows[flow]
        else:
            for index, row in complete_flows[flow].iterrows():
                if row['frame.time_epoch'] - base > flowduration :
                    # if index-start  == 2:
                        # print("Warning - Duration: This traffic contians one length flow without splitting ", flow)
                    splitedFlow[flow + (i,)] = complete_flows[flow].loc[start:index-1, :]
                    start = index
                    base = row['frame.time_epoch']
                    i+=1
            if len(complete_flows[flow]) > index:
                # if len(complete_flows[flow]) - index == 1:
                    # print("Warning - Duration: This traffic contians one length flow without splitting ", flow)
                splitedFlow[flow + (i,)] = complete_flows[flow].loc[index:, :]
    
    return splitedFlow 
    

##
# First it call for networkunitextraction, then call for statistical feature computation
# traffic is the dataframe containing traffic #Traffic is the csv file address.
##    
def extractFeatures_stats(flowthreshold, flowduration, traffic):#traffic):
    
    # networkUnits = extractNetworkUnits(traffic, flowthreshold, flowduration)
    
    # ## latter we should move it to filtering method, I put it here as we have processed data before 
    # print("------- ",traffic)
    # traffic['_ws.col.Protocol'] = traffic['_ws.col.Protocol'].str.replace('\d+', '')
    # traffic['_ws.col.Protocol'] = traffic['_ws.col.Protocol'].str.replace('_', '')
    # traffic['_ws.col.Protocol'] = traffic['_ws.col.Protocol'].str.replace('v', '')
    
    networkUnits = extractNetworkUnits(traffic, flowthreshold, flowduration)
    
    fv = pd.DataFrame()

    for key in networkUnits.keys ():
        duration = networkUnits[key].iloc[-1,0]- networkUnits[key].iloc[0,0]
        features = [networkUnits[key].iloc[0,0], networkUnits[key].iloc[-1,0], key[4], 
                    key[0], key[1], key[2], key[3], duration]#, str(key[5]).replace(".","")] 
        stats = statistical_feature(networkUnits[key]) 
        features.extend(stats) 
        fv = fv.append ( pd.Series( features ).T, ignore_index=True) 
                               
    fv = fv.fillna(0)
        
    header = ['Protocol', 'Srcip','SrcPort', 'Dstip', 'DstPort', 'Duration', 'TotalPkt',# 'FileSource',
              'TotalPktf','TotalLf','MinLf','MeanLf','MaxLf','StdLf', 'MeanIntervalf', "PktCountRatiof", 'DataSizeRatiof', 
              'TotalPktb','TotalLb','MinLb','MeanLb','MaxLb','StdLb', 'MeanIntervalb', "PktCountRatiob", 'DataSizeRatiob']

    colname=['TimeStampStart','TimeStampEnd']  
    
    colname.extend(header)
    fv.columns = colname
    
    # Sort by first packet timestamp
    fv = fv.sort_values(by=['TimeStampStart'])   
   
    return fv
##
# The method responsile for handling statistical feature extraction
##    
def statistical_feature(df):
    features = []
    
    flowlen = df["frame.len"].count()    
    
    size = df["frame.len"].abs().sum()
    #hsize = df["header.len"].sum()
    #bidirection
    #features.extend(computeStatisticalFeature(df, False, flowlen, size, hsize)) 
    
    features.append(flowlen)
    #forward
    features.extend(computeStatisticalFeature(df[df['frame.len'] * 1 > 0], True, flowlen, size))#, hsize))

    #backward
    features.extend(computeStatisticalFeature(df[df['frame.len'] * 1 < 0], True, flowlen, size))#, hsize))

    return features

##
# This method compute statistical features
##    
def computeStatisticalFeature(flow, onedir, count, size): 
    df=flow.copy()
   
    df["frame.len"] = df["frame.len"].abs()
    
    TotalPkt = df["frame.len"].count() 
    TotalL = df["frame.len"].sum()
    MinL = df["frame.len"].min()
    MeanL = df["frame.len"].mean()
    MaxL = df["frame.len"].max()
    StdL = df["frame.len"].std()
        
    ds = df['frame.time_epoch'].diff()
  #  MinIntrval = ds.min()
    MeanInterval = ds.mean()
  #  MaxInterval = ds.max()
  #  StdInterval = ds.std()
    
   # TotalHL = df["header.len"].sum()

    features = [TotalPkt, TotalL, MinL, MeanL, MaxL, StdL,  MeanInterval]#, MinIntrval, MaxInterval, StdInterval, TotalHL]

    if onedir:
       # Pushcnt = pd.Series(df["tcp.flags"]).str.count('P').sum()
        PktCountRatio = TotalPkt/count
        DataSizeRatio = TotalL/size
      #  HDataSizeRatio = TotalHL/hsize
        features.extend([PktCountRatio, DataSizeRatio]) #[Pushcnt, PktCountRatio, DataSizeRatio, HDataSizeRatio]
        
    del df
    return features

# from stackoverflow
def isprivate(ip):
    
    if not isinstance(ip, str):
        ip=str(ip)       
    
    if ip.startswith(static_ip):
        return True
    
    priv_lo = re.compile("^127\.\d{1,3}\.\d{1,3}\.\d{1,3}$")
    priv_24 = re.compile("^10\.\d{1,3}\.\d{1,3}\.\d{1,3}$")
    priv_20 = re.compile("^192\.168\.\d{1,3}.\d{1,3}$")
    priv_16 = re.compile("^172.(1[6-9]|2[0-9]|3[0-1]).[0-9]{1,3}.[0-9]{1,3}$")
    return (priv_lo.match(ip) or priv_24.match(ip) or priv_20.match(ip) or priv_16.match(ip)) is not None

