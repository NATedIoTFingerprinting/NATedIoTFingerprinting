import sys, os
import ast
import numpy as np
import pandas as pd
import binascii
import time
import subprocess
import datetime
from io import StringIO
from datetime import datetime
from dpkt.tcp import parse_opts
import struct


NUM_PACKET_THRESHOLD = 200

## Function add the TCP OPTIONS columns in the dataframe
def tcp_option_list(hexstr):
    options = [0] * 6
    if hexstr == 'N':
        return pd.Series(options,index=['TCP_OPT_NOP', 'TCP_OPT_MSS', 'TCP_OPT_WSCALE','TCP_OPT_SACKOK','TCP_OPT_SACK','TCP_OPT_TIMESTAMP']) 
    else:    
        hexStr = ''.join( hexstr.split(":") )
        hexStr = binascii.unhexlify(hexStr)
        options_list = parse_opts(hexStr)
        for option in options_list:
            try:
                if option[0] == 2:
                    options[1] = struct.unpack(">H", option[1])[0]
                elif option[0] == 3:
                    options[2] = struct.unpack(">B", option[1])[0]
                elif option[0] == 8:
                    options[5] = 1  
                elif int(option[0]) in [1,4,5]:  
                    options[int(option[0])-1] = 1
                else:
                    continue
            except:
                pass
        return pd.Series(options,index=['TCP_OPT_NOP', 'TCP_OPT_MSS', 'TCP_OPT_WSCALE','TCP_OPT_SACKOK','TCP_OPT_SACK','TCP_OPT_TIMESTAMP']) 



def readFlow(df):
    df[['TCP_OPT_NOP', 'TCP_OPT_MSS', 'TCP_OPT_WSCALE','TCP_OPT_SACKOK','TCP_OPT_SACK','TCP_OPT_TIMESTAMP']] = df['tcp_option'].apply(lambda x: tcp_option_list(x))
    return df

## Creating empty dataframes 
df= pd.DataFrame(columns=['packetNum', 'ipsrc', 'prtcl', 'tos', 'tot_len', 'ip_id', 'ttl',
                            'IPsrc_long', 'IPdst_long', 'srcPort', 'dstPort', 'tcp_seq',
                             'tcp_ack_seq', 'tcp_off', 'tcpdatalen', 'tcp_reserve', 'tcp_flag',
                             'tcp_win', 'tcp_urp', 'timestamp', 'tcp_option'])

nated_df= pd.DataFrame(columns=['packetNum','ipsrc','prtcl','tos','tot_len','ip_id','ttl','IPsrc_long',
                                'IPdst_long','srcPort','dstPort','tcp_seq','tcp_ack_seq','tcp_off',
                                'tcpdatalen','tcp_reserve','tcp_flag','tcp_win','tcp_urp','timestamp',
                                'TCP_OPT_NOP', 'TCP_OPT_MSS', 'TCP_OPT_WSCALE','TCP_OPT_SACKOK','TCP_OPT_SACK',
                                'TCP_OPT_TIMESTAMP'])
mirai_unlabeled= pd.DataFrame(columns=['packetNum','ipsrc','prtcl','tos','tot_len','ip_id','ttl','IPsrc_long',
                                'IPdst_long','srcPort','dstPort','tcp_seq','tcp_ack_seq','tcp_off',
                                'tcpdatalen','tcp_reserve','tcp_flag','tcp_win','tcp_urp','timestamp',
                                'TCP_OPT_NOP', 'TCP_OPT_MSS', 'TCP_OPT_WSCALE','TCP_OPT_SACKOK','TCP_OPT_SACK',
                                'TCP_OPT_TIMESTAMP'])
# Loading mirai dump dataframe
mirai_dump= pd.read_csv("/home/Desktop/dump-mirai.csv")

# Print number of events
print(mirai_dump[mirai_dump['packetNum'] == 'packetNum'])

count = 0

for packet in mirai_dump.itertuples():

 if packet.packetNum == "packetNum":

     ### Getting dataframe with additional tcp options columns 
     dataframe = readFlow(df)

     ### Appending each event to mirai_df
     mirai_df=dataframe[['packetNum','ipsrc','prtcl','tos','tot_len','ip_id','ttl','IPsrc_long','IPdst_long','srcPort','dstPort','tcp_seq','tcp_ack_seq','tcp_off','tcpdatalen','tcp_reserve','tcp_flag','tcp_win','tcp_urp','timestamp','TCP_OPT_NOP', 'TCP_OPT_MSS', 'TCP_OPT_WSCALE','TCP_OPT_SACKOK','TCP_OPT_SACK','TCP_OPT_TIMESTAMP']]
     mirai_unlabeled= mirai_unlabeled.append(mirai_df)
     mirai_df.insert(6,'isChecked',False)

     # Checking for nated mirai packets
     for pckt1 in mirai_df.itertuples():

         if pckt1.isChecked == False:
        
             for pckt2 in mirai_df.itertuples():
                
                 if pckt2.ipsrc == pckt1.ipsrc and pckt2.isChecked == False:
                    
                     if not(pckt2.tcp_win == pckt1.tcp_win
                         or pckt2.srcPort == pckt1.srcPort
                         or pckt2.tcp_seq == pckt1.tcp_seq
                         or pckt2.dstPort == pckt1.dstPort):

                         nated_df=nated_df.append(mirai_df.loc[pckt2.Index,['packetNum','ipsrc','prtcl','tos','tot_len','ip_id','ttl','IPsrc_long','IPdst_long','srcPort','dstPort','tcp_seq','tcp_ack_seq','tcp_off','tcpdatalen','tcp_reserve','tcp_flag','tcp_win','tcp_urp','timestamp','TCP_OPT_NOP', 'TCP_OPT_MSS', 'TCP_OPT_WSCALE','TCP_OPT_SACKOK','TCP_OPT_SACK','TCP_OPT_TIMESTAMP']],ignore_index=True)
                         mirai_df.at[pckt2.Index,'isChecked'] = True
                         nated_df=nated_df.append(mirai_df.loc[pckt1.Index,['packetNum','ipsrc','prtcl','tos','tot_len','ip_id','ttl','IPsrc_long','IPdst_long','srcPort','dstPort','tcp_seq','tcp_ack_seq','tcp_off','tcpdatalen','tcp_reserve','tcp_flag','tcp_win','tcp_urp','timestamp','TCP_OPT_NOP', 'TCP_OPT_MSS', 'TCP_OPT_WSCALE','TCP_OPT_SACKOK','TCP_OPT_SACK','TCP_OPT_TIMESTAMP']],ignore_index=True) 

         mirai_df.at[pckt1.Index,'isChecked'] = True
    
     #Save to csv every 5K events
     if count == 5000:

         mirai_unlabeled.to_csv('/home/Desktop/Mirai-November.csv',index=False) 
         nated_df.to_csv('/home/Desktop/Nated-November.csv',index=False)
        
         #To empty dataframes 
         nated_df=nated_df.iloc[0:0] 
         mirai_unlabeled=nated_df.iloc[0:0]

     if count == 10000 or count == 15000 or count == 20000 or count==25000:

         df1= pd.read_csv("/home/Desktop/Mirai-November.csv")
         df1=df1.append(mirai_unlabeled)
         df1.to_csv('/home/Desktop/Mirai-November.csv',index=False) 

         df2= pd.read_csv("/home/Desktop/Nated-November.csv")
         df2=df2.append(nated_df)
         df2.to_csv('/home/Desktop/Nated-November.csv',index=False)

         #To empty dataframes 
         nated_df=nated_df.iloc[0:0] 
         mirai_unlabeled=nated_df.iloc[0:0]
    
     if packet.Index == mirai_dump.index[-1]:
         count+=1
         print("Event " + str(count) + " done")

         df1= pd.read_csv("/home/Desktop/Mirai-November.csv")
         df1=df1.append(mirai_unlabeled)
         df1.to_csv('/home/Desktop/Mirai-November.csv',index=False) 

         df2= pd.read_csv("/home/Desktop/Nated-November.csv")
         df2=df2.append(nated_df)
         df2.to_csv('/home/Desktop/Nated-November.csv',index=False)

     else:
         #To empty dataframes 
         df=df.iloc[0:0]
        
         count+=1
         print("Event " + str(count) + " done")

 else:
     df=df.append(mirai_dump.loc[packet.Index],ignore_index=True)

#Labeling Datasets      
df1= pd.read_csv("/home/Desktop/Mirai-November.csv")  
df2= pd.read_csv("/home/Desktop/Nated-November.csv")       
not_nated_df= pd.concat([df2,df1]).drop_duplicates()
df2.insert(26,'Type','Nated')
not_nated_df.insert(26,'Type','Not-Nated')
    
mirai_df_labeled = pd.concat([df2,not_nated_df], ignore_index=True)
mirai_df_labeled.to_csv('/home/Desktop/Mirai_Labeled-November-2021.csv',index=False) 

