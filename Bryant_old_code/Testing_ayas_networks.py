#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  1 12:31:22 2024

@author: bryant_avila
"""

import repair_direct_ar_prohibit_Binary_Options as rep
import graphs as gra
from os import listdir, makedirs
from os.path import isfile, join
import networkx as nx
import numpy as np
import re
import pandas as pd

def extract_numbers_at_end(s):
    match = re.search(r'(\d+)$', s)
    return int(match.group(1)) if match else 'Other'

#this code calculates the minimal number of balanced partitions 
##gg should be an nx.DiGraph
def FindMP(gg):
    gm = np.transpose(nx.adjacency_matrix(gg))
    network = gra.CoupledCellNetwork(gm.todense())
    grain = network.top_lattice_node() #most partitions
    return(len(set(grain[0]))), network

def graph_to_matrix(G, order_nodes):
    return np.array(nx.adjacency_matrix(G, nodelist=order_nodes).todense())


#Types of community detection
# Community_type = ['200_nodes',"1886_nodes",'9878_nodes']
#Community_type = ['small_test']
Community_type = ['C_H_GC_smaller']
# Community_type = ['C_H_GC_FintS']

meta_data = "Aya_networks.txt"

RM_AD = 3; #1:Add only | 2: Remove only | 3:Add and Remove

with open(meta_data, "a") as file:
    file.write('Type\tSolu_type\tAdd_Penalty\tRem_Penalty\tNumber_of_Colors\tMinimal_Colors\tSolution_type\tSetup_time\tSolving_time\n')

##Basic loop
for comm_type in Community_type:
    #read data
    workpath='/Users/phillips/Documents/GitHub/genes_coloring/Ayas_networks_from_Bryant/'
    
    #subfolders
    dirpath = workpath + 'data/'+comm_type+'/'
    
    #find all the graphfiles
    graphfiles = [f for f in listdir(dirpath) if isfile(join(dirpath, f)) \
                     and f.endswith('graph.txt')]
    
    #find all the color files
    colorfiles = [f for f in listdir(dirpath) if isfile(join(dirpath, f)) \
                     and f.endswith('colors.txt')]
        
    #find all the prohibited edges files
    prohibit = [f for f in listdir(dirpath) if isfile(join(dirpath, f)) \
                     and f.endswith('prohibit.txt')]
    
    #flag indicating we enforce hard constraints on coloring (True) or not
    HardFlag = True
    
    RMOnly = False
    
    InDegOneFlag = True
    
    Return_NetworkX = True
    
    Save_Network_to_files = False
    
    #weights = [[1,1], [0,1], [0,10], [0,100], [0,1000]]
    weights = [[1,1]]
    
    Save_output = True
    
    #main loop
    for Solu_type in ["Linear"]:   
        for gf in graphfiles:

            gpath = dirpath+gf
            df = pd.read_csv(gpath, sep='\t', names = ["Source", "Target", "Weight"])
            OG = nx.DiGraph()
            
            for _, row in df.iterrows():
                OG.add_edge(row['Source'], row['Target'], weight=row['Weight'])

            
            for cf in colorfiles:
                cpath = dirpath+cf
                epsi = extract_numbers_at_end(cf[:-11])
                for pb in prohibit:
                        pbath = dirpath+pb
                        

                        A,B,C,D,E,F,G,H,I,Setup_time = rep.set_rmip(gpath,cpath,'David',HardFlag,[],[],InDegOneFlag,RM_AD,pbath)
    
                        for option in range(0,len(weights)):
                            rm_weight = weights[option][0]
                            add_weight = weights[option][1] 
    
                              
                            #output directory
                            outpath=workpath+'outputs/'+comm_type+'/'+comm_type+'_add'+str(add_weight)+'_rem'+str(rm_weight)+'/'
                            
                            # Create output folder
                            makedirs(outpath, exist_ok=True)
                            
                            outfile = outpath+gf[:-10]+'_'+cf[:-11]+'_'+pb[:-13]+'_'+Solu_type
                            
                            _,idealnum,_,_,rem,add,_,_,_,_,gg,Solving_time = rep.solve_and_write(gpath,cpath,rm_weight,add_weight,outfile,A,B,C,D,E,F,G,H,I,"Linear",HardFlag,[],[],InDegOneFlag,pbath,Save_info=Save_output,NetX=True)
                        
                            numbers = re.findall(r'\d+', cpath)
                            
         
                            if gg.size()==0:
                                minp=0
                                RM_weighted = np.nan
                                Total_weight = np.nan
                            else:
                                minp, net = FindMP(gg)
                                
                                
                                
                                
                                print(f'Success! Minimal balancing achieved with {idealnum} colors.')
                                with open(meta_data, "a") as file:
                                    file.write(comm_type+'\t'+Solu_type+'\t'+str(add_weight)+'\t'+str(rm_weight)+'\t'+str(minp)+'\t'+str(idealnum)+'\t'+'D-minimal'+'\t'+str(Setup_time)+'\t'+str(Solving_time)+'\n')
                            
