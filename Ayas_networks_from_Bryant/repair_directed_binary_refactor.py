#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 26 08:44:15 2024

@author: phillips
"""

"""
language conventions:
    simple variables: x, key 
    lists/dictionaries/sets/tuples: Nodes_l, Color_d, Pairs_s, Nc_t
    constants: EPS
"""

##looks up value in dictionary and returns first key associated
def get_key_from_value(D_d,value):
    for key, Val_l in D_d.items():
        if value in Val_l:
            return key
    
#readdata
#builds network and structures for mip
def readdata():
    
    Data_d = {}

    Node_l = []
    
    
    
    Edge_l = []
    
    
    
#old return
#    return Nodes,Edges,colorpairs,colorsets,NotEdges,colordict,nc_tuples,\
#        outter_imbalance_dict,inner_imbalance_dict, support_num

    return Data_d


#creates the repair MIP, i.e., rmip
def create_rmip(Nodes,Edges,colorpairs,colorsets,outter_imbalance_dict,\
               inner_imbalance_dict,support_num,env, \
               Imbalance,NotEdges,colordict,nc_tuples,HardFlag,\
               FixedEdges,FixedNonEdges,AddRemoveFlag,InDegOneFlag):
    
    
    return rmip,rcons,rvars,remove_edge,add_edge,node_balance_pos,\
        node_balance_neg
        
        
def set_rmip(graphpath,colorpath,Imbalance,HardFlag,\
                 FixedEdges,FixedNonEdges,InDegOneFlag,AddRemoveFlag,prohibit):
    
    
    return rmip,rcons,rvars,setdict,colorsets,remove_edge,add_edge,\
        node_balance_pos,node_balance_neg,setup_time


#runs the rmip
def rmip_optimize(rmip,rcons,rvars,remove_edge,add_edge,node_balance_pos,\
                  node_balance_neg,rm_weight,add_weight,HardFlag,Solu_type,\
                  bal_weight=1):

    return rmip,rcons,rvars,executionTime


def solve_and_write(graphpath,colorpath,rm_weight,add_weight,fname,rmip,rcons,\
                       rvars,setdict,colorsets,remove_edge,add_edge,\
                       node_balance_pos,node_balance_neg,Solu_type,\
                       HardFlag=True,FixedEdges=[],FixedNonEdges=[],\
                       InDegOneFlag=True,prohibit=None,Save_info=True,NetX=False):
    
    return gname,idealnum,EdgesRemoved,EdgesAdded,sumremovals,sumadds,outfname,\
            rmip,rcons,rvars,G_result,executionTime
 