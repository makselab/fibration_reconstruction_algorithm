# Summary 
This repository contain code to perform network repair of directed graph based in-balanced coloring condition. Given a directed graph and color labels for graph find minimal number of edges to add/remove to satisfy in-balance condition. The problem is formalized as a Mixed-Integer Programming (MIP) problem and solved using Gurobi.  

# Repository content 

- repair.py 
    Main script that takes in graph file, node color files and parameter file and uses repair_direct_ar_prohibit_Binary_Options.py to solve MIP. 

- repair_direct_ar_prohibit_Binary_Options.py  
    This script provides a solution to the problem of finding the minimum number of edges to add or remove in a directed network to ensure the coloring is imbalanced. It uses Gurobi for optimization and NetworkX for graph manipulation. Called by repair.py

- 200_nodes/ 
    Contains:  
        - 200.yaml 
            file params.yaml file for the repair code
        - 200_nodes.colors.txt
            tab-separated, no header. Columns: node, label/color
        - 200_nodes.graph.txt
            tab-separated, no header. Columns: source, target, weight(no weight:1)

- graphs.py

# Key features
Graph repair: min number of edges are added/removed such that in-balance coloring condition is satisfied. 
MIP optimization is set up given:
    - variables:
        - remove_edge/add_edge: binary
    - constraints:
        - balance_type: changes the balancing from hard to soft constraints. 
        - InDegOneFlag: Ensure node in-degrees meet specific criteria.
    - objective function:
        minimize the number of edge operations (additions/removals).


# How to use
The main usage is to run the repair file from the command line. Run:
python repair.py -h

for the options. One sample run would be:
python repair.py -i "200_nodes/200_nodes.graph.txt" -p "200_nodes/200.yaml" -c "200_nodes/200_nodes.colors.txt" -o "200_nodes/200_nodes_o_"