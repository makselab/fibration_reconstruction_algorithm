This is a highlevel description of the files in the repository:
genes_coloring

Only files in the main directory will be maintained going forward0
including this readme.txt file

The only code files currently used are:
repair_direct_ar_prohibit_Binary_Options.py
graphs.py
repair.py

The remaining files will be depracated.

The main usage is to run the repair file from the command line. Run:
python repair.py -h

for the options. One sample run would be:
python repair.py -i "200_nodes/200_nodes.graph.txt" -p "200_nodes/200.yaml" -c "200_nodes/200_nodes.colors.txt" -o "200_nodes/200_nodes_o_"