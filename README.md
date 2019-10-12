# Coresets for Clustering with Fairness Constraints
This is the research code for the paper "Coresets for Clustering with Fairness Constraints" that has been accepted by NeurIPS 2019.
This research code consists of the full implementation of all experiments mentioned in the paper, and can be used to reproduce the experiments results.

Our proposed coreset algorithm is implemented in Java. Apart from our algorithm, we also include a C++ implementation for BICO algorithm using the BICO library: http://ls2-www.cs.tu-dortmund.de/grav/en/bico (which is one of our baseline), and modified Python implementations of FairTree: https://github.com/talwagner/fair_clustering and FairLP: https://github.com/nicolasjulioflores/fair_algorithms_for_clustering (which are used to verify the speed-up introduced by our coreset).

Our implementation relies on the following Java libraries which we do not include in this repo:

Apache Common CSV 1.7: http://commons.apache.org/proper/commons-csv/

Apache Common Math 3.6: https://commons.apache.org/proper/commons-math/

CPLEX 12.9: https://www.ibm.com/analytics/cplex-optimizer

JSON.simple: https://code.google.com/archive/p/json-simple/

Please put relevant Jar/dll/so files into /lib folder.

To run the experiments, one also needs to prepare the Python environment that FairTree and FairLP require, and put the compiled binary of BICO into BICO/ folder. One can call the main method in Main.java to run all experiments, which would call BICO, FairTree, FairLP programs automatically.
