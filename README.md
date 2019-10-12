# Coresets for Clustering with Fairness Constraints
This is the research code for the paper "Coresets for Clustering with Fairness Constraints" that has been accepted by NeurIPS 2019.
This research code consists of the full implementation of all experiments mentioned in the paper, and can be used to reproduce the experiments results.

Our proposed coreset algorithm is implemented in Java. Apart from our algorithm, we also include a C++ implementation for BICO algorithm using the BICO library: http://ls2-www.cs.tu-dortmund.de/grav/en/bico (which is one of our baseline), and modified Python implementations of FairTree: https://github.com/talwagner/fair_clustering and FairLP: https://github.com/nicolasjulioflores/fair_algorithms_for_clustering (which are used to verify the speed-up introduced by our coreset).

Our implementation relies on the following Java libraries which we do not include in this repo:

Apache Common CSV 1.7: http://commons.apache.org/proper/commons-csv/

Apache Common Math 3.6: https://commons.apache.org/proper/commons-math/

CPLEX 12.9: https://www.ibm.com/analytics/cplex-optimizer

JSON.simple: https://code.google.com/archive/p/json-simple/

Please put relevant jar/dll/so files into /lib folder.

To run the experiments, one also needs to:

1. Prepare the Python environment that FairTree and FairLP require

2. Put the compiled binary of BICO into BICO/ folder (the source code is located in bico_src)

3. Download the following datasets and compress the CSV files as data.csv.gz files and put them in data/DATASET_NAME/ folder. For instance, for census1990, one should compress & put it in data/census1990/data.csv.gz.

Adult: https://archive.ics.uci.edu/ml/datasets/Adult

Athlete: https://www.kaggle.com/heesoo37/120-years-of-olympic-history-athletes-and-results

Bank: https://archive.ics.uci.edu/ml/datasets/bank+marketing

Diabetes: https://archive.ics.uci.edu/ml/datasets/diabetes

Census1990: https://archive.ics.uci.edu/ml/datasets/US+Census+Data+(1990)

One can call the main method in Main.java to run all experiments, which would call BICO, FairTree, FairLP programs automatically.
