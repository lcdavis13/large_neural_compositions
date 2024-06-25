# large_neural_compositions
 experiments in predicting microbiome compositions from assemblages with large numbers of species

The [Waimea dataset](https://github.com/peterjsadowski/Tutorial-Microbiome/tree/main/data/waimea) has 5747 species, which results in over 66mil trained parameters when using the [cNODE2 architecture](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10055077/). In addition to resource consumption, this huge set of parameters is difficult to train with a reasonable number of samples (1410 in Waimea dataset).

This repository is a sandbox for attempting to model the Waimea dataset. This consists of (a) trying to streamline a cNODE2 implementation for big data, and (b) testing alternate architectures which have fewer parameters than cNODE2 at large scale.

cNODE2, and in particular [this repository](https://github.com/spxuw/DKI), were the starting point for this project.
Additionally, data from the following sources is used:
https://github.com/michel-mata/cNODE-paper/tree/master/Data/Experimental
