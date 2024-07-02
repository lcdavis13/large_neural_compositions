# large_neural_compositions
 experiments in predicting microbiome compositions from assemblages with large numbers of species

The [Waimea dataset](https://github.com/peterjsadowski/Tutorial-Microbiome/tree/main/data/waimea) has 5747 features (OTUs, which are analogous to species). When attempting to use the [cNODE2 architecture](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10055077/) on this, it creates over 66 Million parameters to train. This contrast with the Ocean dataset, the largest that was analyzed in the cNODE/cNODE2 papers, which contains 73 species - corresponding to roughly 10 Thousand parameters. Over 3 orders of magnitude fewer parameters, which is very manageable to train.

cNODE2-Waimea's huge number of parameters is computationally intensive, but more importantly it resists training without a comparatively large sample size. There are only 1410 samples in Waimea dataset.

This repository is a sandbox for attempting to model the Waimea dataset with variations on the cNODE approach. This includes streamlining the cNODE2 implementation, and testing alternate architectures which have fewer parameters than cNODE2 at large scale.

cNODE2, and in particular [the DKI repository](https://github.com/spxuw/DKI), were the starting point for this project.
In addition to the Waimea dataset, this repository includes data from the DKI repository and the [cNODE-paper repository](https://github.com/michel-mata/cNODE-paper).

