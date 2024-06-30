# large_neural_compositions
 experiments in predicting microbiome compositions from assemblages with large numbers of species

The [Waimea dataset](https://github.com/peterjsadowski/Tutorial-Microbiome/tree/main/data/waimea) has 5747 species, which results in over 66mil trained parameters when using the [cNODE2 architecture](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10055077/). Contrast this with the cNODE paper's Ocean dataset, the largest that was analyzed. It contains 73 species resulting in roughly 10 thousand parameters. This is a much more manageable number of parameters. 

In addition to resource consumption, cNODE2-Waimea's huge number of parameters resists training without a comparatively large sample size. There are only 1410 samples in Waimea dataset.

This repository is a sandbox for attempting to model the Waimea dataset with variations on the cNODE approach. This includes streamlining the cNODE2 implementation, and testing alternate architectures which have fewer parameters than cNODE2 at large scale.

cNODE2, and in particular [the DKI repository](https://github.com/spxuw/DKI), were the starting point for this project.
In addition to the Waimea dataset, this repository includes data from the DKI repository and the [cNODE-paper repository](https://github.com/michel-mata/cNODE-paper).

