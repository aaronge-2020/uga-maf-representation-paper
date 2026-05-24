# UGA/MAF Representation Paper Reproduction Repository

This public repository contains the curated code, inputs, generated results, and manuscript figures/tables needed to reproduce the UGA/MAF representation manuscript analyses.

Start with [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md). File checksums and source categories are in [`DATA_MANIFEST.csv`](DATA_MANIFEST.csv). Large files are tracked with Git LFS and listed in [`LFS_MANIFEST.csv`](LFS_MANIFEST.csv).

The main runnable bundle is [`uga_maf_representation_paper/`](uga_maf_representation_paper/).

The bundled GRCh37 reference is compressed as `*.fna.gz`; the one-command workflow expands it automatically on the first real run.
