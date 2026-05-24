# VDKM Encoding Code

This directory contains the core CGR / UGA representation code.

Key file:

- `render.py` - nucleotide bit-pair mapping, CGR encoding/decoding, payload encoding, shared-allele trimming, BiCGR context encoding, and universal variant vector construction.

The active UGA specification is documented in:

```text
docs/UGA_MASTER_SPECIFICATION.md
```

Encoding tests live in:

```text
tests/test_universal_encoding.py
```

