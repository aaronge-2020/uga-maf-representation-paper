# Reader test — Figure 1 (cold read, no caption)

Read cold, the figure says: one tumour's mutation list, a table with locus,
alleles, gene, consequence, and VAF columns, is converted into four feature
tables holding 3, 182, 948, and 1,130 features. Each table is fitted with an
elastic net and an XGBoost, tuned inside training folds only, and the fitted
models are scored against four endpoints (cancer type in 20 types, HRD score,
HRD high/low at three cut-offs, overall survival), with the scoring done on
the same five held-out test groups, drawn as five squares on a shared strip.
Separately, a MuAt-compatible rebuild reads mutation event bags directly and
is benchmarked on the same five groups.

That matches the spec's takeaway sentence point for point. Two places a cold
reader could hesitate, both accepted: the five squares need the strip label to
be read as groups rather than folds (the label says "test groups", and the
caption agrees), and the Burden glyph's two rows against the others' four
could be misread as a count (the printed "3 features" sits on its name line,
which settles it).
