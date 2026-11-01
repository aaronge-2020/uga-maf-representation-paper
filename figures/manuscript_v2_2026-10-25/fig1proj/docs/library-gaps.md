# Library gaps — Figure 1

- No anchor-proxy part: connectors that must land at a non-centre point of a
  wide strip (under a specific box, not the strip's centre) need invisible
  named boxes as anchor stand-ins. Worked around with three transparent
  D.box anchors (stripM, stripE, stripR); a `port` option on box (custom
  anchor offsets) would remove the trick.
- cacheGrid registers only the grid's outer ports. A per-cell anchor (for a
  wire into one specific cell) would help table-style glyphs; not needed here.
- Nothing else was missing: the ruled grid, the cell-grid glyph, pills, dashed
  frames, and anchor connectors covered the whole design.
