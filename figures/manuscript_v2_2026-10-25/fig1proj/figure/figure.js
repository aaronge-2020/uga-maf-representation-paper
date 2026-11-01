// Concept A: horizontal pipeline, MuAt as a dashed lane below.
import * as D from '../engine/cs.js';

export const figure = { width: 170, height: 96, name: 'figure1' };

const INK = '#272727', MID = '#767676';
const BURDEN = '#767676', SPECTRA = '#3775BA', BIOMAF = '#42949E', SIGMAF = '#0F4D92', MUAT = '#E28E2C';
const TINT = '#F3F3F3';

export default function draw(F, L) {
  const a = F.panel('', { x: 0, y: 0, w: 170, h: 96 });
  const T = (x, y, s, o = {}) => L.text(a, x, y, s, { size: 6, color: INK, ...o });

  // ---- input: one tumour's mutation list ------------------------------------
  T(6, 12, 'One tumour', { size: 7, weight: 'bold' });
  T(6, 21, 'Mutation list (MAF)', { size: 5.5, color: MID });
  D.grid(a, 6, 27, [23, 17, 15, 36, 12], [9, 7.5, 7.5, 7.5], {
    size: 5, color: INK, stroke: MID, width: 0.4,
    cells: [
      [{ s: 'Locus', weight: 'bold' }, { s: 'Alleles', weight: 'bold' }, { s: 'Gene', weight: 'bold' }, { s: 'Consequence', weight: 'bold' }, { s: 'VAF', weight: 'bold' }],
      ['…', '…', '…', '…', '…'],
      ['…', '…', '…', '…', '…'],
      ['…', '…', '…', '…', '…'],
    ],
  });
  D.box(a, 6, 27, 103, 31.5, { fill: 'none', stroke: 'none', F, name: 'maf' });

  // ---- feature tables ---------------------------------------------------------
  T(118, 12, 'Feature tables', { size: 7, weight: 'bold' });
  const rows = [
    { name: 'Burden', count: '3 features', color: BURDEN, cols: 2, rows: 2, key: 'g1', mix: null },
    { name: 'Spectra (SBS96 + ID83)', count: '182 features', color: SPECTRA, cols: 6, rows: 4, key: 'g2', mix: null },
    { name: 'Bio MAF', count: '948 features', color: BIOMAF, cols: 9, rows: 4, key: 'g3', mix: null },
    { name: 'Signatures + Bio MAF', count: '1,130 features', color: SIGMAF, cols: 9, rows: 4, key: 'g4', mix: [SPECTRA, BIOMAF] },
  ];
  const rowY = [20, 54, 88, 122];
  rows.forEach((r, i) => {
    const y = rowY[i];
    T(118, y + 1, r.name, { size: 6.5, weight: 'bold' });
    T(252, y + 1, r.count, { size: 6.5, color: r.color, weight: 'bold', anchor: 'end' });
    const g = D.cacheGrid(a, 118, y + 6, r.rows, r.cols, {
      cw: 3.4, ch: 3.4, gap: 1.0, F, name: r.key, stroke: 'none',
      cell: (rr, cc) => r.mix ? (cc % 3 === 2 ? r.mix[1] : r.mix[0]) : r.color,
    });
    r.cy = y + 6 + g.h / 2; r.right = 118 + g.w;
  });

  // ---- models -----------------------------------------------------------------
  D.box(a, 292, 27, 92, 116, { fill: 'none', stroke: MID, width: 0.6, dash: '3 2', F, name: 'models' });
  T(338, 40, 'Models', { size: 7, weight: 'bold', anchor: 'middle' });
  D.box(a, 308, 58, 60, 15, { label: 'Elastic net', pill: true, fill: '#ffffff', stroke: INK, width: 0.6, size: 6.5 });
  D.box(a, 308, 82, 60, 15, { label: 'XGBoost', pill: true, fill: '#ffffff', stroke: INK, width: 0.6, size: 6.5 });
  T(338, 128, 'Tuned inside training folds only', { size: 5.5, color: MID, anchor: 'middle', italic: true });

  // ---- endpoints ---------------------------------------------------------------
  D.box(a, 396, 27, 82, 116, { fill: 'none', stroke: MID, width: 0.6, F, name: 'endpoints' });
  T(404, 40, 'Endpoints', { size: 7, weight: 'bold' });
  const eps = [
    ['Cancer type (20 types)', 'n=8,800', 56, 6],
    ['HRD score', 'n=772', 80, 6],
    ['HRD high/low at 24, 33, 42', 'n=772', 104, 5.5],
    ['Overall survival', 'n=9,986', 128, 6],
  ];
  eps.forEach(([n1, n2, y, sz]) => {
    T(404, y, n1, { size: sz, weight: 'bold' });
    T(404, y + 9, n2, { size: 5.5, color: MID });
  });

  // ---- test-group strip ----------------------------------------------------------
  D.box(a, 292, 176, 186, 18, { fill: TINT, stroke: 'none' });
  D.box(a, 330, 176, 16, 18, { fill: 'none', stroke: 'none', F, name: 'stripM' });
  D.box(a, 429, 176, 16, 18, { fill: 'none', stroke: 'none', F, name: 'stripE' });
  for (let i = 0; i < 5; i++) L.rect(a, 300 + i * 10, 182, 6, 6, { fill: INK });
  T(356, 188, 'Same five held-out test groups', { size: 6 });

  // ---- MuAt lane ------------------------------------------------------------------
  D.box(a, 6, 208, 472, 56, { fill: 'none', stroke: MUAT, width: 0.6, dash: '3 2' });
  T(14, 220, 'Outside the tabular path', { size: 5.5, color: MID, italic: true });
  D.box(a, 10, 225, 64, 32, { fill: 'none', stroke: 'none', F, name: 'bags' });
  const bagPts = [[24, 238], [33, 233], [42, 239], [30, 243], [39, 247], [48, 243], [26, 250], [44, 252], [52, 237], [36, 256]];
  bagPts.forEach(([x, y]) => L.circle(a, x, y, 2.1, { fill: '#FDF1E3', stroke: MUAT, width: 0.5 }));
  T(42, 230, 'Mutation event bags', { size: 6.5, weight: 'bold', anchor: 'middle' });
  D.box(a, 128, 228, 118, 26, { label: 'MuAt-compatible rebuild', fill: '#FDF1E3', stroke: MUAT, width: 0.7, size: 6.5, color: INK, F, name: 'muat' });
  D.box(a, 446, 176, 32, 18, { fill: 'none', stroke: 'none', F, name: 'stripR' });
  T(418, 222, 'Same five test groups', { size: 6, anchor: 'middle' });

  // ---- relations --------------------------------------------------------------------
  F.relate('maf.right', 'g1.left', 'flow');
  F.relate('maf.right', 'g2.left', 'flow');
  F.relate('maf.right', 'g3.left', 'flow');
  F.relate('maf.right', 'g4.left', 'flow');
  rows.forEach((r) => {
    L.connect(F, 'maf.right', `${r.key}.left`, { via: [[112, 42.75], [112, r.cy]], stroke: INK, width: 0.6 });
    F.relate(`${r.key}.right`, 'models.left', 'flow');
    L.connect(F, `${r.key}.right`, 'models.left', { via: [[286, r.cy], [286, 85]], stroke: INK, width: 0.6 });
  });
  F.relate('models.right', 'endpoints.left', 'flow');
  L.connect(F, 'models.right', 'endpoints.left', { stroke: INK, width: 0.6 });
  F.relate('models.bottom', 'stripM.top', 'benchmark');
  L.connect(F, 'models.bottom', 'stripM.top', { stroke: MID, width: 0.5 });
  F.relate('endpoints.bottom', 'stripE.top', 'benchmark');
  L.connect(F, 'endpoints.bottom', 'stripE.top', { stroke: MID, width: 0.5 });
  F.relate('bags.right', 'muat.left', 'flow');
  L.connect(F, 'bags.right', 'muat.left', { stroke: INK, width: 0.6 });
  F.relate('muat.right', 'stripR.bottom', 'benchmark');
  L.connect(F, 'muat.right', 'stripR.bottom', { via: [[462, 241], [462, 194]], stroke: MUAT, width: 0.6 });
}
