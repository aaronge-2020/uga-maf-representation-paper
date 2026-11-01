// Concept B: vertical pipeline, MuAt column at the right.
import * as D from '../engine/cs.js';

export const figure = { width: 170, height: 118, name: 'concept_b' };

const INK = '#272727', MID = '#767676';
const BURDEN = '#767676', SPECTRA = '#3775BA', BIOMAF = '#42949E', SIGMAF = '#0F4D92', MUAT = '#E28E2C';

export default function draw(F, L) {
  const a = F.panel('', { x: 0, y: 0, w: 170, h: 118 });
  const T = (x, y, s, o = {}) => L.text(a, x, y, s, { size: 6, color: INK, ...o });

  // input, top centre-left
  T(60, 12, 'One tumour', { size: 7, weight: 'bold' });
  T(60, 21, 'Mutation list (MAF)', { size: 5.5, color: MID });
  D.grid(a, 60, 27, [22, 14, 14, 32, 12], [9, 7.5, 7.5], {
    size: 5, color: INK, stroke: MID, width: 0.4,
    cells: [
      [{ s: 'Locus', weight: 'bold' }, { s: 'Alleles', weight: 'bold' }, { s: 'Gene', weight: 'bold' }, { s: 'Consequence', weight: 'bold' }, { s: 'VAF', weight: 'bold' }],
      ['…', '…', '…', '…', '…'],
      ['…', '…', '…', '…', '…'],
    ],
  });
  D.box(a, 60, 27, 94, 24, { fill: 'none', stroke: 'none', F, name: 'maf' });

  // four glyphs in a row
  T(14, 78, 'Feature tables', { size: 7, weight: 'bold' });
  const cols = [
    { name: 'Burden', count: '3 features', color: BURDEN, cols: 2, rows: 2, key: 'g1', x: 14 },
    { name: 'Spectra (SBS96 + ID83)', count: '182 features', color: SPECTRA, cols: 6, rows: 4, key: 'g2', x: 92 },
    { name: 'Bio MAF', count: '948 features', color: BIOMAF, cols: 9, rows: 4, key: 'g3', x: 196 },
    { name: 'Signatures + Bio MAF', count: '1,130 features', color: SIGMAF, cols: 9, rows: 4, key: 'g4', x: 300, mix: [SPECTRA, BIOMAF] },
  ];
  cols.forEach((r) => {
    T(r.x, 92, r.name, { size: 6.5, weight: 'bold' });
    const g = D.cacheGrid(a, r.x, 98, r.rows, r.cols, {
      cw: 3.4, ch: 3.4, gap: 1.0, F, name: r.key, stroke: 'none',
      cell: (rr, cc) => r.mix ? (cc % 3 === 2 ? r.mix[1] : r.mix[0]) : r.color,
    });
    T(r.x, 98 + g.h + 10, r.count, { size: 6.5, color: r.color, weight: 'bold' });
    r.cx = r.x + g.w / 2; r.bot = 98 + g.h;
  });

  // models below centre
  D.box(a, 120, 150, 130, 40, { fill: 'none', stroke: MID, width: 0.6, dash: '3 2', F, name: 'models' });
  T(185, 163, 'Models', { size: 7, weight: 'bold', anchor: 'middle' });
  D.box(a, 132, 168, 48, 14, { label: 'Elastic net', pill: true, fill: '#ffffff', stroke: INK, width: 0.6, size: 6 });
  D.box(a, 190, 168, 48, 14, { label: 'XGBoost', pill: true, fill: '#ffffff', stroke: INK, width: 0.6, size: 6 });
  T(185, 204, 'Tuned inside training folds only', { size: 5.5, color: MID, anchor: 'middle', italic: true });

  // test strip + endpoints along the bottom
  D.box(a, 60, 224, 250, 18, { fill: '#F3F3F3', stroke: 'none' });
  D.box(a, 177, 224, 16, 18, { fill: 'none', stroke: 'none', F, name: 'stripM' });
  for (let i = 0; i < 5; i++) L.rect(a, 68 + i * 10, 230, 6, 6, { fill: INK });
  T(124, 236, 'Same five held-out test groups', { size: 6 });
  D.box(a, 60, 258, 340, 62, { fill: 'none', stroke: MID, width: 0.6, F, name: 'endpoints' });
  T(68, 271, 'Endpoints', { size: 7, weight: 'bold' });
  T(68, 288, 'Cancer type (20 types)', { size: 6, weight: 'bold' });
  T(68, 297, 'n=8,800', { size: 5.5, color: MID });
  T(196, 288, 'HRD score', { size: 6, weight: 'bold' });
  T(196, 297, 'n=772', { size: 5.5, color: MID });
  T(68, 312, 'HRD high/low at 24, 33, 42', { size: 5.5, weight: 'bold' });
  T(196, 312, 'n=772', { size: 5.5, color: MID });
  T(282, 288, 'Overall survival', { size: 6, weight: 'bold' });
  T(282, 297, 'n=9,986', { size: 5.5, color: MID });
  D.box(a, 177, 242, 16, 16, { fill: 'none', stroke: 'none', F, name: 'stripB' });

  // MuAt column at right
  D.box(a, 356, 27, 122, 293, { fill: 'none', stroke: MUAT, width: 0.6, dash: '3 2' });
  T(364, 39, 'Outside the tabular path', { size: 5.5, color: MID, italic: true });
  const bagPts = [[392, 66], [401, 61], [410, 67], [398, 71], [407, 75], [416, 71], [394, 78], [412, 80], [420, 65], [404, 84]];
  bagPts.forEach(([x, y]) => L.circle(a, x, y, 2.1, { fill: '#FDF1E3', stroke: MUAT, width: 0.5 }));
  T(417, 56, 'Mutation event bags', { size: 6.5, weight: 'bold', anchor: 'middle' });
  D.box(a, 366, 96, 102, 26, { label: 'MuAt-compatible rebuild', fill: '#FDF1E3', stroke: MUAT, width: 0.7, size: 6, color: INK, F, name: 'muat' });
  D.box(a, 356, 40, 40, 52, { fill: 'none', stroke: 'none', F, name: 'bags' });
  T(417, 300, 'Same five test groups', { size: 6, anchor: 'middle' });

  // relations
  cols.forEach((r) => {
    F.relate('maf.bottom', `${r.key}.top`, 'flow');
    L.connect(F, 'maf.bottom', `${r.key}.top`, { via: [[107, 66], [r.cx, 66]], stroke: INK, width: 0.6 });
    F.relate(`${r.key}.bottom`, 'models.top', 'flow');
    L.connect(F, `${r.key}.bottom`, 'models.top', { via: [[r.cx, 140], [185, 140]], stroke: INK, width: 0.6 });
  });
  F.relate('models.bottom', 'stripM.top', 'benchmark');
  L.connect(F, 'models.bottom', 'stripM.top', { stroke: MID, width: 0.5 });
  F.relate('stripB.bottom', 'endpoints.top', 'flow');
  L.connect(F, 'stripB.bottom', 'endpoints.top', { via: [[185, 250]], stroke: INK, width: 0.6 });
  F.relate('bags.bottom', 'muat.top', 'flow');
  L.connect(F, 'bags.bottom', 'muat.top', { via: [[376, 92]], stroke: INK, width: 0.6 });
  F.relate('muat.bottom', 'stripM.top', 'benchmark');
  L.connect(F, 'muat.bottom', 'stripM.top', { via: [[417, 214], [185, 214]], stroke: MUAT, width: 0.6 });
}
