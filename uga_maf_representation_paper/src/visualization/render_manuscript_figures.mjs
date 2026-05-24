import fs from "node:fs";
import path from "node:path";
import { createRequire } from "node:module";
import { pathToFileURL } from "node:url";

const [,, manuscriptArg, d3Arg] = process.argv;
if (!manuscriptArg) throw new Error("Usage: node render_manuscript_figures.mjs <manuscript_dir> [d3_vendor]");

const manuscriptDir = path.resolve(manuscriptArg);
const plotDir = path.join(manuscriptDir, "plot_data");
const figuresDir = path.join(manuscriptDir, "figures");
const supplementDir = path.join(manuscriptDir, "supplement");
fs.mkdirSync(figuresDir, { recursive: true });
fs.mkdirSync(supplementDir, { recursive: true });

const d3Vendor = d3Arg && fs.existsSync(d3Arg) ? fs.readFileSync(d3Arg, "utf8") : "";
const labelRegistryPath = path.join(process.cwd(), "src", "utils", "label_registry.json");
const labelRegistry = fs.existsSync(labelRegistryPath) ? JSON.parse(fs.readFileSync(labelRegistryPath, "utf8")) : {};

function registryMap(domain, fallback = {}) {
  const entries = Object.entries(labelRegistry[domain] || {}).map(([key, value]) => [key, value.display || key]);
  return entries.length ? Object.fromEntries(entries) : fallback;
}

const palette = {
  burden_only: "#6B7280",
  signatures_only: "#2F6FDB",
  MAF_stack_only: "#159A74",
  signatures_plus_MAF_stack: "#D9822B",
  one_hot_event_KME: "#A855C7",
  UGA_geometry: "#6D5BD0",
  channel_KME: "#0F8B8D",
  COSMIC_NNLS_exposures: "#64748B",
  mechanistic_control: "#B45309",
};

const repLabel = registryMap("representation_family", {
  burden_only: "Burden",
  signatures_only: "Signatures",
  MAF_stack_only: "MAF stack",
  signatures_plus_MAF_stack: "Signatures + MAF stack",
  one_hot_event_KME: "One-hot KME",
  UGA_geometry: "UGA geometry",
  channel_KME: "Channel KME",
  COSMIC_NNLS_exposures: "COSMIC NNLS",
  mechanistic_control: "Controls",
});

const endpointLabel = registryMap("endpoint", {
  damage_class: "Kucab damage class",
  HRD_Score: "HRD score",
  hrd_binary_33: "HRD33 high/low",
  cancer_type_top10: "Cancer type (top 10)",
  os_event: "Overall survival event",
});

const modelLabel = registryMap("model_family", {
  elastic_net: "Elastic net",
  XGBoost: "XGBoost",
});

const metricLabel = registryMap("metric", {
  macro_auroc: "macro-AUROC",
  auroc: "AUROC",
  spearman: "Spearman r",
});

const modelSpecificLabel = registryMap("model_label", {});

function parseCsv(text) {
  const rows = [];
  let row = [], value = "", inQuotes = false;
  for (let i = 0; i < text.length; i++) {
    const c = text[i], n = text[i + 1];
    if (inQuotes) {
      if (c === '"' && n === '"') { value += '"'; i++; }
      else if (c === '"') inQuotes = false;
      else value += c;
    } else if (c === '"') inQuotes = true;
    else if (c === ",") { row.push(value); value = ""; }
    else if (c === "\n") { row.push(value); rows.push(row); row = []; value = ""; }
    else if (c !== "\r") value += c;
  }
  if (value.length || row.length) { row.push(value); rows.push(row); }
  const header = rows.shift() || [];
  return rows.filter(r => r.length && r.some(v => v !== "")).map(r => Object.fromEntries(header.map((h, i) => [h, r[i] ?? ""])));
}

function readCsv(name) {
  const file = path.join(plotDir, name);
  if (!fs.existsSync(file)) return [];
  return parseCsv(fs.readFileSync(file, "utf8"));
}

function esc(value) {
  return String(value ?? "").replace(/[&<>"]/g, ch => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[ch]));
}

function num(value) {
  const x = Number(value);
  return Number.isFinite(x) ? x : NaN;
}

function unique(values) {
  return Array.from(new Set(values.filter(v => v !== "" && v != null)));
}

function labelRep(value, row = null) { return (row && (row.representation_family_display || row.representation_display)) || repLabel[value] || String(value || ""); }
function labelEndpoint(value, row = null) { return (row && row.endpoint_display) || endpointLabel[value] || String(value || ""); }
function labelModel(value, row = null) { return (row && row.model_display) || modelLabel[value] || String(value || ""); }
function labelMetric(value, row = null) { return (row && row.metric_display) || metricLabel[value] || String(value || ""); }
function labelDisplayModel(row) { return row.display_model_display || modelSpecificLabel[row.display_model] || labelModel(row.model_family, row) || String(row.display_model || ""); }

function wrapWords(text, maxChars) {
  const words = String(text || "").split(/\s+/).filter(Boolean);
  const lines = [];
  let line = "";
  for (const word of words) {
    const test = `${line} ${word}`.trim();
    if (test.length > maxChars && line) { lines.push(line); line = word; }
    else line = test;
  }
  if (line) lines.push(line);
  return lines.length ? lines : [""];
}

function textBlock(x, y, lines, opts = {}) {
  const cls = opts.cls || "small";
  const anchor = opts.anchor ? ` text-anchor="${opts.anchor}"` : "";
  const weight = opts.weight ? ` font-weight="${opts.weight}"` : "";
  const fill = opts.fill ? ` fill="${opts.fill}"` : "";
  const lh = opts.lineHeight || 18;
  return lines.map((line, i) => `<text x="${x}" y="${y + i * lh}" class="${cls}"${anchor}${weight}${fill}>${esc(line)}</text>`).join("\n");
}

function svgShell(width, height, body) {
  return `<svg xmlns="http://www.w3.org/2000/svg" width="${width}" height="${height}" viewBox="0 0 ${width} ${height}">
  <defs>
    <marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="8" markerHeight="8" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="#334155"/></marker>
    <linearGradient id="heat" x1="0" x2="1"><stop offset="0%" stop-color="#EFF6FF"/><stop offset="100%" stop-color="#1D4ED8"/></linearGradient>
  </defs>
  <style>
  text{font-family:Inter,Arial,sans-serif;fill:#111827;letter-spacing:0}.title{font-size:30px;font-weight:760}.subtitle{font-size:15px;fill:#475569}.panel-title{font-size:17px;font-weight:760}.label{font-size:14px}.small{font-size:12px;fill:#475569}.tiny{font-size:10px;fill:#64748B}.value{font-size:14px;font-weight:700}.sig{font-size:16px;font-weight:800;fill:#111827}.axis{stroke:#CBD5E1;stroke-width:1}.grid{stroke:#E2E8F0;stroke-width:1}.rule{stroke:#334155;stroke-width:1.8}.card{fill:#FFFFFF;stroke:#CBD5E1;stroke-width:1.2}.soft{fill:#F8FAFC}.shadow{filter:drop-shadow(0 2px 3px rgba(15,23,42,.10))}
  </style>${body}</svg>`;
}

function writeAsset(stem, svg, outDir) {
  if (/undefined|NaN/.test(svg)) throw new Error(`${stem} contains undefined or NaN`);
  fs.mkdirSync(outDir, { recursive: true });
  const svgPath = path.join(outDir, `${stem}.svg`);
  const htmlPath = path.join(outDir, `${stem}.html`);
  fs.writeFileSync(svgPath, svg);
  fs.writeFileSync(htmlPath, `<!doctype html><html><head><meta charset="utf-8"><title>${stem}</title><script>${d3Vendor}</script><style>body{margin:0;background:white}.wrap{padding:24px}</style></head><body><div class="wrap">${svg}</div></body></html>`);
  return { stem, svgPath, htmlPath, outDir };
}

function heatColor(v, min, max) {
  if (!Number.isFinite(v)) return "#E5E7EB";
  const t = Math.max(0, Math.min(1, (v - min) / Math.max(max - min, 1e-9)));
  const stops = [
    [239, 246, 255],
    [147, 197, 253],
    [37, 99, 235],
  ];
  const a = t < 0.5 ? stops[0] : stops[1];
  const b = t < 0.5 ? stops[1] : stops[2];
  const u = t < 0.5 ? t * 2 : (t - 0.5) * 2;
  return `rgb(${Math.round(a[0] + (b[0] - a[0]) * u)},${Math.round(a[1] + (b[1] - a[1]) * u)},${Math.round(a[2] + (b[2] - a[2]) * u)})`;
}

function requireMeasured(rows, stem) {
  const bad = rows.filter(r => String(r.status || "measured") !== "measured");
  if (bad.length) throw new Error(`${stem} contains non-measured rows: ${JSON.stringify(bad.slice(0, 5))}`);
}

function metricSubtitle(rows) {
  const metrics = unique(rows.map(r => r.metric)).map(m => labelMetric(m, rows.find(r => r.metric === m))).join(", ");
  return `Single 5-fold OOF results. Metrics: ${metrics}. Significance markers show BH-adjusted q-values for declared comparisons.`;
}

function barFigure(stem, title, csvName, outDir, options = {}) {
  const rows = readCsv(csvName);
  requireMeasured(rows, stem);
  const endpoints = ["damage_class", "HRD_Score", "hrd_binary_33", "cancer_type_top10", "os_event"].filter(ep => rows.some(r => r.endpoint === ep));
  const models = ["elastic_net", "XGBoost"].filter(m => rows.some(r => r.model_family === m));
  const reps = unique(rows.map(r => r.representation_family));
  const deltaPanel = Boolean(options.deltaPanel);
  const width = deltaPanel ? 1880 : 1650, margin = { left: 210, right: 70, top: 132, bottom: 88 };
  const deltaW = deltaPanel ? 170 : 0;
  const deltaGap = deltaPanel ? 38 : 0;
  const facetGap = 70;
  const facetW = (width - margin.left - margin.right - facetGap * (models.length - 1)) / Math.max(models.length, 1);
  const barW = Math.max(320, facetW - deltaW - deltaGap);
  const groupH = Math.max(68, reps.length * 18 + 20);
  const height = margin.top + margin.bottom + endpoints.length * groupH;
  const vals = rows.map(r => num(r.primary_score)).filter(Number.isFinite);
  const max = Math.max(0.75, Math.min(1.0, Math.max(...vals, 0.1) + 0.08));
  const deltaVals = rows
    .filter(r => r.representation_family === "one_hot_event_KME")
    .map(r => {
      const direct = num(r.delta);
      if (Number.isFinite(direct)) return direct;
      const base = rows.find(d => d.endpoint === r.endpoint && d.model_family === r.model_family && d.representation_family === "signatures_only");
      return base ? num(r.primary_score) - num(base.primary_score) : NaN;
    })
    .filter(Number.isFinite);
  const dMax = Math.max(0.03, Math.min(0.20, Math.max(...deltaVals.map(Math.abs), 0.01) * 1.35));
  const body = [
    `<rect width="${width}" height="${height}" fill="#FFFFFF"/>`,
    `<text x="44" y="50" class="title">${esc(title)}</text>`,
    `<text x="44" y="78" class="subtitle">${esc(metricSubtitle(rows))}</text>`,
  ];
  endpoints.forEach((ep, i) => {
    const y = margin.top + i * groupH + groupH / 2 + 5;
    body.push(textBlock(44, y - 9, wrapWords(labelEndpoint(ep), 20), { cls: "label", lineHeight: 16 }));
  });
  models.forEach((model, mi) => {
    const x0 = margin.left + mi * (facetW + facetGap);
    body.push(`<text x="${x0}" y="110" class="panel-title">${esc(labelModel(model))}</text>`);
    [0, 0.25, 0.5, 0.75, 1].forEach(t => {
      if (t <= max) {
        const x = x0 + t / max * barW;
        body.push(`<line x1="${x}" y1="${margin.top - 18}" x2="${x}" y2="${height - margin.bottom + 8}" class="grid"/>`);
        body.push(`<text x="${x}" y="${height - 42}" text-anchor="middle" class="tiny">${t.toFixed(2)}</text>`);
      }
    });
    endpoints.forEach((ep, ei) => {
      const baseY = margin.top + ei * groupH + 12;
      reps.forEach((rep, ri) => {
        const r = rows.find(d => d.endpoint === ep && d.model_family === model && d.representation_family === rep);
        if (!r) throw new Error(`${stem} missing ${ep}/${model}/${rep}`);
        const y = baseY + ri * 18;
        const w = Math.max(2, num(r.primary_score) / max * barW);
        const q = num(r.q_value);
        const sig = String(r.significance_label || "");
        body.push(`<rect x="${x0}" y="${y}" width="${w}" height="12" rx="2" fill="${palette[rep] || "#334155"}"><title>${esc(labelEndpoint(ep))} | ${esc(labelModel(model))} | ${esc(labelRep(rep))}: ${num(r.primary_score).toFixed(3)}${Number.isFinite(q) ? `; q=${q.toExponential(2)}` : ""}</title></rect>`);
        body.push(`<text x="${x0 + w + 7}" y="${y + 11}" class="tiny">${num(r.primary_score).toFixed(2)}${sig ? ` ${sig}` : ""}</text>`);
      });
    });
    if (deltaPanel) {
      const dx0 = x0 + barW + deltaGap;
      const zero = dx0 + deltaW / 2;
      body.push(`<text x="${zero}" y="110" text-anchor="middle" class="small">KME v2 - signatures</text>`);
      body.push(`<line x1="${zero}" y1="${margin.top - 18}" x2="${zero}" y2="${height - margin.bottom + 8}" stroke="#64748B" stroke-width="1.2"/>`);
      [-dMax, 0, dMax].forEach(t => {
        const x = zero + (t / dMax) * (deltaW / 2);
        body.push(`<line x1="${x}" y1="${height - margin.bottom + 14}" x2="${x}" y2="${height - margin.bottom + 20}" stroke="#64748B"/>`);
        body.push(`<text x="${x}" y="${height - 42}" text-anchor="middle" class="tiny">${t.toFixed(2)}</text>`);
      });
      endpoints.forEach((ep, ei) => {
        const r = rows.find(d => d.endpoint === ep && d.model_family === model && d.representation_family === "one_hot_event_KME");
        const base = rows.find(d => d.endpoint === ep && d.model_family === model && d.representation_family === "signatures_only");
        if (!r || !base) return;
        const direct = num(r.delta);
        const delta = Number.isFinite(direct) ? direct : num(r.primary_score) - num(base.primary_score);
        const y = margin.top + ei * groupH + groupH / 2 + 2;
        const x = zero + Math.max(-1, Math.min(1, delta / dMax)) * (deltaW / 2);
        const fill = delta >= 0 ? "#159A74" : "#B23A48";
        body.push(`<line x1="${zero}" y1="${y}" x2="${x}" y2="${y}" stroke="${fill}" stroke-width="2.2"/>`);
        body.push(`<circle cx="${x}" cy="${y}" r="4.5" fill="${fill}"><title>${esc(labelEndpoint(ep))} | ${esc(labelModel(model))}: delta=${delta.toFixed(3)}</title></circle>`);
      });
    }
  });
  reps.forEach((rep, i) => {
    const x = 44 + i * 170, y = height - 18;
    body.push(`<rect x="${x}" y="${y - 13}" width="14" height="14" rx="2" fill="${palette[rep] || "#334155"}"/><text x="${x + 21}" y="${y}" class="small">${esc(labelRep(rep))}</text>`);
  });
  return writeAsset(stem, svgShell(width, height, body.join("\n")), outDir);
}

function heatFacetFigure(stem, title, csvName, outDir) {
  const rows = readCsv(csvName);
  requireMeasured(rows, stem);
  const endpoints = ["damage_class", "HRD_Score", "hrd_binary_33", "cancer_type_top10", "os_event"].filter(ep => rows.some(r => r.endpoint === ep));
  const reps = ["burden_only", "signatures_only", "MAF_stack_only", "signatures_plus_MAF_stack", "one_hot_event_KME"];
  const models = ["elastic_net", "XGBoost"];
  const cellW = 176, cellH = 50, left = 235, top = 154, facetGap = 70;
  const panelH = 58 + endpoints.length * cellH;
  const width = left + reps.length * cellW + 80;
  const height = top + models.length * panelH + facetGap + 96;
  const vals = rows.map(r => num(r.primary_score)).filter(Number.isFinite);
  const min = Math.min(...vals), max = Math.max(...vals);
  const body = [
    `<rect width="${width}" height="${height}" fill="#FFFFFF"/>`,
    `<text x="44" y="50" class="title">${esc(title)}</text>`,
    `<text x="44" y="78" class="subtitle">Canonical 5-fold OOF values. Figure 5 is faceted by model family, so each cell matches Figures 2-4 exactly.</text>`,
  ];
  reps.forEach((rep, j) => {
    const cx = left + j * cellW + (cellW - 8) / 2;
    body.push(textBlock(cx, 116, wrapWords(labelRep(rep), 14), { cls: "label", anchor: "middle", lineHeight: 16 }));
  });
  models.forEach((model, mi) => {
    const y0 = top + mi * (panelH + facetGap);
    body.push(`<text x="44" y="${y0 - 18}" class="panel-title">${esc(labelModel(model))}</text>`);
    endpoints.forEach((ep, i) => {
      body.push(textBlock(44, y0 + i * cellH + 31, wrapWords(labelEndpoint(ep), 24), { cls: "label", lineHeight: 15 }));
      reps.forEach((rep, j) => {
        const r = rows.find(d => d.endpoint === ep && d.model_family === model && d.representation_family === rep);
        if (!r) throw new Error(`${stem} missing ${ep}/${model}/${rep}`);
        const v = num(r.primary_score);
        const x = left + j * cellW, y = y0 + i * cellH;
        const fill = heatColor(v, min, max);
        const q = num(r.q_value);
        const sig = String(r.significance_label || "");
        body.push(`<rect x="${x}" y="${y}" width="${cellW - 8}" height="${cellH - 8}" rx="6" fill="${fill}" stroke="#FFFFFF"><title>${esc(labelEndpoint(ep))} | ${esc(labelModel(model))} | ${esc(labelRep(rep))}: ${v.toFixed(3)}${Number.isFinite(q) ? `; q=${q.toExponential(2)}` : ""}</title></rect>`);
        body.push(`<text x="${x + (cellW - 8) / 2}" y="${y + 27}" text-anchor="middle" class="value">${v.toFixed(2)}</text>`);
        if (sig) body.push(`<text x="${x + cellW - 24}" y="${y + 17}" class="sig">${esc(sig)}</text>`);
      });
    });
  });
  const legendX = left, legendY = height - 46;
  body.push(`<rect x="${legendX}" y="${legendY - 12}" width="240" height="12" fill="url(#heat)"/><text x="${legendX}" y="${legendY + 20}" class="tiny">lower</text><text x="${legendX + 240}" y="${legendY + 20}" text-anchor="end" class="tiny">higher</text>`);
  return writeAsset(stem, svgShell(width, height, body.join("\n")), outDir);
}

function calibrationFigure(stem, title, csvName, outDir) {
  const rows = readCsv(csvName);
  if (!rows.length) throw new Error(`${stem} has no calibration bins`);
  const endpoints = ["damage_class", "hrd_binary_33", "cancer_type_top10", "os_event"].filter(ep => rows.some(r => r.endpoint === ep));
  const width = 1400, height = 920, panelW = 560, panelH = 300;
  const origins = [[170, 150], [830, 150], [170, 540], [830, 540]];
  const body = [
    `<rect width="${width}" height="${height}" fill="#FFFFFF"/>`,
    `<text x="44" y="50" class="title">${esc(title)}</text>`,
    `<text x="44" y="78" class="subtitle">Reliability curves from canonical OOF predictions for Signatures + MAF stack, XGBoost. Multiclass panels use confidence vs correctness.</text>`,
  ];
  endpoints.forEach((ep, i) => {
    const [x0, y0] = origins[i];
    const plotW = panelW - 100, plotH = panelH - 90;
    body.push(`<text x="${x0}" y="${y0 - 20}" class="panel-title">${esc(labelEndpoint(ep))}</text>`);
    body.push(`<rect x="${x0}" y="${y0}" width="${plotW}" height="${plotH}" fill="#F8FAFC" stroke="#CBD5E1"/>`);
    for (let t = 0; t <= 1.001; t += 0.25) {
      const x = x0 + t * plotW, y = y0 + plotH - t * plotH;
      body.push(`<line x1="${x}" y1="${y0}" x2="${x}" y2="${y0 + plotH}" class="grid"/>`);
      body.push(`<line x1="${x0}" y1="${y}" x2="${x0 + plotW}" y2="${y}" class="grid"/>`);
      body.push(`<text x="${x}" y="${y0 + plotH + 22}" text-anchor="middle" class="tiny">${t.toFixed(2)}</text>`);
      body.push(`<text x="${x0 - 12}" y="${y + 4}" text-anchor="end" class="tiny">${t.toFixed(2)}</text>`);
    }
    body.push(`<line x1="${x0}" y1="${y0 + plotH}" x2="${x0 + plotW}" y2="${y0}" stroke="#334155" stroke-width="1.5" stroke-dasharray="5,5"/>`);
    const points = rows.filter(r => r.endpoint === ep).map(r => ({
      x: x0 + num(r.mean_predicted) * plotW,
      y: y0 + plotH - num(r.observed_frequency) * plotH,
      n: num(r.n_samples),
    })).filter(p => Number.isFinite(p.x) && Number.isFinite(p.y)).sort((a, b) => a.x - b.x);
    if (points.length > 1) {
      body.push(`<polyline points="${points.map(p => `${p.x},${p.y}`).join(" ")}" fill="none" stroke="#2563EB" stroke-width="2.5"/>`);
    }
    points.forEach(p => {
      const r = Math.max(3.5, Math.min(10, Math.sqrt(p.n) * 0.9));
      body.push(`<circle cx="${p.x}" cy="${p.y}" r="${r}" fill="#2563EB" fill-opacity="0.85"><title>n=${p.n}</title></circle>`);
    });
    body.push(`<text x="${x0 + plotW / 2}" y="${y0 + plotH + 48}" text-anchor="middle" class="small">Predicted probability / confidence</text>`);
    body.push(`<text x="${x0}" y="${y0 - 4}" class="tiny">Observed frequency</text>`);
  });
  return writeAsset(stem, svgShell(width, height, body.join("\n")), outDir);
}

function s3MeasuredFigure(stem, title, csvName, outDir) {
  const rows = readCsv(csvName);
  if (!rows.length) throw new Error(`${stem} has no measured rows`);
  requireMeasured(rows, stem);
  if (rows.some(r => String(r.model_family) === "best")) throw new Error(`${stem} contains unlabeled best model rows`);
  const groups = unique(rows.map(r => r.analysis_family));
  const width = 1500, left = 280, right = 360, top = 126;
  const rowsByGroup = groups.map(g => rows.filter(r => r.analysis_family === g));
  const panelHeights = rowsByGroup.map(gRows => 74 + gRows.length * 32);
  const panelGap = 38;
  const height = top + panelHeights.reduce((a, b) => a + b, 0) + panelGap * Math.max(0, groups.length - 1) + 64;
  const vals = rows.map(r => num(r.primary_score)).filter(Number.isFinite);
  const max = Math.max(1, Math.max(...vals));
  const body = [
    `<rect width="${width}" height="${height}" fill="#FFFFFF"/>`,
    `<text x="44" y="50" class="title">${esc(title)}</text>`,
    `<text x="44" y="78" class="subtitle">Measured supplementary results only. Unsupported combinations are documented in Table S3 rather than drawn as placeholders.</text>`,
  ];
  let yCursor = top;
  groups.forEach((group, gi) => {
    const gRows = rowsByGroup[gi];
    const panelH = panelHeights[gi];
    body.push(`<text x="44" y="${yCursor - 16}" class="panel-title">${esc(group)}</text>`);
    [0, 0.25, 0.5, 0.75, 1].forEach(t => {
      const x = left + t / max * (width - left - right);
      body.push(`<line x1="${x}" y1="${yCursor}" x2="${x}" y2="${yCursor + panelH - 40}" class="grid"/>`);
      body.push(`<text x="${x}" y="${yCursor + panelH - 16}" text-anchor="middle" class="tiny">${t.toFixed(2)}</text>`);
    });
    const sortedRows = [...gRows].sort((a, b) => String(a.endpoint).localeCompare(String(b.endpoint)) || String(a.representation_family).localeCompare(String(b.representation_family)));
    const seenEndpoint = new Set();
    sortedRows.forEach((r, i) => {
      const ep = r.endpoint;
      const y = yCursor + 26 + i * 32;
      if (!seenEndpoint.has(ep)) {
        body.push(textBlock(44, y + 4, wrapWords(labelEndpoint(ep), 28), { cls: "small", lineHeight: 14 }));
        seenEndpoint.add(ep);
      }
      const v = num(r.primary_score);
      const x = left + v / max * (width - left - right);
      body.push(`<circle cx="${x}" cy="${y}" r="6" fill="${palette[r.representation_family] || "#334155"}"><title>${esc(labelRep(r.representation_family, r))} | ${esc(labelDisplayModel(r))} | ${v.toFixed(3)}</title></circle>`);
      body.push(`<text x="${x + 12}" y="${y + 4}" class="tiny">${v.toFixed(2)} - ${esc(labelRep(r.representation_family, r))} - ${esc(labelDisplayModel(r))}</text>`);
    });
    yCursor += panelH + panelGap;
  });
  return writeAsset(stem, svgShell(width, height, body.join("\n")), outDir);
}

function conceptualOverview(stem, outDir) {
  const width = 1900, height = 1020;
  const body = [
    `<rect width="${width}" height="${height}" fill="#FFFFFF"/>`,
    `<text x="56" y="54" class="title">Figure 1. Mutation catalogues to representations</text>`,
    textBlock(56, 84, wrapWords("The benchmark asks whether spectra, sequence-context geometry, event-level biology, or their combination is the most useful tabular view of a mutation catalogue.", 145), { cls: "subtitle", lineHeight: 20 }),
  ];

  body.push(`<text x="72" y="150" class="panel-title">A. One sample catalogue</text>`);
  body.push(`<text x="550" y="150" class="panel-title">B. Candidate feature views</text>`);
  body.push(`<text x="1080" y="150" class="panel-title">C. Combined tabular view</text>`);
  body.push(`<text x="1540" y="150" class="panel-title">D. Model comparison</text>`);

  body.push(`<rect x="70" y="178" width="410" height="525" rx="10" class="card shadow"/>`);
  body.push(`<text x="100" y="220" class="label" font-weight="700">MAF / VCF event rows</text>`);
  const cols = ["chr", "pos", "ref", "alt", "gene"];
  const colX = [100, 170, 260, 320, 380];
  cols.forEach((c, i) => body.push(`<text x="${colX[i]}" y="254" class="tiny" font-weight="700">${c}</text>`));
  const muts = [
    ["3", "41.2M", "C", "T", "BRCA1"],
    ["8", "128M", "T", "G", "MYC"],
    ["17", "7.6M", "G", "-", "TP53"],
    ["2", "90.1M", "A", "C", "ALK"],
    ["12", "25.4M", "+", "T", "KRAS"],
  ];
  muts.forEach((m, r) => {
    const y = 292 + r * 58;
    body.push(`<line x1="100" y1="${y + 18}" x2="438" y2="${y + 18}" class="grid"/>`);
    m.forEach((v, i) => body.push(`<text x="${colX[i]}" y="${y}" class="label">${esc(v)}</text>`));
    body.push(`<circle cx="440" cy="${y - 5}" r="${6 + r}" fill="${["#2F6FDB", "#D9822B", "#159A74", "#A855C7", "#6B7280"][r]}" opacity=".92"/>`);
  });
  body.push(textBlock(100, 615, ["Each row keeps genomic locus,", "allele change, consequence,", "VAF, gene and FASTA context."], { cls: "small", lineHeight: 20 }));

  body.push(`<path d="M480 440 C520 440 520 246 550 246" fill="none" class="rule" marker-end="url(#arrow)"/>`);
  body.push(`<path d="M480 440 C520 440 520 430 550 430" fill="none" class="rule" marker-end="url(#arrow)"/>`);
  body.push(`<path d="M480 440 C520 440 520 614 550 614" fill="none" class="rule" marker-end="url(#arrow)"/>`);

  const cards = [
    ["Mutational signatures", "SBS96, DBS78 and ID83 exposure spectra", "#2F6FDB", 550, 178, 390, 138, "hist"],
    ["One-hot sequence KME", "FASTA windows summarized by kernel means", "#A855C7", 550, 362, 390, 138, "kme"],
    ["Event-level MAF stack", "Aggregated gene, locus, consequence and VAF features", "#159A74", 550, 546, 390, 156, "chips"],
    ["Signatures + MAF stack", "Process spectra joined with event-level biology", "#D9822B", 1080, 338, 390, 180, "combo"],
  ];
  cards.forEach(([head, sub, color, x, y, w, h, kind]) => {
    body.push(`<rect x="${x}" y="${y}" width="${w}" height="${h}" rx="10" fill="#F8FAFC" stroke="${color}" stroke-width="2"/>`);
    body.push(`<text x="${x + 22}" y="${y + 34}" class="panel-title">${esc(head)}</text>`);
    body.push(textBlock(x + 22, y + 62, wrapWords(sub, 44), { cls: "small", lineHeight: 17 }));
    if (kind === "hist") for (let i = 0; i < 16; i++) body.push(`<rect x="${x + 24 + i * 18}" y="${y + 116 - ((i * 3) % 7 + 2) * 7}" width="11" height="${((i * 3) % 7 + 2) * 7}" fill="${color}" opacity="${0.42 + i / 36}"/>`);
    if (kind === "kme") {
      body.push(`<text x="${x + 24}" y="${y + 112}" class="tiny">A C G T C T A G</text>`);
      for (let i = 0; i < 12; i++) body.push(`<circle cx="${x + 255 + Math.cos(i * 1.2) * 50}" cy="${y + 104 + Math.sin(i * 1.7) * 24}" r="5" fill="${color}" opacity=".74"/>`);
    }
    if (kind === "chips") {
      ["TP53", "chr8", "missense", "VAF", "splice", "burden"].forEach((t, i) => {
        const px = x + 24 + (i % 3) * 112;
        const py = y + 88 + Math.floor(i / 3) * 30;
        body.push(`<rect x="${px}" y="${py}" width="92" height="22" rx="11" fill="${color}" opacity=".17"/><text x="${px + 46}" y="${py + 15}" text-anchor="middle" class="tiny">${t}</text>`);
      });
    }
    if (kind === "combo") {
      body.push(`<circle cx="${x + 72}" cy="${y + 120}" r="28" fill="#2F6FDB" opacity=".72"/><circle cx="${x + 112}" cy="${y + 120}" r="28" fill="#159A74" opacity=".72"/><circle cx="${x + 152}" cy="${y + 120}" r="28" fill="#D9822B" opacity=".72"/>`);
      ["spectra", "genes", "VAF"].forEach((t, i) => body.push(`<rect x="${x + 220}" y="${y + 86 + i * 28}" width="112" height="20" rx="5" fill="#FFF7ED" stroke="#FDBA74"/><text x="${x + 232}" y="${y + 101 + i * 28}" class="tiny">${t}</text>`));
    }
  });

  body.push(`<path d="M940 247 C990 247 1005 390 1080 390" fill="none" class="rule" marker-end="url(#arrow)"/>`);
  body.push(`<path d="M940 431 C1000 431 1010 428 1080 428" fill="none" class="rule" marker-end="url(#arrow)"/>`);
  body.push(`<path d="M940 615 C990 615 1005 468 1080 468" fill="none" class="rule" marker-end="url(#arrow)"/>`);

  body.push(`<path d="M1470 428 C1500 428 1510 296 1540 296" fill="none" class="rule" marker-end="url(#arrow)"/>`);
  body.push(`<rect x="1540" y="204" width="300" height="185" rx="10" class="card shadow"/>`);
  body.push(`<text x="1564" y="244" class="panel-title">Tabular models</text>`);
  body.push(textBlock(1564, 276, ["Elastic net and XGBoost", "5-fold out-of-fold predictions", "Endpoint-level metrics"], { cls: "small", lineHeight: 22 }));
  body.push(`<rect x="1564" y="342" width="96" height="24" rx="12" fill="#DBEAFE"/><text x="1612" y="359" text-anchor="middle" class="tiny">AUROC</text>`);
  body.push(`<rect x="1674" y="342" width="116" height="24" rx="12" fill="#DCFCE7"/><text x="1732" y="359" text-anchor="middle" class="tiny">Spearman r</text>`);

  body.push(`<rect x="1540" y="604" width="300" height="155" rx="10" fill="#FFF7ED" stroke="#D9822B" stroke-width="1.8"/>`);
  body.push(`<text x="1564" y="644" class="panel-title">End-to-end alternatives</text>`);
  body.push(textBlock(1564, 676, ["MuAt and ATGC operate directly", "on event sets as conceptual", "comparators outside the tabular path."], { cls: "small", lineHeight: 20 }));
  body.push(`<path d="M480 680 C770 890 1280 890 1540 682" fill="none" stroke="#D9822B" stroke-width="2.4" stroke-dasharray="9,9" marker-end="url(#arrow)"/>`);

  body.push(`<rect x="72" y="910" width="1768" height="52" rx="8" fill="#F8FAFC" stroke="#E2E8F0"/>`);
  body.push(`<line x1="100" y1="936" x2="158" y2="936" class="rule" marker-end="url(#arrow)"/><text x="174" y="941" class="small">Solid arrows: feature extraction and tabular benchmarking</text>`);
  body.push(`<line x1="560" y1="936" x2="618" y2="936" stroke="#D9822B" stroke-width="2.4" stroke-dasharray="9,9" marker-end="url(#arrow)"/><text x="634" y="941" class="small">Dashed arrow: direct event-set models used only as conceptual comparators</text>`);
  return writeAsset(stem, svgShell(width, height, body.join("\n")), outDir);
}

function representationConstruction(stem, outDir) {
  const width = 1600, height = 980;
  const body = [
    `<rect width="${width}" height="${height}" fill="#FFFFFF"/>`,
    `<text x="44" y="54" class="title">Supplementary Figure S1. Representation construction</text>`,
    `<text x="44" y="84" class="subtitle">Each expensive feature block is cached with input fingerprints, FASTA provenance, parameters and sample manifests.</text>`,
  ];
  const steps = [
    ["1", "Spectra", "Count SBS/DBS/ID channels and burden", "#2F6FDB", 70, 150],
    ["2", "FASTA windows", "Fetch GRCh37 context and validate REF", "#A855C7", 560, 150],
    ["3", "One-hot KME", "Encode windows and average kernels", "#A855C7", 1050, 150],
    ["4", "MAF stack", "Aggregate gene, locus, consequence and VAF", "#159A74", 70, 555],
    ["5", "UGA variants", "Supplementary atlas/channel geometry", "#6D5BD0", 560, 555],
    ["6", "Checkpointed outputs", "Feature caches, OOF predictions, tests and D3 plot data", "#D9822B", 1050, 555],
  ];
  steps.forEach(([n, head, sub, color, x, y]) => {
    body.push(`<rect x="${x}" y="${y}" width="390" height="260" rx="14" fill="#FFFFFF" stroke="${color}" stroke-width="2" class="shadow"/>`);
    body.push(`<circle cx="${x + 42}" cy="${y + 42}" r="22" fill="${color}"/><text x="${x + 42}" y="${y + 49}" text-anchor="middle" fill="#FFFFFF" font-weight="800">${n}</text>`);
    body.push(`<text x="${x + 78}" y="${y + 40}" class="panel-title">${esc(head)}</text>`);
    body.push(textBlock(x + 78, y + 68, wrapWords(sub, 34), { cls: "small", lineHeight: 17 }));
  });
  for (const [x1, y1, x2, y2] of [[460, 280, 555, 280], [950, 280, 1045, 280], [460, 685, 555, 685], [950, 685, 1045, 685]]) {
    body.push(`<line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" class="rule" marker-end="url(#arrow)"/>`);
  }
  for (let i = 0; i < 14; i++) body.push(`<rect x="${105 + i * 20}" y="${360 - (i % 6 + 2) * 8}" width="11" height="${(i % 6 + 2) * 8}" fill="#2F6FDB" opacity=".72"/>`);
  body.push(`<text x="610" y="340" class="tiny">... A C G T C [variant] A T G ...</text>`);
  for (let i = 0; i < 16; i++) body.push(`<rect x="${620 + i * 16}" y="${365}" width="12" height="24" fill="${["#DBEAFE", "#FEE2E2", "#DCFCE7", "#F3E8FF"][i % 4]}" stroke="#CBD5E1"/>`);
  for (let i = 0; i < 26; i++) body.push(`<circle cx="${1125 + Math.cos(i * 1.7) * (35 + (i % 4) * 11)}" cy="${342 + Math.sin(i * 1.3) * (28 + (i % 3) * 12)}" r="4" fill="#A855C7" opacity=".70"/>`);
  ["TP53", "BRCA1", "chr17p", "splice", "VAF", "HR repair"].forEach((t, i) => body.push(`<rect x="${105 + (i % 3) * 110}" y="${745 + Math.floor(i / 3) * 38}" width="92" height="26" rx="13" fill="#DCFCE7" stroke="#86EFAC"/><text x="${151 + (i % 3) * 110}" y="${763 + Math.floor(i / 3) * 38}" text-anchor="middle" class="tiny">${t}</text>`));
  body.push(`<path d="M630 780 L700 735 L770 775 L840 710" fill="none" stroke="#6D5BD0" stroke-width="2.5"/><circle cx="630" cy="780" r="5" fill="#6D5BD0"/><circle cx="700" cy="735" r="5" fill="#6D5BD0"/><circle cx="770" cy="775" r="5" fill="#6D5BD0"/><circle cx="840" cy="710" r="5" fill="#6D5BD0"/>`);
  ["features.npz", "manifest.json", "oof.csv", "tests.csv", "figure.svg"].forEach((t, i) => body.push(`<rect x="${1110}" y="${720 + i * 34}" width="210" height="24" rx="5" fill="#FFF7ED" stroke="#FDBA74"/><text x="${1122}" y="${737 + i * 34}" class="tiny">${t}</text>`));
  return writeAsset(stem, svgShell(width, height, body.join("\n")), outDir);
}

const assets = [];
assets.push(conceptualOverview("figure_1_conceptual_overview", figuresDir));
assets.push(barFigure("figure_2_signature_baselines", "Figure 2. Burden and signature baselines", "figure_2_signature_baselines.csv", figuresDir));
assets.push(barFigure("figure_3_geometry_vs_signatures", "Figure 3. One-hot sequence KME v2 vs signatures", "figure_3_geometry_vs_signatures.csv", figuresDir, { deltaPanel: true }));
assets.push(barFigure("figure_4_maf_stack_vs_signatures", "Figure 4. Event-level MAF biology", "figure_4_maf_stack_vs_signatures.csv", figuresDir));
assets.push(heatFacetFigure("figure_5_cross_endpoint_summary", "Figure 5. Cross-endpoint representation summary", "figure_5_cross_endpoint_summary.csv", figuresDir));
assets.push(representationConstruction("figure_s1_representation_construction", supplementDir));
assets.push(calibrationFigure("figure_s2_calibration_thresholds", "Supplementary Figure S2. Calibration and reliability", "figure_s2_calibration_thresholds.csv", supplementDir));
assets.push(s3MeasuredFigure("figure_s3_feature_importance", "Supplementary Figure S3. Supplementary representation checks", "figure_s3_feature_importance.csv", supplementDir));

async function qaSvg(page, stem) {
  const issues = await page.locator("svg").evaluate(svg => {
    const view = svg.viewBox.baseVal;
    const textNodes = Array.from(svg.querySelectorAll("text")).filter(t => (t.textContent || "").trim());
    const boxes = textNodes.map((t, i) => {
      const b = t.getBBox();
      return { i, text: (t.textContent || "").trim(), x: b.x, y: b.y, w: b.width, h: b.height };
    }).filter(b => b.w > 0 && b.h > 0);
    const out = [];
    for (const b of boxes) {
      if (b.x < -2 || b.y < -2 || b.x + b.w > view.width + 2 || b.y + b.h > view.height + 2) {
        out.push({ type: "clipped_text", text: b.text, box: b });
      }
    }
    for (let i = 0; i < boxes.length; i++) {
      for (let j = i + 1; j < boxes.length; j++) {
        const a = boxes[i], b = boxes[j];
        const ix = Math.max(0, Math.min(a.x + a.w, b.x + b.w) - Math.max(a.x, b.x));
        const iy = Math.max(0, Math.min(a.y + a.h, b.y + b.h) - Math.max(a.y, b.y));
        if (ix * iy > 18) out.push({ type: "text_overlap", a: a.text, b: b.text });
        if (out.length > 20) return out;
      }
    }
    return out;
  });
  return { stem, status: issues.length ? "failed" : "passed", issues };
}

let playwrightStatus = "not_attempted";
const visualQa = [];
try {
  const require = createRequire(import.meta.url);
  const { chromium } = require("playwright");
  const executableCandidates = [
    process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH,
    "C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe",
    "C:\\Program Files (x86)\\Google\\Chrome\\Application\\chrome.exe",
    "C:\\Program Files\\Microsoft\\Edge\\Application\\msedge.exe",
    "C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe",
  ].filter(Boolean);
  const executablePath = executableCandidates.find(p => fs.existsSync(p));
  const browser = await chromium.launch({ headless: true, ...(executablePath ? { executablePath } : {}) });
  for (const asset of assets) {
    const page = await browser.newPage({ viewport: { width: 1800, height: 1200 }, deviceScaleFactor: 2 });
    await page.goto(pathToFileURL(asset.htmlPath).href);
    const qa = await qaSvg(page, asset.stem);
    visualQa.push(qa);
    if (qa.status !== "passed") throw new Error(`${asset.stem} visual QA failed: ${JSON.stringify(qa.issues.slice(0, 5))}`);
    const svgBox = await page.locator("svg").boundingBox();
    if (svgBox) await page.setViewportSize({ width: Math.ceil(svgBox.width + 80), height: Math.ceil(svgBox.height + 80) });
    await page.screenshot({ path: path.join(asset.outDir, `${asset.stem}.png`), fullPage: true });
    await page.pdf({ path: path.join(asset.outDir, `${asset.stem}.pdf`), printBackground: true, width: `${Math.ceil((svgBox?.width || 1200) + 80)}px`, height: `${Math.ceil((svgBox?.height || 720) + 80)}px` });
    await page.close();
  }
  await browser.close();
  playwrightStatus = "exported_png_pdf";
} catch (error) {
  playwrightStatus = `failed: ${error.message}`;
  fs.writeFileSync(path.join(manuscriptDir, "d3_render_manifest.json"), JSON.stringify({
    created_utc: new Date().toISOString(),
    assets: assets.map(a => ({ stem: a.stem, svg: path.relative(manuscriptDir, a.svgPath), html: path.relative(manuscriptDir, a.htmlPath) })),
    playwright_status: playwrightStatus,
    visual_qa: visualQa,
  }, null, 2));
  throw error;
}

fs.writeFileSync(path.join(manuscriptDir, "d3_render_manifest.json"), JSON.stringify({
  created_utc: new Date().toISOString(),
  assets: assets.map(a => ({ stem: a.stem, svg: path.relative(manuscriptDir, a.svgPath), html: path.relative(manuscriptDir, a.htmlPath) })),
  playwright_status: playwrightStatus,
  visual_qa: visualQa,
}, null, 2));
