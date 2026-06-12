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
  burden_only: "#4B5563",
  signatures_only: "#2563EB",
  MAF_stack_only: "#059669",
  signatures_plus_MAF_stack: "#D97706",
  MuAt_style_attention_MIL: "#334155",
  UGA_geometry: "#6D5BD0",
  channel_KME: "#0F8B8D",
  COSMIC_NNLS_exposures: "#64748B",
};

const gainColor = "#047857";
const lossColor = "#B91C1C";
const neutralText = "#475569";

const repLabel = registryMap("representation_family", {
  burden_only: "Burden",
  signatures_only: "Signatures",
  MAF_stack_only: "Bio MAF v4",
  signatures_plus_MAF_stack: "Signatures + Bio MAF v4",
  MuAt_style_attention_MIL: "MuAt-compatible",
  UGA_geometry: "UGA geometry",
  channel_KME: "Channel KME",
  COSMIC_NNLS_exposures: "COSMIC NNLS",
});

const endpointLabel = registryMap("endpoint", {
  damage_class: "Kucab damage class",
  HRD_Score: "HRD score",
  hrd_binary_24: "HRD24 high/low",
  hrd_binary_33: "HRD33 high/low",
  hrd_binary_42: "HRD42 high/low",
  cancer_type_top20: "Cancer type (top 20)",
  OS: "Overall survival",
  PFI: "Progression-free interval",
});

const modelLabel = registryMap("model_family", {
  elastic_net: "Elastic net",
  XGBoost: "XGBoost",
  cox_ph: "Cox PH",
  "MuAt-compatible reimplementation": "MuAt-compatible",
});

const metricLabel = registryMap("metric", {
  macro_auroc: "macro-AUROC",
  auroc: "AUROC",
  balanced_accuracy: "Balanced accuracy",
  spearman: "Spearman r",
  c_index: "Harrell C-index",
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

function readCsvFile(file) {
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
  text{font-family:Inter,Arial,sans-serif;letter-spacing:0}.title{font-size:30px;font-weight:760;fill:#111827}.subtitle{font-size:15px;fill:#475569}.panel-title{font-size:17px;font-weight:760;fill:#111827}.label{font-size:14px;fill:#111827}.small{font-size:12px;fill:#475569}.tiny{font-size:10px;fill:#64748B}.value{font-size:14px;font-weight:740;fill:#111827}.cell-value{font-size:16px;font-weight:780;fill:#111827}.cell-value-light{font-size:16px;font-weight:780;fill:#FFFFFF}.sig{font-size:13px;font-weight:800;fill:#111827}.badge-text{font-size:10px;font-weight:760;fill:#334155}.axis{stroke:#CBD5E1;stroke-width:1}.grid{stroke:#E2E8F0;stroke-width:1}.rule{stroke:#334155;stroke-width:1.8}.card{fill:#FFFFFF;stroke:#CBD5E1;stroke-width:1.2}.soft{fill:#F8FAFC}.row-band{fill:#F8FAFC}.chip{fill:#F1F5F9;stroke:#CBD5E1}.shadow{filter:drop-shadow(0 2px 3px rgba(15,23,42,.10))}
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

function clamp(value, minValue, maxValue) {
  return Math.max(minValue, Math.min(maxValue, value));
}

function formatScore(value) {
  const x = num(value);
  return Number.isFinite(x) ? x.toFixed(2) : "n/a";
}

function formatInt(value) {
  const x = num(value);
  return Number.isFinite(x) ? Math.round(x).toLocaleString("en-US") : "n/a";
}

function formatSigned(value, digits = 3) {
  let x = num(value);
  if (!Number.isFinite(x)) return "n/a";
  if (Math.abs(x) < 0.5 * Math.pow(10, -digits)) x = 0;
  return `${x >= 0 ? "+" : ""}${x.toFixed(digits)}`;
}

function formatQ(value) {
  const q = num(value);
  if (!Number.isFinite(q)) return "";
  if (q < 0.001) return "q<0.001";
  return `q=${q.toFixed(3)}`;
}

const mainEndpointOrder = ["damage_class", "HRD_Score", "hrd_binary_24", "hrd_binary_33", "hrd_binary_42", "cancer_type_top20", "OS"];
const tabularReps = ["burden_only", "signatures_only", "MAF_stack_only", "signatures_plus_MAF_stack"];
const muatRep = "MuAt_style_attention_MIL";

function mainEndpointsPresent(rows) {
  return mainEndpointOrder.filter(ep => rows.some(r => r.endpoint === ep));
}

function scoreRows(rows) {
  return rows.filter(r => Number.isFinite(num(r.primary_score)) && String(r.status || "measured") === "measured");
}

function bestTabular(rows, endpoint) {
  const candidates = scoreRows(rows).filter(r => r.endpoint === endpoint && tabularReps.includes(r.representation_family));
  candidates.sort((a, b) => num(b.primary_score) - num(a.primary_score));
  return candidates[0] || null;
}

function benchmarkTabular(rows, endpoint) {
  const preferredModel = endpoint === "OS" ? "cox_ph" : "XGBoost";
  return scoreRows(rows).find(r =>
    r.endpoint === endpoint &&
    r.representation_family === "signatures_plus_MAF_stack" &&
    r.model_family === preferredModel
  ) || bestTabular(rows, endpoint);
}

function muatForEndpoint(rows, endpoint) {
  return scoreRows(rows).find(r => r.endpoint === endpoint && r.representation_family === muatRep) || null;
}

function rowTitle(row) {
  if (!row) return "";
  return `${labelRep(row.representation_family, row)} | ${labelModel(row.model_family, row)}`;
}

function metricCode(value, row = null) {
  const label = labelMetric(value, row).toLowerCase();
  if (label.includes("balanced")) return "Bal acc";
  if (label.includes("macro")) return "mAUROC";
  if (label.includes("spearman")) return "rho";
  if (label.includes("auroc")) return "AUROC";
  return labelMetric(value, row);
}

function taskCode(row) {
  const task = String(row?.task || row?.task_display || "").toLowerCase();
  if (task.includes("regression")) return "reg";
  if (task.includes("multiclass")) return "multi";
  if (task.includes("binary")) return "binary";
  if (task.includes("survival")) return "surv";
  return "task";
}

function chip(x, y, text, opts = {}) {
  const w = opts.width || Math.max(42, String(text).length * 6.4 + 18);
  const fill = opts.fill || "#F1F5F9";
  const stroke = opts.stroke || "#CBD5E1";
  const textFill = opts.textFill || "#334155";
  return `<rect x="${x}" y="${y - 13}" width="${w}" height="20" rx="10" fill="${fill}" stroke="${stroke}"/><text x="${x + w / 2}" y="${y + 1}" text-anchor="middle" class="badge-text" fill="${textFill}">${esc(text)}</text>`;
}

function endpointContext(row, x, y) {
  if (!row) return "";
  const n = num(row.n_samples);
  const metric = metricCode(row.metric, row);
  const task = taskCode(row);
  const metricW = Math.max(50, metric.length * 6.8 + 18);
  const compactLongMetric = metricW > 84;
  const parts = [
    chip(x, y, metric, { width: metricW }),
  ];
  if (!compactLongMetric) parts.push(chip(x + 66, y, task, { width: 54 }));
  const nX = compactLongMetric ? x + metricW + 14 : x + 132;
  if (Number.isFinite(n)) parts.push(`<text x="${nX}" y="${y + 1}" class="tiny">n=${Math.round(n).toLocaleString("en-US")}</text>`);
  return parts.join("");
}

function repFeatureSummary(rows, rep) {
  const values = rows.filter(r => r.representation_family === rep).map(r => num(r.n_features)).filter(v => Number.isFinite(v) && v > 0);
  if (!values.length) return "";
  values.sort((a, b) => a - b);
  const median = values[Math.floor(values.length / 2)];
  return `${Math.round(median).toLocaleString("en-US")} feat`;
}

function pairwiseTests(figureId) {
  return readCsvFile(path.join(manuscriptDir, "canonical", "main_panel_pairwise_tests.csv"))
    .filter(r => r.figure_id === figureId && r.test_status === "tested");
}

function findTest(tests, endpoint, model, comparisonName) {
  return tests.find(d => d.endpoint === endpoint && d.model_family === model && d.comparison_name === comparisonName);
}

function contrastRange(tests, fallback = 0.10) {
  const vals = [];
  tests.forEach(r => {
    ["delta", "ci_low", "ci_high"].forEach(k => {
      const v = num(r[k]);
      if (Number.isFinite(v)) vals.push(Math.abs(v));
    });
  });
  return Math.max(fallback, Math.min(0.30, Math.max(...vals, fallback) * 1.18));
}

function significanceLegend(x, y) {
  return [
    `<text x="${x}" y="${y}" class="tiny">Paired-test FDR:</text>`,
    `<text x="${x + 112}" y="${y}" class="tiny">* q&lt;0.05</text>`,
    `<text x="${x + 190}" y="${y}" class="tiny">** q&lt;0.01</text>`,
    `<text x="${x + 274}" y="${y}" class="tiny">*** q&lt;0.001</text>`,
  ].join("");
}

function signedLegend(x, y, positiveText = "positive favors candidate", negativeText = "negative favors baseline") {
  return [
    `<line x1="${x}" y1="${y - 4}" x2="${x + 28}" y2="${y - 4}" stroke="${gainColor}" stroke-width="3"/>`,
    `<text x="${x + 36}" y="${y}" class="tiny">${esc(positiveText)}</text>`,
    `<line x1="${x + 220}" y1="${y - 4}" x2="${x + 248}" y2="${y - 4}" stroke="${lossColor}" stroke-width="3"/>`,
    `<text x="${x + 256}" y="${y}" class="tiny">${esc(negativeText)}</text>`,
  ].join("");
}

function textPill(x, y, text, opts = {}) {
  const anchor = opts.anchor || "middle";
  const fill = opts.fill || "#FFFFFF";
  const stroke = opts.stroke || "#CBD5E1";
  const textFill = opts.textFill || "#334155";
  const w = opts.width || Math.max(48, String(text).length * 6.2 + 16);
  const h = opts.height || 17;
  const left = anchor === "end" ? x - w : anchor === "middle" ? x - w / 2 : x;
  const textX = anchor === "end" ? x - w / 2 : anchor === "middle" ? x : x + w / 2;
  return `<rect x="${left}" y="${y - h + 4}" width="${w}" height="${h}" rx="${Math.min(8, h / 2)}" fill="${fill}" stroke="${stroke}" opacity=".96"/><text x="${textX}" y="${y}" text-anchor="middle" class="tiny" style="fill:${textFill}">${esc(text)}</text>`;
}

function metricSubtitle(rows) {
  const metrics = unique(rows.map(r => r.metric)).map(m => labelMetric(m, rows.find(r => r.metric === m))).join(", ");
  return `5-fold OOF scores with paired tests. Metrics: ${metrics}.`;
}

function requireMeasured(rows, stem) {
  const bad = rows.filter(r => String(r.status || "measured") !== "measured");
  if (bad.length) throw new Error(`${stem} contains non-measured rows: ${JSON.stringify(bad.slice(0, 5))}`);
}

function noMeasuredRowsFigure(stem, title, outDir) {
  const width = 1500, height = 380;
  const body = [
    `<rect width="${width}" height="${height}" fill="#FFFFFF"/>`,
    `<text x="44" y="54" class="title">${esc(title)}</text>`,
    `<text x="44" y="86" class="subtitle">No enabled experiment produced measured supplementary representation-check rows for this strict manuscript build.</text>`,
    `<rect x="44" y="128" width="${width - 88}" height="170" rx="8" fill="#F8FAFC" stroke="#CBD5E1"/>`,
    `<text x="74" y="172" class="panel-title">Strict provenance filter applied</text>`,
    `<text x="74" y="206" class="small">Disabled or optional experiment outputs are excluded from manuscript plot data, even when older CSV files remain in results/tables.</text>`,
    `<text x="74" y="232" class="small">Unsupported combinations are documented in Table S3 rather than drawn from stale disabled-runner artifacts.</text>`,
    `<text x="74" y="266" class="tiny">Regenerate this panel by enabling and rerunning the relevant supplementary experiment before figure rendering.</text>`,
  ];
  return writeAsset(stem, svgShell(width, height, body.join("\n")), outDir);
}

function barFigure(stem, title, csvName, outDir, options = {}) {
  const rows = readCsv(csvName);
  requireMeasured(rows, stem);
  const endpoints = mainEndpointOrder.filter(ep => rows.some(r => r.endpoint === ep));
  const models = ["elastic_net", "XGBoost"].filter(m => rows.some(r => r.model_family === m));
  const reps = unique(rows.map(r => r.representation_family));
  const deltaPanel = Boolean(options.deltaPanel);
  const contrastPanel = Boolean(options.contrastPanel);
  const showSigOnBars = options.showSigOnBars !== false;
  const width = deltaPanel ? 1880 : (contrastPanel ? 2200 : 1650), margin = { left: 210, right: 70, top: 132, bottom: contrastPanel ? 126 : 88 };
  const deltaW = deltaPanel ? 170 : 0;
  const deltaGap = deltaPanel ? 38 : 0;
  const contrastW = contrastPanel ? 310 : 0;
  const contrastGap = contrastPanel ? 34 : 0;
  const facetGap = contrastPanel ? 90 : 70;
  const facetW = (width - margin.left - margin.right - facetGap * (models.length - 1)) / Math.max(models.length, 1);
  const barW = Math.max(320, facetW - deltaW - deltaGap - contrastW - contrastGap);
  const groupH = Math.max(68, reps.length * 18 + 20);
  const height = margin.top + margin.bottom + endpoints.length * groupH;
  const vals = rows.map(r => num(r.primary_score)).filter(Number.isFinite);
  const max = Math.max(0.75, Math.min(1.0, Math.max(...vals, 0.1) + 0.08));
  const body = [
    `<rect width="${width}" height="${height}" fill="#FFFFFF"/>`,
    `<text x="44" y="50" class="title">${esc(title)}</text>`,
    `<text x="44" y="78" class="subtitle">${esc(options.subtitle || metricSubtitle(rows))}</text>`,
  ];
  const contrastRows = contrastPanel
    ? readCsvFile(path.join(manuscriptDir, "canonical", "main_panel_pairwise_tests.csv")).filter(r => r.figure_id === "figure_4" && r.test_status === "tested")
    : [];
  const contrastDefs = [
    ["maf_stack_vs_signatures", "Bio MAF - Sig", "Bio MAF v4 vs mutational signatures"],
    ["sig_maf_vs_signatures", "Sig+Bio MAF - Sig", "Signatures + Bio MAF v4 vs mutational signatures"],
    ["sig_maf_vs_maf_stack", "Sig+Bio MAF - Bio MAF", "Signatures + Bio MAF v4 vs Bio MAF v4"],
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
        const titleBits = [`${labelEndpoint(ep)} | ${labelModel(model)} | ${labelRep(rep)}: ${num(r.primary_score).toFixed(3)}`];
        if (showSigOnBars && Number.isFinite(q)) titleBits.push(`q=${q.toExponential(2)}`);
        body.push(`<rect x="${x0}" y="${y}" width="${w}" height="12" rx="2" fill="${palette[rep] || "#334155"}"><title>${esc(titleBits.join("; "))}</title></rect>`);
        body.push(`<text x="${x0 + w + 7}" y="${y + 11}" class="tiny">${num(r.primary_score).toFixed(2)}${showSigOnBars && sig ? ` ${sig}` : ""}</text>`);
      });
    });
    if (contrastPanel) {
      const cx0 = x0 + barW + contrastGap;
      body.push(`<text x="${cx0}" y="104" class="small">Declared contrasts: Δ metric, q</text>`);
      contrastDefs.forEach(([id, short, long], ci) => {
        const hx = cx0 + ci * 101;
        body.push(textBlock(hx + 42, 122, wrapWords(short, 10), { cls: "tiny", anchor: "middle", lineHeight: 12 }));
        body.push(`<title>${esc(long)}</title>`);
      });
      endpoints.forEach((ep, ei) => {
        const cy = margin.top + ei * groupH + groupH / 2 + 2;
        contrastDefs.forEach(([id, short, long], ci) => {
          const test = contrastRows.find(d => d.endpoint === ep && d.model_family === model && d.comparison_name === id);
          if (!test) return;
          const delta = num(test.delta), qv = num(test.q_value);
          const sig = String(test.significance_label || "");
          const tx = cx0 + ci * 101;
          const fill = Number.isFinite(delta) && delta >= 0 ? "#ECFDF5" : "#FEF2F2";
          const stroke = Number.isFinite(delta) && delta >= 0 ? "#A7F3D0" : "#FECACA";
          const text = `${Number.isFinite(delta) ? (delta >= 0 ? "+" : "") + delta.toFixed(3) : "n/a"}${sig ? ` ${sig}` : ""}`;
          body.push(`<rect x="${tx}" y="${cy - 12}" width="86" height="22" rx="5" fill="${fill}" stroke="${stroke}"><title>${esc(labelEndpoint(ep))} | ${esc(labelModel(model))} | ${long}: delta=${Number.isFinite(delta) ? delta.toFixed(4) : "n/a"}; p=${Number.isFinite(num(test.p_value)) ? num(test.p_value).toExponential(2) : "n/a"}; q=${Number.isFinite(qv) ? qv.toExponential(2) : "n/a"}</title></rect>`);
          body.push(`<text x="${tx + 43}" y="${cy + 4}" text-anchor="middle" class="tiny">${esc(text)}</text>`);
        });
      });
    }
  });
  reps.forEach((rep, i) => {
    const x = 44 + i * 170, y = height - 18;
    body.push(`<rect x="${x}" y="${y - 13}" width="14" height="14" rx="2" fill="${palette[rep] || "#334155"}"/><text x="${x + 21}" y="${y}" class="small">${esc(labelRep(rep))}</text>`);
  });
  return writeAsset(stem, svgShell(width, height, body.join("\n")), outDir);
}

function barFigureV2(stem, title, csvName, outDir, options = {}) {
  const rows = readCsv(csvName);
  requireMeasured(rows, stem);
  const figureId = (stem.match(/^figure_\d+/) || [""])[0];
  const tests = pairwiseTests(figureId);
  const endpoints = mainEndpointOrder.filter(ep => rows.some(r => r.endpoint === ep));
  const models = ["elastic_net", "XGBoost"].filter(m => rows.some(r => r.model_family === m));
  const reps = unique(rows.map(r => r.representation_family));
  const deltaPanel = Boolean(options.deltaPanel);
  const contrastPanel = Boolean(options.contrastPanel);
  const margin = { left: 250, right: 72, top: 154, bottom: 118 };
  const deltaW = deltaPanel ? 292 : 0;
  const deltaGap = deltaPanel ? 42 : 0;
  const contrastW = contrastPanel ? 500 : 0;
  const contrastGap = contrastPanel ? 42 : 0;
  const facetGap = contrastPanel ? 96 : 82;
  const width = contrastPanel ? 2420 : (deltaPanel ? 2120 : 1700);
  const facetW = (width - margin.left - margin.right - facetGap * (models.length - 1)) / Math.max(models.length, 1);
  const barW = Math.max(420, facetW - deltaW - deltaGap - contrastW - contrastGap);
  const groupH = contrastPanel ? Math.max(100, reps.length * 24 + 32) : Math.max(90, reps.length * 24 + 30);
  const height = margin.top + margin.bottom + endpoints.length * groupH;
  const vals = rows.map(r => num(r.primary_score)).filter(Number.isFinite);
  const max = Math.max(0.75, Math.min(1.0, Math.max(...vals, 0.1) + 0.07));
  const dMax = contrastRange(tests.filter(r => deltaPanel ? r.comparison_name === options.deltaComparison : true), options.deltaRange || 0.09);
  const contrastDefs = [
    ["maf_stack_vs_signatures", "Bio MAF - Sig", "Bio MAF v4 minus signatures"],
    ["sig_maf_vs_signatures", "Sig+Bio MAF - Sig", "Signatures + Bio MAF v4 minus signatures"],
    ["sig_maf_vs_maf_stack", "Sig+Bio MAF - Bio MAF", "Signatures + Bio MAF v4 minus Bio MAF v4"],
  ];
  const contrastMax = contrastRange(tests.filter(r => contrastDefs.some(([id]) => id === r.comparison_name)), 0.10);
  const body = [
    `<rect width="${width}" height="${height}" fill="#FFFFFF"/>`,
    `<text x="44" y="50" class="title">${esc(title)}</text>`,
    `<text x="44" y="80" class="subtitle">${esc(options.subtitle || metricSubtitle(rows))}</text>`,
    `<text x="44" y="106" class="tiny">${esc(options.note || "Endpoint labels show metric, task, and sample count. Stars are paired-test FDR markers.")}</text>`,
  ];

  endpoints.forEach((ep, i) => {
    const rowTop = margin.top + i * groupH;
    if (i % 2 === 0) body.push(`<rect x="36" y="${rowTop - 8}" width="${width - 72}" height="${groupH}" class="row-band"/>`);
    const row = rows.find(r => r.endpoint === ep);
    body.push(textBlock(52, rowTop + 30, wrapWords(labelEndpoint(ep, row), 22), { cls: "label", lineHeight: 16 }));
    body.push(endpointContext(row, 52, rowTop + 58));
  });

  models.forEach((model, mi) => {
    const x0 = margin.left + mi * (facetW + facetGap);
    const modelTag = model === "elastic_net" ? "EN" : "XGB";
    const panelLabel = model === "XGBoost" && endpoints.includes("OS") ? "XGBoost / Cox PH for OS" : labelModel(model);
    body.push(`<text x="${x0}" y="126" class="panel-title">${esc(panelLabel)}</text>`);
    body.push(chip(x0 + (model === "XGBoost" && endpoints.includes("OS") ? 278 : 112), 124, model === "XGBoost" && endpoints.includes("OS") ? "XGB+COX" : modelTag, { width: model === "XGBoost" && endpoints.includes("OS") ? 74 : 44, fill: "#E0F2FE", stroke: "#BAE6FD" }));
    [0, 0.25, 0.5, 0.75, 1].forEach(t => {
      if (t <= max + 1e-9) {
        const x = x0 + t / max * barW;
        body.push(`<line x1="${x}" y1="${margin.top - 24}" x2="${x}" y2="${height - margin.bottom + 12}" class="grid"/>`);
        body.push(`<text x="${x}" y="${height - 62}" text-anchor="middle" class="tiny">${t.toFixed(2)}</text>`);
      }
    });
    body.push(`<text x="${x0 + barW / 2}" y="${height - 36}" text-anchor="middle" class="small">OOF primary metric</text>`);
    endpoints.forEach((ep, ei) => {
      if (ep === "OS" && model === "elastic_net") {
        const noteY = margin.top + ei * groupH + groupH / 2 + 4;
        body.push(textBlock(x0 + 18, noteY - 8, ["Survival uses Cox PH", "shown in the right panel"], { cls: "tiny", lineHeight: 13 }));
        return;
      }
      const lookupModel = ep === "OS" && rows.some(r => r.endpoint === ep && r.model_family === "cox_ph") ? "cox_ph" : model;
      const baseY = margin.top + ei * groupH + 18;
      const winner = rows
        .filter(d => d.endpoint === ep && d.model_family === lookupModel)
        .sort((a, b) => num(b.primary_score) - num(a.primary_score))[0]?.representation_family;
      reps.forEach((rep, ri) => {
        let r = rows.find(d => d.endpoint === ep && d.model_family === lookupModel && d.representation_family === rep);
        if (!r && rep === "MuAt_style_attention_MIL") {
          r = rows.find(d => d.endpoint === ep && d.representation_family === rep);
        }
        if (!r) throw new Error(`${stem} missing ${ep}/${model}/${rep}`);
        const y = baseY + ri * 24;
        const w = Math.max(2, num(r.primary_score) / max * barW);
        const stroke = rep === winner ? "#111827" : "none";
        const strokeWidth = rep === winner ? 1.6 : 0;
        const titleBits = [
          `${labelEndpoint(ep, r)} | ${labelModel(model, r)} | ${labelRep(rep, r)}: ${num(r.primary_score).toFixed(3)}`,
          `n=${Math.round(num(r.n_samples)).toLocaleString("en-US")}`,
          `${Math.round(num(r.n_features)).toLocaleString("en-US")} features`,
        ];
        body.push(`<rect x="${x0}" y="${y}" width="${w}" height="15" rx="3" fill="${palette[rep] || "#334155"}" stroke="${stroke}" stroke-width="${strokeWidth}"><title>${esc(titleBits.join("; "))}</title></rect>`);
        body.push(`<text x="${x0 + w + 8}" y="${y + 13}" class="tiny">${formatScore(r.primary_score)}</text>`);
      });
    });

    if (deltaPanel) {
      const dx0 = x0 + barW + deltaGap;
      const zero = dx0 + deltaW / 2;
      body.push(`<text x="${zero}" y="126" text-anchor="middle" class="small">${esc(options.deltaTitle || "Candidate - baseline")}</text>`);
      body.push(`<line x1="${zero}" y1="${margin.top - 24}" x2="${zero}" y2="${height - margin.bottom + 12}" stroke="#64748B" stroke-width="1.4"/>`);
      body.push(`<rect x="${zero - 5}" y="${margin.top - 20}" width="10" height="${height - margin.bottom - margin.top + 32}" fill="#64748B" opacity=".06"/>`);
      [-dMax, 0, dMax].forEach(t => {
        const x = zero + (t / dMax) * (deltaW / 2);
        body.push(`<line x1="${x}" y1="${height - margin.bottom + 20}" x2="${x}" y2="${height - margin.bottom + 27}" stroke="#64748B"/>`);
        body.push(`<text x="${x}" y="${height - 62}" text-anchor="middle" class="tiny">${formatSigned(t, 2)}</text>`);
      });
      body.push(`<text x="${zero}" y="${height - 36}" text-anchor="middle" class="small">Paired delta</text>`);
      endpoints.forEach((ep, ei) => {
        if (ep === "OS" && model === "elastic_net") return;
        const testModel = ep === "OS" && rows.some(r => r.endpoint === ep && r.model_family === "cox_ph") ? "cox_ph" : model;
        const test = findTest(tests, ep, testModel, options.deltaComparison);
        if (!test) return;
        const delta = num(test.delta);
        const low = num(test.ci_low);
        const high = num(test.ci_high);
        const cy = margin.top + ei * groupH + groupH / 2 + 2;
        const glyphY = cy - 8;
        const labelY = cy + 17;
        const x = zero + clamp(delta / dMax, -1, 1) * (deltaW / 2);
        const fill = delta >= 0 ? gainColor : lossColor;
        if (Number.isFinite(low) && Number.isFinite(high)) {
          const x1 = zero + clamp(low / dMax, -1, 1) * (deltaW / 2);
          const x2 = zero + clamp(high / dMax, -1, 1) * (deltaW / 2);
          body.push(`<line x1="${x1}" y1="${glyphY}" x2="${x2}" y2="${glyphY}" stroke="${fill}" stroke-width="2.2" opacity=".75"><title>${esc(`95% interval ${formatSigned(low)} to ${formatSigned(high)}`)}</title></line>`);
          body.push(`<line x1="${x1}" y1="${glyphY - 5}" x2="${x1}" y2="${glyphY + 5}" stroke="${fill}" stroke-width="1.4"/>`);
          body.push(`<line x1="${x2}" y1="${glyphY - 5}" x2="${x2}" y2="${glyphY + 5}" stroke="${fill}" stroke-width="1.4"/>`);
        } else {
          body.push(`<line x1="${zero}" y1="${glyphY}" x2="${x}" y2="${glyphY}" stroke="${fill}" stroke-width="2.2" opacity=".75"/>`);
        }
        body.push(`<circle cx="${x}" cy="${glyphY}" r="5" fill="${fill}"><title>${esc(`${labelEndpoint(ep)} | ${labelModel(model)} | delta=${formatSigned(delta)}; ${formatQ(test.q_value_figure || test.q_value)}`)}</title></circle>`);
        const labelText = `${formatSigned(delta)}${test.significance_label ? ` ${test.significance_label}` : ""}`;
        const labelX = delta >= 0 ? dx0 + deltaW - 4 : dx0 + 4;
        body.push(textPill(labelX, labelY, labelText, {
          anchor: delta >= 0 ? "end" : "start",
          fill: delta >= 0 ? "#ECFDF5" : "#FEF2F2",
          stroke: delta >= 0 ? "#A7F3D0" : "#FECACA",
          textFill: fill,
        }));
      });
    }

    if (contrastPanel) {
      const cx0 = x0 + barW + contrastGap;
      const colW = contrastW / contrastDefs.length;
      body.push(`<text x="${cx0}" y="112" class="small">Pairwise deltas; CI if bootstrapped</text>`);
      contrastDefs.forEach(([id, short, long], ci) => {
        const colX = cx0 + ci * colW;
        const center = colX + colW / 2;
        body.push(textBlock(center, 132, wrapWords(short, 14), { cls: "tiny", anchor: "middle", lineHeight: 12 }));
        body.push(`<line x1="${center}" y1="${margin.top - 12}" x2="${center}" y2="${height - margin.bottom + 8}" stroke="#CBD5E1" stroke-width="1.1"><title>${esc(long)}</title></line>`);
      });
      endpoints.forEach((ep, ei) => {
        if (ep === "OS" && model === "elastic_net") return;
        const cy = margin.top + ei * groupH + groupH / 2 + 1;
        const glyphY = cy - 10;
        const labelY = cy + 17;
        contrastDefs.forEach(([id, short, long], ci) => {
          const testModel = ep === "OS" && rows.some(r => r.endpoint === ep && r.model_family === "cox_ph") ? "cox_ph" : model;
          const test = findTest(tests, ep, testModel, id);
          if (!test) return;
          const colX = cx0 + ci * colW;
          const center = colX + colW / 2;
          const half = colW * 0.36;
          const delta = num(test.delta);
          const low = num(test.ci_low);
          const high = num(test.ci_high);
          const fill = delta >= 0 ? gainColor : lossColor;
          const x = center + clamp(delta / contrastMax, -1, 1) * half;
          body.push(`<rect x="${colX + 8}" y="${cy - 21}" width="${colW - 16}" height="42" rx="5" fill="${delta >= 0 ? "#ECFDF5" : "#FEF2F2"}" stroke="${delta >= 0 ? "#A7F3D0" : "#FECACA"}"/>`);
          if (Number.isFinite(low) && Number.isFinite(high)) {
            const x1 = center + clamp(low / contrastMax, -1, 1) * half;
            const x2 = center + clamp(high / contrastMax, -1, 1) * half;
            body.push(`<line x1="${x1}" y1="${glyphY}" x2="${x2}" y2="${glyphY}" stroke="${fill}" stroke-width="1.8"/>`);
          }
          body.push(`<line x1="${center}" y1="${cy - 17}" x2="${center}" y2="${cy + 17}" stroke="#94A3B8" stroke-width="1"/>`);
          body.push(`<line x1="${center}" y1="${glyphY}" x2="${x}" y2="${glyphY}" stroke="${fill}" stroke-width="3"/>`);
          body.push(`<circle cx="${x}" cy="${glyphY}" r="3.8" fill="${fill}"><title>${esc(`${labelEndpoint(ep)} | ${labelModel(model)} | ${long}: delta=${formatSigned(delta)}; ${formatQ(test.q_value_figure || test.q_value)}`)}</title></circle>`);
          body.push(textPill(center, labelY, `${formatSigned(delta)}${test.significance_label ? ` ${test.significance_label}` : ""}`, {
            width: 72,
            fill: delta >= 0 ? "#ECFDF5" : "#FEF2F2",
            stroke: delta >= 0 ? "#A7F3D0" : "#FECACA",
            textFill: fill,
          }));
        });
      });
    }
  });

  const legendY = height - 20;
  let legendX = 52;
  reps.forEach(rep => {
    const featureText = repFeatureSummary(rows, rep);
    const label = featureText ? `${labelRep(rep)} (${featureText})` : labelRep(rep);
    body.push(`<rect x="${legendX}" y="${legendY - 14}" width="14" height="14" rx="2" fill="${palette[rep] || "#334155"}"/><text x="${legendX + 22}" y="${legendY - 2}" class="small">${esc(label)}</text>`);
    legendX += Math.max(170, label.length * 7 + 42);
  });
  body.push(significanceLegend(width - 430, legendY - 2));
  if (deltaPanel) body.push(signedLegend(width - 720, 106, "positive favors candidate", "negative favors baseline"));
  if (contrastPanel) body.push(signedLegend(width - 820, 82, "positive favors left label", "negative favors baseline"));
  return writeAsset(stem, svgShell(width, height, body.join("\n")), outDir);
}

function muatComparatorFigure(stem, title, csvName, outDir) {
  const rows = readCsv(csvName);
  requireMeasured(rows, stem);
  const endpoints = mainEndpointsPresent(rows);
  const width = 1260;
  const margin = { left: 236, right: 44, top: 142, bottom: 108 };
  const rowH = 88;
  const barW = 520;
  const deltaGap = 50;
  const deltaW = 300;
  const height = margin.top + endpoints.length * rowH + margin.bottom;
  const values = [];
  endpoints.forEach(ep => {
    [bestTabular(rows, ep), benchmarkTabular(rows, ep), muatForEndpoint(rows, ep)].forEach(r => {
      const v = r ? num(r.primary_score) : NaN;
      if (Number.isFinite(v)) values.push(v);
    });
  });
  const max = Math.max(0.75, Math.min(1, Math.max(...values, 0.1) + 0.07));
  const deltas = endpoints.map(ep => {
    const best = bestTabular(rows, ep);
    const muat = muatForEndpoint(rows, ep);
    return best && muat ? num(muat.primary_score) - num(best.primary_score) : NaN;
  }).filter(Number.isFinite);
  const deltaMax = Math.max(0.08, Math.min(0.35, Math.max(...deltas.map(Math.abs), 0.01) * 1.25));
  const deltaX = margin.left + barW + deltaGap;
  const zero = deltaX + deltaW / 2;
  const body = [
    `<rect width="${width}" height="${height}" fill="#FFFFFF"/>`,
    `<text x="40" y="48" class="title">${esc(title)}</text>`,
    textBlock(40, 78, wrapWords("Direct event-bag model, compared with the best tabular row and the Signatures + MAF default on pooled OOF metrics.", 100), { cls: "subtitle", lineHeight: 18 }),
    `<text x="${margin.left}" y="124" class="small">OOF primary metric</text>`,
    `<text x="${zero}" y="124" text-anchor="middle" class="small">MuAt-compatible minus best tabular</text>`,
  ];
  [0, 0.25, 0.5, 0.75, 1].forEach(t => {
    if (t <= max + 1e-9) {
      const x = margin.left + t / max * barW;
      body.push(`<line x1="${x}" y1="${margin.top - 18}" x2="${x}" y2="${height - margin.bottom + 14}" class="grid"/>`);
      body.push(`<text x="${x}" y="${height - 58}" text-anchor="middle" class="tiny">${t.toFixed(2)}</text>`);
    }
  });
  body.push(`<line x1="${zero}" y1="${margin.top - 18}" x2="${zero}" y2="${height - margin.bottom + 14}" stroke="#64748B" stroke-width="1.5"/>`);
  [-deltaMax, 0, deltaMax].forEach(t => {
    const x = zero + (t / deltaMax) * (deltaW / 2);
    body.push(`<line x1="${x}" y1="${height - margin.bottom + 21}" x2="${x}" y2="${height - margin.bottom + 28}" stroke="#64748B"/>`);
    body.push(`<text x="${x}" y="${height - 58}" text-anchor="middle" class="tiny">${formatSigned(t, 2)}</text>`);
  });

  endpoints.forEach((ep, i) => {
    const rowTop = margin.top + i * rowH;
    if (i % 2 === 0) body.push(`<rect x="36" y="${rowTop - 8}" width="${width - 72}" height="${rowH - 4}" class="row-band"/>`);
    const context = rows.find(r => r.endpoint === ep) || {};
    body.push(textBlock(52, rowTop + 29, wrapWords(labelEndpoint(ep, context), 24), { cls: "label", lineHeight: 15 }));
    body.push(endpointContext(context, 52, rowTop + 54));
    const best = bestTabular(rows, ep);
    const defaultTabular = benchmarkTabular(rows, ep);
    const muat = muatForEndpoint(rows, ep);
    const series = [
      { key: "best", label: `Best tabular: ${rowTitle(best)}`, short: "Best tabular", row: best, color: "#111827" },
      { key: "default", label: `Default tabular: ${rowTitle(defaultTabular)}`, short: ep === "OS" ? "Sig+MAF Cox" : "Sig+MAF XGB", row: defaultTabular, color: palette.signatures_plus_MAF_stack },
      { key: "muat", label: "MuAt-compatible event-bag model", short: "MuAt-compatible", row: muat, color: palette.MuAt_style_attention_MIL },
    ];
    series.forEach((item, si) => {
      if (!item.row) return;
      const v = num(item.row.primary_score);
      const y = rowTop + 16 + si * 20;
      const w = Math.max(2, v / max * barW);
      body.push(`<rect x="${margin.left}" y="${y}" width="${w}" height="14" rx="3" fill="${item.color}"><title>${esc(`${labelEndpoint(ep)} | ${item.label}: ${v.toFixed(3)}`)}</title></rect>`);
      body.push(`<text x="${margin.left + w + 7}" y="${y + 12}" class="tiny">${formatScore(v)}</text>`);
    });
    if (best && muat) {
      const delta = num(muat.primary_score) - num(best.primary_score);
      const color = delta >= 0 ? gainColor : lossColor;
      const x = zero + clamp(delta / deltaMax, -1, 1) * (deltaW / 2);
      const y = rowTop + rowH / 2 + 3;
      body.push(`<line x1="${zero}" y1="${y}" x2="${x}" y2="${y}" stroke="${color}" stroke-width="3"/>`);
      body.push(`<circle cx="${x}" cy="${y}" r="5" fill="${color}"><title>${esc(`${labelEndpoint(ep)} | MuAt-compatible minus best tabular: ${formatSigned(delta)}`)}</title></circle>`);
      body.push(textPill(delta >= 0 ? deltaX + deltaW - 4 : deltaX + 4, y + 23, formatSigned(delta), {
        anchor: delta >= 0 ? "end" : "start",
        fill: delta >= 0 ? "#ECFDF5" : "#FEF2F2",
        stroke: delta >= 0 ? "#A7F3D0" : "#FECACA",
        textFill: color,
      }));
    }
  });
  const legendY = height - 22;
  const legend = [
    ["Best tabular", "#111827"],
    ["Sig + MAF default", palette.signatures_plus_MAF_stack],
    ["MuAt-compatible", palette.MuAt_style_attention_MIL],
  ];
  let lx = 52;
  legend.forEach(([label, color]) => {
    body.push(`<rect x="${lx}" y="${legendY - 14}" width="14" height="14" rx="2" fill="${color}"/><text x="${lx + 22}" y="${legendY - 2}" class="small">${esc(label)}</text>`);
    lx += Math.max(160, label.length * 7 + 42);
  });
  const sx = 748;
  body.push(`<line x1="${sx}" y1="${legendY - 6}" x2="${sx + 26}" y2="${legendY - 6}" stroke="${gainColor}" stroke-width="3"/>`);
  body.push(`<text x="${sx + 34}" y="${legendY - 2}" class="tiny">MuAt higher</text>`);
  body.push(`<line x1="${sx + 140}" y1="${legendY - 6}" x2="${sx + 166}" y2="${legendY - 6}" stroke="${lossColor}" stroke-width="3"/>`);
  body.push(`<text x="${sx + 174}" y="${legendY - 2}" class="tiny">MuAt lower than best tabular</text>`);
  return writeAsset(stem, svgShell(width, height, body.join("\n")), outDir);
}

function heatFacetFigure(stem, title, csvName, outDir) {
  const rows = readCsv(csvName);
  requireMeasured(rows, stem);
  const endpoints = ["damage_class", "HRD_Score", "hrd_binary_24", "hrd_binary_33", "hrd_binary_42", "cancer_type_top20", "OS", "PFI"].filter(ep => rows.some(r => r.endpoint === ep));
  const reps = ["burden_only", "signatures_only", "MAF_stack_only", "signatures_plus_MAF_stack"];
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

function heatFacetFigureV2(stem, title, csvName, outDir) {
  const rows = readCsv(csvName);
  requireMeasured(rows, stem);
  const endpoints = mainEndpointOrder.filter(ep => rows.some(r => r.endpoint === ep));
  const reps = ["burden_only", "signatures_only", "MAF_stack_only", "signatures_plus_MAF_stack", "MuAt_style_attention_MIL"];
  const models = ["elastic_net", "XGBoost"];
  const tests = pairwiseTests("figure_5");
  const comparisonForRep = {
    signatures_only: "signatures_vs_burden",
    MAF_stack_only: "maf_stack_vs_signatures",
    signatures_plus_MAF_stack: "sig_maf_vs_signatures",
  };
  const cellW = 196, cellH = 74, left = 315, top = 212, facetGap = 70;
  const panelH = 58 + endpoints.length * cellH;
  const width = left + reps.length * cellW + 260;
  const height = top + models.length * panelH + facetGap + 138;
  const vals = rows.map(r => num(r.primary_score)).filter(Number.isFinite);
  const min = Math.min(...vals), max = Math.max(...vals);
  const body = [
    `<rect width="${width}" height="${height}" fill="#FFFFFF"/>`,
    `<text x="44" y="50" class="title">${esc(title)}</text>`,
    `<text x="44" y="82" class="subtitle">Absolute 5-fold OOF scores with paired deltas. Non-signature columns show delta versus signatures; signatures show delta versus burden.</text>`,
    `<text x="44" y="108" class="tiny">Cell outline marks the best representation within each endpoint and model. Endpoint labels show metric, task, and n.</text>`,
  ];
  body.push(`<text x="${left + 610}" y="132" class="tiny">feature richness increases left to right</text>`);
  body.push(`<line x1="${left + 8}" y1="146" x2="${left + reps.length * cellW - 34}" y2="146" stroke="#94A3B8" stroke-width="1.4" marker-end="url(#arrow)"/>`);
  reps.forEach((rep, j) => {
    const cx = left + j * cellW + (cellW - 10) / 2;
    body.push(textBlock(cx, 166, wrapWords(labelRep(rep), 15), { cls: "label", anchor: "middle", lineHeight: 16 }));
  });
  models.forEach((model, mi) => {
    const y0 = top + mi * (panelH + facetGap);
    body.push(`<text x="44" y="${y0 - 18}" class="panel-title">${esc(labelModel(model))}</text>`);
    body.push(chip(150, y0 - 20, model === "elastic_net" ? "EN" : "XGB", { width: 44, fill: "#E0F2FE", stroke: "#BAE6FD" }));
    endpoints.forEach((ep, i) => {
      const lookupModel = ep === "OS" && rows.some(r => r.endpoint === ep && r.model_family === "cox_ph") ? "cox_ph" : model;
      const rowY = y0 + i * cellH;
      if (i % 2 === 0) body.push(`<rect x="36" y="${rowY - 5}" width="${width - 72}" height="${cellH - 2}" class="row-band"/>`);
      const contextRow = rows.find(r => r.endpoint === ep && r.model_family === lookupModel) || rows.find(r => r.endpoint === ep);
      body.push(textBlock(44, rowY + 27, wrapWords(labelEndpoint(ep, contextRow), 25), { cls: "label", lineHeight: 15 }));
      body.push(endpointContext(contextRow, 44, rowY + 52));
      const winner = rows
        .filter(d => d.endpoint === ep && d.model_family === lookupModel)
        .sort((a, b) => num(b.primary_score) - num(a.primary_score))[0]?.representation_family;
      reps.forEach((rep, j) => {
        let r = rows.find(d => d.endpoint === ep && d.model_family === lookupModel && d.representation_family === rep);
        if (!r && rep === "MuAt_style_attention_MIL") {
          r = rows.find(d => d.endpoint === ep && d.representation_family === rep);
        }
        if (!r) {
          const x = left + j * cellW;
          const y = rowY;
          body.push(`<rect x="${x}" y="${y}" width="${cellW - 10}" height="${cellH - 12}" rx="4" fill="#F8FAFC" stroke="#E2E8F0" stroke-width="1"><title>${esc(labelEndpoint(ep))} | ${esc(labelModel(lookupModel))} | ${esc(labelRep(rep))}: not applicable</title></rect>`);
          body.push(`<text x="${x + (cellW - 10) / 2}" y="${y + 36}" text-anchor="middle" class="tiny" style="fill:#94A3B8">n/a</text>`);
          return;
        }
        const v = num(r.primary_score);
        const x = left + j * cellW;
        const y = rowY;
        const fill = heatColor(v, min, max);
        const darkCell = (v - min) / Math.max(max - min, 1e-9) > 0.62;
        const sig = String(r.significance_label || "");
        const comp = comparisonForRep[rep];
        const test = comp ? findTest(tests, ep, model, comp) : null;
        const delta = test ? num(test.delta) : NaN;
        const stroke = rep === winner ? "#111827" : "#FFFFFF";
        const strokeWidth = rep === winner ? 2.2 : 1;
        body.push(`<rect x="${x}" y="${y}" width="${cellW - 10}" height="${cellH - 12}" rx="4" fill="${fill}" stroke="${stroke}" stroke-width="${strokeWidth}"><title>${esc(labelEndpoint(ep))} | ${esc(labelModel(model))} | ${esc(labelRep(rep))}: ${v.toFixed(3)}${test ? `; delta=${formatSigned(delta)}; ${formatQ(test.q_value_figure || test.q_value)}` : ""}</title></rect>`);
        if (sig) {
          body.push(`<rect x="${x + cellW - 48}" y="${y + 6}" width="32" height="17" rx="8" fill="#FFFFFF" opacity=".82"/>`);
          body.push(`<text x="${x + cellW - 32}" y="${y + 19}" text-anchor="middle" class="sig">${esc(sig)}</text>`);
        }
        body.push(`<text x="${x + (cellW - 10) / 2}" y="${y + 36}" text-anchor="middle" class="${darkCell ? "cell-value-light" : "cell-value"}">${v.toFixed(2)}</text>`);
        if (Number.isFinite(delta)) {
          const color = delta >= 0 ? gainColor : lossColor;
          body.push(`<text x="${x + (cellW - 10) / 2}" y="${y + 59}" text-anchor="middle" class="tiny" style="fill:${color}">${formatSigned(delta)}</text>`);
        } else if (rep === "burden_only") {
          body.push(`<text x="${x + (cellW - 10) / 2}" y="${y + 59}" text-anchor="middle" class="tiny" style="fill:${darkCell ? "#FFFFFF" : "#64748B"};opacity:.82">reference</text>`);
        }
      });
    });
  });
  const legendY = height - 72;
  const legendX = left;
  body.push(`<rect x="${legendX}" y="${legendY - 12}" width="280" height="14" fill="url(#heat)"/>`);
  body.push(`<text x="${legendX}" y="${legendY + 22}" class="tiny">${min.toFixed(2)}</text>`);
  body.push(`<text x="${legendX + 140}" y="${legendY + 22}" text-anchor="middle" class="tiny">OOF score</text>`);
  body.push(`<text x="${legendX + 280}" y="${legendY + 22}" text-anchor="end" class="tiny">${max.toFixed(2)}</text>`);
  body.push(signedLegend(legendX + 380, legendY + 1, "positive delta", "negative delta"));
  body.push(significanceLegend(legendX + 820, legendY + 1));
  return writeAsset(stem, svgShell(width, height, body.join("\n")), outDir);
}

function heatFacetFigureV3(stem, title, csvName, outDir) {
  const rows = readCsv(csvName);
  requireMeasured(rows, stem);
  const endpoints = mainEndpointsPresent(rows);
  const cellW = 126, cellH = 76;
  const left = 310, top = 214, bottom = 128, groupGap = 44;
  const group1X = left;
  const group2X = group1X + tabularReps.length * cellW + groupGap;
  const muatX = group2X + tabularReps.length * cellW + groupGap;
  const muatW = 188;
  const width = muatX + muatW + 70;
  const height = top + endpoints.length * cellH + bottom;
  const visibleRows = [];
  endpoints.forEach(ep => {
    tabularReps.forEach(rep => {
      if (ep !== "OS") {
        const en = rows.find(r => r.endpoint === ep && r.model_family === "elastic_net" && r.representation_family === rep);
        if (en) visibleRows.push(en);
      }
      const model = ep === "OS" ? "cox_ph" : "XGBoost";
      const xgb = rows.find(r => r.endpoint === ep && r.model_family === model && r.representation_family === rep);
      if (xgb) visibleRows.push(xgb);
    });
    const muat = muatForEndpoint(rows, ep);
    if (muat) visibleRows.push(muat);
  });
  const vals = visibleRows.map(r => num(r.primary_score)).filter(Number.isFinite);
  const min = Math.min(...vals), max = Math.max(...vals);
  const body = [
    `<rect width="${width}" height="${height}" fill="#FFFFFF"/>`,
    `<text x="44" y="50" class="title">${esc(title)}</text>`,
    `<text x="44" y="82" class="subtitle">Absolute canonical 5-fold OOF scores. MuAt-compatible is separated as a direct event-bag model, not nested under Elastic net or XGBoost.</text>`,
    `<text x="44" y="108" class="tiny">Black outline marks the best displayed score within each endpoint. OS is evaluated with Cox PH; Elastic net cells are intentionally not applicable for survival.</text>`,
  ];
  const header = (x, w, titleText, subtitleText) => {
    body.push(`<rect x="${x - 10}" y="128" width="${w + 20}" height="64" rx="7" fill="#F8FAFC" stroke="#E2E8F0"/>`);
    body.push(`<text x="${x + w / 2}" y="153" text-anchor="middle" class="panel-title">${esc(titleText)}</text>`);
    body.push(`<text x="${x + w / 2}" y="174" text-anchor="middle" class="tiny">${esc(subtitleText)}</text>`);
  };
  const shortRep = {
    burden_only: "Burden",
    signatures_only: "Signatures",
    MAF_stack_only: "Bio MAF v4",
    signatures_plus_MAF_stack: "Sig + Bio MAF",
  };
  header(group1X, tabularReps.length * cellW - 10, "Tabular linear model", "Elastic net; non-survival endpoints");
  header(group2X, tabularReps.length * cellW - 10, "Tabular nonlinear / survival", "XGBoost; Cox PH for OS");
  header(muatX, muatW - 10, "Direct event-set model", "MuAt-compatible");
  tabularReps.forEach((rep, j) => {
    const lx1 = group1X + j * cellW + (cellW - 10) / 2;
    const lx2 = group2X + j * cellW + (cellW - 10) / 2;
    body.push(textBlock(lx1, 204, [shortRep[rep] || labelRep(rep)], { cls: "tiny", anchor: "middle", lineHeight: 12 }));
    body.push(textBlock(lx2, 204, [shortRep[rep] || labelRep(rep)], { cls: "tiny", anchor: "middle", lineHeight: 12 }));
  });
  body.push(textBlock(muatX + (muatW - 10) / 2, 204, ["MuAt-compatible"], { cls: "tiny", anchor: "middle", lineHeight: 12 }));

  const drawCell = (row, x, y, w, h, winner, opts = {}) => {
    if (!row) {
      body.push(`<rect x="${x}" y="${y}" width="${w}" height="${h}" rx="4" fill="#F8FAFC" stroke="#E2E8F0"/>`);
      body.push(`<text x="${x + w / 2}" y="${y + h / 2 - 2}" text-anchor="middle" class="tiny" style="fill:#94A3B8">${esc(opts.naText || "n/a")}</text>`);
      return;
    }
    const v = num(row.primary_score);
    const fill = heatColor(v, min, max);
    const darkCell = (v - min) / Math.max(max - min, 1e-9) > 0.62;
    const isWinner = winner && row.representation_family === winner.representation_family && row.model_family === winner.model_family;
    body.push(`<rect x="${x}" y="${y}" width="${w}" height="${h}" rx="4" fill="${fill}" stroke="${isWinner ? "#111827" : "#FFFFFF"}" stroke-width="${isWinner ? 2.4 : 1}"><title>${esc(`${labelEndpoint(row.endpoint, row)} | ${rowTitle(row)}: ${v.toFixed(3)}`)}</title></rect>`);
    body.push(`<text x="${x + w / 2}" y="${y + 32}" text-anchor="middle" class="${darkCell ? "cell-value-light" : "cell-value"}">${formatScore(v)}</text>`);
    body.push(`<text x="${x + w / 2}" y="${y + 52}" text-anchor="middle" class="tiny" style="fill:${darkCell ? "#FFFFFF" : "#64748B"}">${esc(labelModel(row.model_family, row))}</text>`);
  };

  endpoints.forEach((ep, i) => {
    const rowY = top + i * cellH;
    if (i % 2 === 0) body.push(`<rect x="36" y="${rowY - 5}" width="${width - 72}" height="${cellH - 3}" class="row-band"/>`);
    const contextRow = rows.find(r => r.endpoint === ep) || {};
    body.push(textBlock(44, rowY + 27, wrapWords(labelEndpoint(ep, contextRow), 25), { cls: "label", lineHeight: 15 }));
    body.push(endpointContext(contextRow, 44, rowY + 52));
    const rowCandidates = [];
    if (ep !== "OS") {
      tabularReps.forEach(rep => {
        const row = rows.find(r => r.endpoint === ep && r.model_family === "elastic_net" && r.representation_family === rep);
        if (row) rowCandidates.push(row);
      });
    }
    const model2 = ep === "OS" ? "cox_ph" : "XGBoost";
    tabularReps.forEach(rep => {
      const row = rows.find(r => r.endpoint === ep && r.model_family === model2 && r.representation_family === rep);
      if (row) rowCandidates.push(row);
    });
    const muat = muatForEndpoint(rows, ep);
    if (muat) rowCandidates.push(muat);
    rowCandidates.sort((a, b) => num(b.primary_score) - num(a.primary_score));
    const winner = rowCandidates[0] || null;
    tabularReps.forEach((rep, j) => {
      const en = ep === "OS" ? null : rows.find(r => r.endpoint === ep && r.model_family === "elastic_net" && r.representation_family === rep);
      drawCell(en, group1X + j * cellW, rowY, cellW - 10, cellH - 12, winner, { naText: "Cox only" });
      const xgb = rows.find(r => r.endpoint === ep && r.model_family === model2 && r.representation_family === rep);
      drawCell(xgb, group2X + j * cellW, rowY, cellW - 10, cellH - 12, winner);
    });
    drawCell(muat, muatX, rowY, muatW - 10, cellH - 12, winner);
    const best = bestTabular(rows, ep);
    if (best && muat) {
      const delta = num(muat.primary_score) - num(best.primary_score);
      const color = delta >= 0 ? gainColor : lossColor;
      body.push(`<text x="${muatX + (muatW - 10) / 2}" y="${rowY + 64}" text-anchor="middle" class="tiny" style="fill:${color}">vs best tabular ${formatSigned(delta)}</text>`);
    }
  });
  const legendY = height - 74;
  const legendX = left;
  body.push(`<rect x="${legendX}" y="${legendY - 12}" width="280" height="14" fill="url(#heat)"/>`);
  body.push(`<text x="${legendX}" y="${legendY + 22}" class="tiny">${min.toFixed(2)}</text>`);
  body.push(`<text x="${legendX + 140}" y="${legendY + 22}" text-anchor="middle" class="tiny">OOF score</text>`);
  body.push(`<text x="${legendX + 280}" y="${legendY + 22}" text-anchor="end" class="tiny">${max.toFixed(2)}</text>`);
  body.push(`<rect x="${legendX + 380}" y="${legendY - 16}" width="22" height="18" fill="#F8FAFC" stroke="#E2E8F0"/><text x="${legendX + 412}" y="${legendY - 2}" class="small">not applicable</text>`);
  body.push(`<rect x="${legendX + 560}" y="${legendY - 17}" width="22" height="18" fill="#FFFFFF" stroke="#111827" stroke-width="2.4"/><text x="${legendX + 592}" y="${legendY - 2}" class="small">best within endpoint</text>`);
  return writeAsset(stem, svgShell(width, height, body.join("\n")), outDir);
}

function calibrationFigure(stem, title, csvName, outDir) {
  const rows = readCsv(csvName);
  if (!rows.length) throw new Error(`${stem} has no calibration bins`);
  const endpoints = ["damage_class", "hrd_binary_24", "hrd_binary_33", "hrd_binary_42", "cancer_type_top20"].filter(ep => rows.some(r => r.endpoint === ep));
  const width = 1680, panelW = 455, panelH = 372;
  const columns = 3, startX = 132, startY = 202, colGap = 512, rowGap = 430;
  const height = startY + Math.max(0, Math.ceil(endpoints.length / columns) - 1) * rowGap + panelH + 120;
  const origins = endpoints.map((_, i) => [startX + (i % columns) * colGap, startY + Math.floor(i / columns) * rowGap]);
  const body = [
    `<rect width="${width}" height="${height}" fill="#FFFFFF"/>`,
    `<text x="44" y="50" class="title">${esc(title)}</text>`,
    `<text x="44" y="78" class="subtitle">Canonical 5-fold OOF reliability for Signatures + Bio MAF v4 with XGBoost. Multiclass panels plot model confidence against correctness.</text>`,
    textBlock(44, 108, [
      "Circle area reflects bin sample count; dashed line is perfect calibration.",
      "Empty probability columns are excluded before multiclass confidence is computed.",
    ], { cls: "tiny", lineHeight: 14 }),
  ];
  endpoints.forEach((ep, i) => {
    const [x0, y0] = origins[i];
    const plotW = panelW - 72, plotH = panelH - 112;
    const epRows = rows.filter(r => r.endpoint === ep);
    const mode = String(epRows[0]?.calibration_mode || "");
    const yLabel = String(epRows[0]?.calibration_y_label || (mode.includes("multiclass") ? "Observed accuracy" : "Observed frequency"));
    const ece = num(epRows[0]?.endpoint_ece);
    const nTotal = num(epRows[0]?.n_total) || epRows.reduce((acc, r) => acc + num(r.n_samples), 0);
    const nClasses = num(epRows[0]?.n_classes);
    body.push(`<text x="${x0}" y="${y0 - 50}" class="panel-title">${esc(labelEndpoint(ep))}</text>`);
    const meta = [`n=${formatInt(nTotal)}`];
    if (nClasses > 2) meta.push(`${formatInt(nClasses)} classes`);
    if (Number.isFinite(ece)) meta.push(`ECE=${ece.toFixed(3)}`);
    body.push(`<text x="${x0}" y="${y0 - 29}" class="tiny">${esc(meta.join(" | "))}</text>`);
    body.push(`<text x="${x0}" y="${y0 - 8}" class="tiny">${esc(yLabel)}</text>`);
    body.push(`<rect x="${x0}" y="${y0}" width="${plotW}" height="${plotH}" fill="#F8FAFC" stroke="#CBD5E1"/>`);
    for (let t = 0; t <= 1.001; t += 0.25) {
      const x = x0 + t * plotW, y = y0 + plotH - t * plotH;
      body.push(`<line x1="${x}" y1="${y0}" x2="${x}" y2="${y0 + plotH}" class="grid"/>`);
      body.push(`<line x1="${x0}" y1="${y}" x2="${x0 + plotW}" y2="${y}" class="grid"/>`);
      body.push(`<text x="${x}" y="${y0 + plotH + 22}" text-anchor="middle" class="tiny">${t.toFixed(2)}</text>`);
      body.push(`<text x="${x0 - 12}" y="${y + 4}" text-anchor="end" class="tiny">${t.toFixed(2)}</text>`);
    }
    body.push(`<line x1="${x0}" y1="${y0 + plotH}" x2="${x0 + plotW}" y2="${y0}" stroke="#334155" stroke-width="1.5" stroke-dasharray="5,5"/>`);
    body.push(`<text x="${x0 + plotW - 30}" y="${y0 + 23}" class="tiny" fill="#475569">ideal</text>`);
    const points = epRows.map(r => ({
      x: x0 + num(r.mean_predicted) * plotW,
      y: y0 + plotH - num(r.observed_frequency) * plotH,
      n: num(r.n_samples),
      mean: num(r.mean_predicted),
      obs: num(r.observed_frequency),
    })).filter(p => Number.isFinite(p.x) && Number.isFinite(p.y)).sort((a, b) => a.x - b.x);
    if (points.length > 1) {
      body.push(`<polyline points="${points.map(p => `${p.x},${p.y}`).join(" ")}" fill="none" stroke="#2563EB" stroke-width="2.6"/>`);
    }
    points.forEach(p => {
      const r = Math.max(4.2, Math.min(14, Math.sqrt(p.n) * 0.82));
      body.push(`<line x1="${p.x}" y1="${y0 + plotH}" x2="${p.x}" y2="${y0 + plotH + 6}" stroke="#94A3B8"/>`);
      body.push(`<circle cx="${p.x}" cy="${p.y}" r="${r}" fill="#2563EB" fill-opacity="0.84" stroke="#1D4ED8"><title>mean=${p.mean.toFixed(3)}, observed=${p.obs.toFixed(3)}, n=${p.n}</title></circle>`);
    });
    body.push(`<text x="${x0 + plotW / 2}" y="${y0 + plotH + 48}" text-anchor="middle" class="small">Predicted probability / confidence</text>`);
    const lowN = points.filter(p => p.n < 10).length;
    if (lowN) body.push(`<text x="${x0 + plotW}" y="${y0 + plotH + 48}" text-anchor="end" class="tiny">${lowN} sparse bin${lowN === 1 ? "" : "s"}</text>`);
  });
  const legendY = height - 52;
  body.push(`<line x1="44" y1="${legendY}" x2="104" y2="${legendY}" stroke="#334155" stroke-width="1.5" stroke-dasharray="5,5"/><text x="118" y="${legendY + 4}" class="small">perfect calibration</text>`);
  body.push(`<line x1="318" y1="${legendY}" x2="378" y2="${legendY}" stroke="#2563EB" stroke-width="2.6"/><circle cx="348" cy="${legendY}" r="8" fill="#2563EB" fill-opacity=".84" stroke="#1D4ED8"/><text x="392" y="${legendY + 4}" class="small">observed OOF bin</text>`);
  return writeAsset(stem, svgShell(width, height, body.join("\n")), outDir);
}

function s3MeasuredFigure(stem, title, csvName, outDir) {
  const rows = readCsv(csvName);
  if (!rows.length) return noMeasuredRowsFigure(stem, title, outDir);
  requireMeasured(rows, stem);
  if (rows.some(r => String(r.model_family) === "best")) throw new Error(`${stem} contains unlabeled best model rows`);
  const groups = unique(rows.map(r => r.analysis_family));
  const width = 1700, left = 380, right = 430, top = 162;
  const rowsByGroup = groups.map(g => rows.filter(r => r.analysis_family === g));
  const rowH = 34;
  const panelHeights = rowsByGroup.map(gRows => 76 + gRows.length * rowH);
  const panelGap = 44;
  const height = top + panelHeights.reduce((a, b) => a + b, 0) + panelGap * Math.max(0, groups.length - 1) + 94;
  const vals = rows.map(r => num(r.primary_score)).filter(Number.isFinite);
  const max = Math.max(1, Math.max(...vals));
  const body = [
    `<rect width="${width}" height="${height}" fill="#FFFFFF"/>`,
    `<text x="44" y="50" class="title">${esc(title)}</text>`,
    `<text x="44" y="78" class="subtitle">Measured supplementary results only; unsupported combinations are documented in Table S3 rather than drawn as empty placeholders.</text>`,
    `<text x="44" y="108" class="tiny">Dots show pooled OOF score. A fixed detail column prevents repeated long model labels from colliding with the plotted values.</text>`,
  ];
  const legendItems = [
    ["UGA_geometry", "UGA geometry"],
    ["channel_KME", "Channel KME"],
    ["COSMIC_NNLS_exposures", "COSMIC NNLS"],
  ].filter(([key]) => rows.some(r => r.representation_family === key));
  let legendX = 1060;
  legendItems.forEach(([key, label]) => {
    body.push(`<circle cx="${legendX}" cy="100" r="6" fill="${palette[key] || "#334155"}"/>`);
    body.push(`<text x="${legendX + 14}" y="104" class="tiny">${esc(label)}</text>`);
    legendX += 150;
  });
  let yCursor = top;
  groups.forEach((group, gi) => {
    const gRows = rowsByGroup[gi];
    const panelH = panelHeights[gi];
    body.push(`<rect x="44" y="${yCursor - 36}" width="${width - 88}" height="${panelH}" rx="8" fill="#FFFFFF" stroke="#E2E8F0"/>`);
    body.push(`<text x="64" y="${yCursor - 12}" class="panel-title">${esc(group)}</text>`);
    body.push(`<text x="${left}" y="${yCursor - 12}" class="tiny">OOF primary metric</text>`);
    body.push(`<text x="${width - right + 30}" y="${yCursor - 12}" class="tiny">selected supplementary row</text>`);
    [0, 0.25, 0.5, 0.75, 1].forEach(t => {
      const x = left + t / max * (width - left - right);
      body.push(`<line x1="${x}" y1="${yCursor}" x2="${x}" y2="${yCursor + panelH - 56}" class="grid"/>`);
      body.push(`<text x="${x}" y="${yCursor + panelH - 16}" text-anchor="middle" class="tiny">${t.toFixed(2)}</text>`);
    });
    const sortedRows = [...gRows].sort((a, b) => String(a.endpoint).localeCompare(String(b.endpoint)) || String(a.representation_family).localeCompare(String(b.representation_family)));
    sortedRows.forEach((r, i) => {
      const ep = r.endpoint;
      const y = yCursor + 28 + i * rowH;
      if (i % 2 === 0) body.push(`<rect x="54" y="${y - 19}" width="${width - 108}" height="${rowH}" fill="#F8FAFC" opacity=".56"/>`);
      body.push(textBlock(64, y + 4, wrapWords(labelEndpoint(ep), 30), { cls: "small", lineHeight: 13 }));
      const v = num(r.primary_score);
      const x = left + v / max * (width - left - right);
      body.push(`<line x1="${left}" y1="${y}" x2="${x}" y2="${y}" stroke="${palette[r.representation_family] || "#334155"}" stroke-width="2" opacity=".18"/>`);
      body.push(`<circle cx="${x}" cy="${y}" r="6.2" fill="${palette[r.representation_family] || "#334155"}"><title>${esc(labelEndpoint(ep))} | ${esc(labelRep(r.representation_family, r))} | ${esc(labelDisplayModel(r))} | ${v.toFixed(3)}</title></circle>`);
      body.push(`<text x="${x + 12}" y="${y + 4}" class="tiny" font-weight="700">${v.toFixed(2)}</text>`);
      const detail = `${labelRep(r.representation_family, r)} | ${labelDisplayModel(r)}`;
      body.push(textBlock(width - right + 30, y + 4, wrapWords(detail, 48).slice(0, 2), { cls: "tiny", lineHeight: 12 }));
    });
    yCursor += panelH + panelGap;
  });
  const footY = height - 48;
  body.push(`<rect x="44" y="${footY - 20}" width="${width - 88}" height="42" rx="8" fill="#F8FAFC" stroke="#E2E8F0"/>`);
  body.push(`<text x="66" y="${footY + 5}" class="tiny">S3 is intentionally scoped to supplementary checks; main endpoint ranking is shown once in Figure 5 to reduce redundant result panels.</text>`);
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
    ["Channel KME", "UGA channels summarized by kernel means", palette.channel_KME, 550, 362, 390, 138, "kme"],
    ["Bio MAF v4", "Nested-selected gene, driver, hotspot, consequence and VAF features", "#159A74", 550, 546, 390, 156, "chips"],
    ["Signatures + Bio MAF v4", "Process spectra joined with nested-selected event-level biology", "#D9822B", 1080, 338, 390, 180, "combo"],
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
  body.push(`<rect x="1564" y="342" width="74" height="24" rx="12" fill="#DBEAFE"/><text x="1601" y="359" text-anchor="middle" class="tiny">AUROC</text>`);
  body.push(`<rect x="1650" y="342" width="104" height="24" rx="12" fill="#DCFCE7"/><text x="1702" y="359" text-anchor="middle" class="tiny">Spearman r</text>`);
  body.push(`<rect x="1766" y="342" width="66" height="24" rx="12" fill="#FDE68A"/><text x="1799" y="359" text-anchor="middle" class="tiny">Bal acc</text>`);

  body.push(`<rect x="1540" y="604" width="300" height="155" rx="10" fill="#FFF7ED" stroke="#D9822B" stroke-width="1.8"/>`);
  body.push(`<text x="1564" y="644" class="panel-title">End-to-end alternatives</text>`);
  body.push(textBlock(1564, 676, ["MuAt and ATGC operate directly", "on event sets as conceptual", "comparators outside the tabular path."], { cls: "small", lineHeight: 20 }));
  body.push(`<path d="M480 680 C770 890 1280 890 1540 682" fill="none" stroke="#D9822B" stroke-width="2.4" stroke-dasharray="9,9" marker-end="url(#arrow)"/>`);

  body.push(`<rect x="72" y="910" width="1768" height="52" rx="8" fill="#F8FAFC" stroke="#E2E8F0"/>`);
  body.push(`<line x1="100" y1="936" x2="158" y2="936" class="rule" marker-end="url(#arrow)"/><text x="174" y="941" class="small">Solid arrows: feature extraction and tabular benchmarking</text>`);
  body.push(`<line x1="560" y1="936" x2="618" y2="936" stroke="#D9822B" stroke-width="2.4" stroke-dasharray="9,9" marker-end="url(#arrow)"/><text x="634" y="941" class="small">Dashed arrow: direct event-set models used only as conceptual comparators</text>`);
  return writeAsset(stem, svgShell(width, height, body.join("\n")), outDir);
}

function conceptualOverviewV2(stem, outDir) {
  const width = 1900, height = 980;
  const body = [
    `<rect width="${width}" height="${height}" fill="#FFFFFF"/>`,
    `<text x="56" y="52" class="title">Figure 1. Mutation catalogues to benchmarked representations</text>`,
    textBlock(56, 82, wrapWords("Primary path: mutation events are converted into tabular feature families and evaluated with paired 5-fold out-of-fold endpoint metrics; the MuAt-compatible event-bag model is shown as a separate direct comparator, not as a tabular learner.", 150), { cls: "subtitle", lineHeight: 20 }),
  ];
  const badge = (x, y, letter, label) => {
    body.push(`<circle cx="${x}" cy="${y}" r="15" fill="#111827"/><text x="${x}" y="${y + 5}" text-anchor="middle" fill="#FFFFFF" font-weight="800">${letter}</text>`);
    body.push(`<text x="${x + 24}" y="${y + 5}" class="panel-title">${esc(label)}</text>`);
  };
  badge(78, 142, "A", "One sample catalogue");
  badge(520, 142, "B", "Candidate feature views");
  badge(1054, 142, "C", "Combined tabular view");
  badge(1532, 142, "D", "Model comparison");

  body.push(`<rect x="70" y="178" width="385" height="540" rx="8" class="card"/>`);
  body.push(`<text x="100" y="218" class="label" font-weight="700">MAF / VCF event rows</text>`);
  body.push(chip(284, 216, "5 example rows", { width: 88, fill: "#EEF2FF", stroke: "#C7D2FE" }));
  const cols = ["chr", "pos", "ref", "alt", "gene"];
  const colX = [100, 168, 252, 312, 374];
  cols.forEach((c, i) => body.push(`<text x="${colX[i]}" y="258" class="tiny" font-weight="700">${c}</text>`));
  const muts = [
    ["3", "41.2M", "C", "T", "BRCA1", palette.signatures_only],
    ["8", "128M", "T", "G", "MYC", palette.signatures_plus_MAF_stack],
    ["17", "7.6M", "G", "-", "TP53", palette.MAF_stack_only],
    ["2", "90.1M", "A", "C", "ALK", palette.channel_KME],
    ["12", "25.4M", "+", "T", "KRAS", palette.burden_only],
  ];
  muts.forEach((m, r) => {
    const y = 296 + r * 56;
    body.push(`<line x1="100" y1="${y + 18}" x2="436" y2="${y + 18}" class="grid"/>`);
    m.slice(0, 5).forEach((v, i) => body.push(`<text x="${colX[i]}" y="${y}" class="label">${esc(v)}</text>`));
    body.push(`<circle cx="440" cy="${y - 5}" r="${7 + r}" fill="${m[5]}" opacity=".92"/>`);
  });
  ["locus", "allele", "gene", "VAF", "effect", "FASTA"].forEach((t, i) => {
    const x = 100 + (i % 3) * 96;
    const y = 598 + Math.floor(i / 3) * 36;
    body.push(`<rect x="${x}" y="${y}" width="74" height="24" rx="12" fill="#F1F5F9" stroke="#CBD5E1"/><text x="${x + 37}" y="${y + 16}" text-anchor="middle" class="tiny">${esc(t)}</text>`);
  });
  body.push(`<text x="100" y="690" class="small">Retained event fields feed each feature family.</text>`);

  const featureCards = [
    ["1", "Mutational signatures", "SBS/DBS/ID spectra plus burden-normalized exposures", palette.signatures_only, 520, 178, "hist"],
    ["2", "Channel KME", "UGA channel contexts summarized by kernel means", palette.channel_KME, 520, 378, "kme"],
    ["3", "Bio MAF v4", "Nested-selected gene, driver, hotspot, consequence, VAF and event biology", palette.MAF_stack_only, 520, 578, "chips"],
  ];
  featureCards.forEach(([n, head, sub, color, x, y, kind]) => {
    body.push(`<rect x="${x}" y="${y}" width="430" height="150" rx="8" fill="#FFFFFF" stroke="${color}" stroke-width="2"/>`);
    body.push(`<circle cx="${x + 30}" cy="${y + 32}" r="16" fill="${color}"/><text x="${x + 30}" y="${y + 38}" text-anchor="middle" fill="#FFFFFF" font-weight="800">${n}</text>`);
    body.push(`<text x="${x + 58}" y="${y + 36}" class="panel-title">${esc(head)}</text>`);
    body.push(textBlock(x + 58, y + 64, wrapWords(sub, 42), { cls: "small", lineHeight: 17 }));
    if (kind === "hist") for (let i = 0; i < 18; i++) body.push(`<rect x="${x + 58 + i * 18}" y="${y + 130 - ((i * 5) % 8 + 2) * 7}" width="11" height="${((i * 5) % 8 + 2) * 7}" fill="${color}" opacity="${0.42 + i / 40}"/>`);
    if (kind === "kme") {
      ["A", "C", "G", "T", "C", "T", "A", "G"].forEach((t, i) => body.push(`<rect x="${x + 58 + i * 22}" y="${y + 108}" width="18" height="18" rx="3" fill="#F3E8FF" stroke="#D8B4FE"/><text x="${x + 67 + i * 22}" y="${y + 122}" text-anchor="middle" class="tiny">${t}</text>`));
      for (let i = 0; i < 13; i++) body.push(`<circle cx="${x + 292 + Math.cos(i * 1.3) * 58}" cy="${y + 112 + Math.sin(i * 1.7) * 24}" r="5" fill="${color}" opacity=".76"/>`);
    }
    if (kind === "chips") {
      ["TP53", "chr8", "missense", "VAF", "splice", "burden"].forEach((t, i) => {
        const px = x + 58 + (i % 3) * 112;
        const py = y + 94 + Math.floor(i / 3) * 30;
        body.push(`<rect x="${px}" y="${py}" width="92" height="22" rx="11" fill="${color}" opacity=".15"/><text x="${px + 46}" y="${py + 15}" text-anchor="middle" class="tiny">${esc(t)}</text>`);
      });
    }
  });

  [["M455 448 C486 448 489 250 520 250"], ["M455 448 C486 448 489 450 520 450"], ["M455 448 C486 448 489 650 520 650"]].forEach(([d]) => body.push(`<path d="${d}" fill="none" class="rule" marker-end="url(#arrow)"/>`));

  body.push(`<rect x="1054" y="278" width="400" height="290" rx="8" fill="#FFFFFF" stroke="${palette.signatures_plus_MAF_stack}" stroke-width="2"/>`);
  body.push(`<text x="1082" y="318" class="panel-title">Signatures + Bio MAF v4</text>`);
  body.push(textBlock(1082, 350, ["Joined table keeps process spectra", "and event-level biology in one", "sample-by-feature matrix."], { cls: "small", lineHeight: 18 }));
  const matrixX = 1082, matrixY = 426;
  const blocks = [
    ["spectra", palette.signatures_only, 96],
    ["genes/loci", palette.MAF_stack_only, 108],
    ["VAF/effect", palette.signatures_plus_MAF_stack, 108],
  ];
  let mx = matrixX;
  blocks.forEach(([label, color, w]) => {
    body.push(`<rect x="${mx}" y="${matrixY}" width="${w}" height="92" rx="5" fill="${color}" opacity=".14" stroke="${color}"/>`);
    body.push(`<text x="${mx + w / 2}" y="${matrixY + 20}" text-anchor="middle" class="tiny">${esc(label)}</text>`);
    for (let r = 0; r < 4; r++) for (let c = 0; c < 3; c++) body.push(`<circle cx="${mx + 24 + c * 22}" cy="${matrixY + 42 + r * 11}" r="2.5" fill="${color}" opacity=".75"/>`);
    mx += w + 12;
  });
  body.push(chip(1082, 544, "nested-selected blocks", { width: 146, fill: "#FFF7ED", stroke: "#FDBA74" }));
  body.push(chip(1232, 544, "single table", { width: 88, fill: "#FFF7ED", stroke: "#FDBA74" }));
  [["M950 252 C992 252 1005 360 1054 360"], ["M950 452 C1000 452 1008 422 1054 422"], ["M950 652 C992 652 1005 490 1054 490"]].forEach(([d]) => body.push(`<path d="${d}" fill="none" class="rule" marker-end="url(#arrow)"/>`));

  body.push(`<rect x="1532" y="210" width="310" height="236" rx="8" class="card"/>`);
  body.push(`<text x="1560" y="250" class="panel-title">Paired tabular models</text>`);
  body.push(textBlock(1560, 284, ["Elastic net and XGBoost", "5-fold out-of-fold predictions", "same endpoint splits and metrics"], { cls: "small", lineHeight: 22 }));
  body.push(chip(1560, 382, "mAUROC", { width: 72, fill: "#DBEAFE", stroke: "#BFDBFE" }));
  body.push(chip(1648, 382, "AUROC", { width: 66, fill: "#DBEAFE", stroke: "#BFDBFE" }));
  body.push(chip(1728, 382, "Bal acc", { width: 70, fill: "#FDE68A", stroke: "#FCD34D" }));
  body.push(chip(1560, 414, "rho", { width: 50, fill: "#DCFCE7", stroke: "#BBF7D0" }));
  body.push(`<path d="M1454 424 C1488 424 1498 328 1532 328" fill="none" class="rule" marker-end="url(#arrow)"/>`);

  body.push(`<rect x="1532" y="592" width="310" height="170" rx="8" fill="#FFF7ED" stroke="${palette.signatures_plus_MAF_stack}" stroke-width="1.8"/>`);
  body.push(`<text x="1560" y="632" class="panel-title">Direct event-set comparator</text>`);
  body.push(textBlock(1560, 664, ["MuAt-compatible consumes event", "bags directly and is benchmarked", "outside the tabular learner path."], { cls: "small", lineHeight: 20 }));
  body.push(`<path d="M455 690 C790 892 1280 884 1532 674" fill="none" stroke="${palette.signatures_plus_MAF_stack}" stroke-width="2.4" stroke-dasharray="9,9" marker-end="url(#arrow)"/>`);

  body.push(`<rect x="70" y="850" width="1772" height="72" rx="8" fill="#F8FAFC" stroke="#E2E8F0"/>`);
  body.push(`<line x1="100" y1="884" x2="158" y2="884" class="rule" marker-end="url(#arrow)"/><text x="176" y="889" class="small">Primary path: feature extraction, joined tabular views, paired endpoint benchmarking</text>`);
  body.push(`<line x1="760" y1="884" x2="818" y2="884" stroke="${palette.signatures_plus_MAF_stack}" stroke-width="2.4" stroke-dasharray="9,9" marker-end="url(#arrow)"/><text x="836" y="889" class="small">Dashed path: direct event-set comparator, evaluated separately from tabular learners</text>`);
  body.push(`<text x="100" y="914" class="tiny">Rendered labels include provenance cues so the static manuscript export does not depend on tooltips.</text>`);
  return writeAsset(stem, svgShell(width, height, body.join("\n")), outDir);
}

function representationConstruction(stem, outDir) {
  const width = 1600, height = 980;
  const body = [
    `<rect width="${width}" height="${height}" fill="#FFFFFF"/>`,
    `<text x="44" y="54" class="title">Supplementary Figure S1. Representation construction</text>`,
    `<text x="44" y="84" class="subtitle">Each feature block is cached with input fingerprints, split provenance, FASTA provenance, parameters and sample manifests.</text>`,
    `<text x="44" y="112" class="tiny">Cards show the reproducible construction path used for strict no-leakage manuscript outputs.</text>`,
  ];
  const cardW = 410, cardH = 276;
  const xs = [70, 595, 1120];
  const ys = [150, 560];
  const steps = [
    ["1", "Spectra", "Count SBS/DBS/ID channels and burden; normalize within each sample.", "#2F6FDB", xs[0], ys[0]],
    ["2", "FASTA windows", "Fetch GRCh37 sequence context and validate REF allele before encoding.", "#A855C7", xs[1], ys[0]],
    ["3", "Channel KME", "Use locked channel geometry and average event-level kernels.", palette.channel_KME, xs[2], ys[0]],
    ["4", "Bio MAF v4", "Select among predeclared gene, driver, hotspot, consequence, VAF and burden blocks inside nested CV.", "#159A74", xs[0], ys[1]],
    ["5", "UGA variants", "Supplementary atlas and channel geometry checks are generated from locked inputs.", "#6D5BD0", xs[1], ys[1]],
    ["6", "Checkpointed outputs", "Every public table and figure traces back to explicit cached artifacts.", "#D9822B", xs[2], ys[1]],
  ];
  steps.forEach(([n, head, sub, color, x, y]) => {
    body.push(`<rect x="${x}" y="${y}" width="${cardW}" height="${cardH}" rx="8" fill="#FFFFFF" stroke="${color}" stroke-width="2" class="shadow"/>`);
    body.push(`<circle cx="${x + 42}" cy="${y + 42}" r="22" fill="${color}"/><text x="${x + 42}" y="${y + 49}" text-anchor="middle" fill="#FFFFFF" font-weight="800">${n}</text>`);
    body.push(`<text x="${x + 78}" y="${y + 40}" class="panel-title">${esc(head)}</text>`);
    body.push(textBlock(x + 78, y + 68, wrapWords(sub, 40), { cls: "small", lineHeight: 17 }));
  });
  for (const [x1, y1, x2, y2] of [
    [xs[0] + cardW + 10, ys[0] + 138, xs[1] - 15, ys[0] + 138],
    [xs[1] + cardW + 10, ys[0] + 138, xs[2] - 15, ys[0] + 138],
    [xs[0] + cardW + 10, ys[1] + 138, xs[1] - 15, ys[1] + 138],
    [xs[1] + cardW + 10, ys[1] + 138, xs[2] - 15, ys[1] + 138],
    [xs[2] + cardW / 2, ys[0] + cardH + 16, xs[2] + cardW / 2, ys[1] - 18],
  ]) {
    body.push(`<line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" class="rule" marker-end="url(#arrow)"/>`);
  }
  for (let i = 0; i < 16; i++) {
    const h = (i % 6 + 2) * 9;
    body.push(`<rect x="${xs[0] + 42 + i * 20}" y="${ys[0] + 226 - h}" width="11" height="${h}" fill="#2F6FDB" opacity=".72"/>`);
  }
  body.push(`<text x="${xs[0] + 42}" y="${ys[0] + 250}" class="tiny">SBS/DBS/ID exposure channels plus burden covariates</text>`);
  body.push(`<text x="${xs[1] + 56}" y="${ys[0] + 174}" class="tiny">... A C G T C [variant] A T G ...</text>`);
  for (let i = 0; i < 17; i++) {
    body.push(`<rect x="${xs[1] + 58 + i * 16}" y="${ys[0] + 202}" width="12" height="24" rx="2" fill="${["#DBEAFE", "#FEE2E2", "#DCFCE7", "#F3E8FF"][i % 4]}" stroke="#CBD5E1"/>`);
  }
  body.push(`<text x="${xs[1] + 58}" y="${ys[0] + 250}" class="tiny">Validated context avoids REF/FASTA mismatches.</text>`);
  for (let i = 0; i < 28; i++) {
    body.push(`<circle cx="${xs[2] + 205 + Math.cos(i * 1.7) * (38 + (i % 4) * 13)}" cy="${ys[0] + 184 + Math.sin(i * 1.3) * (24 + (i % 3) * 10)}" r="4" fill="#A855C7" opacity=".70"/>`);
  }
  body.push(`<text x="${xs[2] + 58}" y="${ys[0] + 252}" class="tiny">Kernel means summarize event geometry without labels.</text>`);
  ["TP53", "BRCA1", "chr17p", "splice", "VAF", "HR repair"].forEach((t, i) => {
    const px = xs[0] + 42 + (i % 3) * 112;
    const py = ys[1] + 170 + Math.floor(i / 3) * 36;
    body.push(`<rect x="${px}" y="${py}" width="94" height="26" rx="13" fill="#DCFCE7" stroke="#86EFAC"/><text x="${px + 47}" y="${py + 18}" text-anchor="middle" class="tiny">${esc(t)}</text>`);
  });
  body.push(`<text x="${xs[0] + 42}" y="${ys[1] + 250}" class="tiny">Aggregates are recomputed only from the current split-safe cache.</text>`);
  const ux = xs[1] + 58, uy = ys[1] + 222;
  body.push(`<path d="M${ux} ${uy} L${ux + 70} ${uy - 45} L${ux + 140} ${uy - 5} L${ux + 210} ${uy - 70} L${ux + 290} ${uy - 24}" fill="none" stroke="#6D5BD0" stroke-width="2.5"/>`);
  [0, 1, 2, 3, 4].forEach((i) => body.push(`<circle cx="${ux + i * 70}" cy="${[uy, uy - 45, uy - 5, uy - 70, uy - 24][i]}" r="5" fill="#6D5BD0"/>`));
  body.push(`<text x="${xs[1] + 58}" y="${ys[1] + 250}" class="tiny">Supplementary checks stay outside the required main table.</text>`);
  ["features.npz", "manifest.json", "oof_predictions.csv", "pairwise_tests.csv", "plot_data.csv"].forEach((t, i) => {
    const py = ys[1] + 132 + i * 26;
    body.push(`<rect x="${xs[2] + 58}" y="${py}" width="248" height="22" rx="5" fill="#FFF7ED" stroke="#FDBA74"/><text x="${xs[2] + 70}" y="${py + 15}" class="tiny">${esc(t)}</text>`);
  });
  body.push(`<rect x="70" y="884" width="1460" height="56" rx="8" fill="#F8FAFC" stroke="#E2E8F0"/>`);
  body.push(`<text x="96" y="910" class="small">Strict reruns invalidate only stale feature caches; checkpointed OOF predictions and tests preserve reproducibility without rerunning unaffected slots.</text>`);
  body.push(`<text x="96" y="930" class="tiny">All artifact names are drawn inside the card bounds to mirror the HTML and Word export layout.</text>`);
  return writeAsset(stem, svgShell(width, height, body.join("\n")), outDir);
}

const assets = [];
assets.push(conceptualOverviewV2("figure_1_conceptual_overview", figuresDir));
assets.push(barFigureV2(
  "figure_2_signature_baselines",
  "Figure 2. Burden and signature baselines",
  "figure_2_signature_baselines.csv",
  figuresDir,
  {
    deltaPanel: true,
    deltaComparison: "signatures_vs_burden",
    deltaTitle: "Signatures - burden",
    note: "Endpoint labels show metric, task, and n. Delta intervals are paired 95% bootstrap CIs where available.",
  },
));
assets.push(muatComparatorFigure(
  "figure_3_geometry_vs_signatures",
  "Figure 3. MuAt-compatible event-bag comparator",
  "figure_3_geometry_vs_signatures.csv",
  figuresDir,
));
assets.push(barFigureV2(
  "figure_4_maf_stack_vs_signatures",
  "Figure 4. MAF-stack biology vs signatures",
  "figure_4_maf_stack_vs_signatures.csv",
  figuresDir,
  {
    contrastPanel: true,
    subtitle: "Bars show canonical 5-fold OOF performance. Right panels show signed pairwise deltas and FDR markers.",
    note: "Contrast signs are candidate minus baseline. Intervals are shown for bootstrap tests; DeLong rows carry p/q markers without bootstrap CIs.",
  },
));
assets.push(heatFacetFigureV3("figure_5_cross_endpoint_summary", "Figure 5. Cross-endpoint representation summary", "figure_5_cross_endpoint_summary.csv", figuresDir));
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
  const programFiles = [process.env.ProgramFiles, process.env["ProgramFiles(x86)"]].filter(Boolean);
  const executableCandidates = [
    process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH,
    ...programFiles.flatMap(base => [
      path.join(base, "Google", "Chrome", "Application", "chrome.exe"),
      path.join(base, "Microsoft", "Edge", "Application", "msedge.exe"),
    ]),
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
