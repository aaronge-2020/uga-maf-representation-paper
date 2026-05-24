import json
import os
from pathlib import Path

def generate_html():
    base_dir = Path(__file__).resolve().parents[1]
    json_path = base_dir / 'bench' / 'signature_visualizer_data.json'
    output_path = base_dir / 'bench' / 'signature_visualizer.html'
    
    if not json_path.exists():
        print(f"Error: {json_path} not found. Run prepare_visualization_data.py first.")
        return

    with open(json_path, 'r') as f:
        data = json.load(f)

    html_template = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Universal-BiCGR Signature Visualizer</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600&family=JetBrains+Mono&display=swap" rel="stylesheet">
    <style>
        :root {{
            --bg-color: #0a0c12;
            --card-bg: rgba(25, 30, 45, 0.7);
            --text-color: #e0e6ed;
            --accent-color: #6366f1;
            --accent-glow: rgba(99, 102, 241, 0.4);
            --border-color: rgba(255, 255, 255, 0.1);
            --sbs-color: #0ea5e9;
            --dbs-color: #ec4899;
        }}

        * {{
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }}

        body {{
            font-family: 'Outfit', sans-serif;
            background-color: var(--bg-color);
            background-image: 
                radial-gradient(circle at 10% 20%, rgba(99, 102, 241, 0.1) 0%, transparent 40%),
                radial-gradient(circle at 90% 80%, rgba(14, 165, 233, 0.1) 0%, transparent 40%);
            color: var(--text-color);
            display: flex;
            flex-direction: column;
            height: 100vh;
            overflow: hidden;
        }}

        header {{
            padding: 1rem 2rem;
            display: flex;
            justify-content: space-between;
            align-items: center;
            background: rgba(10, 12, 18, 0.8);
            backdrop-filter: blur(12px);
            border-bottom: 1px solid var(--border-color);
            z-index: 10;
        }}

        .logo {{
            font-size: 1.4rem;
            font-weight: 600;
            background: linear-gradient(135deg, #6366f1, #0ea5e9);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            letter-spacing: -0.5px;
        }}

        .controls {{
            display: flex;
            gap: 1rem;
            align-items: center;
        }}

        select {{
            background: var(--card-bg);
            border: 1px solid var(--border-color);
            color: white;
            padding: 0.5rem 1rem;
            border-radius: 10px;
            cursor: pointer;
            font-family: inherit;
            transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1);
            outline: none;
            min-width: 150px;
        }}

        select:hover {{
            background: rgba(255, 255, 255, 0.1);
            border-color: var(--accent-color);
            box-shadow: 0 0 15px var(--accent-glow);
        }}

        main {{
            display: grid;
            grid-template-columns: 1fr 1.2fr;
            gap: 1.5rem;
            padding: 1.5rem;
            flex: 1;
            overflow: hidden;
        }}

        .card {{
            background: var(--card-bg);
            backdrop-filter: blur(16px);
            border-radius: 20px;
            border: 1px solid var(--border-color);
            padding: 1.5rem;
            display: flex;
            flex-direction: column;
            position: relative;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        }}

        .card-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 1.2rem;
        }}

        .card-title {{
            font-size: 0.85rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 1.5px;
            color: #94a3b8;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }}

        .card-title::before {{
            content: '';
            display: inline-block;
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background: var(--accent-color);
            box-shadow: 0 0 8px var(--accent-color);
        }}

        .chart-container {{
            flex: 1;
            position: relative;
            min-height: 0;
        }}

        .details-panel {{
            display: flex;
            flex-direction: column;
            gap: 1.5rem;
            overflow-y: auto;
            padding-right: 0.5rem;
        }}

        .details-panel::-webkit-scrollbar {{
            width: 6px;
        }}
        .details-panel::-webkit-scrollbar-thumb {{
            background: rgba(255, 255, 255, 0.1);
            border-radius: 10px;
        }}

        .signature-hero {{
            padding: 1.5rem;
            background: linear-gradient(135deg, rgba(99, 102, 241, 0.1), rgba(14, 165, 233, 0.05));
            border-radius: 16px;
            border-left: 4px solid var(--accent-color);
            margin-bottom: 1rem;
        }}

        .signature-name {{
            font-size: 2.2rem;
            font-weight: 600;
            color: #fff;
            margin-bottom: 0.3rem;
            text-shadow: 0 0 20px rgba(255,255,255,0.1);
        }}

        .signature-type {{
            font-size: 0.9rem;
            color: #94a3b8;
            font-family: 'JetBrains Mono', monospace;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }}

        .legend-item {{
            display: flex;
            align-items: center;
            gap: 0.5rem;
            font-size: 0.75rem;
            color: #64748b;
        }}

        .dot {{
            width: 8px;
            height: 8px;
            border-radius: 2px;
        }}
    </style>
</head>
<body>
    <header>
        <div class="logo">Universal-BiCGR Dashboard</div>
        <div class="controls">
            <select id="type-select">
                <option value="sbs">SBS Signatures</option>
                <option value="dbs">DBS Signatures</option>
            </select>
            <select id="sig-select">
                <!-- Populated by JS -->
            </select>
        </div>
    </header>

    <main>
        <div class="card">
            <div class="card-header">
                <div class="card-title">Embedding Space Projection</div>
                <div class="legend-item" id="projection-type">PCA Analysis</div>
            </div>
            <div class="chart-container">
                <canvas id="pcaChart"></canvas>
            </div>
            <div style="font-size: 0.7rem; color: #475569; margin-top: 1rem; text-align: center;">
                Click points to select a signature
            </div>
        </div>

        <div class="details-panel">
            <div class="card" style="flex: 0 0 auto;">
                <div class="signature-hero">
                    <div id="active-sig-name" class="signature-name">SBS1</div>
                    <div id="active-sig-type" class="signature-type">COSMIC v3.5 SBS</div>
                </div>
                <div class="card-header">
                    <div class="card-title">48D Universal Encoding</div>
                    <div style="display: flex; gap: 0.8rem;">
                        <div class="legend-item"><span class="dot" style="background:#6366f1"></span>L-Ctx</div>
                        <div class="legend-item"><span class="dot" style="background:#10b981"></span>Ref</div>
                        <div class="legend-item"><span class="dot" style="background:#3b82f6"></span>R-Ctx</div>
                        <div class="legend-item"><span class="dot" style="background:#f59e0b"></span>Alt</div>
                    </div>
                </div>
                <div class="chart-container" style="height: 180px;">
                    <canvas id="fingerprintChart"></canvas>
                </div>
            </div>

            <div class="card" style="flex: 1; min-height: 300px;">
                <div class="card-header">
                    <div class="card-title">Categorical Mutational Profile</div>
                </div>
                <div class="chart-container">
                    <canvas id="categoricalChart"></canvas>
                </div>
            </div>
        </div>
    </main>

    <script>
        const data = {json_data};
        
        let activeType = 'sbs';
        let activeSigIndex = 0;
        
        let pcaChart, fingerprintChart, categoricalChart;

        function init() {{
            populateSigSelect();
            initCharts();
            
            document.getElementById('type-select').addEventListener('change', (e) => {{
                activeType = e.target.value;
                activeSigIndex = 0;
                populateSigSelect();
                updateCharts(true);
            }});

            document.getElementById('sig-select').addEventListener('change', (e) => {{
                activeSigIndex = parseInt(e.target.value);
                updateCharts(false);
            }});
        }}

        function populateSigSelect() {{
            const select = document.getElementById('sig-select');
            select.innerHTML = '';
            const names = data[activeType].names;
            names.forEach((name, i) => {{
                const opt = document.createElement('option');
                opt.value = i;
                opt.textContent = name;
                select.appendChild(opt);
            }});
        }}

        function initCharts() {{
            // Custom defaults for dark mode
            Chart.defaults.color = '#94a3b8';
            Chart.defaults.borderColor = 'rgba(255, 255, 255, 0.05)';
            Chart.defaults.font.family = "'Outfit', sans-serif";

            const ctxPca = document.getElementById('pcaChart').getContext('2d');
            pcaChart = new Chart(ctxPca, {{
                type: 'scatter',
                data: {{
                    datasets: [{{
                        label: 'Signatures',
                        data: [], 
                        backgroundColor: (ctx) => {{
                            if (ctx.dataIndex === activeSigIndex) return '#fff';
                            return activeType === 'sbs' ? '#0ea5e966' : '#ec489966';
                        }},
                        pointRadius: (ctx) => ctx.dataIndex === activeSigIndex ? 10 : 6,
                        pointHoverRadius: 12,
                        borderColor: '#fff',
                        borderWidth: (ctx) => ctx.dataIndex === activeSigIndex ? 2 : 0,
                    }}]
                }},
                options: {{
                    responsive: true,
                    maintainAspectRatio: false,
                    onClick: (e, elements) => {{
                        if (elements.length > 0) {{
                            activeSigIndex = elements[0].index;
                            document.getElementById('sig-select').value = activeSigIndex;
                            updateCharts(false);
                        }}
                    }},
                    scales: {{
                        x: {{ grid: {{ display: false }}, ticks: {{ display: false }} }},
                        y: {{ grid: {{ display: false }}, ticks: {{ display: false }} }}
                    }},
                    plugins: {{
                        legend: {{ display: false }},
                        tooltip: {{
                            backgroundColor: 'rgba(15, 23, 42, 0.9)',
                            titleColor: '#fff',
                            bodyColor: '#94a3b8',
                            padding: 10,
                            borderRadius: 8,
                            displayColors: false,
                            callbacks: {{
                                label: (ctx) => ' ' + data[activeType].names[ctx.dataIndex]
                            }}
                        }}
                    }}
                }}
            }});

            const ctxFinger = document.getElementById('fingerprintChart').getContext('2d');
            fingerprintChart = new Chart(ctxFinger, {{
                type: 'bar',
                data: {{
                    labels: data.metadata.labels,
                    datasets: [{{
                        data: [],
                        backgroundColor: (ctx) => {{
                            const label = data.metadata.labels[ctx.dataIndex];
                            if (label.startsWith('Lx') || label.startsWith('Ly')) return '#6366f1';
                            if (label.startsWith('Ref')) return '#10b981';
                            if (label.startsWith('Rx') || label.startsWith('Ry')) return '#3b82f6';
                            if (label.startsWith('Alt')) return '#f59e0b';
                            return '#fff';
                        }},
                        borderRadius: 4
                    }}]
                }},
                options: {{
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {{
                        y: {{ beginAtZero: true, max: 1, grid: {{ color: 'rgba(255,255,255,0.05)' }} }},
                        x: {{ display: false }}
                    }},
                    plugins: {{ legend: {{ display: false }} }}
                }}
            }});

            const ctxCat = document.getElementById('categoricalChart').getContext('2d');
            categoricalChart = new Chart(ctxCat, {{
                type: 'bar',
                data: {{
                    labels: [],
                    datasets: [{{
                        data: [],
                        backgroundColor: activeType === 'sbs' ? '#0ea5e9' : '#ec4899',
                        borderRadius: 2
                    }}]
                }},
                options: {{
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {{
                        y: {{ beginAtZero: true, grid: {{ color: 'rgba(255,255,255,0.05)' }} }},
                        x: {{ display: false }}
                    }},
                    plugins: {{ legend: {{ display: false }} }}
                }}
            }});

            updateCharts(true);
        }}

        function updateCharts(typeChanged) {{
            const currentData = data[activeType];
            const name = currentData.names[activeSigIndex];
            
            document.getElementById('active-sig-name').textContent = name;
            document.getElementById('active-sig-type').textContent = `COSMIC v3.5 ${{activeType.toUpperCase()}} Signature`;

            // Update PCA
            if (typeChanged) {{
                pcaChart.data.datasets[0].data = currentData.pca.map(p => ({{x: p[0], y: p[1]}}));
            }}
            pcaChart.update('none'); // Update without full animation for performance on highlight

            // Update Fingerprint
            fingerprintChart.data.datasets[0].data = currentData.embeddings[activeSigIndex];
            fingerprintChart.update();

            // Update Categorical
            categoricalChart.data.labels = Array.from({{length: currentData.standard[activeSigIndex].length}}, (_, i) => i);
            categoricalChart.data.datasets[0].data = currentData.standard[activeSigIndex];
            categoricalChart.data.datasets[0].backgroundColor = activeType === 'sbs' ? '#0ea5e9' : '#ec4899';
            categoricalChart.update();
        }}

        init();
    </script>
</body>
</html>
"""
    
    final_html = html_template.format(json_data=json.dumps(data))
    
    with open(output_path, 'w') as f:
        f.write(final_html)
    
    print(f"Successfully generated {output_path}")

if __name__ == "__main__":
    generate_html()
