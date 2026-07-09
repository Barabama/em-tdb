import { useState } from "react";
import { invoke } from "@tauri-apps/api/core";
import { open } from "@tauri-apps/plugin-dialog";
import "./App.css";

// ── types ──
interface FitResult {
  params: number[];
  r2: number;
  expression: string;
  tdb_parameter: string;
  t_data: number[];
  g_data: number[];
  g_fit: number[];
}

interface DatasetInfo {
  folder_name: string;
  file_path: string;
}

// ── helpers ──
function fmtParams(p: number[]): string {
  if (!p || p.length < 6) return "";
  return p
    .map((v, i) => `${"ABCDEF"[i]}=${v >= 0 ? "+" : ""}${v.toExponential(4)}`)
    .join("\n");
}

function parseFolderName(name: string): { phase: string; metrics: string; atomNum: string } {
  const parts = name.split("-");
  const phase = parts[0] || "BCC";
  let atomNum = "1";
  const last = parts[parts.length - 1];
  if (last && /^\d/.test(last)) {
    atomNum = last.replace(/\D.*$/, "");
  }
  const elems = parts.slice(1, last && /^\d/.test(last) ? -1 : parts.length);
  const metrics = elems.map(() => "1").join(" ");
  return { phase, metrics, atomNum };
}

// ── SVG chart ──
function FitChart({ result }: { result: FitResult }) {
  const { t_data, g_data, g_fit } = result;
  if (!t_data.length) return null;

  const w = 600, h = 400;
  const pad = { l: 60, r: 20, t: 20, b: 40 };

  const tMin = Math.min(...t_data);
  const tMax = Math.max(...t_data);
  const gAll = [...g_data, ...g_fit];
  const gMin = Math.min(...gAll);
  const gMax = Math.max(...gAll);

  const sx = (t: number) => pad.l + ((t - tMin) / (tMax - tMin || 1)) * (w - pad.l - pad.r);
  const sy = (g: number) => h - pad.b - ((g - gMin) / (gMax - gMin || 1)) * (h - pad.t - pad.b);

  const pts = t_data.map((t, i) => `${sx(t)},${sy(g_data[i])}`).join(" ");
  const line = t_data
    .map((t, i) => `${i === 0 ? "M" : "L"}${sx(t)},${sy(g_fit[i])}`)
    .join(" ");

  return (
    <svg viewBox={`0 0 ${w} ${h}`} style={{ maxWidth: "100%", height: "auto" }}>
      <line x1={pad.l} y1={h - pad.b} x2={w - pad.r} y2={h - pad.b} stroke="#ccc" />
      <line x1={pad.l} y1={pad.t} x2={pad.l} y2={h - pad.b} stroke="#ccc" />
      <polyline points={pts} fill="none" stroke="#2196F3" strokeWidth="1" opacity="0.4" />
      {t_data.map((t, i) => i % 5 !== 0 ? null : (
        <circle key={i} cx={sx(t)} cy={sy(g_data[i])} r="2" fill="#2196F3" opacity="0.6" />
      ))}
      <path d={line} fill="none" stroke="#F44336" strokeWidth="2" />
      <text x={w / 2} y={h - 5} textAnchor="middle" fontSize="10">T (K)</text>
      <text x={12} y={h / 2} textAnchor="middle" fontSize="10" transform={`rotate(-90, 12, ${h / 2})`}>G (J/mol)</text>
    </svg>
  );
}

// ── shared result card (used by both single and batch) ──
function ResultCard({ label, result }: { label: string; result: FitResult }) {
  return (
    <details className="result-card" open>
      <summary><strong>{label}</strong>  R² = {result.r2.toFixed(6)}</summary>
      <div className="result-body">
        <div className="card">
          <p><strong>TDB:</strong></p>
          <pre className="tdb-line">{result.tdb_parameter}</pre>
          <p><strong>Expression:</strong></p>
          <pre>{result.expression}</pre>
          <p><strong>Parameters:</strong></p>
          <pre>{fmtParams(result.params)}</pre>
        </div>
        <div className="chart">
          <FitChart result={result} />
        </div>
      </div>
    </details>
  );
}

// ── main App ──
function App() {
  const [mode, setMode] = useState<"single" | "batch">("single");

  // single mode (independent state)
  const [filepath, setFilepath] = useState("");
  const [phase, setPhase] = useState("BCC");
  const [elem, setElem] = useState("Fe Cr");
  const [metrics, setMetrics] = useState("1 1");
  const [atomNum, setAtomNum] = useState("2");
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<FitResult | null>(null);
  const [error, setError] = useState("");

  // batch mode (independent state)
  const [folderPath, setFolderPath] = useState("");
  const [datasets, setDatasets] = useState<DatasetInfo[]>([]);
  const [selected, setSelected] = useState<Set<number>>(new Set());
  const [batchLoading, setBatchLoading] = useState(false);
  const [batchResults, setBatchResults] = useState<{ label: string; result: FitResult }[]>([]);
  const [batchError, setBatchError] = useState("");

  // ── single mode file picker ──
  async function pickFile() {
    const selected = await open({
      multiple: false,
      filters: [
        { name: "Thermo data", extensions: ["dat", "DAT"] },
        { name: "All files", extensions: ["*"] },
      ],
    });
    if (selected) setFilepath(selected as string);
  }

  // ── batch mode folder picker ──
  async function pickFolder() {
    const selected = await open({ multiple: false, directory: true });
    if (!selected) return;
    const fp = selected as string;
    setFolderPath(fp);
    try {
      const list: DatasetInfo[] = await invoke("scan_folder", { folderPath: fp });
      setDatasets(list);
      setSelected(new Set(list.map((_, i) => i)));
    } catch (e: any) {
      setBatchError(e?.toString() || "Failed to scan folder");
    }
  }

  function toggleSel(idx: number) {
    setSelected((prev) => {
      const next = new Set(prev);
      if (next.has(idx)) next.delete(idx);
      else next.add(idx);
      return next;
    });
  }

  // ── single mode fit ──
  async function handleRun() {
    setError("");
    setResult(null);
    if (!filepath) { setError("Please select a file first"); return; }
    setLoading(true);
    try {
      const r: FitResult = await invoke("run_fit", {
        filepath,
        phase,
        elem: elem.split(/\s+/).filter(Boolean),
        metrics: metrics.split(/\s+/).filter(Boolean).map(Number),
        atomNum: parseInt(atomNum) || 1,
      });
      setResult(r);
    } catch (e: any) {
      setError(e?.toString() || "Unknown error");
    } finally {
      setLoading(false);
    }
  }

  // ── batch mode fit ──
  async function handleBatchRun() {
    setBatchError("");
    setBatchResults([]);
    const indices = [...selected].sort();
    if (!indices.length) { setBatchError("No datasets selected"); return; }
    setBatchLoading(true);

    const tasks = indices.map(idx => {
      const ds = datasets[idx];
      const { phase: p, metrics: m, atomNum: a } = parseFolderName(ds.folder_name);
      return invoke("run_fit", {
        filepath: ds.file_path,
        phase: p,
        elem: ds.folder_name.split("-").slice(1),
        metrics: m.split(" ").map(Number),
        atomNum: parseInt(a) || 1,
      })
        .then((r) => ({ label: ds.folder_name, result: r as FitResult }))
        .catch(() => ({ label: ds.folder_name, result: null as never }));
    });

    const settled = await Promise.allSettled(tasks);
    const valid = settled
      .map(s => s.status === "fulfilled" ? s.value : null)
      .filter((v): v is { label: string; result: FitResult } =>
        v !== null && v.result !== null
      );
    setBatchResults(valid);
    setBatchLoading(false);
  }

  return (
    <main className="container">
      <h1>Gibbs-Temperature Fitter</h1>
      <p className="subtitle">SGTE polynomial: A + B·T + C·T·LN(T) + D·T² + E·T³ + F/T</p>

      {/* ── mode toggle ── */}
      <div className="mode-toggle">
        <button className={`mode-btn ${mode === "single" ? "active" : ""}`} onClick={() => setMode("single")}>Single</button>
        <button className={`mode-btn ${mode === "batch" ? "active" : ""}`} onClick={() => setMode("batch")}>Batch</button>
      </div>

      {/* ── single mode ── */}
      {mode === "single" && (
        <>
          <div className="controls">
            <div className="row">
              <button onClick={pickFile}>📁 Select File</button>
              <span className="filepath">{filepath || "(no file selected)"}</span>
            </div>

            <div className="grid">
              <label>Phase<select value={phase} onChange={(e) => setPhase(e.target.value)}><option>SER</option><option>BCC</option><option>FCC</option><option>HCP</option><option>OTH</option></select></label>
              <label>Elements<input value={elem} onChange={(e) => setElem(e.target.value)} placeholder="Fe Cr" /></label>
              <label>Metrics<input value={metrics} onChange={(e) => setMetrics(e.target.value)} placeholder="1 1" /></label>
              <label>Atom count<input type="number" value={atomNum} onChange={(e) => setAtomNum(e.target.value)} min={1} /></label>
            </div>

            <button className="run-btn" onClick={handleRun} disabled={loading}>
              {loading ? "⏳ Fitting..." : "🚀 Run Fit"}
            </button>
          </div>

          {error && <pre className="error">{error}</pre>}

          {result && (
            <div className="batch-results">
              <ResultCard label="Result" result={result} />
            </div>
          )}
        </>
      )}

      {/* ── batch mode ── */}
      {mode === "batch" && (
        <>
          <div className="controls">
            <div className="row">
              <button onClick={pickFolder}>📁 Select Folder</button>
              <span className="filepath">{folderPath || "(no folder selected)"}</span>
            </div>

            {datasets.length > 0 && (
              <>
                <p className="dataset-count">📦 Found {datasets.length} datasets</p>
                <div className="dataset-list">
                  {datasets.map((ds, i) => (
                    <label key={i} className="dataset-item">
                      <input type="checkbox" checked={selected.has(i)} onChange={() => toggleSel(i)} />
                      <span>{ds.folder_name}</span>
                    </label>
                  ))}
                </div>
                <button className="run-btn" onClick={handleBatchRun} disabled={batchLoading}>
                  {batchLoading ? "⏳ Fitting..." : `🚀 Run Selected (${selected.size})`}
                </button>
              </>
            )}
          </div>

          {batchError && <pre className="error">{batchError}</pre>}

          {batchResults.length > 0 && (
            <div className="batch-results">
              <h3>Batch Results ({batchResults.length})</h3>
              {batchResults.map((br, i) => (
                <ResultCard key={i} label={br.label} result={br.result} />
              ))}
            </div>
          )}
        </>
      )}
    </main>
  );
}

export default App;
