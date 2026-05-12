import { useState, useCallback } from "react"
import { useDropzone } from "react-dropzone"

const API = "http://localhost:8000"

// ── Palette ──────────────────────────────────────────────────
const C = {
  bg:      "#0B0F1A",
  surface: "#111827",
  card:    "#1A2235",
  border:  "#1E2D45",
  indigo:  "#6366F1",
  indigoL: "#818CF8",
  red:     "#EF4444",
  amber:   "#F59E0B",
  green:   "#22C55E",
  text:    "#F1F5F9",
  muted:   "#64748B",
  subtle:  "#334155",
}

// ── Tiny helpers ─────────────────────────────────────────────
const Badge = ({ level }) => {
  const map = { High: [C.red,"#3B0000"], Medium: [C.amber,"#2D1F00"], Low: [C.green,"#002D12"] }
  const [fg, bg] = map[level] || [C.muted, C.surface]
  return (
    <span style={{ background: bg, color: fg, border: `1px solid ${fg}44`,
      borderRadius: 6, padding: "2px 10px", fontSize: 12, fontWeight: 700, letterSpacing: 0.5 }}>
      {level}
    </span>
  )
}

const StatCard = ({ label, value, sub, color }) => (
  <div style={{ background: C.card, border: `1px solid ${C.border}`, borderRadius: 14,
    padding: "20px 24px", flex: 1, minWidth: 140 }}>
    <div style={{ fontSize: 13, color: C.muted, marginBottom: 6, letterSpacing: 0.5 }}>{label}</div>
    <div style={{ fontSize: 32, fontWeight: 800, color: color || C.text, fontFamily:"'DM Mono',monospace" }}>{value}</div>
    {sub && <div style={{ fontSize: 12, color: C.muted, marginTop: 4 }}>{sub}</div>}
  </div>
)

// ── Dropzone ─────────────────────────────────────────────────
function DropZone({ onFile, loading }) {
  const [hover, setHover] = useState(false)
  const onDrop = useCallback(files => { if (files[0]) onFile(files[0]) }, [onFile])
  const { getRootProps, getInputProps, isDragActive } = useDropzone({ onDrop, accept: { "text/csv": [".csv"] } })

  return (
    <div {...getRootProps()} onMouseEnter={() => setHover(true)} onMouseLeave={() => setHover(false)}
      style={{ border: `2px dashed ${isDragActive || hover ? C.indigo : C.border}`,
        borderRadius: 16, padding: "48px 32px", textAlign: "center", cursor: "pointer",
        background: isDragActive ? "#1A1F3A" : C.surface,
        transition: "all 0.2s", userSelect: "none" }}>
      <input {...getInputProps()} />
      <div style={{ fontSize: 40, marginBottom: 12 }}>📂</div>
      <div style={{ fontSize: 17, fontWeight: 600, color: C.text, marginBottom: 6 }}>
        {isDragActive ? "Drop it here" : "Drag & drop your CSV dataset"}
      </div>
      <div style={{ fontSize: 13, color: C.muted }}>or click to browse — must be a .csv file</div>
      {loading && (
        <div style={{ marginTop: 20, color: C.indigoL, fontSize: 14, fontWeight: 600 }}>
          ⏳ Processing...
        </div>
      )}
    </div>
  )
}

// ── Chart image ───────────────────────────────────────────────
const ChartImg = ({ src, title }) => src ? (
  <div style={{ background: C.card, border: `1px solid ${C.border}`, borderRadius: 14,
    padding: 20, flex: 1, minWidth: 280 }}>
    <div style={{ fontSize: 13, fontWeight: 600, color: C.muted, marginBottom: 12,
      letterSpacing: 0.5, textTransform: "uppercase" }}>{title}</div>
    <img src={`data:image/png;base64,${src}`} alt={title}
      style={{ width: "100%", borderRadius: 8, objectFit: "contain" }} />
  </div>
) : null

// ── Main App ─────────────────────────────────────────────────
export default function App() {
  const [mode,    setMode]    = useState("predict") // "predict" | "retrain"
  const [file,    setFile]    = useState(null)
  const [loading, setLoading] = useState(false)
  const [error,   setError]   = useState(null)
  const [result,  setResult]  = useState(null)
  const [retrain, setRetrain] = useState(null)
  const [search,  setSearch]  = useState("")
  const [filter,  setFilter]  = useState("All")  // All | High | Medium | Low
  const [page,    setPage]    = useState(1)
  const PER_PAGE = 15

  const handleFile = async (f) => {
    setFile(f); setError(null); setResult(null); setRetrain(null)
  }

  const handleRun = async () => {
    if (!file) return
    setLoading(true); setError(null)
    const form = new FormData()
    form.append("file", file)
    try {
      const endpoint = mode === "retrain" ? `${API}/retrain` : `${API}/predict`
      const res = await fetch(endpoint, { method: "POST", body: form })
      if (!res.ok) {
        const err = await res.json()
        throw new Error(err.detail || "Server error")
      }
      const data = await res.json()
      if (mode === "retrain") {
        setRetrain(data)
        // After retraining, run predict too
        const form2 = new FormData(); form2.append("file", file)
        const res2 = await fetch(`${API}/predict`, { method: "POST", body: form2 })
        if (res2.ok) setResult(await res2.json())
      } else {
        setResult(data)
      }
    } catch (e) {
      setError(e.message)
    } finally {
      setLoading(false)
    }
  }

  // Filter + search customers
  const customers = result?.customers || []
  const filtered = customers.filter(c => {
    const matchSearch = String(c.id).toLowerCase().includes(search.toLowerCase())
    const matchFilter = filter === "All" || c.risk_level === filter
    return matchSearch && matchFilter
  })
  const totalPages = Math.ceil(filtered.length / PER_PAGE)
  const paginated  = filtered.slice((page - 1) * PER_PAGE, page * PER_PAGE)

  const s = result?.summary

  return (
    <div style={{ minHeight: "100vh", background: C.bg, color: C.text,
      fontFamily: "'DM Sans', 'Segoe UI', sans-serif" }}>

      {/* ── Header ── */}
      <div style={{ borderBottom: `1px solid ${C.border}`, padding: "18px 40px",
        display: "flex", alignItems: "center", gap: 16,
        background: C.surface, position: "sticky", top: 0, zIndex: 100 }}>
        <div style={{ width: 36, height: 36, borderRadius: 10, background: C.indigo,
          display: "flex", alignItems: "center", justifyContent: "center", fontSize: 18 }}>📉</div>
        <div>
          <div style={{ fontWeight: 800, fontSize: 18, letterSpacing: -0.5 }}>ChurnIQ</div>
          <div style={{ fontSize: 12, color: C.muted }}>Customer Churn Intelligence Platform</div>
        </div>
      </div>

      <div style={{ maxWidth: 1300, margin: "0 auto", padding: "40px 32px" }}>

        {/* ── Upload section ── */}
        <div style={{ background: C.surface, border: `1px solid ${C.border}`,
          borderRadius: 18, padding: 32, marginBottom: 32 }}>

          <div style={{ display: "flex", gap: 12, marginBottom: 24 }}>
            {["predict","retrain"].map(m => (
              <button key={m} onClick={() => { setMode(m); setResult(null); setRetrain(null); setError(null) }}
                style={{ padding: "10px 24px", borderRadius: 10, border: "none", cursor: "pointer",
                  fontWeight: 700, fontSize: 14, letterSpacing: 0.3,
                  background: mode === m ? C.indigo : C.card,
                  color: mode === m ? "#fff" : C.muted,
                  transition: "all 0.2s" }}>
                {m === "predict" ? "⚡ Predict (Use Existing Model)" : "🔁 Retrain + Predict"}
              </button>
            ))}
          </div>

          <div style={{ fontSize: 13, color: C.muted, marginBottom: 20, lineHeight: 1.6 }}>
            {mode === "predict"
              ? "Upload a customer CSV to predict churn using the already-trained model (best_model.pkl must exist)."
              : "Upload a CSV with a 'Churn' column — the app will retrain all 3 models, pick the best one, then predict."}
          </div>

          <DropZone onFile={handleFile} loading={false} />

          {file && (
            <div style={{ marginTop: 16, display: "flex", alignItems: "center", gap: 12 }}>
              <div style={{ background: C.card, border: `1px solid ${C.border}`, borderRadius: 8,
                padding: "8px 16px", fontSize: 13, color: C.indigoL, flex: 1 }}>
                📄 {file.name} &nbsp;·&nbsp; {(file.size / 1024).toFixed(1)} KB
              </div>
              <button onClick={handleRun} disabled={loading}
                style={{ padding: "10px 28px", borderRadius: 10, border: "none", cursor: loading ? "not-allowed" : "pointer",
                  fontWeight: 700, fontSize: 14, background: loading ? C.subtle : C.indigo,
                  color: loading ? C.muted : "#fff", transition: "all 0.2s" }}>
                {loading ? "Running..." : mode === "predict" ? "Run Prediction" : "Retrain & Predict"}
              </button>
            </div>
          )}

          {error && (
            <div style={{ marginTop: 16, background: "#2D0A0A", border: `1px solid ${C.red}44`,
              borderRadius: 10, padding: "12px 18px", color: C.red, fontSize: 13 }}>
              ⚠️ {error}
            </div>
          )}
        </div>

        {/* ── Retrain results ── */}
        {retrain && (
          <div style={{ background: C.surface, border: `1px solid ${C.border}`,
            borderRadius: 18, padding: 28, marginBottom: 32 }}>
            <div style={{ fontWeight: 800, fontSize: 18, marginBottom: 18 }}>
              🔁 Retraining Complete — Best Model: <span style={{ color: C.indigoL }}>{retrain.best_model}</span>
              &nbsp;<span style={{ fontSize: 14, fontWeight: 500, color: C.green }}>ROC-AUC: {retrain.best_roc_auc}%</span>
            </div>
            <div style={{ display: "flex", gap: 16, flexWrap: "wrap" }}>
              {Object.entries(retrain.all_results || {}).map(([name, m]) => (
                <div key={name} style={{ background: C.card, border: `1px solid ${name === retrain.best_model ? C.indigo : C.border}`,
                  borderRadius: 12, padding: "16px 20px", minWidth: 200, flex: 1 }}>
                  <div style={{ fontWeight: 700, marginBottom: 10, color: name === retrain.best_model ? C.indigoL : C.text }}>
                    {name === retrain.best_model ? "🏆 " : ""}{name}
                  </div>
                  {[["Accuracy", m.accuracy], ["Precision", m.precision],
                    ["Recall", m.recall], ["F1", m.f1], ["ROC-AUC", m.roc_auc]].map(([k,v]) => (
                    <div key={k} style={{ display:"flex", justifyContent:"space-between",
                      fontSize: 13, marginBottom: 4 }}>
                      <span style={{ color: C.muted }}>{k}</span>
                      <span style={{ fontWeight: 600 }}>{v}%</span>
                    </div>
                  ))}
                </div>
              ))}
            </div>
          </div>
        )}

        {/* ── Summary stats ── */}
        {s && (
          <>
            <div style={{ display: "flex", gap: 16, flexWrap: "wrap", marginBottom: 24 }}>
              <StatCard label="TOTAL CUSTOMERS"     value={s.total_customers}                  />
              <StatCard label="PREDICTED TO CHURN"  value={s.churn_count}  color={C.red}       sub={`${s.churn_rate}% churn rate`} />
              <StatCard label="WILL STAY"           value={s.stay_count}   color={C.green}     />
              <StatCard label="HIGH RISK"           value={s.high_risk}    color={C.red}       sub="≥70% probability" />
              <StatCard label="REVENUE AT RISK"     value={s.revenue_at_risk ? `$${s.revenue_at_risk.toLocaleString()}` : "N/A"} color={C.amber} sub="monthly charges" />
              <StatCard label="AVG CHURN PROB"      value={`${s.avg_churn_probability}%`}      />
            </div>

            {/* ── Charts ── */}
            <div style={{ display: "flex", gap: 20, flexWrap: "wrap", marginBottom: 32 }}>
              <ChartImg src={result.charts?.risk_donut}  title="Risk Distribution" />
              <ChartImg src={result.charts?.churn_bar}   title="Churn vs Stay" />
              <ChartImg src={result.charts?.prob_hist}   title="Probability Distribution" />
              {result.charts?.shap_bar && <ChartImg src={result.charts?.shap_bar} title="Top Features (SHAP)" />}
            </div>

            {/* ── Customer table ── */}
            <div style={{ background: C.surface, border: `1px solid ${C.border}`,
              borderRadius: 18, padding: 28 }}>
              <div style={{ display: "flex", justifyContent: "space-between",
                alignItems: "center", marginBottom: 20, flexWrap: "wrap", gap: 12 }}>
                <div style={{ fontWeight: 800, fontSize: 18 }}>
                  Customer Predictions &nbsp;
                  <span style={{ fontSize: 13, fontWeight: 500, color: C.muted }}>
                    ({filtered.length} customers)
                  </span>
                </div>
                <div style={{ display: "flex", gap: 10 }}>
                  <input value={search} onChange={e => { setSearch(e.target.value); setPage(1) }}
                    placeholder="Search customer ID..."
                    style={{ padding: "8px 14px", borderRadius: 8, border: `1px solid ${C.border}`,
                      background: C.card, color: C.text, fontSize: 13, outline: "none", width: 200 }} />
                  {["All","High","Medium","Low"].map(f => (
                    <button key={f} onClick={() => { setFilter(f); setPage(1) }}
                      style={{ padding: "8px 14px", borderRadius: 8, border: "none", cursor: "pointer",
                        fontSize: 13, fontWeight: 600,
                        background: filter === f ? C.indigo : C.card,
                        color: filter === f ? "#fff" : C.muted }}>
                      {f}
                    </button>
                  ))}
                </div>
              </div>

              {/* Table */}
              <div style={{ overflowX: "auto" }}>
                <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 13 }}>
                  <thead>
                    <tr style={{ borderBottom: `1px solid ${C.border}` }}>
                      {["Customer ID","Prediction","Risk Level","Churn Prob.","Tenure (mo.)","Monthly Charges","Contract"].map(h => (
                        <th key={h} style={{ textAlign: "left", padding: "10px 16px",
                          color: C.muted, fontWeight: 600, letterSpacing: 0.5,
                          textTransform: "uppercase", fontSize: 11 }}>{h}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {paginated.map((c, i) => (
                      <tr key={i} style={{ borderBottom: `1px solid ${C.border}22`,
                        background: i % 2 === 0 ? "transparent" : "#0D1424" }}>
                        <td style={{ padding: "12px 16px", fontWeight: 600, color: C.indigoL }}>{c.id}</td>
                        <td style={{ padding: "12px 16px" }}>
                          <span style={{ color: c.prediction === "Will Churn" ? C.red : C.green, fontWeight: 700 }}>
                            {c.prediction === "Will Churn" ? "⚠ Will Churn" : "✓ Will Stay"}
                          </span>
                        </td>
                        <td style={{ padding: "12px 16px" }}><Badge level={c.risk_level} /></td>
                        <td style={{ padding: "12px 16px" }}>
                          <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                            <div style={{ width: 60, height: 6, borderRadius: 3, background: C.border, overflow: "hidden" }}>
                              <div style={{ width: `${c.churn_probability}%`, height: "100%",
                                background: c.churn_probability >= 70 ? C.red : c.churn_probability >= 40 ? C.amber : C.green,
                                borderRadius: 3 }} />
                            </div>
                            <span style={{ fontFamily: "'DM Mono',monospace", fontWeight: 600 }}>
                              {c.churn_probability}%
                            </span>
                          </div>
                        </td>
                        <td style={{ padding: "12px 16px", color: C.muted }}>{c.tenure ?? "—"}</td>
                        <td style={{ padding: "12px 16px", fontFamily: "'DM Mono',monospace" }}>
                          {c.monthly_charges != null ? `$${c.monthly_charges.toFixed(2)}` : "—"}
                        </td>
                        <td style={{ padding: "12px 16px", color: C.muted }}>{c.contract ?? "—"}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>

              {/* Pagination */}
              {totalPages > 1 && (
                <div style={{ display: "flex", justifyContent: "center", gap: 8, marginTop: 20 }}>
                  <button onClick={() => setPage(p => Math.max(1, p-1))} disabled={page===1}
                    style={{ padding: "6px 14px", borderRadius: 8, border: `1px solid ${C.border}`,
                      background: C.card, color: page===1 ? C.muted : C.text, cursor: page===1 ? "not-allowed":"pointer" }}>
                    ‹
                  </button>
                  {Array.from({length: totalPages}, (_,i) => i+1)
                    .filter(p => p===1 || p===totalPages || Math.abs(p-page)<=2)
                    .map((p, idx, arr) => (
                      <>
                        {idx > 0 && arr[idx-1] !== p-1 && <span key={`e${p}`} style={{color:C.muted,padding:"6px 4px"}}>…</span>}
                        <button key={p} onClick={() => setPage(p)}
                          style={{ padding: "6px 12px", borderRadius: 8, border: "none", cursor: "pointer",
                            background: page===p ? C.indigo : C.card,
                            color: page===p ? "#fff" : C.muted, fontWeight: page===p ? 700 : 400 }}>
                          {p}
                        </button>
                      </>
                    ))}
                  <button onClick={() => setPage(p => Math.min(totalPages, p+1))} disabled={page===totalPages}
                    style={{ padding: "6px 14px", borderRadius: 8, border: `1px solid ${C.border}`,
                      background: C.card, color: page===totalPages ? C.muted : C.text, cursor: page===totalPages ? "not-allowed":"pointer" }}>
                    ›
                  </button>
                </div>
              )}
            </div>
          </>
        )}
      </div>
    </div>
  )
}
