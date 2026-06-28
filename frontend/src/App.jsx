import { useState, useCallback } from "react";

const API_URL = "http://localhost:8000";


const CLASS_COLORS = {
  "glioma":      { bg: "#fef2f2", border: "#ef4444", text: "#b91c1c", badge: "#ef4444" },
  "meningioma":  { bg: "#fff7ed", border: "#f97316", text: "#c2410c", badge: "#f97316" },
  "notumor":     { bg: "#f0fdf4", border: "#22c55e", text: "#15803d", badge: "#22c55e" },
  "pituitary":   { bg: "#fefce8", border: "#eab308", text: "#a16207", badge: "#eab308" },
};

const SEVERITY_LABELS = {
  "High":   { label: "High Risk",     color: "#ef4444" },
  "Medium": { label: "Moderate Risk", color: "#f97316" },
  "None":   { label: "No Risk",       color: "#22c55e" },
};

export default function App() {
  const [file, setFile]         = useState(null);
  const [preview, setPreview]   = useState(null);
  const [result, setResult]     = useState(null);
  const [loading, setLoading]   = useState(false);
  const [error, setError]       = useState(null);
  const [dragging, setDragging] = useState(false);

  const handleFile = (f) => {
    if (!f) return;
    setFile(f);
    setResult(null);
    setError(null);
    setPreview(URL.createObjectURL(f));
  };

  const onDrop = useCallback((e) => {
    e.preventDefault();
    setDragging(false);
    const f = e.dataTransfer.files[0];
    if (f && (f.type === "image/jpeg" || f.type === "image/png")) handleFile(f);
  }, []);

  const onDragOver  = (e) => { e.preventDefault(); setDragging(true); };
  const onDragLeave = () => setDragging(false);

  const handlePredict = async () => {
    if (!file) return;
    setLoading(true);
    setError(null);
    setResult(null);

    const formData = new FormData();
    formData.append("file", file);

    try {
      const res = await fetch(`${API_URL}/predict`, { method: "POST", body: formData });
      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail || "Prediction failed");
      }
      const data = await res.json();
      setResult(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const reset = () => {
    setFile(null);
    setPreview(null);
    setResult(null);
    setError(null);
  };

  
  const colors   = result ? CLASS_COLORS[result.predicted_class] : null;
  const severity = result ? SEVERITY_LABELS[result.info.severity] : null;
  const sortedProbs = result
    ? Object.entries(result.probabilities).sort((a, b) => b[1] - a[1])
    : [];

  return (
    <div style={{
      minHeight: "100vh",
      background: "linear-gradient(135deg, #0f172a 0%, #1e1b4b 50%, #0f172a 100%)",
      fontFamily: "'Inter', system-ui, sans-serif",
      color: "#f8fafc",
    }}>

      {/* ── Header ── */}
      <header style={{
        borderBottom: "1px solid rgba(255,255,255,0.08)",
        padding: "18px 32px",
        display: "flex", alignItems: "center", gap: "12px",
        backdropFilter: "blur(10px)",
        background: "rgba(15,23,42,0.6)",
        position: "sticky", top: 0, zIndex: 10,
      }}>
        <div style={{
          width: 38, height: 38, borderRadius: "10px",
          background: "linear-gradient(135deg, #6366f1, #8b5cf6)",
          display: "flex", alignItems: "center", justifyContent: "center",
          fontSize: 20,
        }}>🧠</div>
        <div>
          <div style={{ fontWeight: 700, fontSize: 16, letterSpacing: "-0.3px" }}>
            NeuroScan AI
          </div>
          <div style={{ fontSize: 11, color: "#94a3b8", letterSpacing: "0.5px" }}>
            BRAIN TUMOR DETECTION SYSTEM
          </div>
        </div>
        <div style={{ marginLeft: "auto" }}>
          <span style={{
            background: "#22c55e22", color: "#4ade80",
            border: "1px solid #22c55e44",
            borderRadius: 20, padding: "4px 12px",
            fontSize: 11, fontWeight: 600,
          }}>● LIVE</span>
        </div>
      </header>

      <main style={{ maxWidth: 900, margin: "0 auto", padding: "40px 24px" }}>

        {/* ── Hero ── */}
        <div style={{ textAlign: "center", marginBottom: 40 }}>
          <h1 style={{
            fontSize: "clamp(28px, 5vw, 42px)",
            fontWeight: 800, margin: "0 0 12px",
            background: "linear-gradient(135deg, #e2e8f0, #a5b4fc)",
            WebkitBackgroundClip: "text",
            WebkitTextFillColor: "transparent",
            letterSpacing: "-1px",
          }}>
            Brain Tumor MRI Analysis
          </h1>
          <p style={{ color: "#94a3b8", fontSize: 15, margin: 0 }}>
            Upload an MRI scan for instant AI-powered tumor classification
          </p>
        </div>

        <div style={{
          display: "grid",
          gridTemplateColumns: result ? "1fr 1fr" : "1fr",
          gap: 24, alignItems: "start"
        }}>

          {/* ── Upload Panel ── */}
          <div style={{
            background: "rgba(255,255,255,0.04)",
            border: "1px solid rgba(255,255,255,0.08)",
            borderRadius: 20, padding: 28,
          }}>
            <h2 style={{ margin: "0 0 20px", fontSize: 15, fontWeight: 600, color: "#cbd5e1" }}>
              Upload MRI Scan
            </h2>

            {/* Drop Zone */}
            <div
              onDrop={onDrop}
              onDragOver={onDragOver}
              onDragLeave={onDragLeave}
              onClick={() => !preview && document.getElementById("fileInput").click()}
              style={{
                border: `2px dashed ${dragging ? "#6366f1" : preview ? "#6366f144" : "rgba(255,255,255,0.15)"}`,
                borderRadius: 14,
                padding: preview ? 0 : "40px 20px",
                textAlign: "center",
                cursor: preview ? "default" : "pointer",
                transition: "all 0.2s",
                background: dragging ? "rgba(99,102,241,0.08)" : "transparent",
                overflow: "hidden",
                minHeight: preview ? 0 : 180,
                display: "flex", alignItems: "center", justifyContent: "center",
              }}
            >
              {preview ? (
                <img src={preview} alt="MRI Preview"
                  style={{ width: "100%", borderRadius: 12, display: "block" }} />
              ) : (
                <div>
                  <div style={{ fontSize: 40, marginBottom: 12 }}>🧠</div>
                  <div style={{ fontWeight: 600, fontSize: 14, color: "#e2e8f0", marginBottom: 6 }}>
                    Drop MRI image here
                  </div>
                  <div style={{ fontSize: 12, color: "#64748b" }}>or click to browse · JPG, PNG</div>
                </div>
              )}
            </div>

            <input id="fileInput" type="file" accept="image/jpeg,image/png"
              style={{ display: "none" }}
              onChange={(e) => handleFile(e.target.files[0])} />

            {/* Buttons */}
            {preview ? (
              <div style={{ marginTop: 14, display: "flex", gap: 10 }}>
                <button onClick={() => document.getElementById("fileInput").click()} style={{
                  flex: 1, padding: "10px", borderRadius: 10,
                  background: "rgba(255,255,255,0.06)",
                  border: "1px solid rgba(255,255,255,0.1)",
                  color: "#94a3b8", cursor: "pointer", fontSize: 13,
                }}>Change Image</button>

                <button onClick={handlePredict} disabled={loading} style={{
                  flex: 2, padding: "10px 20px", borderRadius: 10,
                  background: loading ? "#4338ca99" : "linear-gradient(135deg, #6366f1, #8b5cf6)",
                  border: "none", color: "white",
                  cursor: loading ? "not-allowed" : "pointer",
                  fontWeight: 700, fontSize: 14,
                  display: "flex", alignItems: "center", justifyContent: "center", gap: 8,
                }}>
                  {loading ? (
                    <>
                      <span style={{ animation: "spin 1s linear infinite", display: "inline-block" }}>⟳</span>
                      Analysing...
                    </>
                  ) : "Analyse Scan →"}
                </button>
              </div>
            ) : (
              <button onClick={() => document.getElementById("fileInput").click()} style={{
                width: "100%", marginTop: 14, padding: "12px", borderRadius: 12,
                background: "linear-gradient(135deg, #6366f1, #8b5cf6)",
                border: "none", color: "white",
                cursor: "pointer", fontWeight: 700, fontSize: 14,
              }}>
                Select MRI Image
              </button>
            )}

            {/* Error */}
            {error && (
              <div style={{
                marginTop: 14, padding: "12px 16px",
                background: "#fef2f2", border: "1px solid #ef4444",
                borderRadius: 10, color: "#b91c1c", fontSize: 13,
              }}>
                ⚠️ {error}
              </div>
            )}


          </div>

          {/* ── Result Panel ── */}
          {result && colors && (
            <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>

              {/* Main Result Card */}
              <div style={{
                background: colors.bg,
                border: `1px solid ${colors.border}`,
                borderRadius: 20, padding: 24,
              }}>
                <div style={{
                  display: "flex", justifyContent: "space-between",
                  alignItems: "flex-start", marginBottom: 16
                }}>
                  <div>
                    <div style={{
                      fontSize: 11, fontWeight: 600,
                      color: colors.text, letterSpacing: 1, marginBottom: 4
                    }}>
                      PREDICTION RESULT
                    </div>
                    {/* Use display_name for friendly name */}
                    <div style={{
                      fontSize: 28, fontWeight: 800,
                      color: colors.text, letterSpacing: "-0.5px"
                    }}>
                      {result.display_name}
                    </div>
                  </div>
                  <div style={{
                    background: colors.badge, color: "white",
                    borderRadius: 20, padding: "6px 14px",
                    fontSize: 13, fontWeight: 700,
                  }}>
                    {result.confidence.toFixed(1)}% confident
                  </div>
                </div>

                {/* Confidence Bar */}
                <div style={{ marginBottom: 14 }}>
                  <div style={{
                    height: 8, borderRadius: 4,
                    background: `${colors.border}30`, overflow: "hidden",
                  }}>
                    <div style={{
                      height: "100%", borderRadius: 4,
                      background: colors.badge,
                      width: `${result.confidence}%`,
                      transition: "width 1s ease",
                    }} />
                  </div>
                </div>

                <p style={{ margin: "0 0 12px", fontSize: 13, color: colors.text, lineHeight: 1.5 }}>
                  {result.info.description}
                </p>

                {severity && (
                  <span style={{
                    background: `${severity.color}20`, color: severity.color,
                    border: `1px solid ${severity.color}44`,
                    borderRadius: 20, padding: "4px 12px",
                    fontSize: 11, fontWeight: 600,
                  }}>
                    {severity.label}
                  </span>
                )}
              </div>

              {/* Probability Breakdown */}
              <div style={{
                background: "rgba(255,255,255,0.04)",
                border: "1px solid rgba(255,255,255,0.08)",
                borderRadius: 20, padding: 20,
              }}>
                <div style={{ fontSize: 13, fontWeight: 600, color: "#cbd5e1", marginBottom: 16 }}>
                  Class Probabilities
                </div>
                <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
                  {sortedProbs.map(([cls, prob]) => {
                    const c = CLASS_COLORS[cls] || CLASS_COLORS["notumor"];
                    return (
                      <div key={cls}>
                        <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 4 }}>
                          <span style={{ fontSize: 12, color: "#94a3b8", textTransform: "capitalize" }}>
                            {cls === "notumor" ? "No Tumor" : cls}
                          </span>
                          <span style={{ fontSize: 12, fontWeight: 600, color: "#e2e8f0" }}>
                            {prob.toFixed(1)}%
                          </span>
                        </div>
                        <div style={{ height: 6, borderRadius: 3, background: "rgba(255,255,255,0.06)" }}>
                          <div style={{
                            height: "100%", borderRadius: 3,
                            background: c.badge,
                            width: `${prob}%`,
                            transition: "width 0.8s ease",
                          }} />
                        </div>
                      </div>
                    );
                  })}
                </div>
              </div>

              {/* Reset */}
              <button onClick={reset} style={{
                padding: "12px", borderRadius: 12,
                background: "transparent",
                border: "1px solid rgba(255,255,255,0.12)",
                color: "#94a3b8", cursor: "pointer", fontSize: 13,
              }}>
                ← Analyse Another Scan
              </button>
            </div>
          )}
        </div>

        {/* ── How It Works ── */}
        {!result && (
          <div style={{
            marginTop: 40,
            display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: 16,
          }}>
            {[
              { icon: "🧠", title: "Upload MRI",   desc: "Upload a brain MRI scan in JPG or PNG format" },
              { icon: "⚡", title: "AI Analysis",  desc: "Vision Transformer (ViT) analyses the scan in seconds" },
              { icon: "📊", title: "Get Results",  desc: "Receive classification with confidence scores" },
            ].map((step, i) => (
              <div key={i} style={{
                background: "rgba(255,255,255,0.03)",
                border: "1px solid rgba(255,255,255,0.06)",
                borderRadius: 16, padding: 20, textAlign: "center",
              }}>
                <div style={{ fontSize: 28, marginBottom: 10 }}>{step.icon}</div>
                <div style={{ fontWeight: 600, fontSize: 14, marginBottom: 6 }}>{step.title}</div>
                <div style={{ fontSize: 12, color: "#64748b", lineHeight: 1.5 }}>{step.desc}</div>
              </div>
            ))}
          </div>
        )}
      </main>

      <style>{`
        @keyframes spin { from { transform: rotate(0deg); } to { transform: rotate(360deg); } }
        * { box-sizing: border-box; }
      `}</style>
    </div>
  );
}
