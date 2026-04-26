import { useEffect, useMemo, useState } from 'react'
import './App.css'

type DashboardMetricsModel = {
  display_name: string
  color: string
  metrics: {
    acc_test_plus: number | null
    acc_test_21: number | null
    attack_recall_21: number | null
    train_val_gap: number | null
  }
}

type DashboardMetricsResponse = {
  generated_at: string
  models: Record<string, DashboardMetricsModel>
}

type RealtimeRow = {
  row_number: number
  stage1_decision: string
  stage1_p_attack: number | null
  stage2_category: string | null
  stage2_confidence: number | null
  final_prediction: string
  decision_path: string | null
}

type RealtimeModelResult = {
  available: boolean
  error?: string
  stage1: {
    total: number
    normal: number
    attack: number
  }
  stage2: {
    total_attacks: number
    categories: Record<string, number>
  }
  preview_rows: RealtimeRow[]
}

type RealtimeResponse = {
  generated_at: string
  input_file: string
  models: Record<string, RealtimeModelResult>
}

type DatasetModelResult = {
  available: boolean
  source: string | null
  pipeline: string | null
  warning?: string
  error?: string
  stage1: {
    total: number
    normal: number
    attack: number
  }
  stage2: {
    total_attacks: number
    categories: Record<string, number>
    mode?: 'categorized' | 'binary_only'
  }
}

type DatasetResultItem = {
  name: string
  path: string
  models: Record<string, DatasetModelResult>
}

type DatasetResultsResponse = {
  generated_at: string
  datasets: Record<string, DatasetResultItem>
}

const MODEL_ORDER = ['xgboost', 'svm', 'random_forest']

function labelFromKey(key: string) {
  if (key === 'xgboost') return 'XGBoost'
  if (key === 'svm') return 'SVM'
  if (key === 'random_forest') return 'Random Forest'
  return key
}

function scoreText(value: number | null) {
  if (value === null || Number.isNaN(value)) return 'N/A'
  return `${value.toFixed(2)}%`
}

function App() {
  const [theme, setTheme] = useState<'light' | 'dark'>('light')
  const [metrics, setMetrics] = useState<DashboardMetricsResponse | null>(null)
  const [realtime, setRealtime] = useState<RealtimeResponse | null>(null)
  const [datasetResults, setDatasetResults] = useState<DatasetResultsResponse | null>(null)
  const [selectedFile, setSelectedFile] = useState<File | null>(null)
  const [livePolling, setLivePolling] = useState(true)
  const [loadingRealtime, setLoadingRealtime] = useState(false)
  const [loadingDatasetResults, setLoadingDatasetResults] = useState(false)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    document.documentElement.setAttribute('data-theme', theme)
  }, [theme])

  useEffect(() => {
    const loadMetrics = async () => {
      try {
        const response = await fetch('/api/dashboard-metrics')
        if (!response.ok) throw new Error('Failed to load dashboard metrics')
        const data = (await response.json()) as DashboardMetricsResponse
        setMetrics(data)
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Failed to fetch metrics')
      }
    }

    void loadMetrics()
    const timer = window.setInterval(loadMetrics, 8000)
    return () => window.clearInterval(timer)
  }, [])

  const fetchDatasetResults = async () => {
    setLoadingDatasetResults(true)
    try {
      const response = await fetch('/api/dataset-results')
      const data = (await response.json()) as DatasetResultsResponse | { error: string }
      if (!response.ok || 'error' in data) {
        throw new Error('error' in data ? data.error : 'Failed to load dataset results')
      }
      setDatasetResults(data)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load dataset results')
    } finally {
      setLoadingDatasetResults(false)
    }
  }

  useEffect(() => {
    void fetchDatasetResults()
  }, [])

  const fetchRealtime = async () => {
    setLoadingRealtime(true)
    setError(null)
    try {
      const response = await fetch('/api/realtime/predict-all?limit=20')
      const data = (await response.json()) as RealtimeResponse | { error: string }
      if (!response.ok || 'error' in data) {
        throw new Error('error' in data ? data.error : 'Prediction failed')
      }
      setRealtime(data)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Prediction failed')
    } finally {
      setLoadingRealtime(false)
    }
  }

  useEffect(() => {
    if (!livePolling) return
    const timer = window.setInterval(() => {
      void fetchRealtime()
    }, 5000)
    return () => window.clearInterval(timer)
  }, [livePolling])

  const uploadAndRun = async () => {
    if (!selectedFile) {
      setError('Choose a CSV file first')
      return
    }

    setError(null)
    setLoadingRealtime(true)

    try {
      const formData = new FormData()
      formData.append('file', selectedFile)

      const uploadResponse = await fetch('/api/realtime/upload', {
        method: 'POST',
        body: formData,
      })

      const uploadData = (await uploadResponse.json()) as { error?: string }
      if (!uploadResponse.ok || uploadData.error) {
        throw new Error(uploadData.error || 'File upload failed')
      }

      await fetchRealtime()
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unable to upload and run predictions')
      setLoadingRealtime(false)
    }
  }

  const runtimeCards = useMemo(() => {
    if (!realtime) return []
    return MODEL_ORDER.map((key) => ({ key, data: realtime.models[key] }))
  }, [realtime])

  return (
    <div className="app-shell">
      <header className="topbar">
        <div>
          <p className="eyebrow">CyberSentinel</p>
          <h1>Intrusion Analytics Dashboard</h1>
          <p className="subtitle">
            Live model view for XGBoost, SVM, and Random Forest with two-stage attack intelligence.
          </p>
        </div>
        <div className="topbar-actions">
          <button
            className="ghost-btn"
            onClick={() => setTheme((prev) => (prev === 'dark' ? 'light' : 'dark'))}
          >
            {theme === 'dark' ? 'Switch to Light' : 'Switch to Dark'}
          </button>
          <button className="primary-btn" onClick={() => void fetchRealtime()}>
            Run Now
          </button>
        </div>
      </header>

      <section className="panel upload-panel">
        <div className="upload-grid">
          <label className="file-picker">
            <span>Upload input CSV for scoring</span>
            <input
              type="file"
              accept=".csv"
              onChange={(event) => setSelectedFile(event.target.files?.[0] || null)}
            />
          </label>
          <button className="primary-btn" onClick={() => void uploadAndRun()} disabled={loadingRealtime}>
            {loadingRealtime ? 'Processing...' : 'Upload and Predict All Models'}
          </button>
          <label className="toggle-row">
            <input
              type="checkbox"
              checked={livePolling}
              onChange={(event) => setLivePolling(event.target.checked)}
            />
            <span>Auto refresh every 5 seconds</span>
          </label>
        </div>
        {error && <p className="error-text">{error}</p>}
      </section>

      <section className="panel">
        <h2>Model Quality Snapshot</h2>
        <div className="cards-grid">
          {MODEL_ORDER.map((key) => {
            const card = metrics?.models[key]
            return (
              <article className="metric-card" key={key}>
                <h3>{card?.display_name || labelFromKey(key)}</h3>
                <p>Accuracy (KDDTest+): {scoreText(card?.metrics.acc_test_plus ?? null)}</p>
                <p>Accuracy (KDDTest-21): {scoreText(card?.metrics.acc_test_21 ?? null)}</p>
                <p>Attack Recall: {scoreText(card?.metrics.attack_recall_21 ?? null)}</p>
                <p>Train/Val Gap: {scoreText(card?.metrics.train_val_gap ?? null)}</p>
              </article>
            )
          })}
        </div>
      </section>

      <section className="panel">
        <h2>Result Process (Two Stages)</h2>
        <div className="stage-format-box">
          <p>
            <strong>Stage 1: Binary Classification</strong>
          </p>
          <p>Normal</p>
          <p>Attack</p>
          <p>
            <strong>Stage 2: Attack Classification</strong>
          </p>
          <p>If attack is detected, classify into attack category (DoS, Probe, R2L, U2R, and others if present).</p>
        </div>
      </section>

      <section className="panel">
        <h2>Two-Stage Inference</h2>
        <p className="subtitle small">
          Stage 1 = Binary (Normal vs Attack), Stage 2 = Attack Category (DoS, Probe, R2L, U2R, or other trained class).
        </p>

        <div className="cards-grid realtime">
          {runtimeCards.map(({ key, data }) => (
            <article className="metric-card realtime-card" key={key}>
              <h3>{labelFromKey(key)}</h3>
              {!data?.available && <p className="error-text">{data?.error || 'Model unavailable'}</p>}
              {data?.available && (
                <>
                  <div className="stage-block">
                    <h4>Stage 1 Binary</h4>
                    <p>Total: {data.stage1.total}</p>
                    <p>Normal: {data.stage1.normal}</p>
                    <p>Attack: {data.stage1.attack}</p>
                  </div>

                  <div className="stage-block">
                    <h4>Stage 2 Categories</h4>
                    <p>Total Attacks: {data.stage2.total_attacks}</p>
                    {Object.keys(data.stage2.categories).length === 0 && <p>No attack categories detected.</p>}
                    {Object.entries(data.stage2.categories).map(([category, count]) => (
                      <p key={category}>
                        {category}: {count}
                      </p>
                    ))}
                  </div>

                  <div className="table-wrap">
                    <table>
                      <thead>
                        <tr>
                          <th>#</th>
                          <th>Stage 1</th>
                          <th>P(Attack)</th>
                          <th>Stage 2</th>
                          <th>Final</th>
                        </tr>
                      </thead>
                      <tbody>
                        {data.preview_rows.slice(0, 8).map((row) => (
                          <tr key={row.row_number}>
                            <td>{row.row_number}</td>
                            <td>{row.stage1_decision}</td>
                            <td>{row.stage1_p_attack === null ? 'N/A' : row.stage1_p_attack.toFixed(3)}</td>
                            <td>{row.stage2_category || 'N/A'}</td>
                            <td>{row.final_prediction}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </>
              )}
            </article>
          ))}
        </div>
      </section>

      <section className="panel">
        <div className="panel-header-row">
          <h2>Results From Existing data/test Outputs</h2>
          <button className="ghost-btn" onClick={() => void fetchDatasetResults()} disabled={loadingDatasetResults}>
            {loadingDatasetResults ? 'Refreshing...' : 'Refresh Dataset Results'}
          </button>
        </div>
        <p className="subtitle small">
          These cards are built directly from the two project datasets in data/test (KDDTest+ and KDDTest-21),
          and from existing model output files when found.
        </p>

        {datasetResults &&
          Object.entries(datasetResults.datasets).map(([datasetKey, dataset]) => (
            <div key={datasetKey} className="dataset-group">
              <h3>{dataset.name}</h3>
              <p className="subtitle tiny">Source: {dataset.path}</p>

              <div className="cards-grid realtime">
                {MODEL_ORDER.map((modelKey) => {
                  const item = dataset.models[modelKey]
                  return (
                    <article className="metric-card realtime-card" key={`${datasetKey}-${modelKey}`}>
                      <h4>{labelFromKey(modelKey)}</h4>
                      {!item?.available && <p className="error-text">{item?.error || 'No output available'}</p>}
                      {item?.available && (
                        <>
                          <p className="subtitle tiny">Pipeline: {item.pipeline || 'unknown'}</p>
                          <p className="subtitle tiny">Output: {item.source || 'n/a'}</p>
                          {item.warning && <p className="error-text">{item.warning}</p>}
                          <div className="stage-block">
                            <h4>Stage 1 Binary</h4>
                            <p>Total: {item.stage1.total}</p>
                            <p>Normal: {item.stage1.normal}</p>
                            <p>Attack: {item.stage1.attack}</p>
                          </div>
                          <div className="stage-block">
                            <h4>Stage 2 Categories</h4>
                            <p>Total Attacks: {item.stage2.total_attacks}</p>
                            <p className="subtitle tiny">Mode: {item.stage2.mode || 'categorized'}</p>
                            {Object.keys(item.stage2.categories).length === 0 && <p>No attack categories detected.</p>}
                            {Object.entries(item.stage2.categories).map(([category, count]) => (
                              <p key={category}>
                                {category}: {count}
                              </p>
                            ))}
                          </div>
                        </>
                      )}
                    </article>
                  )
                })}
              </div>
            </div>
          ))}
      </section>
    </div>
  )
}

export default App
