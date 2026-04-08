import { useState } from 'react'
import './App.css'

const API_BASE = '/analyze'

function App() {
  const [text, setText] = useState('')
  const [results, setResults] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [analysisType, setAnalysisType] = useState(null)

  async function callApi(endpoint, body) {
    setLoading(true)
    setError(null)
    setResults(null)
    try {
      const res = await fetch(`${API_BASE}${endpoint}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      })
      if (!res.ok) {
        const err = await res.json()
        throw new Error(err.detail || 'Request failed')
      }
      return await res.json()
    } catch (e) {
      setError(e.message)
      return null
    } finally {
      setLoading(false)
    }
  }

  async function handleSentiment() {
    const data = await callApi('/sentiment/', { text })
    if (data) { setResults(data); setAnalysisType('sentiment') }
  }

  async function handleNer() {
    const data = await callApi('/ner/', { text })
    if (data) { setResults(data); setAnalysisType('ner') }
  }

  async function handleSummarize() {
    const data = await callApi('/summarize/', { text })
    if (data) { setResults(data); setAnalysisType('summarize') }
  }

  async function handleAll() {
    const data = await callApi('/all/', { text })
    if (data) { setResults(data); setAnalysisType('all') }
  }

  return (
    <>
      <header className="header">
        <h1>NLP Analysis API</h1>
        <p>Sentiment Analysis &middot; Named Entity Recognition &middot; Text Summarization</p>
      </header>

      <div className="app-container">
        <div className="input-section glass-panel">
          <h2>Input Text</h2>
          <textarea
            className="input-area"
            placeholder="Enter or paste your text here..."
            value={text}
            onChange={(e) => setText(e.target.value)}
          />
          <div className="controls">
            <button className="btn-primary" onClick={handleSentiment} disabled={loading || !text.trim()}>
              Sentiment
            </button>
            <button className="btn-primary" onClick={handleNer} disabled={loading || !text.trim()}>
              NER
            </button>
            <button className="btn-primary" onClick={handleSummarize} disabled={loading || !text.trim()}>
              Summarize
            </button>
            <button className="btn-primary" onClick={handleAll} disabled={loading || !text.trim()}>
              Analyze All
            </button>
          </div>
        </div>

        <div className="result-section glass-panel">
          <h2>Analysis Results</h2>

          {loading && <div className="loader"></div>}

          {error && (
            <div className="result-card">
              <p style={{ color: '#ff4757' }}>{error}</p>
            </div>
          )}

          {!loading && !error && !results && (
            <div className="result-card">
              <p style={{ color: 'var(--text-muted)' }}>Enter text and select an analysis type to see results.</p>
            </div>
          )}

          {results && analysisType === 'sentiment' && <SentimentResult data={results} />}
          {results && analysisType === 'ner' && <NerResult data={results} />}
          {results && analysisType === 'summarize' && <SummarizeResult data={results} />}
          {results && analysisType === 'all' && <AllResult data={results} />}
        </div>
      </div>
    </>
  )
}

function SentimentResult({ data }) {
  const badge = data.label === 'POSITIVE' ? 'positive' : 'negative'
  return (
    <div className="result-card">
      <h3>Sentiment Analysis</h3>
      <span className={`sentiment-badge ${badge}`}>{data.label}</span>
      <p style={{ marginTop: '1rem', color: 'var(--text-muted)' }}>
        Confidence: {(data.confidence * 100).toFixed(2)}%
      </p>
    </div>
  )
}

function NerResult({ data }) {
  return (
    <div className="result-card">
      <h3>Named Entity Recognition</h3>
      {data.entities.length === 0 ? (
        <p style={{ color: 'var(--text-muted)' }}>No entities found.</p>
      ) : (
        <>
          <p style={{ color: 'var(--text-muted)', marginBottom: '0.8rem' }}>
            Found {data.count} {data.count === 1 ? 'entity' : 'entities'}
          </p>
          <div>
            {data.entities.map((e, i) => (
              <span className="entity-pill" key={i}>
                <span>{e.entity}</span>{e.word}
              </span>
            ))}
          </div>
        </>
      )}
    </div>
  )
}

function SummarizeResult({ data }) {
  return (
    <div className="result-card">
      <h3>Text Summarization</h3>
      <p style={{ lineHeight: 1.7 }}>{data.summary}</p>
      <p style={{ marginTop: '1rem', color: 'var(--text-muted)', fontSize: '0.9rem' }}>
        {data.original_length} words → {data.summary_length} words
      </p>
    </div>
  )
}

function AllResult({ data }) {
  const badge = data.sentiment.label === 'POSITIVE' ? 'positive' : 'negative'
  return (
    <>
      <div className="result-card">
        <h3>Sentiment</h3>
        <span className={`sentiment-badge ${badge}`}>{data.sentiment.label}</span>
        <span style={{ marginLeft: '1rem', color: 'var(--text-muted)' }}>
          {(data.sentiment.confidence * 100).toFixed(2)}%
        </span>
      </div>

      <div className="result-card">
        <h3>Named Entities</h3>
        {data.entities.length === 0 ? (
          <p style={{ color: 'var(--text-muted)' }}>No entities found.</p>
        ) : (
          <div>
            {data.entities.map((e, i) => (
              <span className="entity-pill" key={i}>
                <span>{e.entity}</span>{e.word}
              </span>
            ))}
          </div>
        )}
      </div>

      {data.summary ? (
        <div className="result-card">
          <h3>Summary</h3>
          <p style={{ lineHeight: 1.7 }}>{data.summary}</p>
        </div>
      ) : (
        <div className="result-card">
          <h3>Summary</h3>
          <p style={{ color: 'var(--text-muted)' }}>Text too short to summarize (need 30+ words).</p>
        </div>
      )}
    </>
  )
}

export default App
