import { useEffect, useMemo, useRef, useState } from 'react'

function makeWsUrl(apiKey) {
  const protocol = window.location.protocol === 'https:' ? 'wss' : 'ws'
  const keyPart = apiKey ? `?api_key=${encodeURIComponent(apiKey)}` : ''
  return `${protocol}://${window.location.host}/ws${keyPart}`
}

function buildPayloadTemplate(requiredFields) {
  const row = {}
  for (const field of requiredFields) {
    if (field === 'DateTime') {
      row[field] = new Date().toISOString().slice(0, 16).replace('T', ' ')
    } else {
      row[field] = null
    }
  }
  return { rows: [row] }
}

export default function App() {
  const wsRef = useRef(null)
  const [connected, setConnected] = useState(false)
  const [apiKey, setApiKey] = useState('')
  const [modelInfo, setModelInfo] = useState(null)
  const [lastMessage, setLastMessage] = useState('{}')
  const [payloadText, setPayloadText] = useState(JSON.stringify({ rows: [] }, null, 2))

  const requiredFields = useMemo(
    () => (modelInfo?.required_predict_fields ? modelInfo.required_predict_fields : []),
    [modelInfo],
  )

  function connect() {
    if (wsRef.current) {
      wsRef.current.close()
    }

    const ws = new WebSocket(makeWsUrl(apiKey))
    wsRef.current = ws

    ws.onopen = () => {
      setConnected(true)
      sendMessage({ type: 'health' })
      sendMessage({ type: 'model_info' })
    }

    ws.onclose = () => setConnected(false)

    ws.onmessage = (event) => {
      setLastMessage(event.data)
      try {
        const msg = JSON.parse(event.data)
        if (msg.type === 'model_info') {
          setModelInfo(msg.data || null)
        }
      } catch {
        // ignore malformed messages
      }
    }
  }

  function sendMessage(message) {
    if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) {
      setLastMessage(JSON.stringify({ type: 'error', detail: 'WebSocket is not connected' }, null, 2))
      return
    }
    wsRef.current.send(JSON.stringify(message))
  }

  function fillTemplateFromModelInfo() {
    const template = buildPayloadTemplate(requiredFields)
    setPayloadText(JSON.stringify(template, null, 2))
  }

  function sendPredict() {
    try {
      const payload = JSON.parse(payloadText)
      sendMessage({
        type: 'predict',
        request_id: `ui-${Date.now()}`,
        payload,
      })
    } catch {
      setLastMessage(JSON.stringify({ type: 'error', detail: 'Invalid JSON payload' }, null, 2))
    }
  }

  useEffect(() => {
    connect()
    return () => {
      if (wsRef.current) {
        wsRef.current.close()
      }
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  return (
    <div className="container">
      <h1 className="title">CTR Demo Web UI</h1>
      <p className="subtitle">Production build (Vite) + WebSocket client</p>

      <div className="card controls">
        <div className="status-row">
          <span className={connected ? 'status-ok' : 'status-bad'}>
            {connected ? 'Connected' : 'Disconnected'}
          </span>
          <input
            type="password"
            placeholder="API key (optional)"
            value={apiKey}
            onChange={(e) => setApiKey(e.target.value)}
          />
          <button onClick={connect}>Reconnect</button>
          <button onClick={() => sendMessage({ type: 'health' })} disabled={!connected}>Health</button>
          <button onClick={() => sendMessage({ type: 'model_info' })} disabled={!connected}>Model Info</button>
          <button onClick={sendPredict} disabled={!connected}>Predict</button>
        </div>
      </div>

      <div className="grid">
        <section className="card">
          <div className="section-head">
            <h2>Payload</h2>
            <button className="secondary" onClick={fillTemplateFromModelInfo} disabled={requiredFields.length === 0}>
              Fill required fields
            </button>
          </div>
          <p className="muted">required fields: {requiredFields.join(', ') || 'n/a'}</p>
          <textarea value={payloadText} onChange={(e) => setPayloadText(e.target.value)} />
        </section>

        <section className="card">
          <h2>Model Info</h2>
          <pre>{JSON.stringify(modelInfo || {}, null, 2)}</pre>
          <h2>Last WS Message</h2>
          <pre>{lastMessage}</pre>
        </section>
      </div>
    </div>
  )
}
