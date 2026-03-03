import { useEffect, useMemo, useRef, useState } from 'react'

function makeWsUrl(apiKey) {
  const protocol = window.location.protocol === 'https:' ? 'wss' : 'ws'
  const keyPart = apiKey ? `?api_key=${encodeURIComponent(apiKey)}` : ''
  return `${protocol}://${window.location.host}/ws${keyPart}`
}

function defaultValueForField(field) {
  if (field === 'DateTime') {
    return new Date().toISOString().slice(0, 16).replace('T', ' ')
  }
  return ''
}

function buildRowFromFields(fields) {
  const row = {}
  fields.forEach((field) => {
    row[field] = defaultValueForField(field)
  })
  return row
}

function normalizeRowForPayload(row) {
  const normalized = {}
  Object.entries(row).forEach(([key, value]) => {
    if (value === '') {
      normalized[key] = null
      return
    }
    if (typeof value !== 'string') {
      normalized[key] = value
      return
    }

    const asNum = Number(value)
    if (!Number.isNaN(asNum) && value.trim() !== '') {
      normalized[key] = asNum
      return
    }
    normalized[key] = value
  })
  return normalized
}

export default function App() {
  const wsRef = useRef(null)
  const [connected, setConnected] = useState(false)
  const [apiKey, setApiKey] = useState('')
  const [modelInfo, setModelInfo] = useState(null)
  const [rows, setRows] = useState([])
  const [selectedRow, setSelectedRow] = useState(0)
  const [rawPayloadText, setRawPayloadText] = useState(JSON.stringify({ rows: [] }, null, 2))
  const [useRawMode, setUseRawMode] = useState(false)
  const [events, setEvents] = useState([])
  const [lastResponse, setLastResponse] = useState('{}')

  const requiredFields = useMemo(
    () => (modelInfo?.required_predict_fields ? modelInfo.required_predict_fields : []),
    [modelInfo],
  )

  function appendEvent(level, text, payload = null) {
    const event = {
      ts: new Date().toISOString(),
      level,
      text,
      payload,
    }
    setEvents((prev) => [event, ...prev].slice(0, 200))
  }

  function connect() {
    if (wsRef.current) {
      wsRef.current.close()
    }

    const ws = new WebSocket(makeWsUrl(apiKey))
    wsRef.current = ws

    ws.onopen = () => {
      setConnected(true)
      appendEvent('ok', 'WebSocket connected')
      sendMessage({ type: 'health' })
      sendMessage({ type: 'model_info' })
    }

    ws.onclose = () => {
      setConnected(false)
      appendEvent('warn', 'WebSocket disconnected')
    }

    ws.onmessage = (event) => {
      setLastResponse(event.data)
      try {
        const msg = JSON.parse(event.data)
        if (msg.type === 'model_info') {
          setModelInfo(msg.data || null)
          appendEvent('ok', 'Model info updated', msg.data)
        } else if (msg.type === 'predict_result') {
          appendEvent('ok', `Predict completed (request_id: ${msg.request_id || 'n/a'})`, msg.data)
        } else if (msg.type === 'error') {
          appendEvent('error', `Server error ${msg.status || ''}`.trim(), msg.detail)
        } else if (msg.type === 'health') {
          appendEvent('ok', 'Health response received', msg.data)
        } else {
          appendEvent('info', `Message: ${msg.type || 'unknown'}`, msg)
        }
      } catch {
        appendEvent('warn', 'Received non-JSON message')
      }
    }
  }

  function sendMessage(message) {
    if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) {
      appendEvent('error', 'WebSocket is not connected')
      return
    }
    wsRef.current.send(JSON.stringify(message))
  }

  function bootstrapRowsFromModelInfo() {
    if (requiredFields.length === 0) {
      appendEvent('warn', 'required_predict_fields is empty')
      return
    }
    const row = buildRowFromFields(requiredFields)
    setRows([row])
    setSelectedRow(0)
    appendEvent('ok', 'Initialized editable batch from model schema')
  }

  function addRow() {
    if (requiredFields.length === 0) {
      appendEvent('warn', 'Load model-info first')
      return
    }
    setRows((prev) => [...prev, buildRowFromFields(requiredFields)])
    setSelectedRow(rows.length)
  }

  function removeRow(index) {
    setRows((prev) => prev.filter((_, i) => i !== index))
    setSelectedRow((prev) => Math.max(0, prev - (index <= prev ? 1 : 0)))
  }

  function updateCell(rowIndex, field, value) {
    setRows((prev) => {
      const next = [...prev]
      next[rowIndex] = { ...next[rowIndex], [field]: value }
      return next
    })
  }

  function clearBatch() {
    setRows([])
    setSelectedRow(0)
  }

  function buildPayloadFromRows() {
    return {
      rows: rows.map((row) => normalizeRowForPayload(row)),
    }
  }

  function syncRawFromRows() {
    const payload = buildPayloadFromRows()
    setRawPayloadText(JSON.stringify(payload, null, 2))
  }

  function sendPredictFromTable() {
    if (rows.length === 0) {
      appendEvent('warn', 'Batch is empty')
      return
    }
    const payload = buildPayloadFromRows()
    sendMessage({
      type: 'predict',
      request_id: `ui-${Date.now()}`,
      payload,
    })
  }

  function sendPredictFromRaw() {
    try {
      const payload = JSON.parse(rawPayloadText)
      sendMessage({
        type: 'predict',
        request_id: `ui-${Date.now()}`,
        payload,
      })
    } catch {
      appendEvent('error', 'Invalid JSON payload')
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
    <div className="console-root">
      <header className="console-header">
        <div>
          <h1>CTR Ops Console</h1>
          <p>Batch prediction workspace with schema-driven editor and live WS events.</p>
        </div>
        <div className="conn-box">
          <span className={connected ? 'pill ok' : 'pill bad'}>
            {connected ? 'WS online' : 'WS offline'}
          </span>
          <input
            type="password"
            placeholder="API key"
            value={apiKey}
            onChange={(e) => setApiKey(e.target.value)}
          />
          <button onClick={connect}>Reconnect</button>
          <button className="secondary" onClick={() => sendMessage({ type: 'health' })} disabled={!connected}>Health</button>
          <button className="secondary" onClick={() => sendMessage({ type: 'model_info' })} disabled={!connected}>Model Info</button>
        </div>
      </header>

      <main className="console-grid">
        <section className="panel workspace">
          <div className="panel-head">
            <h2>Batch Builder</h2>
            <div className="row">
              <button className="secondary" onClick={bootstrapRowsFromModelInfo} disabled={requiredFields.length === 0}>Init from schema</button>
              <button className="secondary" onClick={addRow} disabled={requiredFields.length === 0}>Add row</button>
              <button className="secondary" onClick={clearBatch}>Clear</button>
              <button onClick={syncRawFromRows}>Export JSON</button>
              <button onClick={sendPredictFromTable} disabled={!connected}>Predict batch</button>
            </div>
          </div>

          <div className="schema-line">
            <strong>Required fields:</strong> {requiredFields.join(', ') || 'n/a'}
          </div>

          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>#</th>
                  {requiredFields.map((f) => (
                    <th key={f}>{f}</th>
                  ))}
                  <th>actions</th>
                </tr>
              </thead>
              <tbody>
                {rows.length === 0 ? (
                  <tr>
                    <td colSpan={requiredFields.length + 2} className="empty">No rows. Click "Init from schema".</td>
                  </tr>
                ) : (
                  rows.map((row, idx) => (
                    <tr key={`row-${idx}`} className={idx === selectedRow ? 'selected' : ''}>
                      <td>{idx + 1}</td>
                      {requiredFields.map((field) => (
                        <td key={`${idx}-${field}`}>
                          <input
                            value={row[field] ?? ''}
                            onFocus={() => setSelectedRow(idx)}
                            onChange={(e) => updateCell(idx, field, e.target.value)}
                            placeholder={field === 'DateTime' ? 'YYYY-MM-DD HH:MM' : 'value or empty -> null'}
                          />
                        </td>
                      ))}
                      <td>
                        <button className="danger" onClick={() => removeRow(idx)}>Delete</button>
                      </td>
                    </tr>
                  ))
                )}
              </tbody>
            </table>
          </div>
        </section>

        <section className="panel inspector">
          <div className="panel-head">
            <h2>Inspector</h2>
            <div className="row">
              <button className={useRawMode ? 'secondary' : ''} onClick={() => setUseRawMode(false)}>Events</button>
              <button className={!useRawMode ? 'secondary' : ''} onClick={() => setUseRawMode(true)}>Raw JSON</button>
            </div>
          </div>

          {useRawMode ? (
            <>
              <p className="muted">Raw payload mode for low-level debugging.</p>
              <textarea
                className="raw"
                value={rawPayloadText}
                onChange={(e) => setRawPayloadText(e.target.value)}
              />
              <button onClick={sendPredictFromRaw} disabled={!connected}>Predict from raw JSON</button>
              <h3>Last response</h3>
              <pre>{lastResponse}</pre>
            </>
          ) : (
            <>
              <h3>Model Info</h3>
              <pre>{JSON.stringify(modelInfo || {}, null, 2)}</pre>
              <h3>Event Stream</h3>
              <div className="event-list">
                {events.length === 0 ? (
                  <div className="event empty">No events yet.</div>
                ) : (
                  events.map((event, i) => (
                    <div className={`event ${event.level}`} key={`event-${i}`}>
                      <div className="event-top">
                        <span>{event.ts}</span>
                        <strong>{event.level.toUpperCase()}</strong>
                      </div>
                      <div>{event.text}</div>
                      {event.payload ? <pre>{JSON.stringify(event.payload, null, 2)}</pre> : null}
                    </div>
                  ))
                )}
              </div>
            </>
          )}
        </section>
      </main>
    </div>
  )
}
