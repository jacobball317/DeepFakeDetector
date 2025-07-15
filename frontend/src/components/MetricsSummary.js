// frontend/src/components/MetricsSummary.js
import React, { useEffect, useState } from 'react';
import Papa from 'papaparse';

const metricsCSV = process.env.PUBLIC_URL + '/metrics_summary.csv';

function MetricsSummary() {
  const [metrics, setMetrics] = useState(null);

  useEffect(() => {
    fetch(metricsCSV)
      .then((res) => res.text())
      .then((text) => {
        const parsed = Papa.parse(text, { header: true });
        setMetrics(parsed.data[0]); // Only one row
      });
  }, []);

  if (!metrics) return <p>Loading metrics...</p>;

  return (
    <div style={{ marginTop: '2rem' }}>
      <h2>Model Performance Summary</h2>
      <table style={{ borderCollapse: 'collapse', border: '1px solid black' }}>
        <thead>
          <tr>
            {Object.keys(metrics).map((key) => (
              <th key={key} style={{ border: '1px solid black', padding: '8px' }}>{key}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          <tr>
            {Object.values(metrics).map((value, idx) => (
              <td key={idx} style={{ border: '1px solid black', padding: '8px' }}>{parseFloat(value).toFixed(4)}</td>
            ))}
          </tr>
        </tbody>
      </table>
    </div>
  );
}

export default MetricsSummary;