import React from 'react';
// ✅ Correct default import
import PredictionsTable from './components/PredictionsTable.js';
import MetricsSummary from './components/MetricsSummary';

function App() {
  return (
    <div className="App">
      <h1>DeepFake Detector Results</h1>
      <PredictionsTable />
      <MetricsSummary />

    </div>
  );
}

export default App;