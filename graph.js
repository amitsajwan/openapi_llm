import React, { useEffect, useState } from 'https://esm.sh/react';
import ReactDOM from 'https://esm.sh/react-dom';
import ReactFlow, { Background, Controls } from 'https://esm.sh/react-flow-renderer@10.3.17';

function GraphViewer({ graph }) {
  const nodes = graph.nodes.map((node, i) => ({
    id: node.id,
    data: { label: `${node.type.toUpperCase()} ${node.endpoint}` },
    position: { x: 100 + i * 150, y: 100 },
    style: { padding: 10, border: '1px solid #777', borderRadius: 8 }
  }));

  const edges = graph.edges.map(edge => ({
    id: `${edge.from}-${edge.to}`,
    source: edge.from,
    target: edge.to,
    animated: true,
    style: { stroke: '#555' }
  }));

  return (
    <div style={{ width: '100%', height: '100%' }}>
      <ReactFlow nodes={nodes} edges={edges} fitView>
        <Background />
        <Controls />
      </ReactFlow>
    </div>
  );
}

function App() {
  const [graph, setGraph] = useState(null);

  useEffect(() => {
    const ws = new WebSocket('ws://localhost:8000/chat');

    ws.onopen = () => {
      console.log('WebSocket connected');
    };

    ws.onmessage = (event) => {
      const data = JSON.parse(event.data);
      if (data.type === 'workflow_graph') {
        setGraph(data.graph);
      }
    };

    ws.onclose = (event) => {
      console.log('WebSocket closed:', event.code, event.reason);
    };

    ws.onerror = (error) => {
      console.error('WebSocket error:', error);
    };

    return () => {
      if (ws.readyState === WebSocket.OPEN || ws.readyState === WebSocket.CONNECTING) {
        ws.close();
      }
    };
  }, []);

  return graph ? <GraphViewer graph={graph} /> : <div style={{ padding: 20 }}>Waiting for graph...</div>;
}

ReactDOM.render(<App />, document.getElementById('root'));
