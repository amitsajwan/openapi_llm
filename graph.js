import React from "https://esm.sh/react@18.2.0";
import ReactDOM from "https://esm.sh/react-dom@18.2.0/client";
import ReactFlow, {
  Background,
  Controls
} from "https://esm.sh/reactflow@11.11.4";

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

  return React.createElement(
    "div",
    { style: { width: '100%', height: '100%' } },
    React.createElement(
      ReactFlow,
      { nodes, edges, fitView: true },
      React.createElement(Background, null),
      React.createElement(Controls, null)
    )
  );
}

function App() {
  const [graph, setGraph] = React.useState(null);

  React.useEffect(() => {
    const ws = new WebSocket('ws://localhost:9091/ws/submit_openapi');
    ws.onmessage = (event) => {
      const data = JSON.parse(event.data);
      if (data.graph) {
        setGraph(data.graph);
      }
    };
    return () => ws.close();
  }, []);

  return graph
    ? React.createElement(GraphViewer, { graph })
    : React.createElement("div", null, "Waiting for graph...");
}

// Mount using React 18+ way
const root = ReactDOM.createRoot(document.getElementById("root"));
root.render(React.create
