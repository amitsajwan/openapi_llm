import React from "https://esm.sh/react";
import ReactDOM from "https://esm.sh/react-dom";
import ReactFlow, {
  Background,
  Controls
} from "https://esm.sh/reactflow@11.6.5";

function GraphViewer({ graph }) {
  const nodes = graph.nodes.map((node, i) => ({
    id: node.id,
    data: { label: `${node.type.toUpperCase()} ${node.endpoint}` },
    position: { x: 100 + i * 180, y: 100 },
    style: { padding: 10, border: "1px solid #777", borderRadius: 8 },
  }));

  const edges = graph.edges.map((edge) => ({
    id: `${edge.from}-${edge.to}`,
    source: edge.from,
    target: edge.to,
    animated: true,
    style: { stroke: "#555" },
  }));

  return React.createElement(
    "div",
    { style: { width: "100%", height: "100%" } },
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
    const ws = new WebSocket("ws://localhost:9091/ws/submit_openapi");
    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        if (data.graph) {
          setGraph(data.graph);
        }
      } catch (err) {
        console.error("Invalid message:", event.data);
      }
    };
    return () => {
      // Optional: don't close immediately to avoid 1001 errors
      // ws.close();
    };
  }, []);

  return graph
    ? React.createElement(GraphViewer, { graph })
    : React.createElement("div", null, "Waiting for graph...");
}

ReactDOM.render(
  React.createElement(App),
  document.getElementById("root")
);
