import React, { useEffect, useState } from "react";
import { createRoot } from "react-dom/client";
import ReactFlow, { Background, Controls } from "react-flow-renderer";

const App = () => {
  const [nodes, setNodes] = useState([]);
  const [edges, setEdges] = useState([]);

  useEffect(() => {
    const ws = new WebSocket("ws://localhost:9091/ws/submit_openapi");
    ws.onmessage = (event) => {
      const data = JSON.parse(event.data);
      if (data.graph) {
        const nodeList = data.graph.nodes.map((node, i) => ({
          id: node.id,
          data: { label: `${node.type.toUpperCase()} ${node.endpoint}` },
          position: { x: 100 + i * 150, y: 100 }
        }));
        const edgeList = data.graph.edges.map((edge) => ({
          id: `${edge.from}-${edge.to}`,
          source: edge.from,
          target: edge.to,
          animated: true,
        }));
        setNodes(nodeList);
        setEdges(edgeList);
      }
    };
    return () => ws.close();
  }, []);

  return (
    <div style={{ width: "100vw", height: "100vh" }}>
      <ReactFlow nodes={nodes} edges={edges} fitView>
        <Background />
        <Controls />
      </ReactFlow>
    </div>
  );
};

const container = document.getElementById("root");
const root = createRoot(container);
root.render(<App />);
