import { useEffect, useState } from "react";

function GameAnalysis() {
  const [health, setHealth] = useState(null);

  useEffect(() => {
    fetch("http://localhost:8000/health")
      .then((res) => res.json())
      .then((data) => setHealth(data.status));
  }, []);

  return (
    <div>
      <p>Backend health: {health}</p>
    </div>
  );
}

export default GameAnalysis;
