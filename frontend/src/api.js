const API_BASE = import.meta.env.VITE_API_BASE_URL || "http://localhost:8000";

async function parseResponse(response) {
  const contentType = response.headers.get("content-type") || "";
  const data = contentType.includes("application/json")
    ? await response.json().catch(() => ({}))
    : await response.text().catch(() => "");
  if (!response.ok) {
    const detail = typeof data === "string" ? data : data.detail || data.message || "Request failed";
    throw new Error(typeof detail === "string" ? detail : JSON.stringify(detail));
  }
  if (typeof data === "string") {
    throw new Error("Backend returned HTML/text instead of JSON. Check that FastAPI is running on port 8000 and Vite proxy is active.");
  }
  return data;
}

export async function fetchHealth() {
  const response = await fetch(`${API_BASE}/api/health`);
  const data = await parseResponse(response);
  if (data.status !== "ok") {
    throw new Error("Backend health check did not return ok status.");
  }
  return data;
}

export async function fetchRuns() {
  const response = await fetch(`${API_BASE}/api/runs`);
  const data = await parseResponse(response);
  if (!Array.isArray(data)) {
    throw new Error(`Expected /api/runs to return an array, received: ${JSON.stringify(data)}`);
  }
  return data;
}

export async function fetchLatestRun() {
  const response = await fetch(`${API_BASE}/api/runs/latest`);
  return parseResponse(response);
}

export async function trainModel(file) {
  const formData = new FormData();
  formData.append("file", file);
  const response = await fetch(`${API_BASE}/api/train`, {
    method: "POST",
    body: formData
  });
  return parseResponse(response);
}

export async function predictPatient(payload) {
  const response = await fetch(`${API_BASE}/api/predict`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json"
    },
    body: JSON.stringify(payload)
  });
  return parseResponse(response);
}

export async function fetchPredictions(runId = "") {
  const params = new URLSearchParams();
  if (runId) params.set("run_id", runId);
  const query = params.toString();
  const response = await fetch(`${API_BASE}/api/predictions${query ? `?${query}` : ""}`);
  const data = await parseResponse(response);
  if (!Array.isArray(data)) {
    throw new Error(`Expected /api/predictions to return an array, received: ${JSON.stringify(data)}`);
  }
  return data;
}

export async function fetchPrediction(predictionId) {
  const response = await fetch(`${API_BASE}/api/predictions/${predictionId}`);
  return parseResponse(response);
}

export async function deletePrediction(predictionId) {
  const response = await fetch(`${API_BASE}/api/predictions/${predictionId}`, {
    method: "DELETE"
  });
  return parseResponse(response);
}

export async function analyzePrediction(predictionId) {
  const response = await fetch(`${API_BASE}/api/predictions/${predictionId}/analysis`, {
    method: "POST"
  });
  return parseResponse(response);
}

export async function chatWithPrediction(predictionId, messages) {
  const response = await fetch(`${API_BASE}/api/chat`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json"
    },
    body: JSON.stringify({
      prediction_id: predictionId,
      messages
    })
  });
  return parseResponse(response);
}
