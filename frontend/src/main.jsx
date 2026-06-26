import React, { Component, useEffect, useMemo, useState } from "react";
import { createRoot } from "react-dom/client";
import { AnimatePresence, motion } from "framer-motion";
import {
  Activity,
  AlertCircle,
  BarChart3,
  BrainCircuit,
  CheckCircle2,
  ChevronRight,
  Database,
  FileUp,
  HeartPulse,
  MessageCircle,
  Moon,
  Play,
  Sparkles,
  Stethoscope,
  Sun,
  Trash2,
  Zap,
  UploadCloud
} from "lucide-react";
import "./index.css";
import {
  analyzePrediction,
  chatWithPrediction,
  deletePrediction,
  fetchHealth,
  fetchLatestRun,
  fetchPrediction,
  fetchPredictions,
  fetchRuns,
  predictPatient,
  trainModel
} from "./api";
import Markdown from 'react-markdown'

const fields = [
  { name: "age", label: "Age", type: "number", value: 52 },
  { name: "sex", label: "Sex", type: "select", value: 1, options: [["0", "Female"], ["1", "Male"]] },
  { name: "cp", label: "Chest pain", type: "select", value: 0, options: [["0", "Type 0"], ["1", "Type 1"], ["2", "Type 2"], ["3", "Type 3"]] },
  { name: "trestbps", label: "Resting BP", type: "number", value: 125 },
  { name: "chol", label: "Cholesterol", type: "number", value: 212 },
  { name: "fbs", label: "Fasting sugar", type: "select", value: 0, options: [["0", "False"], ["1", "True"]] },
  { name: "restecg", label: "Rest ECG", type: "select", value: 1, options: [["0", "0"], ["1", "1"], ["2", "2"]] },
  { name: "thalach", label: "Max heart rate", type: "number", value: 168 },
  { name: "exang", label: "Exercise angina", type: "select", value: 0, options: [["0", "No"], ["1", "Yes"]] },
  { name: "oldpeak", label: "Oldpeak", type: "number", value: 1 },
  { name: "slope", label: "Slope", type: "select", value: 2, options: [["0", "0"], ["1", "1"], ["2", "2"]] },
  { name: "ca", label: "Major vessels", type: "select", value: 2, options: [["0", "0"], ["1", "1"], ["2", "2"], ["3", "3"], ["4", "4"]] },
  { name: "thal", label: "Thal", type: "select", value: 3, options: [["0", "0"], ["1", "1"], ["2", "2"], ["3", "3"]] }
];

const tabs = [
  { id: "train", label: "CSV Training", icon: FileUp },
  { id: "predict", label: "Prediction", icon: Stethoscope }
];

function emptyPatient() {
  return Object.fromEntries(fields.map((field) => [field.name, field.value]));
}

function getInitialTheme() {
  const stored = localStorage.getItem("heart-ui-theme");
  if (stored === "light" || stored === "dark") return stored;
  return window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
}

function App() {
  const [theme, setTheme] = useState(getInitialTheme);
  const [route, setRoute] = useState(window.location.pathname);
  const [backendReady, setBackendReady] = useState(false);
  const [bootError, setBootError] = useState("");
  const [bootLoading, setBootLoading] = useState(true);

  useEffect(() => {
    document.documentElement.classList.toggle("dark", theme === "dark");
    localStorage.setItem("heart-ui-theme", theme);
  }, [theme]);

  useEffect(() => {
    const query = window.matchMedia("(prefers-color-scheme: dark)");
    const onChange = (event) => {
      if (!localStorage.getItem("heart-ui-theme")) {
        setTheme(event.matches ? "dark" : "light");
      }
    };
    query.addEventListener("change", onChange);
    return () => query.removeEventListener("change", onChange);
  }, []);

  useEffect(() => {
    const onRouteChange = () => setRoute(window.location.pathname);
    window.addEventListener("popstate", onRouteChange);
    return () => window.removeEventListener("popstate", onRouteChange);
  }, []);

  async function checkBackend() {
    setBootLoading(true);
    setBootError("");
    try {
      await fetchHealth();
      setBackendReady(true);
    } catch (error) {
      setBackendReady(false);
      setBootError(error.message);
    } finally {
      setBootLoading(false);
    }
  }

  useEffect(() => {
    checkBackend();
  }, []);

  if (!backendReady) {
    return (
      <AppLoader
        error={bootError}
        loading={bootLoading}
        onRetry={checkBackend}
        theme={theme}
        onTheme={setTheme}
      />
    );
  }

  if (route === "/chatbot") {
    return <ChatbotPage theme={theme} onTheme={setTheme} />;
  }

  return <DashboardApp theme={theme} onTheme={setTheme} />;
}

function DashboardApp({ theme, onTheme }) {
  const [activeTab, setActiveTab] = useState("train");
  const [runs, setRuns] = useState([]);
  const [runsLoading, setRunsLoading] = useState(true);
  const [selectedRunId, setSelectedRunId] = useState("");
  const [currentRun, setCurrentRun] = useState(null);
  const [file, setFile] = useState(null);
  const [patient, setPatient] = useState(emptyPatient);
  const [prediction, setPrediction] = useState(null);
  const [predictionRows, setPredictionRows] = useState([]);
  const [predictionsLoading, setPredictionsLoading] = useState(false);
  const [analysis, setAnalysis] = useState(null);
  const [analysisOpen, setAnalysisOpen] = useState(false);
  const [analysisLoading, setAnalysisLoading] = useState(false);
  const [loading, setLoading] = useState(false);
  const [message, setMessage] = useState("");

  async function refreshRuns() {
    setRunsLoading(true);
    try {
      const data = await fetchRuns();
      setRuns(data);
      const completed = data.find((run) => run.status === "completed");
      if (completed) {
        setCurrentRun(completed);
        setSelectedRunId((value) => value || completed.id);
      }
    } catch (error) {
      setRuns([]);
      setMessage(error.message);
    } finally {
      setRunsLoading(false);
    }
  }

  async function refreshPredictions(runId = selectedRunId) {
    setPredictionsLoading(true);
    try {
      const data = await fetchPredictions(runId);
      setPredictionRows(data);
    } catch (error) {
      setPredictionRows([]);
      setMessage(error.message);
    } finally {
      setPredictionsLoading(false);
    }
  }

  useEffect(() => {
    refreshRuns();
  }, []);

  useEffect(() => {
    if (activeTab === "predict") refreshPredictions();
  }, [activeTab, selectedRunId]);

  async function handleTrain(event) {
    event.preventDefault();
    if (!file) {
      setMessage("Choose a CSV file first.");
      return;
    }
    setLoading(true);
    setMessage("Training model. This can take a minute.");
    setPrediction(null);
    try {
      const run = await trainModel(file);
      setCurrentRun(run);
      setSelectedRunId(run.id);
      setMessage("Training completed.");
      await refreshRuns();
      await refreshPredictions(run.id);
    } catch (error) {
      setMessage(error.message);
    } finally {
      setLoading(false);
    }
  }

  async function handlePredict(event) {
    event.preventDefault();
    setLoading(true);
    setMessage("");
    try {
      let runId = selectedRunId;
      if (!runId) {
        const latest = await fetchLatestRun();
        runId = latest.id;
      }
      const payload = {
        run_id: runId,
        ...Object.fromEntries(Object.entries(patient).map(([key, value]) => [key, Number(value)]))
      };
      const result = await predictPatient(payload);
      setPrediction(result);
      await refreshPredictions(runId);
      setActiveTab("predict");
    } catch (error) {
      setMessage(error.message);
    } finally {
      setLoading(false);
    }
  }

  async function handleDeletePrediction(predictionId) {
    if (!window.confirm("Delete this saved prediction?")) return;
    try {
      await deletePrediction(predictionId);
      await refreshPredictions();
      setMessage("Prediction deleted.");
    } catch (error) {
      setMessage(error.message);
    }
  }

  async function handleAnalyzePrediction(predictionId) {
    setAnalysisOpen(true);
    setAnalysis(null);
    setAnalysisLoading(true);
    try {
      const result = await analyzePrediction(predictionId);
      setAnalysis(result);
    } catch (error) {
      setMessage(error.message);
      setAnalysisOpen(false);
    } finally {
      setAnalysisLoading(false);
    }
  }

  const metricCards = useMemo(() => {
    if (!currentRun) return [];
    const evalStats = currentRun.evaluation_stats || {};
    return [
      { label: "Rows retained", value: `${currentRun.rows_after ?? "-"} / ${currentRun.rows_before ?? "-"}`, icon: Database, tone: "primary" },
      { label: "Feature count", value: currentRun.feature_count ?? "-", icon: BarChart3, tone: "success" },
      { label: "Champion model", value: currentRun.champion_model || "-", icon: BrainCircuit, tone: "primary" },
      { label: "Recall", value: formatMetric(evalStats.recall), icon: Activity, tone: "success" },
      { label: "ROC-AUC", value: formatMetric(evalStats.roc_auc), icon: Sparkles, tone: "primary" },
      { label: "Threshold", value: formatMetric(currentRun.optimal_threshold), icon: CheckCircle2, tone: "warning" }
    ];
  }, [currentRun]);

  return (
    <div className="min-h-screen bg-background text-foreground">
      <Sidebar activeTab={activeTab} currentRun={currentRun} onTab={setActiveTab} theme={theme} onTheme={onTheme} />

      <main className="relative lg:pl-[280px]">
        <Topbar activeTab={activeTab} theme={theme} onTab={setActiveTab} onTheme={onTheme} />

        <motion.section
          key={activeTab}
          initial={{ opacity: 0, y: 16 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.45, ease: "easeOut" }}
          className="mx-auto max-w-[1500px] px-4 py-6 sm:px-6 lg:px-8"
        >
          <AnimatePresence>
            {message && (
              <Toast key={message} message={message} loading={loading} onClose={() => setMessage("")} />
            )}
          </AnimatePresence>

          {activeTab === "train" ? (
            <TrainingView
              currentRun={currentRun}
              file={file}
              loading={loading}
              dataLoading={runsLoading}
              metricCards={metricCards}
              onFile={setFile}
              onTrain={handleTrain}
            />
          ) : (
            <PredictionView
              fields={fields}
              loading={loading}
              patient={patient}
              prediction={prediction}
              predictionRows={predictionRows}
              predictionsLoading={predictionsLoading}
              runs={runs}
              runsLoading={runsLoading}
              selectedRunId={selectedRunId}
              setPatient={setPatient}
              setSelectedRunId={setSelectedRunId}
              onPredict={handlePredict}
              onAnalyze={handleAnalyzePrediction}
              onDelete={handleDeletePrediction}
            />
          )}
        </motion.section>
      </main>

      <AnalysisDialog
        analysis={analysis}
        loading={analysisLoading}
        open={analysisOpen}
        onClose={() => setAnalysisOpen(false)}
      />
    </div>
  );
}

function Sidebar({ activeTab, currentRun, onTab, theme, onTheme }) {
  return (
    <aside className="fixed inset-y-0 left-0 z-20 hidden w-[280px] border-r border-line bg-card px-4 py-5 lg:block">
      <div className="flex items-center gap-3 px-2">
        <motion.div whileHover={{ scale: 1.03 }} className={`grid h-10 w-10 place-items-center rounded-lg ${theme === "dark" ? "bg-primary/10 text-primary" : "bg-primary/20 text-primary"}`}>
          <HeartPulse size={23} />
        </motion.div>
        <div>
          <h1 className="text-base font-semibold tracking-tight">Cardio Intelligence</h1>
          <p className="text-xs font-medium text-muted">Clinical ML console</p>
        </div>
      </div>

      <nav className="mt-8 space-y-2" aria-label="Primary">
        {tabs.map((tab) => (
          <SideButton key={tab.id} tab={tab} active={activeTab === tab.id} onClick={() => onTab(tab.id)} />
        ))}
      </nav>

      <div className="latest-run-card mt-8 rounded-xl border p-4 shadow-sm">
        <div className="flex items-center justify-between gap-3">
          <div className="flex items-center gap-2 text-sm font-semibold text-foreground">
            <span className="icon-surface grid h-7 w-7 place-items-center rounded-lg">
              <Database size={15} />
            </span>
            Latest run
          </div>
          <StatusBadge status={currentRun?.status || "idle"} />
        </div>
        <p className="latest-run-id mt-3 break-all rounded-lg px-3 py-2 text-xs font-medium leading-5">
          {currentRun?.id || "Train a model to create the first run."}
        </p>
      </div>

      <ThemeToggle theme={theme} onTheme={onTheme} className="mt-4 w-full" />
    </aside>
  );
}

function Topbar({ activeTab, theme, onTab, onTheme }) {
  return (
    <header className="sticky top-0 z-10 border-b border-line bg-background/90 px-4 py-4 backdrop-blur-md sm:px-6 lg:px-8">
      <div className="mx-auto flex max-w-[1500px] flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
        <div>
          <div className="mb-2 flex items-center gap-2 text-xs font-semibold uppercase tracking-[0.18em] text-primary">
            <span>Dashboard</span>
            <ChevronRight size={14} />
            <span>{activeTab === "train" ? "Training" : "Prediction"}</span>
          </div>
          <h2 className="text-2xl font-semibold tracking-tight">
            {activeTab === "train" ? "Train model and inspect pipeline stats" : "Predict patient risk"}
          </h2>
        </div>

        <div className="flex items-center gap-3">
          <div className="flex rounded-xl border border-line bg-card p-1 shadow-sm lg:hidden">
            {tabs.map((tab) => (
              <button
                key={tab.id}
                className={mobileTabClass(activeTab === tab.id)}
                onClick={() => onTab(tab.id)}
                type="button"
              >
                {tab.label.replace("CSV ", "")}
              </button>
            ))}
          </div>
          <ThemeToggle theme={theme} onTheme={onTheme} />
        </div>
      </div>
    </header>
  );
}

function TrainingView({ currentRun, file, loading, dataLoading, metricCards, onFile, onTrain }) {
  return (
    <div className="space-y-6">
      <Card>
        <form onSubmit={onTrain} className="grid gap-5 lg:grid-cols-[1fr_auto] lg:items-end">
          <div>
            <SectionTitle icon={UploadCloud} title="Dataset upload" eyebrow="CSV intake" />
            <label className="mt-4 block">
              <span className="text-sm font-medium text-muted">Heart disease CSV file</span>
              <input
                className="form-control mt-2 min-h-12 w-full rounded-xl border px-4 py-3 text-sm outline-none transition focus:border-primary focus:ring-4 focus:ring-primary/15"
                type="file"
                accept=".csv"
                onChange={(event) => onFile(event.target.files?.[0] || null)}
              />
            </label>
          </div>
          <Button disabled={loading || !file} loading={loading} icon={Play}>
            Train model
          </Button>
        </form>
      </Card>

      {loading || dataLoading ? <MetricSkeleton /> : <MetricGrid metrics={metricCards} />}
      {dataLoading ? <StatsSkeleton /> : currentRun ? <StatsPanel run={currentRun} /> : <EmptyState title="No completed training run" text="Upload a CSV to populate preprocessing, feature engineering, model selection, and evaluation cards." icon={BrainCircuit} />}
    </div>
  );
}

function PredictionView({ fields, loading, patient, prediction, predictionRows, predictionsLoading, runs, runsLoading, selectedRunId, setPatient, setSelectedRunId, onPredict, onAnalyze, onDelete }) {
  const safeRuns = Array.isArray(runs) ? runs : [];
  const completedRuns = safeRuns.filter((run) => run?.status === "completed");

  return (
    <div className="space-y-6">
      <div className="grid gap-6 xl:grid-cols-[minmax(0,1fr)_380px]">
        <Card>
          <form onSubmit={onPredict}>
            <SectionTitle icon={Stethoscope} title="Patient input" eyebrow="Risk inference" />
            <label className="mt-5 block">
              <span className="text-sm font-medium text-muted">Model run</span>
              <select
                className="form-control mt-2 min-h-12 w-full rounded-xl border px-4 text-sm outline-none transition focus:border-primary focus:ring-4 focus:ring-primary/15"
                value={selectedRunId}
                onChange={(event) => setSelectedRunId(event.target.value)}
              >
                <option value="">Latest completed run</option>
                {completedRuns.map((run) => (
                  <option key={run.id} value={run.id}>
                    {run.champion_model || "Model"} - {formatDate(run.created_at)}
                  </option>
                ))}
              </select>
              {runsLoading ? (
                <p className="mt-2 flex items-center gap-2 text-sm text-muted">
                  <SkeletonDot dark />
                  Loading model runs...
                </p>
              ) : completedRuns.length === 0 && (
                <p className="mt-2 text-sm text-muted">No completed training run yet. Train a model before predicting.</p>
              )}
            </label>

            <div className="mt-5 grid gap-4 md:grid-cols-2 xl:grid-cols-3">
              {fields.map((field) => (
                <FieldControl key={field.name} field={field} value={patient[field.name]} onChange={(value) => setPatient({ ...patient, [field.name]: value })} />
              ))}
            </div>

            <div className="mt-6">
              <Button disabled={loading} loading={loading} icon={BrainCircuit} tone="success">
                Predict risk
              </Button>
            </div>
          </form>
        </Card>

        <PredictionResult prediction={prediction} />
      </div>

      <PredictionTable rows={predictionRows} loading={predictionsLoading} onAnalyze={onAnalyze} onDelete={onDelete} />
    </div>
  );
}

function FieldControl({ field, value, onChange }) {
  return (
    <label className="block">
      <span className="text-sm font-medium text-muted">{field.label}</span>
      {field.type === "select" ? (
        <select
          className="form-control mt-2 min-h-11 w-full rounded-xl border px-3 text-sm outline-none transition focus:border-primary focus:ring-4 focus:ring-primary/15"
          value={value}
          onChange={(event) => onChange(event.target.value)}
        >
          {field.options.map(([optionValue, label]) => (
            <option key={optionValue} value={optionValue}>{label}</option>
          ))}
        </select>
      ) : (
        <input
          className="form-control mt-2 min-h-11 w-full rounded-xl border px-3 text-sm outline-none transition focus:border-primary focus:ring-4 focus:ring-primary/15"
          type="number"
          step="any"
          value={value}
          onChange={(event) => onChange(event.target.value)}
        />
      )}
    </label>
  );
}

function MetricGrid({ metrics }) {
  if (!metrics.length) {
    return <EmptyState title="Waiting for training stats" text="Metrics appear here after a successful training run." icon={BarChart3} compact />;
  }

  return (
    <motion.div variants={stagger} initial="hidden" animate="show" className="grid gap-4 sm:grid-cols-2 xl:grid-cols-3">
      {metrics.map((metric) => (
        <MetricCard key={metric.label} metric={metric} />
      ))}
    </motion.div>
  );
}

function MetricCard({ metric }) {
  const Icon = metric.icon;
  return (
    <motion.div variants={fadeUp} whileHover={{ y: -2 }} className="rounded-xl border border-line bg-card p-5 shadow-sm transition hover:border-primary/30 hover:shadow-soft">
      <div className="flex items-center justify-between gap-3">
        <p className="text-sm font-medium text-muted">{metric.label}</p>
        <div className={`grid h-9 w-9 place-items-center rounded-lg ${toneClass(metric.tone)}`}>
          <Icon size={17} />
        </div>
      </div>
      <p className="mt-3 truncate text-2xl font-semibold tracking-tight">{metric.value}</p>
    </motion.div>
  );
}

function PredictionResult({ prediction }) {
  return (
    <Card>
      <SectionTitle icon={Activity} title="Prediction result" eyebrow="Latest inference" />
      {prediction ? (
        <motion.div initial={{ opacity: 0, scale: 0.96 }} animate={{ opacity: 1, scale: 1 }} transition={{ type: "spring", stiffness: 180, damping: 18 }} className="mt-5">
          <div className={`rounded-xl border p-5 ${prediction.prediction ? "border-danger/20 bg-danger/10 text-danger" : "border-success/20 bg-success/10 text-success"}`}>
            <p className="text-sm font-semibold">{prediction.label}</p>
            <p className="mt-2 text-5xl font-bold tracking-tight">{formatPercent(prediction.probability)}</p>
            <div className="mt-4 h-2 overflow-hidden rounded-full bg-card">
              <motion.div className="h-full rounded-full bg-current" initial={{ width: 0 }} animate={{ width: `${Math.round(prediction.probability * 100)}%` }} transition={{ duration: 0.6 }} />
            </div>
          </div>
          <dl className="mt-5 space-y-3 text-sm">
            <InfoRow label="Threshold" value={formatMetric(prediction.threshold)} />
            <InfoRow label="Probability" value={formatMetric(prediction.probability)} />
            <InfoRow label="Run ID" value={prediction.run_id} />
          </dl>
        </motion.div>
      ) : (
        <EmptyState title="No prediction yet" text="Submit patient values after training a model." icon={BrainCircuit} compact />
      )}
    </Card>
  );
}

function PredictionTable({ rows, loading, onAnalyze, onDelete }) {
  const safeRows = Array.isArray(rows) ? rows : [];

  return (
    <Card>
      <SectionTitle icon={Database} title="Saved predictions" eyebrow="Database history" />
      <div className="mt-4 overflow-x-auto">
        <table className="w-full min-w-[900px] border-separate border-spacing-0 text-left text-sm">
          <thead>
            <tr className="text-muted">
              {["Time", "Age", "Sex", "BP", "Chol", "Probability", "Threshold", "Prediction", "Actions"].map((head) => (
                <th key={head} className="border-b border-line py-3 pr-4 font-semibold">{head}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {loading ? (
              <TableSkeletonRows columns={9} rows={3} />
            ) : safeRows.length === 0 ? (
              <tr>
                <td className="py-8 text-muted" colSpan="9">No saved predictions yet.</td>
              </tr>
            ) : (
              safeRows.map((row, index) => (
                <motion.tr
                  key={row.id}
                  initial={{ opacity: 0, y: 8 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: index * 0.025 }}
                  className="group transition hover:bg-primary/5"
                >
                  <td className="border-b border-line py-3 pr-4 text-muted">{formatDate(row.created_at)}</td>
                  <td className="border-b border-line py-3 pr-4">{row.input_data?.age ?? "-"}</td>
                  <td className="border-b border-line py-3 pr-4">{formatSex(row.input_data?.sex)}</td>
                  <td className="border-b border-line py-3 pr-4">{row.input_data?.trestbps ?? "-"}</td>
                  <td className="border-b border-line py-3 pr-4">{row.input_data?.chol ?? "-"}</td>
                  <td className="border-b border-line py-3 pr-4 font-semibold">{formatPercent(row.probability)}</td>
                  <td className="border-b border-line py-3 pr-4">{formatMetric(row.threshold)}</td>
                  <td className="border-b border-line py-3 pr-4">
                    <ResultBadge prediction={row.prediction} label={row.label} />
                  </td>
                  <td className="border-b border-line py-3 pr-4">
                    <div className="flex items-center gap-2">
                      <IconButton title="Analyze with Groq" onClick={() => onAnalyze(row.id)}>
                        <Zap size={16} />
                      </IconButton>
                      <IconButton title="Open chatbot" onClick={() => window.location.assign(`/chatbot?prediction_id=${row.id}`)}>
                        <BrainCircuit size={16} />
                      </IconButton>
                      <IconButton title="Delete prediction" danger onClick={() => onDelete(row.id)}>
                        <Trash2 size={16} />
                      </IconButton>
                    </div>
                  </td>
                </motion.tr>
              ))
            )}
          </tbody>
        </table>
      </div>
    </Card>
  );
}

function AnalysisDialog({ analysis, loading, open, onClose }) {
  if (!open) return null;

  return (
    <div className="modal-backdrop fixed inset-0 z-50 grid place-items-center p-4 backdrop-blur-md">
      <motion.div
        initial={{ opacity: 0, scale: 0.94, y: 12 }}
        animate={{ opacity: 1, scale: 1, y: 0 }}
        className="modal-surface max-h-[86vh] w-full max-w-2xl overflow-auto rounded-2xl border p-0"
      >
        <div className="border-b border-line px-6 py-5">
          <div className="flex items-start justify-between gap-4">
            <div className="flex items-center gap-3">
              <span className="icon-surface grid h-11 w-11 place-items-center rounded-xl">
                <Zap size={20} />
              </span>
              <div>
                <p className="text-xs font-semibold uppercase tracking-[0.2em] text-primary">AI assistant</p>
                <h3 className="mt-1 text-xl font-semibold tracking-tight">Groq risk analysis</h3>
              </div>
            </div>
            <button className="secondary-action rounded-xl border px-3 py-2 text-sm font-semibold" onClick={onClose} type="button">Close</button>
          </div>
        </div>
        <div className="px-6 py-5">
          {loading ? (
            <div className="space-y-3">
              <div className="h-4 w-3/4 animate-pulse rounded bg-panel" />
              <div className="h-4 w-full animate-pulse rounded bg-panel" />
              <div className="h-4 w-5/6 animate-pulse rounded bg-panel" />
              <div className="h-28 animate-pulse rounded-xl bg-panel" />
            </div>
          ) : analysis ? (
            <div className="space-y-4 text-sm leading-6">
              <div className="modal-section rounded-xl border p-4">
                <h4 className="font-semibold">Summary</h4>
                <p className="mt-2 text-muted">{analysis.summary}</p>
              </div>
              <ListBlock title="Possible reasons" items={analysis.possible_reasons} />
              <ListBlock title="Suggested next steps" items={analysis.suggested_steps} />
              <p className="safety-note rounded-xl border p-4 text-sm font-medium leading-6">{analysis.safety_note}</p>
            </div>
          ) : null}
        </div>
      </motion.div>
    </div>
  );
}

function ListBlock({ title, items }) {
  const safeItems = Array.isArray(items) ? items.filter(Boolean) : [];

  return (
    <div className="modal-section rounded-xl border p-4">
      <h4 className="font-semibold">{title}</h4>
      {safeItems.length > 0 ? (
        <ul className="mt-2 space-y-2 text-muted">
          {safeItems.map((item, index) => (
            <li key={index} className="flex gap-2">
              <span className="mt-2 h-1.5 w-1.5 shrink-0 rounded-full bg-primary" />
              <span>{String(item)}</span>
            </li>
          ))}
        </ul>
      ) : (
        <p className="mt-2 text-sm text-muted">Groq did not return structured items for this section. Try running the analysis again.</p>
      )}
    </div>
  );
}

function ChatbotPage({ theme, onTheme }) {
  const [prediction, setPrediction] = useState(null);
  const [messages, setMessages] = useState([
    {
      role: "assistant",
      content: "I can help explain this saved prediction, likely risk contributors, and what questions to ask a clinician."
    }
  ]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const predictionId = new URLSearchParams(window.location.search).get("prediction_id");

  useEffect(() => {
    async function loadPrediction() {
      if (!predictionId) {
        setError("Missing prediction_id in URL.");
        return;
      }
      try {
        const data = await fetchPrediction(predictionId);
        setPrediction(data);
      } catch (err) {
        setError(err.message);
      }
    }
    loadPrediction();
  }, [predictionId]);

  async function sendMessage(event) {
    event.preventDefault();
    if (!input.trim() || !predictionId) return;
    const nextMessages = [...messages, { role: "user", content: input.trim() }];
    setMessages(nextMessages);
    setInput("");
    setLoading(true);
    setError("");
    try {
      const result = await chatWithPrediction(predictionId, nextMessages);
      setMessages([...nextMessages, { role: "assistant", content: result.reply }]);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="min-h-screen bg-background text-foreground">
      <header className="border-b border-line bg-card px-5 py-4">
        <div className="mx-auto flex max-w-5xl items-center justify-between gap-4">
          <div className="flex items-center gap-3">
            <span className="icon-surface grid h-10 w-10 place-items-center rounded-xl">
              <MessageCircle size={20} />
            </span>
            <div>
              <p className="text-xs font-semibold uppercase tracking-[0.18em] text-primary">Groq chatbot</p>
              <h1 className="text-xl font-semibold">Prediction assistant</h1>
            </div>
          </div>
          <div className="flex items-center gap-2">
            <button className="secondary-action rounded-xl border px-4 py-2 text-sm font-semibold" onClick={() => window.location.assign("/")} type="button">Dashboard</button>
            <ThemeToggle theme={theme} onTheme={onTheme} />
          </div>
        </div>
      </header>

      <main className="mx-auto grid max-w-5xl gap-5 px-5 py-6 lg:grid-cols-[320px_1fr]">
        <Card>
          <SectionTitle icon={BrainCircuit} title="Loaded prediction" eyebrow="Context" />
          {prediction ? (
            <dl className="mt-4 space-y-3 text-sm">
              <InfoRow label="Risk" value={formatPercent(prediction.probability)} />
              <InfoRow label="Label" value={prediction.label} />
              <InfoRow label="Age" value={prediction.input_data?.age ?? "-"} />
              <InfoRow label="BP" value={prediction.input_data?.trestbps ?? "-"} />
              <InfoRow label="Chol" value={prediction.input_data?.chol ?? "-"} />
            </dl>
          ) : (
            <div className="mt-4 space-y-3">
              {error ? (
                <p className="text-sm text-muted">{error}</p>
              ) : (
                <>
                  <div className="h-4 w-32 animate-pulse rounded-full bg-panel" />
                  <div className="h-4 w-48 animate-pulse rounded-full bg-panel" />
                  <div className="h-4 w-40 animate-pulse rounded-full bg-panel" />
                </>
              )}
            </div>
          )}
        </Card>

        <Card>
          <div className="space-y-3">
            {messages.map((message, index) => (
              <div key={index} className={`rounded-xl border border-line p-3 text-sm leading-6 ${message.role === "user" ? "ml-10 bg-primarySoft" : "mr-10 bg-panel"}`}>
                <p className="mb-1 text-xs font-semibold uppercase tracking-[0.16em] text-muted">{message.role}</p>
                <Markdown>{message.content}</Markdown>
              </div>
            ))}
            {loading && (
              <div className="mr-10 rounded-xl border border-line bg-panel p-3">
                <div className="mb-2 h-3 w-24 animate-pulse rounded-full bg-card" />
                <div className="h-4 w-3/4 animate-pulse rounded-full bg-card" />
              </div>
            )}
            {error && <p className="rounded-xl border border-danger/25 bg-danger/10 p-3 text-sm text-danger">{error}</p>}
          </div>
          <form onSubmit={sendMessage} className="mt-5 flex gap-3">
            <input
              className="form-control min-h-12 flex-1 rounded-xl border px-4 text-sm outline-none focus:border-primary focus:ring-4 focus:ring-primary/15"
              value={input}
              onChange={(event) => setInput(event.target.value)}
              placeholder="Ask about this prediction..."
            />
            <Button disabled={loading || !input.trim()} loading={loading} icon={MessageCircle}>Send</Button>
          </form>
        </Card>
      </main>
    </div>
  );
}

function StatsPanel({ run }) {
  const preprocessing = run.preprocessing_stats || {};
  const featureEngineering = run.feature_engineering_stats || {};
  const modelSelection = run.model_selection_stats || {};
  const evaluation = run.evaluation_stats || {};

  return (
    <motion.div variants={stagger} initial="hidden" animate="show" className="grid gap-5 xl:grid-cols-2">
      <PipelineCard
        title="Preprocessing"
        icon={CheckCircle2}
        rows={[
          ["Rows before", preprocessing.rows_before],
          ["Rows after", preprocessing.rows_after],
          ["Duplicates removed", preprocessing.duplicates_removed],
          ["Scaled columns", preprocessing.continuous_columns_scaled?.join(", ") || "-"],
          ["Encoded columns", preprocessing.categorical_columns_encoded?.join(", ") || "-"]
        ]}
      />
      <PipelineCard
        title="Feature engineering"
        icon={BarChart3}
        rows={[
          ["Engineered features", featureEngineering.engineered_features?.join(", ") || "-"],
          ["Feature count", featureEngineering.feature_count],
          ["Train shape", featureEngineering.train_shape?.join(" x ") || "-"],
          ["Test shape", featureEngineering.test_shape?.join(" x ") || "-"]
        ]}
      />
      <PipelineCard title="Model selection" icon={BrainCircuit}>
        {Object.keys(modelSelection.baseline_results || {}).length > 0 ? (
          <div className="mt-4 overflow-x-auto">
            <table className="w-full min-w-[520px] text-left text-sm">
              <thead className="text-muted">
                <tr>
                  <th className="border-b border-line py-2 pr-4 font-medium">Model</th>
                  <th className="border-b border-line py-2 pr-4 font-medium">Recall</th>
                  <th className="border-b border-line py-2 pr-4 font-medium">Accuracy</th>
                  <th className="border-b border-line py-2 pr-4 font-medium">ROC-AUC</th>
                </tr>
              </thead>
              <tbody>
                {Object.entries(modelSelection.baseline_results || {}).map(([model, metrics]) => (
                  <tr key={model}>
                    <td className="border-b border-line py-2 pr-4 font-medium">{model}</td>
                    <td className="border-b border-line py-2 pr-4">{formatMetric(metrics.mean_recall)}</td>
                    <td className="border-b border-line py-2 pr-4">{formatMetric(metrics.mean_accuracy)}</td>
                    <td className="border-b border-line py-2 pr-4">{formatMetric(metrics.mean_roc_auc)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <p className="mt-4 rounded-lg border border-line bg-panel p-3 text-sm text-muted">Model comparison appears after training.</p>
        )}
        <div className="mt-4 rounded-lg border border-line bg-panel p-3 text-sm dark:bg-zinc-950/40">
          <span className="text-muted">Champion:</span> <span className="font-semibold">{modelSelection.champion_model || "-"}</span>
        </div>
      </PipelineCard>
      <PipelineCard
        title="Evaluation"
        icon={Activity}
        rows={[
          ["Accuracy", formatMetric(evaluation.accuracy)],
          ["Precision", formatMetric(evaluation.precision)],
          ["Recall", formatMetric(evaluation.recall)],
          ["F1 score", formatMetric(evaluation.f1)],
          ["ROC-AUC", formatMetric(evaluation.roc_auc)],
          ["Specificity", formatMetric(evaluation.specificity)]
        ]}
      />
    </motion.div>
  );
}

function PipelineCard({ title, icon: Icon, rows = [], children }) {
  return (
    <motion.section variants={fadeUp} whileHover={{ y: -2 }} className="rounded-xl border border-line bg-card p-5 shadow-sm transition hover:border-primary/30 hover:shadow-soft">
      <SectionTitle icon={Icon} title={title} eyebrow="Pipeline stats" />
      {rows.length > 0 && (
        <dl className="mt-4 divide-y divide-line rounded-lg border border-line">
          {rows.map(([label, value]) => (
            <div key={label} className="grid gap-1 px-3 py-3 text-sm sm:grid-cols-[160px_1fr]">
              <dt className="text-muted">{label}</dt>
              <dd className="break-words font-medium">{value ?? "-"}</dd>
            </div>
          ))}
        </dl>
      )}
      {children}
    </motion.section>
  );
}

function Card({ children }) {
  return (
    <motion.section
      variants={fadeUp}
      initial="hidden"
      animate="show"
      transition={{ duration: 0.35 }}
      className="rounded-xl border border-line bg-card p-5 shadow-sm sm:p-6"
    >
      {children}
    </motion.section>
  );
}

function SectionTitle({ icon: Icon, title, eyebrow }) {
  return (
    <div className="flex items-center justify-between gap-4">
      <div>
        <p className="text-xs font-semibold uppercase tracking-[0.2em] text-primary">{eyebrow}</p>
        <h3 className="mt-1 text-lg font-semibold tracking-tight">{title}</h3>
      </div>
      <motion.div whileHover={{ scale: 1.04 }} className="grid h-10 w-10 place-items-center rounded-xl bg-primarySoft text-primary">
        <Icon size={18} />
      </motion.div>
    </div>
  );
}

function Button({ children, disabled, loading, icon: Icon, tone = "primary", type = "submit", onClick }) {
  return (
    <motion.button
      whileHover={!disabled ? { y: -1, scale: 1.01 } : undefined}
      whileTap={!disabled ? { scale: 0.98 } : undefined}
      className={`action-button inline-flex min-h-12 items-center justify-center gap-2 rounded-xl px-5 text-sm font-semibold text-white shadow-sm outline-none transition focus:ring-4 disabled:cursor-not-allowed ${
        tone === "success" ? "bg-success focus:ring-success/20" : "bg-primary focus:ring-primary/20"
      }`}
      disabled={disabled}
      type={type}
      onClick={onClick}
    >
      {loading ? <SkeletonDot /> : <Icon size={18} />}
      {children}
    </motion.button>
  );
}

function ThemeToggle({ theme, onTheme, className = "" }) {
  const isDark = theme === "dark";
  return (
    <motion.button
      whileTap={{ scale: 0.96 }}
      className={`secondary-action inline-flex min-h-11 items-center justify-center gap-2 rounded-xl border px-4 text-sm font-semibold shadow-sm outline-none transition hover:border-primary/50 focus:ring-4 focus:ring-primary/15 ${className}`}
      type="button"
      aria-label="Toggle theme"
      onClick={() => onTheme(isDark ? "light" : "dark")}
    >
      <motion.span animate={{ rotate: isDark ? 180 : 0 }} transition={{ duration: 0.3 }}>
        {isDark ? <Moon size={17} /> : <Sun size={17} />}
      </motion.span>
      {isDark ? "Dark" : "Light"}
    </motion.button>
  );
}

function IconButton({ children, danger = false, title, onClick }) {
  return (
    <button
      aria-label={title}
      className={`grid h-9 w-9 place-items-center rounded-lg border text-sm transition focus:ring-4 focus:ring-primary/15 ${
        danger
          ? "border-danger/25 bg-danger/10 text-danger hover:bg-danger/15"
          : "border-line bg-card text-foreground hover:border-primary/40 hover:text-primary"
      }`}
      title={title}
      type="button"
      onClick={onClick}
    >
      {children}
    </button>
  );
}

function SideButton({ tab, active, onClick }) {
  const Icon = tab.icon;
  return (
    <motion.button
      whileHover={{ x: 2 }}
      whileTap={{ scale: 0.98 }}
      className={`relative flex min-h-11 w-full items-center gap-3 rounded-xl px-3 text-left text-sm font-semibold outline-none transition focus:ring-4 focus:ring-primary/15 ${
        active ? "text-foreground" : "text-muted hover:bg-panel/80 hover:text-foreground"
      }`}
      onClick={onClick}
      type="button"
    >
      {active && <motion.span layoutId="active-pill" className="absolute inset-0 rounded-xl bg-primarySoft" />}
      <span className={`relative z-10 grid h-8 w-8 place-items-center rounded-lg ${active ? "icon-surface" : "bg-primarySoft text-primary dark:bg-zinc-900 dark:text-blue-300"}`}>
        <Icon size={17} />
      </span>
      <span className="relative z-10">{tab.label}</span>
    </motion.button>
  );
}

function StatusBadge({ status }) {
  const normalized = status || "idle";
  const tone = normalized === "completed" ? "success" : normalized === "failed" ? "danger" : normalized === "running" ? "warning" : "muted";
  return <span className={`rounded-full px-2.5 py-1 text-xs font-semibold ${badgeTone(tone)}`}>{normalized}</span>;
}

function ResultBadge({ prediction, label }) {
  return <span className={`rounded-full px-2.5 py-1 text-xs font-semibold ${prediction ? "bg-danger/10 text-danger" : "bg-success/10 text-success"}`}>{label}</span>;
}

function Toast({ message, loading, onClose }) {
  return (
    <motion.div
      initial={{ opacity: 0, y: -10, scale: 0.98 }}
      animate={{ opacity: 1, y: 0, scale: 1 }}
      exit={{ opacity: 0, y: -10, scale: 0.98 }}
      className="mb-5 flex items-center justify-between gap-3 rounded-xl border border-line bg-card px-4 py-3 text-sm shadow-sm"
      role="status"
    >
      <span className="flex items-center gap-2">
        {loading ? <SkeletonDot /> : <AlertCircle size={18} className="text-primary" />}
        {message}
      </span>
      <button className="rounded-lg px-2 py-1 text-muted outline-none transition hover:bg-panel focus:ring-4 focus:ring-primary/15" onClick={onClose} type="button">Dismiss</button>
    </motion.div>
  );
}

function EmptyState({ title, text, icon: Icon, compact = false }) {
  return (
    <motion.div initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} className={`rounded-xl border border-dashed border-line bg-card text-center shadow-sm ${compact ? "p-5" : "p-8"}`}>
      <div className="mx-auto grid h-12 w-12 place-items-center rounded-xl bg-primarySoft text-primary">
        <Icon size={22} />
      </div>
      <h3 className="mt-4 text-base font-semibold">{title}</h3>
      <p className="mx-auto mt-2 max-w-xl text-sm leading-6 text-muted">{text}</p>
    </motion.div>
  );
}

function AppLoader({ error, loading, onRetry, theme, onTheme }) {
  return (
    <div className="grid min-h-screen place-items-center bg-background px-5 text-foreground">
      <motion.div
        initial={{ opacity: 0, y: 14, scale: 0.98 }}
        animate={{ opacity: 1, y: 0, scale: 1 }}
        transition={{ duration: 0.35 }}
        className="w-full max-w-md rounded-2xl border border-line bg-card p-6 shadow-soft"
      >
        <div className="flex items-start justify-between gap-4">
          <div className="flex items-center gap-3">
            <span className="icon-surface grid h-11 w-11 place-items-center rounded-xl">
              <HeartPulse size={22} />
            </span>
            <div>
              <p className="text-xs font-semibold uppercase tracking-[0.18em] text-primary">Backend status</p>
              <h1 className="mt-1 text-xl font-semibold tracking-tight">
                {error ? "FastAPI is not ready" : "Connecting to FastAPI"}
              </h1>
            </div>
          </div>
          <ThemeToggle theme={theme} onTheme={onTheme} />
        </div>

        <div className="mt-6 rounded-xl border border-line bg-panel p-4">
          {error ? (
            <div className="space-y-4">
              <div className="flex items-start gap-3 text-sm leading-6 text-danger">
                <AlertCircle className="mt-0.5 shrink-0" size={18} />
                <span>{error}</span>
              </div>
              <Button disabled={loading} loading={loading} icon={Activity} onClick={onRetry} type="button">
                Check again
              </Button>
            </div>
          ) : (
            <div className="space-y-4">
              <div className="flex items-center gap-3 text-sm font-medium text-muted">
                <SkeletonDot dark />
                Checking http://localhost:8000/
              </div>
              <div className="space-y-2">
                <div className="h-3 w-full overflow-hidden rounded-full bg-card">
                  <motion.div
                    className="h-full rounded-full bg-primary"
                    initial={{ x: "-100%" }}
                    animate={{ x: "100%" }}
                    transition={{ duration: 1.1, repeat: Infinity, ease: "easeInOut" }}
                  />
                </div>
                <p className="text-xs text-muted">Dashboard will open automatically after status returns ok.</p>
              </div>
            </div>
          )}
        </div>
      </motion.div>
    </div>
  );
}

function MetricSkeleton() {
  return (
    <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-3">
      {Array.from({ length: 6 }).map((_, index) => (
        <div key={index} className="relative overflow-hidden rounded-2xl border border-line bg-card/80 p-5 shadow-soft">
          <div className="h-4 w-28 rounded-full bg-panel" />
          <div className="mt-5 h-8 w-36 rounded-full bg-panel" />
          <div className="absolute inset-0 -translate-x-full bg-gradient-to-r from-transparent via-white/20 to-transparent animate-shimmer" />
        </div>
      ))}
    </div>
  );
}

function StatsSkeleton() {
  return (
    <div className="grid gap-5 xl:grid-cols-2">
      {Array.from({ length: 4 }).map((_, index) => (
        <div key={index} className="relative overflow-hidden rounded-xl border border-line bg-card p-5 shadow-sm">
          <div className="h-3 w-28 rounded-full bg-panel" />
          <div className="mt-3 h-5 w-48 rounded-full bg-panel" />
          <div className="mt-5 space-y-3 rounded-lg border border-line p-3">
            <div className="h-4 w-full rounded-full bg-panel" />
            <div className="h-4 w-5/6 rounded-full bg-panel" />
            <div className="h-4 w-4/6 rounded-full bg-panel" />
          </div>
          <div className="absolute inset-0 -translate-x-full bg-gradient-to-r from-transparent via-white/20 to-transparent animate-shimmer" />
        </div>
      ))}
    </div>
  );
}

function TableSkeletonRows({ columns, rows }) {
  return Array.from({ length: rows }).map((_, rowIndex) => (
    <tr key={rowIndex}>
      {Array.from({ length: columns }).map((__, columnIndex) => (
        <td key={columnIndex} className="border-b border-line py-3 pr-4">
          <div className={`h-4 animate-pulse rounded-full bg-panel ${columnIndex === 0 ? "w-36" : "w-20"}`} />
        </td>
      ))}
    </tr>
  ));
}

function SkeletonDot({ dark = false }) {
  return <span className={`h-4 w-4 animate-pulse rounded-full ${dark ? "bg-primary" : "bg-white/80"}`} />;
}

function InfoRow({ label, value }) {
  return (
    <div className="flex items-start justify-between gap-4 border-b border-line pb-3">
      <dt className="text-muted">{label}</dt>
      <dd className="break-all text-right font-semibold">{value}</dd>
    </div>
  );
}

function toneClass(tone) {
  if (tone === "success") return "border border-success/20 bg-success/10 text-success dark:border-success/30";
  if (tone === "warning") return "border border-warning/20 bg-warning/10 text-warning dark:border-warning/30";
  return "icon-surface";
}

function badgeTone(tone) {
  if (tone === "success") return "border border-success/25 bg-success/10 text-success";
  if (tone === "danger") return "border border-danger/25 bg-danger/10 text-danger";
  if (tone === "warning") return "border border-warning/25 bg-warning/10 text-warning";
  return "border border-line bg-card text-muted";
}

function formatMetric(value) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "-";
  return Number(value).toFixed(4);
}

function formatPercent(value) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "-";
  return `${Math.round(Number(value) * 100)}%`;
}

function formatSex(value) {
  if (Number(value) === 1) return "Male";
  if (Number(value) === 0) return "Female";
  return "-";
}

function formatDate(value) {
  if (!value) return "unknown date";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return "unknown date";
  return date.toLocaleString();
}

function mobileTabClass(active) {
  return `min-h-9 rounded-xl px-4 text-sm font-semibold transition ${active ? "bg-card text-foreground shadow-sm" : "text-muted"}`;
}

const fadeUp = {
  hidden: { opacity: 0, y: 14 },
  show: { opacity: 1, y: 0 }
};

const stagger = {
  hidden: {},
  show: {
    transition: {
      staggerChildren: 0.06
    }
  }
};

class ErrorBoundary extends Component {
  constructor(props) {
    super(props);
    this.state = { error: null };
  }

  static getDerivedStateFromError(error) {
    return { error };
  }

  render() {
    if (this.state.error) {
      return (
        <div className="min-h-screen bg-background p-6 text-foreground">
          <div className="mx-auto max-w-3xl rounded-2xl border border-line bg-card p-6 shadow-soft">
            <div className="flex items-center gap-2 text-sm font-semibold text-danger">
              <AlertCircle size={18} />
              UI error
            </div>
            <p className="mt-3 text-sm text-muted">{this.state.error.message}</p>
          </div>
        </div>
      );
    }

    return this.props.children;
  }
}

createRoot(document.getElementById("root")).render(
  <ErrorBoundary>
    <App />
  </ErrorBoundary>
);
