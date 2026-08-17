import { FormEvent, useEffect, useState } from "react";

const API_URL = import.meta.env.VITE_API_URL ?? "http://localhost:8000";

type Mode = "qa" | "triage";

interface RepoInfo {
  collection: string;
  issue_count: number;
}

interface QAResult {
  question: string;
  answer: string;
  collection: string;
}

interface TriageResult {
  priority: "high" | "medium" | "low";
  is_duplicate: boolean;
  duplicate_of: number | null;
  reasoning: string;
  collection: string;
}

const PRIORITY_COLOR: Record<string, string> = {
  high: "var(--high)",
  medium: "var(--mid)",
  low: "var(--low)",
};

// The signature element: a vertical signal bar whose height/color encodes a
// 0-1 strength value -- used for retrieval relevance in Ask mode and priority
// level in Triage mode. Same visual language, same underlying idea: how
// strong is this signal.
function SignalBar({ strength, color }: { strength: number; color: string }) {
  return (
    <div className="signal-bar-track" aria-hidden="true">
      <div
        className="signal-bar-fill"
        style={{ height: `${Math.max(8, strength * 100)}%`, background: color }}
      />
    </div>
  );
}

function HealthPing() {
  const [status, setStatus] = useState<"checking" | "up" | "down">("checking");

  useEffect(() => {
    fetch(`${API_URL}/health`)
      .then((r) => (r.ok ? setStatus("up") : setStatus("down")))
      .catch(() => setStatus("down"));
  }, []);

  return (
    <div className="health-ping" data-status={status}>
      <span className="dot" />
      {status === "checking" ? "connecting..." : status === "up" ? "api online" : "api offline"}
    </div>
  );
}

// Repo picker + "add a new repo" flow. Polls the background ingest job until
// it finishes, then refreshes the repo list and selects the new one.
function RepoBar({
  repos,
  selected,
  onSelect,
  onReposChanged,
}: {
  repos: RepoInfo[];
  selected: string;
  onSelect: (collection: string) => void;
  onReposChanged: () => void;
}) {
  const [showAddForm, setShowAddForm] = useState(false);
  const [newRepo, setNewRepo] = useState("");
  const [jobStatus, setJobStatus] = useState<string | null>(null);
  const [jobError, setJobError] = useState<string | null>(null);
  const [adding, setAdding] = useState(false);

  async function handleAddRepo(e: FormEvent) {
    e.preventDefault();
    setAdding(true);
    setJobError(null);
    setJobStatus("queuing...");

    try {
      const res = await fetch(`${API_URL}/repos`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ repo: newRepo.trim(), max_issues: 500 }),
      });
      if (!res.ok) throw new Error((await res.json()).error ?? `Request failed (${res.status})`);
      const { job_id } = await res.json();

      const poll = setInterval(async () => {
        const jobRes = await fetch(`${API_URL}/repos/jobs/${job_id}`);
        const job = await jobRes.json();
        setJobStatus(job.message ?? job.status);

        if (job.status === "done") {
          clearInterval(poll);
          setAdding(false);
          setNewRepo("");
          onReposChanged();
          if (job.result?.collection) onSelect(job.result.collection);
          setTimeout(() => setJobStatus(null), 3000);
        } else if (job.status === "error") {
          clearInterval(poll);
          setAdding(false);
          setJobError(job.message ?? "Ingest failed.");
        }
      }, 2000);
    } catch (err) {
      setAdding(false);
      setJobError(err instanceof Error ? err.message : "Something went wrong.");
    }
  }

  return (
    <div className="repo-bar">
      <div className="repo-select-row">
        <select
          className="repo-select"
          value={selected}
          onChange={(e) => onSelect(e.target.value)}
          disabled={repos.length === 0}
        >
          {repos.length === 0 ? (
            <option>no repos ingested yet</option>
          ) : (
            repos.map((r) => (
              <option key={r.collection} value={r.collection}>
                {r.collection} ({r.issue_count} issues)
              </option>
            ))
          )}
        </select>
        <button type="button" className="add-repo-btn" onClick={() => setShowAddForm((s) => !s)}>
          + add repo
        </button>
      </div>

      {showAddForm && (
        <form onSubmit={handleAddRepo} className="add-repo-form">
          <input
            placeholder="owner/name - e.g. facebook/react"
            value={newRepo}
            onChange={(e) => setNewRepo(e.target.value)}
            disabled={adding}
          />
          <button type="submit" className="submit-btn small" disabled={adding || newRepo.trim().length < 3}>
            {adding ? "ingesting..." : "ingest"}
          </button>
        </form>
      )}

      {jobStatus && <div className="job-status">{jobStatus}</div>}
      {jobError && <div className="error-panel">[!] {jobError}</div>}
    </div>
  );
}

export default function App() {
  const [mode, setMode] = useState<Mode>("qa");
  const [repos, setRepos] = useState<RepoInfo[]>([]);
  const [selectedCollection, setSelectedCollection] = useState<string>("");
  const [question, setQuestion] = useState("");
  const [title, setTitle] = useState("");
  const [body, setBody] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [qaResult, setQaResult] = useState<QAResult | null>(null);
  const [triageResult, setTriageResult] = useState<TriageResult | null>(null);

  async function loadRepos(preferCollection?: string) {
    try {
      const res = await fetch(`${API_URL}/repos`);
      const data = await res.json();
      setRepos(data.repos ?? []);
      if (preferCollection) {
        setSelectedCollection(preferCollection);
      } else if (data.repos?.length && !selectedCollection) {
        setSelectedCollection(data.repos[0].collection);
      }
    } catch {
      // API might not be up yet -- HealthPing already surfaces that
    }
  }

  useEffect(() => {
    loadRepos();
  }, []);

  async function handleSubmit(e: FormEvent) {
    e.preventDefault();
    setLoading(true);
    setError(null);
    setQaResult(null);
    setTriageResult(null);

    try {
      if (mode === "qa") {
        const res = await fetch(`${API_URL}/qa`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ question, collection: selectedCollection || undefined }),
        });
        if (!res.ok) throw new Error((await res.json()).error ?? `Request failed (${res.status})`);
        setQaResult(await res.json());
      } else {
        const res = await fetch(`${API_URL}/triage`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ title, body, collection: selectedCollection || undefined }),
        });
        if (!res.ok) throw new Error((await res.json()).error ?? `Request failed (${res.status})`);
        setTriageResult(await res.json());
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "Something went wrong.");
    } finally {
      setLoading(false);
    }
  }

  const canSubmit =
    (mode === "qa" ? question.trim().length > 2 : title.trim().length > 2 && body.trim().length > 2) &&
    repos.length > 0;

  return (
    <div className="page">
      <header className="topbar">
        <div className="wordmark">
          REPO<span className="slash">//</span>TRIAGE
        </div>
        <HealthPing />
      </header>

      <RepoBar
        repos={repos}
        selected={selectedCollection}
        onSelect={setSelectedCollection}
        onReposChanged={() => loadRepos()}
      />

      <main className="console">
        <div className="tabs" role="tablist">
          <button
            role="tab"
            aria-selected={mode === "qa"}
            className={mode === "qa" ? "tab active" : "tab"}
            onClick={() => setMode("qa")}
          >
            01 - Ask
          </button>
          <button
            role="tab"
            aria-selected={mode === "triage"}
            className={mode === "triage" ? "tab active" : "tab"}
            onClick={() => setMode("triage")}
          >
            02 - Triage
          </button>
        </div>

        <form onSubmit={handleSubmit} className="console-form">
          {mode === "qa" ? (
            <div className="prompt-line">
              <span className="prompt-char">&gt;</span>
              <input
                autoFocus
                placeholder="What are common causes of crash reports?"
                value={question}
                onChange={(e) => setQuestion(e.target.value)}
                maxLength={1000}
              />
            </div>
          ) : (
            <div className="triage-fields">
              <div className="prompt-line">
                <span className="prompt-char">&gt;</span>
                <input
                  autoFocus
                  placeholder="Issue title - e.g. App crashes on save"
                  value={title}
                  onChange={(e) => setTitle(e.target.value)}
                  maxLength={300}
                />
              </div>
              <textarea
                placeholder="Describe what happens..."
                value={body}
                onChange={(e) => setBody(e.target.value)}
                rows={3}
                maxLength={5000}
              />
            </div>
          )}
          <button type="submit" className="submit-btn" disabled={!canSubmit || loading}>
            {loading ? "processing..." : mode === "qa" ? "ask" : "triage"}
          </button>
        </form>

        {repos.length === 0 && (
          <div className="loading-log">no repos ingested yet -- use "+ add repo" above to get started</div>
        )}

        {error && <div className="error-panel">[!] {error}</div>}

        {loading && (
          <div className="loading-log">
            <span className="cursor">_</span> querying vector index...
          </div>
        )}

        {qaResult && (
          <section className="result-card">
            <div className="result-row">
              <SignalBar strength={0.85} color="var(--accent)" />
              <div className="result-body">
                <div className="result-label">answer -- {qaResult.collection}</div>
                <p className="result-text">{qaResult.answer}</p>
              </div>
            </div>
          </section>
        )}

        {triageResult && (
          <section className="result-card">
            <div className="result-row">
              <SignalBar
                strength={triageResult.priority === "high" ? 1 : triageResult.priority === "medium" ? 0.6 : 0.3}
                color={PRIORITY_COLOR[triageResult.priority]}
              />
              <div className="result-body">
                <div className="badges">
                  <span
                    className="priority-badge"
                    style={{ color: PRIORITY_COLOR[triageResult.priority], borderColor: PRIORITY_COLOR[triageResult.priority] }}
                  >
                    {triageResult.priority} priority
                  </span>
                  {triageResult.is_duplicate && (
                    <span className="dup-badge">duplicate of #{triageResult.duplicate_of}</span>
                  )}
                </div>
                <div className="result-label">reasoning -- {triageResult.collection}</div>
                <p className="result-text">{triageResult.reasoning}</p>
              </div>
            </div>
          </section>
        )}
      </main>

      <footer className="footer">retrieval-augmented triage over any public github repo - groq + qdrant</footer>
    </div>
  );
}