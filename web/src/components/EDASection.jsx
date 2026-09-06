import { useEffect, useState } from "react";
import { api, API_BASE } from "../api/client";
import { BuildNote, Card, ErrorBox, Spinner } from "./ui";

const KEY_TAKEAWAYS = [
  "Only about 8% of applicants had trouble paying. That is a real imbalance in the data, so the model needs special handling for it instead of relying on plain accuracy.",
  "External credit scores (EXT_SOURCE_1, 2 and 3) are the strongest single predictor. A lower score is consistently linked to more payment trouble across all three.",
  "Younger applicants have more payment trouble than older applicants. Age on its own is a simple, easy to explain risk factor.",
  "Income type and education level both separate risk cleanly. Applicants who are unemployed, on maternity leave, or have lower education run well above the average trouble rate.",
  "Loan to income ratio matters, but it is only one signal among several. On its own the effect is real but modest.",
  "A data quality issue turned into a useful feature. The placeholder value in the employment column actually flags a lower risk group of retirees, and having zero credit bureau history is itself a mild risk signal, not something neutral.",
];

export default function EDASection() {
  const [summary, setSummary] = useState(null);
  const [insights, setInsights] = useState([]);
  const [error, setError] = useState(null);

  useEffect(() => {
    Promise.all([api.edaSummary(), api.edaInsights()])
      .then(([s, i]) => {
        setSummary(s);
        setInsights(i);
      })
      .catch((e) => setError(e.message));
  }, []);

  return (
    <div>
      <h2>Exploratory Data Analysis</h2>
      <p style={{ color: "var(--text-muted)" }}>
        A first look at who is applying for these loans, what shape the data is in, and
        the patterns that matter most for spotting risk. Computed once in the analysis
        notebook and shown here as is.
      </p>

      <BuildNote>
        We loaded every loan application into a table, checked which pieces of
        information were missing or unusual, and drew charts to spot patterns before
        building any prediction model. Nothing here uses AI, it is careful counting and
        charting of the real data.
      </BuildNote>

      <ErrorBox message={error} />

      {!summary && !error && <Spinner label="Loading summary..." />}

      {summary && (
        <div
          style={{
            display: "grid",
            gridTemplateColumns: "repeat(auto-fit, minmax(160px, 1fr))",
            gap: "1rem",
            margin: "1.5rem 0",
          }}
        >
          <StatCard label="Applicants" value={summary.n_applicants.toLocaleString()} />
          <StatCard label="Features" value={summary.n_features} />
          <StatCard label="Had payment trouble" value={`${summary.default_rate_pct}%`} />
          <StatCard label="Columns with missing data" value={summary.n_columns_with_missing} />
        </div>
      )}

      <div
        style={{
          display: "grid",
          gridTemplateColumns: "repeat(auto-fit, minmax(420px, 1fr))",
          gap: "1.25rem",
          alignItems: "start",
        }}
      >
        {insights.map((item) => (
          <Card key={item.chart_url}>
            <h3 style={{ marginTop: 0, fontSize: "1.05rem" }}>{item.title}</h3>
            <p style={{ color: "var(--text-muted)", fontSize: "0.9rem" }}>{item.description}</p>
            <img
              src={`${API_BASE}${item.chart_url}`}
              alt={item.title}
              style={{ maxWidth: "100%", borderRadius: 8, border: "1px solid var(--border)" }}
            />
          </Card>
        ))}
      </div>

      <Card style={{ marginTop: "1.5rem", background: "var(--accent-soft)" }}>
        <h3 style={{ marginTop: 0 }}>Summary: the key takeaways</h3>
        <ol style={{ paddingLeft: "1.25rem", margin: 0 }}>
          {KEY_TAKEAWAYS.map((point, i) => (
            <li key={i} style={{ marginBottom: "0.6rem" }}>
              {point}
            </li>
          ))}
        </ol>
        <p style={{ color: "var(--text-muted)", fontSize: "0.9rem", marginTop: "1rem", marginBottom: 0 }}>
          These findings directly shaped the feature engineering, the model, and the
          business rules covered in the other sections.
        </p>
      </Card>
    </div>
  );
}

function StatCard({ label, value }) {
  return (
    <Card style={{ textAlign: "center" }}>
      <div style={{ fontSize: "1.6rem", fontWeight: 700, color: "var(--accent-dark)" }}>
        {value}
      </div>
      <div style={{ color: "var(--text-muted)", fontSize: "0.9rem", marginTop: 4 }}>{label}</div>
    </Card>
  );
}
