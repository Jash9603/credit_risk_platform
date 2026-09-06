import { useEffect, useState } from "react";
import { api } from "../api/client";
import { BuildNote, Card, ErrorBox, Spinner } from "./ui";

export default function RulesSection() {
  const [rules, setRules] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    api
      .rules()
      .then((r) => setRules(r.rules))
      .catch((e) => setError(e.message));
  }, []);

  return (
    <div>
      <h2>Decision Rules</h2>
      <p style={{ color: "var(--text-muted)" }}>
        The model's overall logic, distilled into a small set of plain "if this, then
        that" statements. A static summary a credit policy team could review on its own,
        without needing to run any prediction. Sorted from highest to lowest observed risk.
      </p>

      <BuildNote>
        We asked a much simpler, easy to read version of the model to copy the behaviour
        of the real, more complicated model. That simpler version naturally organises
        itself into a short list of plain rules, which is what you see below. Each rule
        also shows how many real applicants fall into it and what actually happened to
        them, so the rules can be checked against reality, not just trusted blindly.
      </BuildNote>

      <ErrorBox message={error} />
      {!rules && !error && <Spinner label="Loading rules..." />}

      <div style={{ display: "grid", gap: "0.9rem", marginTop: "1.25rem" }}>
        {rules?.map((rule, i) => (
          <Card key={i} style={{ borderLeft: `5px solid ${riskColor(rule.observed_default_rate)}` }}>
            <p style={{ margin: 0 }}>{rule.readable}</p>
          </Card>
        ))}
      </div>
    </div>
  );
}

function riskColor(rate) {
  if (rate >= 0.15) return "var(--high)";
  if (rate >= 0.08) return "var(--medium)";
  return "var(--low)";
}
