import { useState } from "react";
import { api } from "../api/client";
import { EXAMPLE_APPLICANT_IDS, RISK_BAND_COLORS } from "../constants";
import { Badge, BuildNote, Card, ChipButton, ErrorBox, PrimaryButton, Spinner } from "./ui";

export default function PredictSection() {
  const [skId, setSkId] = useState("");
  const [result, setResult] = useState(null);
  const [explanations, setExplanations] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const runAnalysis = (id) => {
    const applicantId = Number(id);
    if (!applicantId) {
      setError("Enter a numeric applicant ID.");
      return;
    }
    setLoading(true);
    setError(null);
    setResult(null);
    setExplanations(null);

    Promise.all([api.predict(applicantId), api.explain(applicantId)])
      .then(([predictRes, explainRes]) => {
        setResult(predictRes);
        setExplanations(explainRes.explanations);
      })
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false));
  };

  const maxAbs = explanations ? Math.max(...explanations.map((i) => Math.abs(i.shap_value))) : 1;

  return (
    <div>
      <h2>Risk Prediction and Explainability</h2>
      <p style={{ color: "var(--text-muted)" }}>
        Enter an applicant's ID to get a default probability, a 0-1000 risk score, a Low,
        Medium or High risk band, and a plain English breakdown of what drove that result.
        Try one of the real applicants below, or type your own ID.
      </p>

      <BuildNote>
        A machine learning model studied thousands of past loan applications where we
        already know what happened, meaning who repaid and who struggled, and learned the
        patterns that separated the two groups. For every guess it makes, a second
        technique called SHAP works backward through that same calculation to show which
        pieces of information pushed the guess up, which pushed it down, and roughly by
        how much, so the result is never just a mystery number.
      </BuildNote>

      <p style={{ fontWeight: 600, marginBottom: "0.5rem" }}>Example applicants</p>
      <div style={{ display: "flex", flexWrap: "wrap", gap: "0.5rem", marginBottom: "1.25rem" }}>
        {EXAMPLE_APPLICANT_IDS.map((id) => (
          <ChipButton
            key={id}
            active={skId === String(id)}
            disabled={loading}
            onClick={() => setSkId(String(id))}
          >
            {id}
          </ChipButton>
        ))}
      </div>

      <form
        onSubmit={(e) => {
          e.preventDefault();
          runAnalysis(skId);
        }}
        style={{ display: "flex", gap: "0.75rem", marginBottom: "1.5rem" }}
      >
        <input
          value={skId}
          onChange={(e) => setSkId(e.target.value)}
          disabled={loading}
          placeholder="e.g. 100002"
          style={{
            flex: 1,
            maxWidth: 240,
            padding: "0.6rem 0.9rem",
            borderRadius: 8,
            border: "1px solid var(--border)",
            background: loading ? "var(--surface-alt)" : "var(--surface)",
          }}
        />
        <PrimaryButton type="submit" disabled={loading}>
          {loading ? "Analyzing..." : "Analyze"}
        </PrimaryButton>
      </form>

      <ErrorBox message={error} />
      {loading && <Spinner label="Scoring applicant and computing the explanation..." />}

      <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(420px, 1fr))", gap: "1.25rem", alignItems: "start" }}>
        {result && (
          <Card>
            <h3 style={{ marginTop: 0 }}>Risk score</h3>
            <div style={{ display: "flex", alignItems: "center", gap: "1.5rem", flexWrap: "wrap" }}>
              <div>
                <div style={{ color: "var(--text-muted)", fontSize: "0.9rem" }}>Applicant</div>
                <div style={{ fontSize: "1.4rem", fontWeight: 700 }}>#{result.sk_id_curr}</div>
              </div>
              <div>
                <div style={{ color: "var(--text-muted)", fontSize: "0.9rem" }}>Risk score</div>
                <div style={{ fontSize: "1.4rem", fontWeight: 700 }}>{result.risk_score} / 1000</div>
              </div>
              <div>
                <div style={{ color: "var(--text-muted)", fontSize: "0.9rem" }}>Default probability</div>
                <div style={{ fontSize: "1.4rem", fontWeight: 700 }}>
                  {(result.default_probability * 100).toFixed(1)}%
                </div>
              </div>
              <div>
                <div style={{ color: "var(--text-muted)", fontSize: "0.9rem", marginBottom: 4 }}>Risk band</div>
                <Badge
                  color={RISK_BAND_COLORS[result.risk_band]?.text}
                  bg={RISK_BAND_COLORS[result.risk_band]?.bg}
                >
                  {result.risk_band}
                </Badge>
              </div>
            </div>
            {result.actual_target !== null && result.actual_target !== undefined && (
              <p style={{ marginTop: "1rem", color: "var(--text-muted)", fontSize: "0.9rem" }}>
                This is a real applicant from the training data. Their actual recorded outcome was{" "}
                <strong>{result.actual_target === 1 ? "had payment trouble" : "repaid fine"}</strong>.
              </p>
            )}
          </Card>
        )}

        {explanations && (
          <Card>
            <h3 style={{ marginTop: 0 }}>Top drivers</h3>
            <div style={{ display: "grid", gap: "0.85rem" }}>
              {explanations.map((item) => {
                const increase = item.shap_value > 0;
                const widthPct = (Math.abs(item.shap_value) / maxAbs) * 100;
                return (
                  <div key={item.feature}>
                    <div style={{ fontSize: "0.92rem" }}>{item.sentence}</div>
                    <div
                      style={{
                        height: 8,
                        borderRadius: 4,
                        background: "var(--surface-alt)",
                        marginTop: 4,
                        overflow: "hidden",
                      }}
                    >
                      <div
                        style={{
                          height: "100%",
                          width: `${widthPct}%`,
                          background: increase ? "var(--high)" : "var(--low)",
                          borderRadius: 4,
                        }}
                      />
                    </div>
                  </div>
                );
              })}
            </div>
            <p style={{ color: "var(--text-muted)", fontSize: "0.85rem", marginTop: "1rem", marginBottom: 0 }}>
              Red bars pushed the risk score up, green bars pulled it down. Bar length
              shows how much each factor mattered relative to the others shown.
            </p>
          </Card>
        )}
      </div>
    </div>
  );
}
