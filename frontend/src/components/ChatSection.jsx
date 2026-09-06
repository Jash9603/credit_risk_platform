import { useEffect, useRef, useState } from "react";
import { api } from "../api/client";
import { EXAMPLE_QUESTIONS } from "../constants";
import { BuildNote, Card, ChipButton, ErrorBox, PrimaryButton } from "./ui";

function UserBubble({ children }) {
  return (
    <div style={{ display: "flex", justifyContent: "flex-end", marginBottom: "0.6rem" }}>
      <div
        style={{
          background: "var(--accent)",
          color: "#fff",
          padding: "0.65rem 1rem",
          borderRadius: "14px 14px 4px 14px",
          maxWidth: "70%",
          fontSize: "0.95rem",
        }}
      >
        {children}
      </div>
    </div>
  );
}

function AssistantBubble({ children }) {
  return (
    <div style={{ display: "flex", justifyContent: "flex-start", marginBottom: "1.25rem" }}>
      <div style={{ maxWidth: "78%" }}>
        <Card style={{ borderRadius: "14px 14px 14px 4px" }}>{children}</Card>
      </div>
    </div>
  );
}

export default function ChatSection() {
  const [turns, setTurns] = useState([]); // completed {question, sql, rows, answer, refused}
  const [pendingQuestion, setPendingQuestion] = useState(null);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const bottomRef = useRef(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [turns, pendingQuestion]);

  const send = (question) => {
    const q = question.trim();
    if (!q || loading) return;
    setLoading(true);
    setError(null);
    setPendingQuestion(q);
    setInput("");

    const history = turns.flatMap((t) => [
      { role: "user", content: t.question },
      { role: "assistant", content: t.answer },
    ]);

    api
      .chat(q, history)
      .then((result) => setTurns((prev) => [...prev, result]))
      .catch((e) => setError(e.message))
      .finally(() => {
        setLoading(false);
        setPendingQuestion(null);
      });
  };

  const fillInput = (text) => {
    if (loading) return;
    setInput(text);
  };

  return (
    <div style={{ display: "flex", flexDirection: "column", height: "100%", minHeight: 0 }}>
      <div style={{ flex: 1, minHeight: 0, overflowY: "auto", padding: "2rem 2rem 0" }}>
        <div style={{ maxWidth: 880, margin: "0 auto" }}>
          <h2>Talk to Your Data</h2>
          <p style={{ color: "var(--text-muted)" }}>
            Ask a question in plain English. It is turned into a database query, run
            against the real data, and answered using only what the query returns.
          </p>

          <BuildNote>
            When you type a question, it is sent to an AI language model that turns your
            words into a database query, similar to how a person would write one by hand.
            The query runs on the real data, and the same AI turns the results back into a
            plain English answer. Nothing is invented, every number you see comes straight
            from the query results, and the query itself is always shown below the answer
            so you can check it.
          </BuildNote>

          <Card style={{ marginBottom: "1.5rem", background: "var(--surface-alt)" }}>
            <p style={{ margin: 0, fontWeight: 600 }}>This assistant reliably handles 5 kinds of questions:</p>
            <ol style={{ paddingLeft: "1.25rem", margin: "0.5rem 0 0" }}>
              <li>Totals and averages across all applicants (e.g. average income)</li>
              <li>Comparisons across groups (e.g. default rate by education level)</li>
              <li>Rankings and filters (e.g. the 10 highest earners who had payment difficulty)</li>
              <li>Ratios between two groups (e.g. loan to income ratio, defaulted vs not)</li>
              <li>Questions spanning multiple tables (e.g. applicants with bad debt on their credit bureau record)</li>
            </ol>
          </Card>

          <Card style={{ marginBottom: "1.5rem", background: "var(--surface-alt)" }}>
            <p style={{ margin: 0, fontWeight: 600 }}>How this stays fast and cheap to run (token optimization):</p>
            <ul style={{ paddingLeft: "1.25rem", margin: "0.5rem 0 0" }}>
              <li>The AI only ever sees a compact list of table and column names, not long descriptions of every field, so each request stays small.</li>
              <li>Query results are capped at 200 rows by the database, and only the first 50 are ever shown to the step that writes the final answer, so one huge result never balloons a request.</li>
              <li>Follow-up questions are first condensed into a single, self-contained question by a small dedicated step, instead of replaying the entire conversation into every request as the chat grows longer.</li>
            </ul>
          </Card>

          <div>
            {turns.map((t, i) => (
              <div key={i}>
                <UserBubble>{t.question}</UserBubble>
                <AssistantBubble>
                  {t.standalone_question && t.standalone_question !== t.question && (
                    <p style={{ margin: "0 0 0.5rem", color: "var(--text-muted)", fontSize: "0.85rem", fontStyle: "italic" }}>
                      Interpreted as: "{t.standalone_question}"
                    </p>
                  )}
                  <p style={{ margin: 0, whiteSpace: "pre-line" }}>{t.answer}</p>
                  {t.sql && (
                    <details style={{ marginTop: "0.75rem" }}>
                      <summary style={{ cursor: "pointer", color: "var(--text-muted)", fontSize: "0.9rem" }}>
                        Show SQL and raw rows ({t.rows.length} row{t.rows.length === 1 ? "" : "s"})
                      </summary>
                      <pre
                        style={{
                          background: "var(--surface-alt)",
                          padding: "0.75rem",
                          borderRadius: 6,
                          overflowX: "auto",
                          fontSize: "0.85rem",
                        }}
                      >
                        {t.sql}
                      </pre>
                      <pre
                        style={{
                          background: "var(--surface-alt)",
                          padding: "0.75rem",
                          borderRadius: 6,
                          overflowX: "auto",
                          fontSize: "0.85rem",
                        }}
                      >
                        {JSON.stringify(t.rows.slice(0, 20), null, 2)}
                      </pre>
                    </details>
                  )}
                </AssistantBubble>
              </div>
            ))}

            {pendingQuestion && (
              <div>
                <UserBubble>{pendingQuestion}</UserBubble>
                <AssistantBubble>
                  <span style={{ color: "var(--text-muted)" }}>Thinking...</span>
                </AssistantBubble>
              </div>
            )}
          </div>

          <ErrorBox message={error} />
          <div ref={bottomRef} style={{ height: "1.5rem" }} />
        </div>
      </div>

      <div
        style={{
          flexShrink: 0,
          borderTop: "1px solid var(--border)",
          background: "var(--surface)",
          padding: "1rem 2rem",
        }}
      >
        <div style={{ maxWidth: 880, margin: "0 auto" }}>
          <p style={{ fontWeight: 600, marginBottom: "0.5rem", fontSize: "0.9rem" }}>Try asking</p>
          <div style={{ display: "flex", flexWrap: "wrap", gap: "0.5rem", marginBottom: "0.85rem" }}>
            {EXAMPLE_QUESTIONS.map((q) => (
              <ChipButton key={q} disabled={loading} onClick={() => fillInput(q)}>
                {q}
              </ChipButton>
            ))}
          </div>

          <form
            onSubmit={(e) => {
              e.preventDefault();
              send(input);
            }}
            style={{ display: "flex", gap: "0.75rem" }}
          >
            <input
              value={input}
              onChange={(e) => setInput(e.target.value)}
              disabled={loading}
              placeholder={loading ? "Waiting for the previous answer..." : "Ask a question about the data..."}
              style={{
                flex: 1,
                padding: "0.7rem 1rem",
                borderRadius: 8,
                border: "1px solid var(--border)",
                background: loading ? "var(--surface-alt)" : "var(--bg)",
              }}
            />
            <PrimaryButton type="submit" disabled={loading}>
              Send
            </PrimaryButton>
          </form>
        </div>
      </div>
    </div>
  );
}
