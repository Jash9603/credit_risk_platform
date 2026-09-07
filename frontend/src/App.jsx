import { useState } from "react";
import ChatSection from "./components/ChatSection";
import EDASection from "./components/EDASection";
import PredictSection from "./components/PredictSection";
import RulesSection from "./components/RulesSection";

const TABS = [
  { id: "eda", label: "EDA", hint: "Explore the data", component: EDASection },
  { id: "predict", label: "Risk Prediction", hint: "Score and explain an applicant", component: PredictSection },
  { id: "rules", label: "Decision Rules", hint: "The model's policy", component: RulesSection },
  { id: "chat", label: "Chatbot", hint: "Ask in plain English", component: ChatSection },
];

export default function App() {
  const [activeTab, setActiveTab] = useState("eda");
  const active = TABS.find((t) => t.id === activeTab);
  const Active = active.component;
  const isChat = activeTab === "chat";

  return (
    <div style={{ display: "flex", height: "100vh", background: "var(--bg)" }}>
      <aside
        style={{
          width: "30%",
          maxWidth: 340,
          minWidth: 220,
          background: "var(--surface)",
          borderRight: "1px solid var(--border)",
          display: "flex",
          flexDirection: "column",
          padding: "1.5rem 1.25rem",
        }}
      >
        <h1 style={{ margin: 0, fontSize: "1.25rem" }}>Credit Risk Intelligence Platform</h1>
        <p style={{ margin: "0.5rem 0 1.75rem", color: "var(--text-muted)", fontSize: "0.9rem" }}>
          Home Credit Default Risk: exploratory analysis, risk scoring, explainability,
          policy rules, and a talk to data assistant.
        </p>

        <nav style={{ display: "flex", flexDirection: "column", gap: "0.4rem" }}>
          {TABS.map((tab) => {
            const isActive = tab.id === activeTab;
            return (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                style={{
                  textAlign: "left",
                  padding: "0.75rem 1rem",
                  borderRadius: 8,
                  border: "none",
                  background: isActive ? "var(--accent-soft)" : "transparent",
                  color: isActive ? "var(--accent-dark)" : "var(--text)",
                  fontWeight: isActive ? 700 : 500,
                }}
              >
                <div style={{ fontSize: "1rem" }}>{tab.label}</div>
                <div style={{ fontSize: "0.8rem", color: "var(--text-muted)", fontWeight: 400 }}>
                  {tab.hint}
                </div>
              </button>
            );
          })}
        </nav>

        <div style={{ marginTop: "auto", paddingTop: "1.5rem", color: "var(--text-muted)", fontSize: "0.78rem" }}>
          Built end to end: data pipeline, ML model, explainability, business rules, and a
          talk to data chatbot, all working on the real dataset.
        </div>

        <div
          style={{
            marginTop: "1rem",
            padding: "0.65rem 0.75rem",
            borderRadius: 8,
            background: "var(--high-bg)",
            color: "var(--high)",
            fontSize: "0.75rem",
            lineHeight: 1.45,
          }}
        >
          This deployment runs on free-tier hosting with a free-tier database. Because the
          dataset is large, predictions and chatbot answers can take a while to come back.
          Running it locally with Docker is much faster.
        </div>
      </aside>

      <main
        style={{
          flex: 1,
          height: "100vh",
          overflowY: isChat ? "hidden" : "auto",
          display: "flex",
          flexDirection: "column",
        }}
      >
        <div style={{ maxWidth: isChat ? "none" : 1200, width: "100%", margin: isChat ? 0 : "0 auto", padding: isChat ? 0 : "2rem", flex: 1, minHeight: 0, display: "flex", flexDirection: "column" }}>
          <Active />
        </div>
      </main>
    </div>
  );
}
