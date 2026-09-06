export function Card({ children, style }) {
  return (
    <div
      style={{
        background: "var(--surface)",
        border: "1px solid var(--border)",
        borderRadius: 10,
        padding: "1.25rem 1.5rem",
        ...style,
      }}
    >
      {children}
    </div>
  );
}

export function Badge({ children, color = "var(--text)", bg = "var(--surface-alt)" }) {
  return (
    <span
      style={{
        display: "inline-block",
        padding: "0.25rem 0.75rem",
        borderRadius: 999,
        fontSize: "0.85rem",
        fontWeight: 600,
        color,
        background: bg,
      }}
    >
      {children}
    </span>
  );
}

export function ChipButton({ children, onClick, active, disabled }) {
  return (
    <button
      onClick={onClick}
      disabled={disabled}
      style={{
        padding: "0.4rem 0.9rem",
        borderRadius: 999,
        border: `1px solid ${active ? "var(--accent)" : "var(--border)"}`,
        background: active ? "var(--accent-soft)" : "var(--surface)",
        color: disabled ? "var(--text-muted)" : active ? "var(--accent-dark)" : "var(--text)",
        fontSize: "0.9rem",
        fontWeight: 500,
        opacity: disabled ? 0.55 : 1,
        cursor: disabled ? "not-allowed" : "pointer",
      }}
    >
      {children}
    </button>
  );
}

export function PrimaryButton({ children, onClick, disabled, type = "button" }) {
  return (
    <button
      type={type}
      onClick={onClick}
      disabled={disabled}
      style={{
        padding: "0.6rem 1.4rem",
        borderRadius: 8,
        border: "none",
        background: disabled ? "var(--border)" : "var(--accent)",
        color: disabled ? "var(--text-muted)" : "#fff",
        fontSize: "1rem",
        fontWeight: 600,
      }}
    >
      {children}
    </button>
  );
}

export function Spinner({ label = "Loading..." }) {
  return <p style={{ color: "var(--text-muted)" }}>{label}</p>;
}

export function ErrorBox({ message }) {
  if (!message) return null;
  return (
    <div
      style={{
        background: "var(--high-bg)",
        color: "var(--high)",
        borderRadius: 8,
        padding: "0.75rem 1rem",
        marginTop: "0.75rem",
        fontSize: "0.95rem",
      }}
    >
      {message}
    </div>
  );
}

export function BuildNote({ children }) {
  return (
    <div
      style={{
        background: "var(--surface-alt)",
        border: "1px solid var(--border)",
        borderRadius: 8,
        padding: "0.85rem 1.1rem",
        margin: "1rem 0 1.5rem",
        fontSize: "0.9rem",
        color: "var(--text-muted)",
      }}
    >
      <strong style={{ color: "var(--text)" }}>How this was built (in plain terms): </strong>
      {children}
    </div>
  );
}
