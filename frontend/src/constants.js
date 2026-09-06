// Real applicant IDs from the dataset, for one-click demo without typing an ID.
export const EXAMPLE_APPLICANT_IDS = [
  100002, 100003, 451071, 429655, 166551, 130914, 150369, 437201, 298124, 290903,
];

// The 5 documented, verified-working query patterns for the chatbot.
export const EXAMPLE_QUESTIONS = [
  "What is the average annual income of all applicants?",
  "What is the default rate by education level?",
  "Show the 10 applicants with the highest income who had payment difficulty.",
  "What is the average loan-to-income ratio for applicants who defaulted versus those who didn't?",
  "How many applicants have at least one bad debt record with the credit bureau?",
];

export const RISK_BAND_COLORS = {
  Low: { text: "var(--low)", bg: "var(--low-bg)" },
  Medium: { text: "var(--medium)", bg: "var(--medium-bg)" },
  High: { text: "var(--high)", bg: "var(--high-bg)" },
};
