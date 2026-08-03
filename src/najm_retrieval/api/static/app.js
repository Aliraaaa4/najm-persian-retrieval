"use strict";

const form = document.querySelector("#search-form");
const queryInput = document.querySelector("#query");
const limitInput = document.querySelector("#limit");
const submitButton = document.querySelector("#submit-button");
const buttonLabel = submitButton.querySelector(".button-label");
const buttonProgress = submitButton.querySelector(".button-progress");
const statusPanel = document.querySelector("#status-panel");
const suggestionsSection = document.querySelector("#suggestions-section");
const suggestionsContainer = document.querySelector("#suggestions");
const resultsSection = document.querySelector("#results-section");
const resultsContainer = document.querySelector("#results");
const resultCount = document.querySelector("#result-count");

function setLoading(isLoading) {
  submitButton.disabled = isLoading;
  queryInput.disabled = isLoading;
  limitInput.disabled = isLoading;
  buttonLabel.hidden = isLoading;
  buttonProgress.hidden = !isLoading;
}

function clearOutput() {
  statusPanel.hidden = true;
  statusPanel.className = "status-panel";
  statusPanel.replaceChildren();
  suggestionsSection.hidden = true;
  suggestionsContainer.replaceChildren();
  resultsSection.hidden = true;
  resultsContainer.replaceChildren();
  resultCount.textContent = "";
}

function showStatus(title, detail = "", kind = "success") {
  statusPanel.replaceChildren();

  const titleElement = document.createElement("p");
  titleElement.className = "status-title";
  titleElement.textContent = title;
  statusPanel.append(titleElement);

  if (detail) {
    const detailElement = document.createElement("p");
    detailElement.className = "status-detail";
    detailElement.textContent = detail;
    statusPanel.append(detailElement);
  }

  statusPanel.className = "status-panel";

  if (kind === "abstain") {
    statusPanel.classList.add("status-abstain");
  }

  if (kind === "error") {
    statusPanel.classList.add("status-error");
  }

  statusPanel.hidden = false;
}

function addReferenceItem(grid, label, value, useCode = false) {
  const labelElement = document.createElement("span");
  labelElement.textContent = label;

  const valueElement = useCode
    ? document.createElement("code")
    : document.createElement("span");

  valueElement.textContent = value ?? "—";
  grid.append(labelElement, valueElement);
}

function renderResult(result) {
  const article = document.createElement("article");
  article.className = "result-card";

  const header = document.createElement("div");
  header.className = "result-header";

  const rank = document.createElement("span");
  rank.className = "result-rank";
  rank.textContent = String(result.rank);

  const titleGroup = document.createElement("div");
  titleGroup.className = "result-title";

  const title = document.createElement("h3");
  title.textContent = result.reference.work_title;

  const author = document.createElement("p");
  author.className = "result-author";
  author.textContent = `نویسنده: ${result.reference.author_name}`;

  titleGroup.append(title, author);
  header.append(rank, titleGroup);

  const snippet = document.createElement("p");
  snippet.className = "result-snippet";
  snippet.textContent = result.snippet;

  const details = document.createElement("details");
  details.className = "result-details";

  const summary = document.createElement("summary");
  summary.textContent = "مشاهده منبع";

  const fullText = document.createElement("p");
  fullText.className = "result-text";
  fullText.textContent = result.display_text;

  const referenceGrid = document.createElement("div");
  referenceGrid.className = "reference-grid";

  addReferenceItem(referenceGrid, "شناسه قطعه", result.reference.passage_id, true);
  addReferenceItem(referenceGrid, "شناسه نسخه", result.reference.version_id, true);
  addReferenceItem(referenceGrid, "نوع متن", result.reference.kind);
  addReferenceItem(referenceGrid, "رتبه واژگانی", result.scores.lexical_rank);
  addReferenceItem(referenceGrid, "رتبه معنایی", result.scores.dense_rank);
  addReferenceItem(
    referenceGrid,
    "امتیاز تلفیقی",
    Number(result.scores.fusion_score).toFixed(6),
  );

  details.append(summary, fullText, referenceGrid);
  article.append(header, snippet, details);

  return article;
}

function renderSuggestions(suggestions) {
  suggestionsContainer.replaceChildren();

  if (!Array.isArray(suggestions) || suggestions.length === 0) {
    suggestionsSection.hidden = true;
    return;
  }

  for (const suggestion of suggestions) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "suggestion-button";
    button.textContent = suggestion.label;
    button.dataset.query = suggestion.query;

    button.addEventListener("click", () => {
      queryInput.value = button.dataset.query;
      queryInput.focus();
      form.requestSubmit();
    });

    suggestionsContainer.append(button);
  }

  suggestionsSection.hidden = false;
}

function renderResponse(payload) {
  if (payload.return_results) {
    showStatus(payload.message);

    const results = Array.isArray(payload.results) ? payload.results : [];

    for (const result of results) {
      resultsContainer.append(renderResult(result));
    }

    resultCount.textContent = `${results.length} نتیجه`;
    resultsSection.hidden = results.length === 0;
    renderSuggestions([]);
    return;
  }

  showStatus(
    "نتیجه قابل اعتمادی نمایش داده نشد",
    payload.message,
    "abstain",
  );

  resultsSection.hidden = true;
  renderSuggestions(payload.suggestions);
}

async function submitQuery(query, limit) {
  const response = await fetch("/v1/retrieve", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ query, limit }),
  });

  let payload = null;

  try {
    payload = await response.json();
  } catch {
    throw new Error("پاسخ سامانه قابل خواندن نبود.");
  }

  if (!response.ok) {
    const message =
      payload?.detail?.message ??
      "در اجرای درخواست خطایی رخ داد.";

    throw new Error(message);
  }

  return payload;
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();

  const query = queryInput.value.trim();
  const limit = Number(limitInput.value);

  clearOutput();

  if (!query) {
    showStatus(
      "پرسش خالی است",
      "لطفاً یک عبارت یا سؤال وارد کنید.",
      "error",
    );
    queryInput.focus();
    return;
  }

  setLoading(true);

  try {
    const payload = await submitQuery(query, limit);
    renderResponse(payload);
  } catch (error) {
    const message =
      error instanceof Error
        ? error.message
        : "خطای ناشناخته در ارتباط با سامانه.";

    showStatus("درخواست تکمیل نشد", message, "error");
  } finally {
    setLoading(false);
  }
});
