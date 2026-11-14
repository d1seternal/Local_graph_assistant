"use strict";

const $ = id => document.getElementById(id);
const states = {uploaded: "Загружен", processing: "Индексируется", ready: "Готов", failed: "Не проиндексирован"};
const pageSize = 50;
let token = null;
let busy = false;
let offset = 0;
let hasNext = false;
let documents = [];
let activeCollection = "";

class ApiError extends Error {
  constructor(status, detail) {
    super(detail);
    this.status = status;
  }
}

async function api(path, {method = "GET", json, body} = {}) {
  const headers = new Headers();
  if (token) headers.set("Authorization", `Bearer ${token}`);
  if (json !== undefined) {
    headers.set("Content-Type", "application/json");
    body = JSON.stringify(json);
  }
  let response;
  try {
    response = await fetch(path, {method, headers, body, cache: "no-store"});
  } catch {
    throw new Error("Нет связи с приложением. Проверьте, что API запущен.");
  }
  const data = await response.json().catch(() => null);
  if (!response.ok) {
    const detail = typeof data?.detail === "string" ? data.detail :
      response.status === 422 ? "Проверьте поля запроса: ввод не прошёл проверку." : "Не удалось выполнить запрос.";
    throw new ApiError(response.status, detail);
  }
  return data;
}

function status(message = "", kind = "info") {
  $("status").textContent = message;
  $("status").dataset.kind = kind;
  $("status").hidden = !message;
}

function showError(error) {
  const hadSession = Boolean(token);
  if (error.status === 401 && hadSession) resetSession();
  status(error.status === 401 && hadSession
    ? "Сеанс завершён. Войдите снова."
    : `${error.status ? `HTTP ${error.status}: ` : ""}${error.message}`, "error");
}

function syncControls() {
  for (const form of document.querySelectorAll("form")) {
    for (const field of form.elements) field.disabled = busy;
  }
  $("logout").disabled = busy;
  $("collections").disabled = busy || !$("collections").options.length;
  $("previous").disabled = busy || offset === 0;
  $("next").disabled = busy || !hasNext;
  for (const button of $("documents").querySelectorAll("button")) button.disabled = busy;
}

async function run(action) {
  if (busy) return;
  busy = true;
  status();
  syncControls();
  try {
    await action();
  } catch (error) {
    showError(error);
  } finally {
    busy = false;
    syncControls();
  }
}

function resetSession() {
  token = null;
  activeCollection = "";
  offset = 0;
  hasNext = false;
  $("identity").textContent = "";
  $("collections").replaceChildren();
  $("workspace").hidden = true;
  $("sign-in").hidden = false;
  $("upload").reset();
  $("ask").reset();
  clearResult();
  documents = [];
  renderDocuments();
}

function clearResult() {
  $("result").hidden = true;
  $("answer").textContent = "";
  $("usage").textContent = "";
  $("sources").replaceChildren();
}

function collectionUrl() {
  return `/collections/${encodeURIComponent(activeCollection)}`;
}

function renderDocuments() {
  $("documents").replaceChildren();
  $("empty-docs").hidden = documents.length > 0;
  $("pagination").hidden = !hasNext && offset === 0;
  for (const item of documents) {
    const entry = document.createElement("li");
    const name = document.createElement("strong");
    name.textContent = item.original_name;
    const label = document.createElement("div");
    label.className = "muted";
    label.textContent = states[item.status] || item.status;
    const button = document.createElement("button");
    button.type = "button";
    button.className = "secondary";
    button.textContent = item.status === "ready" ? "Переиндексировать" : "Индексировать";
    button.addEventListener("click", () => run(() => index(item.id)));
    entry.append(name, label, button);
    $("documents").append(entry);
  }
  syncControls();
}

async function loadDocuments(newOffset = 0) {
  const rows = await api(`${collectionUrl()}/documents?limit=${pageSize + 1}&offset=${newOffset}`);
  offset = newOffset;
  hasNext = rows.length > pageSize;
  documents = rows.slice(0, pageSize);
  renderDocuments();
}

async function index(id) {
  clearResult();
  status("Индексация: извлечение текста и расчёт эмбеддингов…");
  try {
    await api(`${collectionUrl()}/documents/${encodeURIComponent(id)}/index`, {method: "POST"});
  } catch (error) {
    if (error.status !== 401) {
      await loadDocuments(offset).catch(() => {});
    }
    throw error;
  }
  await loadDocuments(offset);
  status("Документ проиндексирован. Можно задать вопрос.");
}

function locationText(source) {
  const s = source.source || {};
  if (s.type === "txt") return s.line_start === s.line_end ? `строка ${s.line_start}` : `строки ${s.line_start}–${s.line_end}`;
  if (s.type === "pdf") return `страница ${s.page}`;
  if (s.type === "docx_paragraph") return `абзац ${s.paragraph}`;
  if (s.type === "docx_cell") return `таблица ${s.table}, строка ${s.row}, столбец ${s.column}`;
  return "фрагмент";
}

$("login").addEventListener("submit", event => {
  event.preventDefault();
  run(async () => {
    let session;
    try {
      session = await api("/auth/login", {method: "POST", json: {
        username: $("username").value.trim(), password: $("password").value,
      }});
    } finally {
      $("password").value = "";
    }
    token = session.access_token;
    const [user, collections] = await Promise.all([api("/auth/me"), api("/collections")]);
    $("identity").textContent = user.username;
    $("collections").replaceChildren(...collections.map(item => {
      const option = document.createElement("option");
      option.value = item.id;
      option.textContent = item.name;
      return option;
    }));
    activeCollection = $("collections").value;
    $("workspace").hidden = false;
    $("sign-in").hidden = true;
    $("collection-workspace").hidden = !activeCollection;
    if (activeCollection) await loadDocuments();
    else status("У вашей учётной записи пока нет доступных коллекций.");
  });
});

$("logout").addEventListener("click", () => run(async () => {
  try {
    await api("/auth/logout", {method: "POST"});
  } finally {
    resetSession();
  }
  status("Вы вышли из приложения.");
}));

$("collections").addEventListener("change", () => run(async () => {
  activeCollection = $("collections").value;
  documents = [];
  offset = 0;
  hasNext = false;
  renderDocuments();
  $("upload").reset();
  $("ask").reset();
  clearResult();
  await loadDocuments();
}));
$("previous").addEventListener("click", () => run(() => loadDocuments(offset - pageSize)));
$("next").addEventListener("click", () => run(() => loadDocuments(offset + pageSize)));

$("upload").addEventListener("submit", event => {
  event.preventDefault();
  run(async () => {
    const file = $("file").files[0];
    if (!file) return;
    clearResult();
    status("Загрузка документа…");
    const body = new FormData();
    body.append("file", file);
    const uploaded = await api(`${collectionUrl()}/documents`, {method: "POST", body});
    $("upload").reset();
    await loadDocuments();
    if (uploaded.status === "ready") status("Этот документ уже загружен и проиндексирован.");
    else await index(uploaded.id);
  });
});

$("ask").addEventListener("submit", event => {
  event.preventDefault();
  run(async () => {
    const question = $("question").value.trim();
    if (!question) throw new Error("Введите вопрос.");
    clearResult();
    status("Ищу фрагменты и готовлю ответ…");
    const result = await api(`${collectionUrl()}/ask`, {method: "POST", json: {question, top_k: 3}});
    $("answer").textContent = result.answer;
    for (const source of result.sources) {
      const entry = document.createElement("li");
      const title = document.createElement("strong");
      title.textContent = `[${source.citation_id}] ${source.document_name} — ${locationText(source)}`;
      const excerpt = document.createElement("div");
      excerpt.className = "preserve";
      excerpt.textContent = source.text;
      entry.append(title, excerpt);
      $("sources").append(entry);
    }
    if (!result.sources.length) $("sources").append(Object.assign(document.createElement("li"), {textContent: "Источников нет."}));
    $("usage").textContent = result.usage?.total_tokens == null ? "" : `Токенов всего: ${result.usage.total_tokens}`;
    $("result").hidden = false;
    status();
  });
});
