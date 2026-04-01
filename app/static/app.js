const state = {
  token: localStorage.getItem("chromosome_portal_token") || "",
  user: null,
  datasets: [],
  jobs: [],
  optimizer: null,
  currentAnnotation: {
    datasetId: "",
    imageKey: "",
    imageUrl: "",
    polygons: [],
    draftPoints: [],
    selectedClassId: 0,
    canvasImage: null,
  },
};

const canvas = document.getElementById("annotation-canvas");
const ctx = canvas.getContext("2d");

async function api(path, options = {}) {
  const headers = new Headers(options.headers || {});
  if (state.token) headers.set("Authorization", `Bearer ${state.token}`);
  const response = await fetch(path, { ...options, headers });
  if (!response.ok) {
    let detail = response.statusText;
    try {
      const data = await response.json();
      detail = data.detail || JSON.stringify(data);
    } catch (error) {}
    throw new Error(detail);
  }
  const contentType = response.headers.get("content-type") || "";
  if (contentType.includes("application/json")) return response.json();
  return response;
}

function setStatus(id, message, isError = false) {
  const target = document.getElementById(id);
  if (!target) return;
  target.textContent = message || "";
  target.style.color = isError ? "#b91c1c" : "";
}

function activateTab(targetId) {
  document.querySelectorAll(".tab").forEach((button) => {
    button.classList.toggle("active", button.dataset.target === targetId);
  });
  document.querySelectorAll(".panel-section").forEach((section) => {
    section.classList.toggle("hidden", section.id !== targetId);
  });
}

function gateByRole() {
  const role = state.user?.role || "";
  document.querySelectorAll(".admin-only").forEach((element) => {
    element.classList.toggle("hidden", role !== "admin");
  });
  document.querySelectorAll(".annotator-only").forEach((element) => {
    element.classList.toggle("hidden", !["admin", "annotator"].includes(role));
  });
}

function renderOverview(data) {
  document.getElementById("overview-datasets").textContent = data.dataset_count;
  document.getElementById("overview-completed").textContent = data.completed_jobs;
  document.getElementById("overview-queued").textContent = data.queued_jobs;
  document.getElementById("overview-role").textContent = state.user?.role || "-";

  const list = document.getElementById("overview-dataset-list");
  list.innerHTML = state.datasets.map((dataset) => `
    <div class="card-item">
      <h4>${dataset.name}</h4>
      <div class="muted">${dataset.image_count} 張影像，${dataset.labeled_image_count} 張已標註，task: ${dataset.task}，revision: r${dataset.revision || 1}</div>
    </div>
  `).join("") || `<div class="list-item muted">目前尚無資料集。</div>`;

  const optimizerSummary = document.getElementById("optimizer-summary");
  if (!state.optimizer) {
    optimizerSummary.innerHTML = `<div class="list-item muted">Optimizer 尚未啟動。</div>`;
    return;
  }
  optimizerSummary.innerHTML = `
    <div class="list-item">
      <strong>${state.optimizer.enabled ? "執行中" : "已停止"}</strong>
      <div class="muted">dataset=${state.optimizer.dataset_id || "-"} / revision=r${state.optimizer.dataset_revision || 1} / ${state.optimizer.trials_started || 0} of ${state.optimizer.max_trials || 0}</div>
      <div class="muted">total trials=${state.optimizer.total_trials_started || 0}</div>
      <div class="muted">${state.optimizer.last_suggestion || ""}</div>
    </div>
  `;
}

function renderRecommendations(items = []) {
  if (!items.length) return `<div class="muted">尚無建議。</div>`;
  return `<div class="recommendations">${items.map((item) => `
    <div class="recommendation ${item.severity}">
      <strong>${item.title}</strong><br>${item.detail}
    </div>
  `).join("")}</div>`;
}

function renderDatasets() {
  const container = document.getElementById("dataset-list");
  container.innerHTML = state.datasets.map((dataset) => `
    <article class="card-item">
      <h4>${dataset.name}</h4>
      <div class="muted">來源：${dataset.source_filename}</div>
      <div class="muted">revision: r${dataset.revision || 1} / updated: ${dataset.updated_at || dataset.created_at}</div>
      <div class="muted">影像：${dataset.image_count} / 已標註：${dataset.labeled_image_count} / 未標註：${dataset.unlabeled_image_count}</div>
      <div class="muted">類別：${(dataset.classes || []).join(", ")}</div>
      ${renderRecommendations(dataset.recommendations || [])}
    </article>
  `).join("") || `<div class="list-item muted">尚無資料集。</div>`;

  const selects = [
    "annotation-dataset-select",
    "training-dataset-select",
    "optimizer-dataset-select",
    "compare-dataset-select",
  ];
  for (const selectId of selects) {
    const select = document.getElementById(selectId);
    if (!select) continue;
    const currentValue = select.value;
    select.innerHTML = state.datasets.map((dataset) => `
      <option value="${dataset.id}">${dataset.name}</option>
    `).join("");
    if (currentValue && state.datasets.some((dataset) => dataset.id === currentValue)) {
      select.value = currentValue;
    }
  }
}

async function loadOverview() {
  const overview = await api("/api/overview");
  renderOverview(overview);
}

async function loadDatasets() {
  state.datasets = await api("/api/datasets");
  renderDatasets();
}

async function loadJobs() {
  state.jobs = await api("/api/training/jobs");
  const container = document.getElementById("job-list");
  container.innerHTML = state.jobs.map((job) => `
    <div class="list-item">
        <div>
          <h4>${job.model_label}</h4>
        <div class="muted">dataset=${job.dataset_id} / revision=r${job.dataset_revision || 1} / status=${job.status} / epochs=${job.epochs} / imgsz=${job.imgsz}</div>
        <div class="muted">${job.metric_name || "-"} ${job.metric_value ?? ""}</div>
      </div>
      <button class="btn-secondary" data-log-job="${job.id}">查看 log</button>
    </div>
  `).join("") || `<div class="list-item muted">尚無工作。</div>`;
  container.querySelectorAll("[data-log-job]").forEach((button) => {
    button.addEventListener("click", async () => {
      const data = await api(`/api/training/jobs/${button.dataset.logJob}/log`);
      alert(data.log || "目前沒有 log");
    });
  });
}

async function loadOptimizer() {
  state.optimizer = await api("/api/optimizer/status");
  renderOverview({
    dataset_count: state.datasets.length,
    completed_jobs: state.jobs.filter((job) => job.status === "completed").length,
    queued_jobs: state.jobs.filter((job) => ["queued", "running"].includes(job.status)).length,
  });
}

async function renderCompare() {
  const datasetId = document.getElementById("compare-dataset-select").value;
  if (!datasetId) {
    document.getElementById("compare-summary").innerHTML = `<div class="list-item muted">尚無資料集。</div>`;
    document.getElementById("compare-table").innerHTML = "";
    document.getElementById("compare-preview").innerHTML = "";
    return;
  }
  const summary = await api(`/api/models/compare?dataset_id=${encodeURIComponent(datasetId)}`);
  document.getElementById("compare-summary").innerHTML = renderRecommendations(summary.recommendations || []);
  const table = document.getElementById("compare-table");
  table.innerHTML = summary.completed_jobs.map((job) => `
    <div class="list-item">
      <div>
        <h4>${job.model_label}</h4>
        <div class="muted">revision=r${job.dataset_revision || 1} / ${job.metric_name || "-"} = ${job.metric_value ?? "-"}</div>
      </div>
      <button class="btn-secondary" data-preview-job="${job.id}">預覽</button>
    </div>
  `).join("") || `<div class="list-item muted">還沒有完成的比較結果。</div>`;
  table.querySelectorAll("[data-preview-job]").forEach((button) => {
    button.addEventListener("click", async () => {
      const job = await api(`/api/training/jobs/${button.dataset.previewJob}`);
      const preview = document.getElementById("compare-preview");
      preview.innerHTML = (job.preview_files || []).map((filename) => `
        <img src="/api/training/jobs/${job.id}/preview/${encodeURIComponent(filename)}?token=${encodeURIComponent(state.token)}" alt="${filename}">
      `).join("") || `<div class="list-item muted">這個工作還沒有 preview。</div>`;
    });
  });
}

async function loadUsers() {
  if (state.user?.role !== "admin") return;
  const users = await api("/api/users");
  document.getElementById("user-list").innerHTML = users.map((user) => `
    <div class="list-item">
      <h4>${user.display_name}</h4>
      <div class="muted">${user.username} / ${user.role}</div>
    </div>
  `).join("");
}

function canvasToNorm(event) {
  const rect = canvas.getBoundingClientRect();
  return {
    x: (event.clientX - rect.left) / rect.width,
    y: (event.clientY - rect.top) / rect.height,
  };
}

function renderCanvas() {
  const { canvasImage, polygons, draftPoints } = state.currentAnnotation;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  if (canvasImage) {
    ctx.drawImage(canvasImage, 0, 0, canvas.width, canvas.height);
  }

  const drawPolygon = (points, strokeStyle, fillStyle) => {
    if (!points.length) return;
    ctx.beginPath();
    ctx.moveTo(points[0].x * canvas.width, points[0].y * canvas.height);
    points.slice(1).forEach((point) => ctx.lineTo(point.x * canvas.width, point.y * canvas.height));
    if (points.length >= 3) ctx.closePath();
    ctx.strokeStyle = strokeStyle;
    ctx.lineWidth = 2;
    ctx.stroke();
    if (points.length >= 3) {
      ctx.fillStyle = fillStyle;
      ctx.fill();
    }
  };

  polygons.forEach((polygon) => drawPolygon(polygon.points, "#0f766e", "rgba(15, 118, 110, 0.12)"));
  drawPolygon(draftPoints, "#1d4ed8", "rgba(29, 78, 216, 0.08)");

  const list = document.getElementById("annotation-polygon-list");
  list.innerHTML = polygons.map((polygon, index) => `
    <div class="list-item">#${index + 1} class=${polygon.class_id} / points=${polygon.points.length}</div>
  `).join("") + (draftPoints.length ? `<div class="list-item">草稿點位：${draftPoints.length}</div>` : "");
}

async function loadAnnotationImages() {
  const datasetId = document.getElementById("annotation-dataset-select").value;
  if (!datasetId) return;
  state.currentAnnotation.datasetId = datasetId;
  const filter = document.getElementById("annotation-filter").value;
  const images = await api(`/api/datasets/${datasetId}/images?labeled=${encodeURIComponent(filter)}`);
  const container = document.getElementById("annotation-image-list");
  container.innerHTML = images.map((image) => `
    <div class="thumb-item ${state.currentAnnotation.imageKey === image.key ? "active" : ""}" data-image-key="${image.key}">
      <img src="/api/datasets/${datasetId}/images/${encodeURIComponent(image.key)}/file?token=${encodeURIComponent(state.token)}" alt="${image.filename}">
      <div>
        <strong>${image.filename}</strong>
        <div class="muted">${image.split} / ${image.labeled ? "已標註" : "未標註"}</div>
      </div>
    </div>
  `).join("") || `<div class="list-item muted">沒有符合條件的影像。</div>`;

  container.querySelectorAll("[data-image-key]").forEach((element) => {
    element.addEventListener("click", () => openAnnotationImage(datasetId, element.dataset.imageKey));
  });
}

async function openAnnotationImage(datasetId, imageKey) {
  state.currentAnnotation.datasetId = datasetId;
  state.currentAnnotation.imageKey = imageKey;
  const data = await api(`/api/datasets/${datasetId}/images/${encodeURIComponent(imageKey)}/annotation`);
  state.currentAnnotation.polygons = data.polygons || [];
  state.currentAnnotation.draftPoints = [];
  state.currentAnnotation.selectedClassId = 0;
  const classSelect = document.getElementById("annotation-class-select");
  classSelect.innerHTML = (data.classes || []).map((item, index) => `<option value="${index}">${index}: ${item}</option>`).join("");
  classSelect.value = "0";

  const imageResponse = await fetch(`/api/datasets/${datasetId}/images/${encodeURIComponent(imageKey)}/file?token=${encodeURIComponent(state.token)}`);
  const blob = await imageResponse.blob();
  const bitmap = await createImageBitmap(blob);
  state.currentAnnotation.canvasImage = bitmap;
  const maxWidth = 980;
  const maxHeight = 620;
  const scale = Math.min(maxWidth / bitmap.width, maxHeight / bitmap.height, 1);
  canvas.width = bitmap.width * scale;
  canvas.height = bitmap.height * scale;
  document.getElementById("annotation-title").textContent = data.filename;
  renderCanvas();
  loadAnnotationImages();
}

async function saveAnnotation() {
  if (!state.currentAnnotation.datasetId || !state.currentAnnotation.imageKey) return;
  const payload = { polygons: state.currentAnnotation.polygons };
  const data = await api(
    `/api/datasets/${state.currentAnnotation.datasetId}/images/${encodeURIComponent(state.currentAnnotation.imageKey)}/annotation`,
    {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    }
  );
  state.currentAnnotation.polygons = data.polygons || [];
  state.currentAnnotation.draftPoints = [];
  setStatus("annotation-status", "標註已保存");
  await loadDatasets();
  await loadAnnotationImages();
  renderCanvas();
}

canvas.addEventListener("click", (event) => {
  if (!state.currentAnnotation.canvasImage) return;
  state.currentAnnotation.draftPoints.push(canvasToNorm(event));
  renderCanvas();
});

document.getElementById("annotation-complete-btn").addEventListener("click", () => {
  if (state.currentAnnotation.draftPoints.length < 3) return;
  state.currentAnnotation.polygons.push({
    class_id: Number(document.getElementById("annotation-class-select").value),
    points: [...state.currentAnnotation.draftPoints],
  });
  state.currentAnnotation.draftPoints = [];
  renderCanvas();
});

document.getElementById("annotation-undo-btn").addEventListener("click", () => {
  state.currentAnnotation.draftPoints.pop();
  renderCanvas();
});

document.getElementById("annotation-remove-btn").addEventListener("click", () => {
  if (state.currentAnnotation.draftPoints.length) {
    state.currentAnnotation.draftPoints = [];
  } else {
    state.currentAnnotation.polygons.pop();
  }
  renderCanvas();
});

document.getElementById("annotation-save-btn").addEventListener("click", saveAnnotation);

async function bootstrap() {
  document.getElementById("login-view").classList.add("hidden");
  document.getElementById("app-view").classList.remove("hidden");
  document.getElementById("user-pill").textContent = `${state.user.display_name} (${state.user.role})`;
  gateByRole();
  await loadDatasets();
  await loadJobs();
  await loadOptimizer();
  await loadOverview();
  if (["admin", "annotator"].includes(state.user.role) && state.datasets.length) {
    await loadAnnotationImages();
  }
  if (state.user.role === "admin") {
    await loadUsers();
  }
  if (state.datasets.length) {
    await renderCompare();
  }
}

document.querySelectorAll(".tab").forEach((button) => {
  button.addEventListener("click", () => activateTab(button.dataset.target));
});

document.getElementById("login-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  try {
    const data = await api("/api/auth/login", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        username: document.getElementById("login-username").value,
        password: document.getElementById("login-password").value,
      }),
    });
    state.token = data.token;
    state.user = data.user;
    localStorage.setItem("chromosome_portal_token", state.token);
    setStatus("login-status", "");
    await bootstrap();
  } catch (error) {
    setStatus("login-status", error.message, true);
  }
});

document.getElementById("logout-btn").addEventListener("click", async () => {
  try {
    await api("/api/auth/logout", { method: "POST" });
  } catch (error) {}
  localStorage.removeItem("chromosome_portal_token");
  location.reload();
});

document.getElementById("refresh-overview-btn").addEventListener("click", async () => {
  await loadDatasets();
  await loadJobs();
  await loadOptimizer();
  await loadOverview();
});

document.getElementById("refresh-datasets-btn").addEventListener("click", loadDatasets);
document.getElementById("refresh-jobs-btn").addEventListener("click", loadJobs);
document.getElementById("refresh-compare-btn").addEventListener("click", renderCompare);
document.getElementById("refresh-users-btn").addEventListener("click", loadUsers);
document.getElementById("optimizer-refresh-btn").addEventListener("click", loadOptimizer);
document.getElementById("optimizer-stop-btn").addEventListener("click", async () => {
  await api("/api/optimizer/stop", { method: "POST" });
  await loadOptimizer();
});

document.getElementById("dataset-upload-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  try {
    const file = document.getElementById("dataset-file").files[0];
    if (!file) return;
    const formData = new FormData();
    formData.append("name", document.getElementById("dataset-name").value || file.name.replace(/\.zip$/i, ""));
    formData.append("file", file);
    const response = await fetch("/api/datasets/upload", {
      method: "POST",
      headers: { Authorization: `Bearer ${state.token}` },
      body: formData,
    });
    if (!response.ok) {
      const data = await response.json();
      throw new Error(data.detail || "Upload failed");
    }
    setStatus("dataset-upload-status", "資料集已上傳");
    await loadDatasets();
    await loadOverview();
  } catch (error) {
    setStatus("dataset-upload-status", error.message, true);
  }
});

document.getElementById("annotation-dataset-select").addEventListener("change", loadAnnotationImages);
document.getElementById("annotation-filter").addEventListener("change", loadAnnotationImages);

document.getElementById("training-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const modelLabels = [...document.querySelectorAll('#training-form input[type="checkbox"]:checked')].map((item) => item.value);
  try {
    await api("/api/training/jobs", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        dataset_id: document.getElementById("training-dataset-select").value,
        model_labels: modelLabels,
        epochs: Number(document.getElementById("training-epochs").value),
        imgsz: Number(document.getElementById("training-imgsz").value),
        batch: Number(document.getElementById("training-batch").value),
        device: document.getElementById("training-device").value,
      }),
    });
    setStatus("training-status", "比較工作已加入佇列");
    await loadJobs();
  } catch (error) {
    setStatus("training-status", error.message, true);
  }
});

document.getElementById("optimizer-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  try {
    await api("/api/optimizer/start", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        dataset_id: document.getElementById("optimizer-dataset-select").value,
        max_trials: Number(document.getElementById("optimizer-trials").value),
        epochs: Number(document.getElementById("optimizer-epochs").value),
        imgsz_options: document.getElementById("optimizer-imgsz").value.split(",").map((item) => Number(item.trim())).filter(Boolean),
        batch: Number(document.getElementById("optimizer-batch").value),
        device: document.getElementById("training-device").value,
      }),
    });
    setStatus("optimizer-status-line", "Optimizer 已啟動");
    await loadOptimizer();
    await loadJobs();
  } catch (error) {
    setStatus("optimizer-status-line", error.message, true);
  }
});

document.getElementById("compare-dataset-select").addEventListener("change", renderCompare);

document.getElementById("user-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  try {
    await api("/api/users", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        username: document.getElementById("user-username").value,
        display_name: document.getElementById("user-display-name").value,
        role: document.getElementById("user-role").value,
        password: document.getElementById("user-password").value,
      }),
    });
    setStatus("user-status", "使用者已建立");
    await loadUsers();
  } catch (error) {
    setStatus("user-status", error.message, true);
  }
});

async function tryResume() {
  if (!state.token) return;
  try {
    state.user = await api("/api/auth/me");
    await bootstrap();
  } catch (error) {
    localStorage.removeItem("chromosome_portal_token");
    state.token = "";
  }
}

setInterval(async () => {
  if (!state.token) return;
  try {
    await loadJobs();
    await loadOptimizer();
  } catch (error) {}
}, 5000);

tryResume();
