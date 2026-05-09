const form = document.querySelector("#uploadForm");
const videoInput = document.querySelector("#videoInput");
const fileLabel = document.querySelector("#fileLabel");
const statusTitle = document.querySelector("#statusTitle");
const statusBadge = document.querySelector("#statusBadge");
const summary = document.querySelector("#summary");
const media = document.querySelector("#media");
const downloads = document.querySelector("#downloads");

let activeJob = null;
let timer = null;

videoInput.addEventListener("change", () => {
  fileLabel.textContent = videoInput.files[0]?.name || "Choose video";
});

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const data = new FormData(form);
  setStatus("Uploading", "running");
  summary.innerHTML = "";
  media.innerHTML = "";
  downloads.innerHTML = "";

  const response = await fetch("/api/jobs", { method: "POST", body: data });
  const payload = await response.json();
  if (!response.ok) {
    setStatus(payload.detail || "Upload failed", "failed");
    return;
  }

  activeJob = payload.job_id;
  pollJob();
  timer = setInterval(pollJob, 2500);
});

async function pollJob() {
  if (!activeJob) return;
  const response = await fetch(`/api/jobs/${activeJob}`);
  const job = await response.json();
  if (!response.ok) {
    setStatus(job.detail || "Job missing", "failed");
    clearInterval(timer);
    return;
  }

  setStatus(job.progress || job.status, job.status);
  if (job.status === "done") {
    clearInterval(timer);
    renderResult(job);
  }
  if (job.status === "failed") {
    clearInterval(timer);
    setStatus(job.error || "Processing failed", "failed");
  }
}

function setStatus(text, state) {
  statusTitle.textContent = text;
  statusBadge.textContent = state;
  statusBadge.dataset.state = state;
}

function renderResult(job) {
  const s = job.summary;
  summary.innerHTML = [
    ["Tracks", s.tracked_people],
    ["Frames", s.processed_frames],
    ["Duration", `${s.duration_seconds.toFixed(1)}s`],
    ["Processing", `${s.processing_seconds}s`],
    ["Model", s.model],
  ]
    .map(([label, value]) => `<div><span>${label}</span><strong>${value}</strong></div>`)
    .join("");

  media.innerHTML = `
    <video controls src="${job.links.video}"></video>
    <img src="${job.links.trajectory_map}" alt="Trajectory map" />
  `;

  downloads.innerHTML = `
    <a href="${job.links.video}">MP4</a>
    <a href="${job.links.csv}">CSV</a>
    <a href="${job.links.json}">JSON</a>
    <a href="${job.links.summary}">Summary</a>
  `;
}

