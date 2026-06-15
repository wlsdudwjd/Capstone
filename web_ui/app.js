const isLogsPage = window.location.pathname === "/logs";
const monitorPage = document.getElementById("monitorPage");
const logsPage = document.getElementById("logsPage");
const monitorLink = document.getElementById("monitorLink");
const logsLink = document.getElementById("logsLink");
const cameraGrid = document.getElementById("cameraGrid");
const cameraTemplate = document.getElementById("cameraTemplate");
const overallResultPanel = document.getElementById("overallResultPanel");
const overallResultValue = document.getElementById("overallResultValue");
const dateTabs = document.getElementById("dateTabs");
const selectedLogFolder = document.getElementById("selectedLogFolder");
const csvDownload = document.getElementById("csvDownload");
const logCount = document.getElementById("logCount");
const logRows = document.getElementById("logRows");
const cameraCards = new Map();
const cameraLabels = new Map([
  [0, "전면 카메라"],
  [1, "후면 카메라"],
]);

monitorPage.hidden = isLogsPage;
logsPage.hidden = !isLogsPage;
monitorLink.classList.toggle("active", !isLogsPage);
logsLink.classList.toggle("active", isLogsPage);

function cameraLabel(cameraIndex) {
  return cameraLabels.get(Number(cameraIndex)) || `CAM ${cameraIndex}`;
}

function formatPercent(value) {
  const number = Number(value);
  if (!Number.isFinite(number)) return "-";
  return `${Math.round(number * 100)}%`;
}

function cleanLabel(value) {
  if (!value) return "-";
  return String(value).replaceAll("_", " ");
}

function koreanFinal(value) {
  return koreanFinalResult(value);
}

function koreanFinalResult(value) {
  const normalized = String(value || "").toLowerCase();
  if (normalized === "violation") return "위규";
  if (normalized === "normal") return "정상";
  if (normalized === "review") return "검토";
  if (normalized === "error") return "오류";
  if (normalized === "no_detection") return "미탐지";
  if (normalized === "waiting") return "대기 중";
  return cleanLabel(value);
}

function formatDetection(row, side) {
  const phones = row[`${side}_phones`] || "0";
  const lenses = row[`${side}_lenses`] || "0";
  const stickers = row[`${side}_stickers`] || "0";
  return `P ${phones} / L ${lenses} / S ${stickers}`;
}

function formatErrorDetail(error, cameraIndex) {
  if (!error) return "";
  const firstLine = String(error).split("\n")[0] || "";
  const cameraPrefix = new RegExp(`^Camera\\s+${cameraIndex}:\\s*`, "i");
  const detail = firstLine
    .replace(cameraPrefix, "")
    .replace(/\.$/, "")
    .trim();
  return detail ? `:${detail}` : "";
}

function createCameraCard(cameraIndex) {
  const fragment = cameraTemplate.content.cloneNode(true);
  const card = fragment.querySelector(".camera-column");
  card.dataset.cameraIndex = cameraIndex;
  const label = cameraLabel(cameraIndex);

  card.querySelector(".camera-title").textContent = label;
  card.querySelector(".final-pill").textContent = "대기 중";
  card.querySelector(".final-pill").classList.add("waiting");

  const fullStream = card.querySelector(".full-stream");
  const cropStream = card.querySelector(".crop-stream");
  const cropErrorTitle = card.querySelector(".crop-error-title");
  if (cropErrorTitle) cropErrorTitle.textContent = `${label} Crop Error`;
  card.querySelector(".full-error-title").textContent = `${label} Full Error`;
  fullStream.src = `/stream/${cameraIndex}/full`;
  if (cropStream) cropStream.src = `/stream/${cameraIndex}/crop`;
  fullStream.alt = `${label} full webcam stream`;
  if (cropStream) cropStream.alt = `${label} crop stream`;

  cameraGrid.appendChild(fragment);
  cameraCards.set(cameraIndex, card);
  return card;
}

function ensureCameraCards(cameraIndexes) {
  cameraIndexes.forEach((cameraIndex) => {
    if (!cameraCards.has(cameraIndex)) {
      createCameraCard(cameraIndex);
    }
  });
}

function setFinalPill(pill, finalValue) {
  const normalized = String(finalValue || "unknown").toLowerCase();
  pill.classList.remove("normal", "violation", "waiting", "unknown", "error", "review", "no_detection");
  if (["normal", "violation", "waiting", "error", "review", "no_detection"].includes(normalized)) {
    pill.classList.add(normalized);
  } else {
    pill.classList.add("unknown");
  }
  pill.textContent = koreanFinalResult(finalValue);
}

function setResultPanel(panel, valueElement, finalValue) {
  const normalizedFinal = String(finalValue || "unknown").toLowerCase();
  panel.classList.remove("normal", "violation", "review", "error", "waiting", "unknown", "no_detection");
  panel.classList.add(["normal", "violation", "review", "error", "waiting", "no_detection"].includes(normalizedFinal) ? normalizedFinal : "unknown");
  valueElement.textContent = koreanFinalResult(finalValue);
}

function updateCameraCard(camera, forceNoDetection = false) {
  const cameraIndex = camera.camera_index;
  const card = cameraCards.get(cameraIndex) || createCameraCard(cameraIndex);
  const hasCameraError = Boolean(camera.error);

  const displayFinal = hasCameraError
    ? "error"
    : forceNoDetection
      ? "no_detection"
      : (camera.display_fused_final || camera.fused_final);
  setFinalPill(card.querySelector(".final-pill"), displayFinal);
  card.classList.toggle("has-camera-error", hasCameraError);
  const cropMeta = card.querySelector(".crop-meta");
  if (cropMeta) cropMeta.textContent = cleanLabel(camera.crop_reason);

  const cropPred = camera.display_crop_pred || camera.crop_pred;
  const cropConf = camera.display_crop_conf ?? camera.crop_conf;
  const cropText = `${cleanLabel(cropPred)} ${formatPercent(cropConf)}`;
  const detectionText = `P ${camera.phones ?? 0} / L ${camera.lenses ?? 0} / S ${camera.stickers ?? 0}`;

  card.querySelector(".crop-value").textContent = cropText;
  card.querySelector(".detect-value").textContent = detectionText;
  const cropErrorDetail = card.querySelector(".crop-error-detail");
  if (cropErrorDetail) cropErrorDetail.textContent = formatErrorDetail(camera.error, cameraIndex);
  card.querySelector(".full-error-detail").textContent = formatErrorDetail(camera.error, cameraIndex);

  if (hasCameraError) {
    if (cropMeta) cropMeta.textContent = "";
    card.querySelector(".crop-value").textContent = "-";
    card.querySelector(".detect-value").textContent = "-";
  }
}

function updateOverallResult(payload) {
  if (!overallResultPanel || !overallResultValue) return;
  setResultPanel(
    overallResultPanel,
    overallResultValue,
    payload.overall_final || "waiting",
  );
}

async function refreshStatus() {
  try {
    const response = await fetch("/api/status", { cache: "no-store" });
    if (!response.ok) return;
    const payload = await response.json();
    updateOverallResult(payload);
    ensureCameraCards(payload.camera_indexes || []);
    const cameras = payload.cameras || [];
    const pairedCameras = cameras.slice(0, 2);
    const forceNoDetection =
      pairedCameras.length >= 2 && pairedCameras.some((camera) => Number(camera.phones ?? 0) <= 0);
    cameras.forEach((camera) => updateCameraCard(camera, forceNoDetection));
  } catch (error) {
    return;
  }
}

function renderEmptyLogRows(message) {
  logRows.innerHTML = `<tr><td colspan="7">${message}</td></tr>`;
}

async function loadLogDate(date) {
  const response = await fetch(`/api/logs/${encodeURIComponent(date)}`, { cache: "no-store" });
  if (!response.ok) {
    renderEmptyLogRows("로그를 불러오지 못했습니다.");
    return;
  }

  const payload = await response.json();
  selectedLogFolder.textContent = `폴더: ${payload.path || "-"}`;
  logCount.textContent = `저장된 로그 ${payload.count || 0}건을 표시합니다.`;
  csvDownload.hidden = !payload.csv_download_url;
  csvDownload.href = payload.csv_download_url || "#";

  if (!payload.rows || payload.rows.length === 0) {
    renderEmptyLogRows("해당 날짜에 저장된 로그가 없습니다.");
    return;
  }

  logRows.innerHTML = payload.rows
    .map(
      (row) => `
        <tr>
          <td>${row.timestamp || "-"}</td>
          <td>${koreanFinal(row.overall_final)}</td>
          <td>${row.front_crop_conf || "-"}</td>
          <td>${row.back_crop_conf || "-"}</td>
          <td>${formatDetection(row, "front")}</td>
          <td>${formatDetection(row, "back")}</td>
          <td>${row.snapshot_path || "-"}</td>
        </tr>
      `,
    )
    .join("");
}

async function initLogsPage() {
  try {
    const response = await fetch("/api/logs", { cache: "no-store" });
    if (!response.ok) {
      renderEmptyLogRows("로그 날짜를 불러오지 못했습니다.");
      return;
    }

    const payload = await response.json();
    const dates = payload.dates || [];
    if (dates.length === 0) {
      dateTabs.innerHTML = "";
      selectedLogFolder.textContent = `폴더: ${payload.log_root || "-"}`;
      csvDownload.hidden = true;
      renderEmptyLogRows("저장된 로그가 없습니다.");
      return;
    }

    dateTabs.innerHTML = dates
      .map((item, index) => `<button class="${index === 0 ? "active" : ""}" data-date="${item.date}">${item.date}</button>`)
      .join("");

    dateTabs.querySelectorAll("button").forEach((button) => {
      button.addEventListener("click", () => {
        dateTabs.querySelectorAll("button").forEach((tab) => tab.classList.remove("active"));
        button.classList.add("active");
        loadLogDate(button.dataset.date);
      });
    });

    loadLogDate(dates[0].date);
  } catch (error) {
    renderEmptyLogRows("로그를 불러오지 못했습니다.");
  }
}

if (isLogsPage) {
  initLogsPage();
} else {
  refreshStatus();
  setInterval(refreshStatus, 1000);
}
