const el = (id) => document.getElementById(id);

const btnStart = el("btn-start");
const btnStop = el("btn-stop");
const btnSave = el("btn-save");
const btnReset = el("btn-reset");
const errorBanner = el("error-banner");
const errorText = el("error-text");
const toast = el("toast");
const gridLink = el("grid-link");
const resetFlash = el("reset-flash");
const spasiLabel = el("spasi-label");
const spasiProgressBar = el("spasi-progress-bar");

function showToast(msg, ok = true) {
  toast.textContent = msg;
  toast.className = `text-xs text-center ${ok ? "text-emerald-400" : "text-rose-400"}`;
  clearTimeout(showToast._t);
  showToast._t = setTimeout(() => toast.classList.add("hidden"), 2500);
}

function setRunningUI(running) {
  btnStart.disabled = running;
  btnStop.disabled = !running;
}

async function pollStatus() {
  try {
    const res = await fetch("/status");
    const data = await res.json();

    el("huruf-yolo").textContent = data.huruf_yolo
      ? `${data.huruf_yolo} (${data.conf.toFixed(2)})`
      : "-";
    el("huruf-pose").textContent = data.huruf_pose || "-";
    el("kandidat").textContent = data.kandidat || "-";
    el("kalimat").innerHTML = data.kalimat ? data.kalimat : "&nbsp;";
    el("saved-count").textContent = `${data.saved_count}/4`;

    const pct = data.min_votes ? Math.min((data.suara / data.min_votes) * 100, 100) : 0;
    el("progress-bar").style.width = `${pct}%`;

    const spasiPct = Math.min((data.spasi_progress || 0) * 100, 100);
    spasiProgressBar.style.width = `${spasiPct}%`;
    if (data.spasi_cooldown) {
      spasiLabel.textContent = "jeda...";
    } else if (spasiPct > 0) {
      spasiLabel.textContent = "menahan...";
    } else {
      spasiLabel.textContent = "-";
    }

    resetFlash.classList.toggle("hidden", !data.reset_flash);

    setRunningUI(data.running);

    if (data.error) {
      errorText.textContent = data.error;
      errorBanner.classList.remove("hidden");
    } else {
      errorBanner.classList.add("hidden");
    }

    if (data.grid_ready) {
      gridLink.href = `/output/grid_final.jpg?t=${Date.now()}`;
      gridLink.classList.remove("hidden");
      gridLink.classList.add("flex");
    }
  } catch (e) {
    // server belum siap / network sempat putus, abaikan tick ini
  }
}

btnStart.addEventListener("click", async () => {
  btnStart.disabled = true;
  const res = await fetch("/start", { method: "POST" });
  const data = await res.json();
  if (!data.ok) {
    showToast(data.error || "Gagal membuka kamera", false);
    btnStart.disabled = false;
  }
});

btnStop.addEventListener("click", async () => {
  btnStop.disabled = true;
  await fetch("/stop", { method: "POST" });
});

btnSave.addEventListener("click", async () => {
  const res = await fetch("/save", { method: "POST" });
  const data = await res.json();
  if (data.ok) {
    showToast(`Tersimpan: ${data.message}`);
    if (data.info && data.info.grid_created) {
      showToast("Grid 2x2 selesai dibuat!");
    }
  } else {
    showToast(data.message || "Gagal menyimpan", false);
  }
});

btnReset.addEventListener("click", async () => {
  await fetch("/reset", { method: "POST" });
  gridLink.classList.add("hidden");
  gridLink.classList.remove("flex");
  showToast("Kalimat & gambar direset");
});

setInterval(pollStatus, 250);
pollStatus();
