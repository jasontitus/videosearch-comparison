/* Video Search Eval Workbench — frontend */

(function () {
  "use strict";

  const form     = document.getElementById("search-form");
  const input    = document.getElementById("search-input");
  const results  = document.getElementById("results");
  const statusEl = document.getElementById("status");

  // Media base URL — empty for local, GCS URL for cloud.
  // Default to GCS if served from Firebase/Cloud Run (detected by hostname).
  let mediaBase = location.hostname.includes("web.app") || location.hostname.includes("run.app")
    ? "https://storage.googleapis.com/videosearch-comparison-media"
    : "";

  // ---- Load status on page load ----
  function loadStatus() {
    fetch("/api/status")
      .then((r) => r.json())
      .then((d) => {
        if (d.media_base) mediaBase = d.media_base;
        const parts = d.pipelines.map(
          (p) => `${p.display_name}: ${p.count} vectors`
        );
        statusEl.textContent =
          `${d.total_videos} video(s) ingested  —  ${parts.join("  |  ")}`;
      })
      .catch(() => {
        statusEl.textContent = "Could not reach backend";
      });
  }
  loadStatus();

  // ---- Upload ----
  const uploadZone = document.getElementById("upload-zone");
  const fileInput  = document.getElementById("file-input");
  const uploadProg = document.getElementById("upload-progress");
  const uploadBar  = document.getElementById("upload-bar");
  const uploadStat = document.getElementById("upload-status");
  const videoList  = document.getElementById("video-list");

  uploadZone.addEventListener("click", () => fileInput.click());
  uploadZone.addEventListener("dragover", (e) => {
    e.preventDefault();
    uploadZone.classList.add("border-blue-400", "bg-blue-50");
  });
  uploadZone.addEventListener("dragleave", () => {
    uploadZone.classList.remove("border-blue-400", "bg-blue-50");
  });
  uploadZone.addEventListener("drop", (e) => {
    e.preventDefault();
    uploadZone.classList.remove("border-blue-400", "bg-blue-50");
    const files = [...e.dataTransfer.files].filter((f) => f.name.endsWith(".mp4"));
    if (files.length) uploadFiles(files);
  });
  fileInput.addEventListener("change", () => {
    if (fileInput.files.length) uploadFiles([...fileInput.files]);
  });

  async function uploadFiles(files) {
    uploadProg.classList.remove("hidden");
    for (let i = 0; i < files.length; i++) {
      const file = files[i];
      uploadStat.textContent = `Uploading ${file.name} (${i + 1}/${files.length})…`;

      const xhr = new XMLHttpRequest();
      const formData = new FormData();
      formData.append("file", file);

      await new Promise((resolve) => {
        xhr.upload.onprogress = (e) => {
          if (e.lengthComputable) {
            uploadBar.style.width = Math.round((e.loaded / e.total) * 100) + "%";
          }
        };
        xhr.onload = () => {
          const resp = JSON.parse(xhr.responseText);
          if (resp.error) {
            uploadStat.textContent = `${file.name}: ${resp.error}`;
          }
          resolve();
        };
        xhr.onerror = () => {
          uploadStat.textContent = `Upload failed: ${file.name}`;
          resolve();
        };
        xhr.open("POST", "/api/upload", true);
        xhr.send(formData);
      });
    }
    uploadStat.textContent = "Upload complete!";
    uploadBar.style.width = "100%";
    setTimeout(() => uploadProg.classList.add("hidden"), 2000);
    loadStatus();
    loadVideoList();
  }

  async function loadVideoList() {
    try {
      const res = await fetch("/api/videos");
      const videos = await res.json();
      if (!videos.length) {
        videoList.innerHTML = '<p class="text-gray-400 text-sm">No videos uploaded</p>';
        return;
      }
      let html = `<table class="w-full text-xs">
        <thead><tr class="text-left text-gray-500 border-b">
          <th class="py-1">Filename</th><th class="py-1">Duration</th><th class="py-1">Embeddings</th>
        </tr></thead><tbody>`;
      for (const v of videos) {
        const dur = v.duration ? fmtTime(v.duration) : "—";
        const embs = Object.entries(v.pipelines).map(([k, c]) => `${k}: ${c}`).join(", ") || "none";
        html += `<tr class="border-b border-gray-50">
          <td class="py-1.5 truncate max-w-[300px]">${esc(v.filename)}</td>
          <td class="py-1.5">${dur}</td>
          <td class="py-1.5 text-gray-500">${embs}</td>
        </tr>`;
      }
      html += "</tbody></table>";
      videoList.innerHTML = html;
    } catch {
      videoList.innerHTML = "";
    }
  }

  // Load video list when Manage Videos is opened
  document.querySelector("details").addEventListener("toggle", (e) => {
    if (e.target.open) loadVideoList();
  });

  // ---- Search ----
  form.addEventListener("submit", (e) => {
    e.preventDefault();
    performSearch();
  });

  async function performSearch() {
    const q = input.value.trim();
    if (!q) return;

    results.innerHTML = `
      <div class="text-center py-16 text-gray-400">
        <svg class="animate-spin inline-block w-6 h-6 mr-2 -mt-1" xmlns="http://www.w3.org/2000/svg"
             fill="none" viewBox="0 0 24 24">
          <circle class="opacity-25" cx="12" cy="12" r="10"
                  stroke="currentColor" stroke-width="4"></circle>
          <path class="opacity-75" fill="currentColor"
                d="M4 12a8 8 0 018-8v4a4 4 0 00-4 4H4z"></path>
        </svg>
        Searching... (first query may take a while as models load)
      </div>`;

    try {
      const res = await fetch(`/api/search?q=${encodeURIComponent(q)}`);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      renderResults(data);
    } catch (err) {
      results.innerHTML = `<div class="text-red-600 text-center py-8">${err.message}</div>`;
    }
  }

  // ---- Render ----
  function renderResults(data) {
    const names = Object.keys(data);
    if (names.length === 0) {
      results.innerHTML =
        '<p class="text-gray-400 text-center py-8">No pipelines registered</p>';
      return;
    }

    let html = `<div class="grid gap-5" style="grid-template-columns:repeat(${names.length},minmax(240px,1fr))">`;

    for (const name of names) {
      const p = data[name];
      html += `
        <div class="rounded-xl border border-gray-200 bg-white shadow-sm overflow-hidden flex flex-col">
          <div class="bg-gray-800 text-white px-4 py-2.5 text-center text-sm font-semibold tracking-wide">
            ${p.display_name}
          </div>
          <div class="p-3 space-y-3 flex-1">`;

      if (p.error) {
        html += `<p class="text-red-500 text-xs break-words">${p.error}</p>`;
      } else if (p.results.length === 0) {
        html += '<p class="text-gray-400 text-center py-8 text-sm">No results</p>';
      } else {
        for (const r of p.results) {
          const ts = fmtTime(r.timestamp_start);
          const pct = (r.similarity * 100).toFixed(1);
          html += `
            <div class="result-card rounded-lg border border-gray-100 overflow-hidden cursor-pointer"
                 data-file="${esc(r.filename)}" data-ts="${r.timestamp_start}">
              <div class="relative bg-gray-200">
                <img src="${mediaBase ? mediaBase + '/thumbnails/' : '/thumbnails/'}${encPath(r.thumbnail_path)}"
                     alt="${ts}" loading="lazy"
                     class="w-full aspect-video object-cover" />
                <span class="absolute bottom-1 right-1 bg-black/70 text-white text-[11px]
                             px-1.5 py-0.5 rounded font-mono">${ts}</span>
              </div>
              <div class="px-2 py-1.5 flex justify-between items-center text-xs">
                <span class="text-gray-500 truncate mr-2">${esc(r.filename)}</span>
                <span class="text-blue-600 font-semibold whitespace-nowrap">${pct}%</span>
              </div>
            </div>`;
        }
      }

      html += "</div></div>";
    }

    html += "</div>";
    results.innerHTML = html;

    results.querySelectorAll(".result-card[data-file]").forEach((card) => {
      card.addEventListener("click", () => swapToVideo(card));
    });
  }

  // ---- Click-to-play ----
  function swapToVideo(card) {
    const container = card.querySelector(".relative");
    if (container.querySelector("video")) return;

    const img = container.querySelector("img");
    if (img) img.style.display = "none";

    const ts = card.getAttribute("data-ts");
    const file = card.getAttribute("data-file");

    const video = document.createElement("video");
    video.src = `${mediaBase ? mediaBase + '/videos/' : '/videos/'}${encPath(file)}#t=${ts}`;
    video.className = "w-full aspect-video object-cover bg-black";
    video.controls = true;
    video.autoplay = true;
    container.appendChild(video);
  }

  // ---- Helpers ----
  function fmtTime(sec) {
    const h = Math.floor(sec / 3600);
    const m = Math.floor((sec % 3600) / 60);
    const s = Math.floor(sec % 60);
    return (
      String(h).padStart(2, "0") +
      ":" +
      String(m).padStart(2, "0") +
      ":" +
      String(s).padStart(2, "0")
    );
  }

  function esc(s) {
    const d = document.createElement("div");
    d.textContent = s;
    return d.innerHTML;
  }

  function encPath(p) {
    return p.split("/").map(encodeURIComponent).join("/");
  }
})();
