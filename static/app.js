/* Video Search Eval Workbench — frontend */

(function () {
  "use strict";

  const form     = document.getElementById("search-form");
  const input    = document.getElementById("search-input");
  const results  = document.getElementById("results");
  const statusEl = document.getElementById("status");

  // ---- Load status on page load ----
  fetch("/api/status")
    .then((r) => r.json())
    .then((d) => {
      const parts = d.pipelines.map(
        (p) => `${p.display_name}: ${p.count} vectors`
      );
      statusEl.textContent =
        `${d.total_videos} video(s) ingested  —  ${parts.join("  |  ")}`;
    })
    .catch(() => {
      statusEl.textContent = "Could not reach backend";
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
        Searching… (first query may take a while as models load)
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
                <img src="/thumbnails/${encPath(r.thumbnail_path)}"
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

    // Attach click handlers for video playback
    results.querySelectorAll(".result-card[data-file]").forEach((card) => {
      card.addEventListener("click", () => swapToVideo(card));
    });
  }

  // ---- Click-to-play ----
  function swapToVideo(card) {
    const container = card.querySelector(".relative");
    if (container.querySelector("video")) return; // already playing

    const img = container.querySelector("img");
    if (img) img.style.display = "none";

    const ts = card.getAttribute("data-ts");
    const file = card.getAttribute("data-file");

    const video = document.createElement("video");
    video.src = `/videos/${encPath(file)}#t=${ts}`;
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
