/* HermesTeacher front-end: Markdown + math rendering, answer previews,
   "je ne sais pas" toggles, file drop zone, and the live progress overlay
   for slow actions (background jobs polled from /api/jobs/{id}). */
(function () {
  "use strict";

  const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  /* ---------------------------------------------------------------- */
  /* Markdown + KaTeX                                                  */
  /* ---------------------------------------------------------------- */
  // Math is cut out before Markdown (so "_" and "\" are not mangled),
  // the HTML is sanitized, then the formulas are rendered back in.
  const MATH = /\$\$([\s\S]+?)\$\$|\\\[([\s\S]+?)\\\]|\\\(([\s\S]+?)\\\)|\$((?:\\\$|[^$\n])+?)\$/g;

  function renderMath(tex, display) {
    if (typeof katex === "undefined") return escapeHtml(display ? "$$" + tex + "$$" : "$" + tex + "$");
    try {
      return katex.renderToString(tex, { displayMode: display, throwOnError: false, strict: "ignore", output: "html" });
    } catch (error) {
      return escapeHtml(tex);
    }
  }

  function escapeHtml(text) {
    return String(text).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  }

  function renderMarkdown(source) {
    const formulas = [];
    const protectedText = source.replace(MATH, (match, block, bracket, paren, inline) => {
      const display = block !== undefined || bracket !== undefined;
      const tex = block ?? bracket ?? paren ?? inline;
      formulas.push(renderMath(tex.trim(), display));
      return "KTXPLACEHOLDER" + (formulas.length - 1) + "X";
    });
    let html;
    if (typeof marked !== "undefined") {
      html = marked.parse(protectedText, { gfm: true, breaks: true });
    } else {
      html = escapeHtml(protectedText).replace(/\n/g, "<br>");
    }
    if (typeof DOMPurify !== "undefined") {
      html = DOMPurify.sanitize(html, { USE_PROFILES: { html: true } });
    }
    return html.replace(/KTXPLACEHOLDER(\d+)X/g, (m, i) => formulas[Number(i)] || "");
  }

  function slug(text, used) {
    let base = text.toLowerCase().normalize("NFD").replace(/[̀-ͯ]/g, "").replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, "") || "section";
    let id = base;
    let n = 2;
    while (used.has(id)) id = base + "-" + n++;
    used.add(id);
    return id;
  }

  function renderAllMarkdown() {
    document.querySelectorAll("[data-md]").forEach((el) => {
      el.innerHTML = renderMarkdown(el.textContent.trim());
    });
    const source = document.querySelector("[data-outline-source]");
    const outline = document.querySelector("[data-outline]");
    if (source && outline) {
      const used = new Set();
      const headings = source.querySelectorAll("h2");
      outline.innerHTML = "";
      if (headings.length > 1) {
        headings.forEach((h) => {
          h.id = slug(h.textContent, used);
          const link = document.createElement("a");
          link.href = "#" + h.id;
          link.textContent = h.textContent.replace(/^\s*\d+[.)]\s*/, "");
          outline.appendChild(link);
        });
      }
    }
  }

  /* ---------------------------------------------------------------- */
  /* answers: live preview and "je ne sais pas"                        */
  /* ---------------------------------------------------------------- */
  function bindAnswers() {
    document.querySelectorAll(".answer").forEach((answer) => {
      const area = answer.querySelector("textarea");
      const preview = answer.querySelector(".preview");
      const box = answer.querySelector(".dont-know input");
      let timer;
      const update = () => {
        if (!preview || !area) return;
        const text = area.value.trim();
        // preview only when it adds something: formulas or Markdown
        preview.innerHTML = text && /[$\\*#_`]/.test(text) && !area.disabled ? renderMarkdown(text) : "";
      };
      area?.addEventListener("input", () => {
        clearTimeout(timer);
        timer = setTimeout(update, 180);
      });
      const sync = () => {
        if (!area || !box) return;
        area.disabled = box.checked;
        update();
      };
      box?.addEventListener("change", sync);
      sync();
    });
  }

  /* ---------------------------------------------------------------- */
  /* file drop zone                                                     */
  /* ---------------------------------------------------------------- */
  function bindDropzones() {
    document.querySelectorAll("[data-dropzone]").forEach((zone) => {
      const input = zone.querySelector("input[type=file]");
      const list = zone.querySelector("[data-file-list]");
      const show = () => {
        list.innerHTML = "";
        Array.from(input.files).forEach((file) => {
          const chip = document.createElement("span");
          chip.className = "file-chip";
          chip.textContent = file.name;
          list.appendChild(chip);
        });
      };
      input.addEventListener("change", show);
      ["dragenter", "dragover"].forEach((type) => zone.addEventListener(type, () => zone.classList.add("is-over")));
      ["dragleave", "drop"].forEach((type) => zone.addEventListener(type, () => zone.classList.remove("is-over")));
    });
  }

  /* ---------------------------------------------------------------- */
  /* live progress overlay                                              */
  /* ---------------------------------------------------------------- */
  const SVG = "http://www.w3.org/2000/svg";

  function svg(tag, attrs, parent) {
    const el = document.createElementNS(SVG, tag);
    Object.entries(attrs || {}).forEach(([k, v]) => el.setAttribute(k, v));
    if (parent) parent.appendChild(el);
    return el;
  }

  function formatClock(seconds) {
    const s = Math.max(0, Math.round(seconds));
    return Math.floor(s / 60) + ":" + String(s % 60).padStart(2, "0");
  }

  class ProgressView {
    constructor(title, steps) {
      this.steps = steps;
      this.overlay = document.createElement("div");
      this.overlay.className = "progress-overlay";
      this.overlay.setAttribute("role", "dialog");
      this.overlay.setAttribute("aria-modal", "true");
      this.overlay.setAttribute("aria-label", title);
      this.overlay.innerHTML =
        '<div class="progress-card">' +
        '  <div class="progress-head"><h2></h2><span class="progress-clock" aria-hidden="true">0:00</span></div>' +
        '  <div class="steps"></div>' +
        '  <ul class="feed" aria-live="polite"><li class="waiting">Connexion au tuteur…</li></ul>' +
        '  <div class="progress-slot"></div>' +
        '  <div class="progress-foot"><p>Vous pouvez laisser cette page ouverte : le travail continue côté serveur.</p></div>' +
        "</div>";
      this.overlay.querySelector("h2").textContent = title;
      this.clock = this.overlay.querySelector(".progress-clock");
      this.feed = this.overlay.querySelector(".feed");
      this.slot = this.overlay.querySelector(".progress-slot");
      this.foot = this.overlay.querySelector(".progress-foot");
      this.buildSteps(this.overlay.querySelector(".steps"));
      document.body.appendChild(this.overlay);
      requestAnimationFrame(() => this.overlay.classList.add("is-open"));
      this.lastFeed = null;
    }

    buildSteps(holder) {
      const n = this.steps.length;
      const width = 600;
      // nodes sit at the centre of each label column
      const gap = width / n;
      const pad = gap / 2;
      const y = 30;
      const root = svg("svg", { viewBox: "0 0 " + width + " 60", preserveAspectRatio: "xMidYMid meet", "aria-hidden": "true" }, holder);
      this.edges = [];
      for (let i = 0; i < n - 1; i++) {
        const x1 = pad + i * gap + 16;
        const x2 = pad + (i + 1) * gap - 16;
        svg("line", { class: "edge", x1, y1: y, x2, y2: y }, root);
        const fill = svg("line", { class: "edge-fill", x1, y1: y, x2, y2: y, pathLength: 1, "stroke-dasharray": 1, "stroke-dashoffset": 1 }, root);
        const comet = svg("circle", { class: "comet", r: 3.2, cx: x1, cy: y, opacity: 0 }, root);
        this.edges.push({ fill, comet, x1, x2 });
      }
      this.nodes = this.steps.map((step, i) => {
        const g = svg("g", { class: "step-node" }, root);
        const cx = pad + i * gap;
        svg("circle", { class: "node-halo", cx, cy: y, r: 15 }, g);
        svg("circle", { class: "node-core", cx, cy: y, r: 13 }, g);
        svg("path", { class: "node-mark", d: "M " + (cx - 5) + " " + y + " l 3.6 3.8 l 6.6 -7.4" }, g);
        return g;
      });
      const labels = document.createElement("div");
      labels.className = "step-labels";
      labels.style.gridTemplateColumns = "repeat(" + n + ", 1fr)";
      if (n === 1) labels.style.gridTemplateColumns = "1fr";
      this.labels = this.steps.map((step) => {
        const label = document.createElement("div");
        label.className = "step-label";
        label.innerHTML = "<strong></strong><span></span>";
        label.querySelector("strong").textContent = step.label;
        labels.appendChild(label);
        return label;
      });
      holder.appendChild(labels);
      this.animateComets();
    }

    animateComets() {
      if (reduceMotion) return;
      const tick = (time) => {
        if (!this.overlay.isConnected) return;
        this.edges.forEach((edge) => {
          if (edge.running) {
            const t = (time % 1600) / 1600;
            edge.comet.setAttribute("cx", edge.x1 + (edge.x2 - edge.x1) * t);
            edge.comet.setAttribute("opacity", String(Math.sin(Math.PI * t)));
          } else {
            edge.comet.setAttribute("opacity", "0");
          }
        });
        requestAnimationFrame(tick);
      };
      requestAnimationFrame(tick);
    }

    update(job) {
      this.clock.textContent = formatClock(job.elapsed);
      job.steps.forEach((step, i) => {
        const state = step.status;
        this.nodes[i].setAttribute("class", "step-node is-" + state);
        this.labels[i].className = "step-label is-" + state;
        this.labels[i].querySelector("span").textContent =
          step.elapsed != null && state !== "pending" ? formatClock(step.elapsed) : "";
        const edge = this.edges[i];
        if (edge) {
          const next = job.steps[i + 1];
          const done = state === "done" && next && next.status !== "pending";
          edge.fill.setAttribute("stroke-dashoffset", done ? "0" : "1");
          edge.running = state === "active" && job.status === "running";
        }
      });
      const items = job.activity.slice().reverse();
      const key = items.map((a) => a.seconds + a.text).join("|");
      if (key !== this.lastFeed) {
        this.lastFeed = key;
        this.feed.innerHTML = "";
        if (!items.length) {
          const li = document.createElement("li");
          li.className = "waiting";
          li.textContent = "Le tuteur travaille… les détails apparaissent ici.";
          this.feed.appendChild(li);
        }
        items.slice(0, 6).forEach((item) => {
          const li = document.createElement("li");
          const time = document.createElement("time");
          time.textContent = formatClock(item.seconds);
          li.appendChild(time);
          li.appendChild(document.createTextNode(item.text));
          this.feed.appendChild(li);
        });
      }
    }

    fail(job, onClose) {
      this.update(job);
      this.edges.forEach((edge) => (edge.running = false));
      const box = document.createElement("div");
      box.className = "errors progress-error";
      box.setAttribute("role", "alert");
      box.textContent = job.error || "L'action a échoué.";
      this.slot.innerHTML = "";
      this.slot.appendChild(box);
      this.foot.innerHTML = "";
      const close = document.createElement("button");
      close.className = "btn";
      close.type = "button";
      close.textContent = job.redirect ? "Ouvrir le concept" : "Fermer et réessayer";
      close.addEventListener("click", () => (job.redirect ? (window.location.href = job.redirect) : onClose()));
      this.foot.appendChild(document.createElement("p")).textContent =
        job.redirect ? "Ce qui a déjà été fait est enregistré." : "Vos réponses sont conservées dans la page.";
      this.foot.appendChild(close);
      close.focus();
    }

    showExerciseResult(job) {
      const r = job.result;
      const score = Math.round((r.score || 0) * 100);
      const before = r.level_before != null ? Math.round(r.level_before * 100) : null;
      const after = r.level != null ? Math.round(r.level * 100) : null;
      const wrap = document.createElement("div");
      wrap.className = "result";
      wrap.innerHTML =
        '<div class="result-top">' +
        '  <div class="result-score">' + score + "<small>/100</small></div>" +
        '  <div class="level-move">' +
        '    <div class="status-line"><span class="concept"></span><span class="change"></span></div>' +
        '    <div class="bar"><span style="width:' + (before ?? 0) + '%"></span></div>' +
        "  </div>" +
        "</div>" +
        '<div class="result-feedback prose prose-tight"></div>';
      wrap.querySelector(".concept").textContent = r.concept_label || "Niveau du sous-concept";
      wrap.querySelector(".change").textContent =
        before != null && after != null ? before + " % → " + after + " %" : after != null ? after + " %" : "";
      wrap.querySelector(".result-feedback").innerHTML = renderMarkdown(r.feedback || "Pas de commentaire.");
      this.overlay.querySelector(".steps").remove();
      this.feed.remove();
      this.overlay.querySelector("h2").textContent = "Correction";
      this.slot.innerHTML = "";
      this.slot.appendChild(wrap);
      requestAnimationFrame(() =>
        setTimeout(() => {
          const bar = wrap.querySelector(".bar > span");
          if (bar && after != null) bar.style.width = after + "%";
        }, 120)
      );
      this.foot.innerHTML = "";
      const next = document.createElement("button");
      next.className = "btn btn-primary";
      next.type = "button";
      next.textContent = "Continuer";
      next.addEventListener("click", () => (window.location.href = job.redirect || window.location.pathname));
      this.foot.appendChild(document.createElement("p")).textContent =
        r.progress != null ? "Progression du concept : " + Math.round(r.progress * 100) + " %" : "";
      this.foot.appendChild(next);
      next.focus();
    }

    close() {
      this.overlay.classList.remove("is-open");
      setTimeout(() => this.overlay.remove(), 250);
    }
  }

  async function poll(jobId, view, form) {
    let delay = 700;
    for (;;) {
      await new Promise((resolve) => setTimeout(resolve, delay));
      delay = Math.min(1500, delay + 100);
      let job;
      try {
        const response = await fetch("/api/jobs/" + jobId, { headers: { Accept: "application/json" } });
        if (!response.ok) throw new Error("HTTP " + response.status);
        job = await response.json();
      } catch (error) {
        // the server may be restarting: keep trying quietly
        continue;
      }
      view.update(job);
      if (job.status === "done") {
        if (job.result && job.result.card === "exercise") {
          view.showExerciseResult(job);
        } else {
          window.location.href = job.redirect || window.location.pathname;
        }
        return;
      }
      if (job.status === "error") {
        view.fail(job, () => {
          view.close();
          setBusy(form, false);
        });
        return;
      }
    }
  }

  function setBusy(form, busy) {
    form.querySelectorAll("button[type=submit]").forEach((b) => (b.disabled = busy));
  }

  function bindJobs() {
    document.querySelectorAll("form[data-job]").forEach((form) => {
      form.addEventListener("submit", async (event) => {
        if (typeof fetch === "undefined") return; // plain form submission
        event.preventDefault();
        setBusy(form, true);
        let response;
        try {
          response = await fetch(form.dataset.job, { method: "POST", body: new FormData(form), headers: { Accept: "application/json" } });
        } catch (error) {
          form.submit(); // network trouble: fall back to the classic route
          return;
        }
        const data = await response.json().catch(() => ({}));
        if (!response.ok) {
          setBusy(form, false);
          const box = document.createElement("div");
          box.className = "errors";
          box.setAttribute("role", "alert");
          box.textContent = data.error || data.detail || "L'action n'a pas pu démarrer.";
          form.prepend(box);
          return;
        }
        const view = new ProgressView(form.dataset.jobTitle || "Le tuteur travaille", data.steps);
        view.update(data);
        poll(data.job_id, view, form);
      });
    });
  }

  function start() {
    renderAllMarkdown();
    bindAnswers();
    bindDropzones();
    bindJobs();
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", start);
  } else {
    start();
  }
})();
