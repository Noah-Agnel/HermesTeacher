/* Interactive concept graph: zoom/pan, prerequisite chain on hover,
   mastery colours and locked notions. Data comes from #graph-data. */
(function () {
  "use strict";

  const RAMP = ["#2b3a6c", "#3c58ad", "#4f84ee", "#6cc3ff", "#a4f6ff"];
  const STATUS_TEXT = {
    mastered: "Maîtrisé",
    current: "En cours",
    locked: "Prérequis à consolider",
    open: "Disponible",
  };
  const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  function mix(a, b, t) {
    const pa = parseInt(a.slice(1), 16);
    const pb = parseInt(b.slice(1), 16);
    const ch = (shift) => Math.round(((pa >> shift) & 255) * (1 - t) + ((pb >> shift) & 255) * t);
    return "#" + ((1 << 24) | (ch(16) << 16) | (ch(8) << 8) | ch(0)).toString(16).slice(1);
  }

  function rampColor(level) {
    const x = Math.max(0, Math.min(1, level)) * (RAMP.length - 1);
    const i = Math.min(RAMP.length - 2, Math.floor(x));
    return mix(RAMP[i], RAMP[i + 1], x - i);
  }

  function escapeHtml(text) {
    const div = document.createElement("div");
    div.textContent = text;
    return div.innerHTML;
  }

  function start() {
    const holder = document.getElementById("graph");
    const source = document.getElementById("graph-data");
    if (!holder || !source || typeof cytoscape === "undefined") return;
    if (typeof cytoscapeDagre !== "undefined") cytoscape.use(cytoscapeDagre);

    const data = JSON.parse(source.textContent);
    const byId = Object.fromEntries(data.nodes.map((n) => [n.id, n]));
    let direction = window.innerWidth < 700 ? "TB" : "LR";

    const cy = cytoscape({
      container: holder,
      minZoom: 0.2,
      maxZoom: 2.5,
      elements: [
        ...data.nodes.map((n) => ({
          data: { id: n.id, label: n.label, color: rampColor(n.level / (n.target || 1)) },
          classes: n.status,
        })),
        ...data.edges
          .filter((e) => byId[e.source] && byId[e.target])
          .map((e) => ({ data: { id: e.source + "->" + e.target, source: e.source, target: e.target } })),
      ],
      style: [
        {
          selector: "node",
          style: {
            "background-color": "data(color)",
            width: 24,
            height: 24,
            "border-width": 2,
            "border-color": "#070c1c",
            label: "data(label)",
            color: "#c9d4ff",
            "font-family": "Manrope, system-ui, sans-serif",
            "font-size": 13,
            "font-weight": 600,
            "text-valign": "bottom",
            "text-margin-y": 7,
            "text-wrap": "wrap",
            "text-max-width": 150,
            "text-outline-color": "#070c1c",
            "text-outline-width": 3,
            "transition-property": "opacity, border-color, background-color",
            "transition-duration": reduceMotion ? 0 : 180,
          },
        },
        {
          selector: "node.mastered",
          style: { "underlay-color": "#86efff", "underlay-opacity": 0.22, "underlay-padding": 7, "underlay-shape": "ellipse" },
        },
        {
          selector: "node.current",
          style: { "border-color": "#86efff", "border-width": 3, width: 28, height: 28, color: "#ffffff" },
        },
        {
          selector: "node.locked",
          style: { "border-style": "dashed", "border-color": "#66749f", "border-width": 1.5, opacity: 0.6 },
        },
        {
          selector: "edge",
          style: {
            width: 1.3,
            "line-color": "rgba(148, 174, 255, 0.32)",
            "target-arrow-color": "rgba(148, 174, 255, 0.5)",
            "target-arrow-shape": "triangle",
            "arrow-scale": 0.8,
            "curve-style": "bezier",
            "transition-property": "opacity, line-color, width",
            "transition-duration": reduceMotion ? 0 : 180,
          },
        },
        { selector: ".faded", style: { opacity: 0.12 } },
        { selector: "node.chain", style: { opacity: 1, "border-color": "#86efff" } },
        { selector: "node.focus", style: { opacity: 1, "border-color": "#ffffff", "border-width": 3, color: "#ffffff" } },
        {
          selector: "edge.chain",
          style: { opacity: 1, width: 2.2, "line-color": "#86efff", "target-arrow-color": "#86efff" },
        },
      ],
    });

    function runLayout(animate) {
      cy.layout({
        name: typeof cytoscapeDagre !== "undefined" ? "dagre" : "breadthfirst",
        rankDir: direction,
        nodeSep: direction === "LR" ? 34 : 60,
        rankSep: direction === "LR" ? 130 : 90,
        padding: 40,
        directed: true,
        animate: animate && !reduceMotion,
        animationDuration: 450,
        fit: true,
      }).run();
    }
    runLayout(false);
    // keep labels readable on large graphs: never start below 70 % zoom
    if (cy.zoom() < 0.7) {
      cy.zoom(0.7);
      const first = cy.nodes().roots().first();
      if (first.length) cy.pan({ x: 60 - first.position("x") * 0.7 + 40, y: holder.clientHeight / 2 - first.position("y") * 0.7 });
    }

    const card = document.getElementById("node-card");

    function showCard(node) {
      const n = byId[node.id()];
      const pct = Math.round(n.level * 100);
      const target = Math.round((n.target || 1) * 100);
      const prereqs = n.prerequisites.map((id) => (byId[id] ? byId[id].label : id));
      card.innerHTML =
        "<h3>" + escapeHtml(n.label) + "</h3>" +
        '<div class="status-line"><span>' + STATUS_TEXT[n.status] + "</span><span>" + pct + " % / " + target + " %</span></div>" +
        '<div class="bar"><span style="width:' + Math.min(100, (pct / target) * 100) + "%\"></span></div>" +
        (n.description ? "<p>" + escapeHtml(n.description) + "</p>" : "") +
        (prereqs.length ? "<p>Demande : " + prereqs.map(escapeHtml).join(", ") + "</p>" : "<p>Point de départ : aucun prérequis.</p>");
      const pos = node.renderedPosition();
      const frame = holder.getBoundingClientRect();
      const width = 280;
      let left = pos.x + 24;
      if (left + width > frame.width - 12) left = pos.x - width - 24;
      let top = Math.max(12, Math.min(pos.y - 40, frame.height - card.offsetHeight - 12));
      card.style.left = Math.max(12, left) + "px";
      card.style.top = top + "px";
      card.classList.add("is-visible");
    }

    function highlight(node) {
      const chain = node.predecessors().union(node);
      cy.elements().addClass("faded");
      chain.removeClass("faded").addClass("chain");
      node.addClass("focus");
      showCard(node);
    }

    function clear() {
      cy.elements().removeClass("faded chain focus");
      card.classList.remove("is-visible");
    }

    cy.on("mouseover", "node", (event) => highlight(event.target));
    cy.on("mouseout", "node", clear);
    cy.on("tap", "node", (event) => highlight(event.target));
    cy.on("tap", (event) => { if (event.target === cy) clear(); });
    cy.on("pan zoom", () => card.classList.remove("is-visible"));

    document.querySelector('[data-graph="fit"]')?.addEventListener("click", () => {
      cy.animate({ fit: { padding: 40 } }, { duration: reduceMotion ? 0 : 350 });
    });
    document.querySelector('[data-graph="direction"]')?.addEventListener("click", () => {
      direction = direction === "LR" ? "TB" : "LR";
      runLayout(true);
    });
    window.addEventListener("resize", () => cy.resize());
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", start);
  } else {
    start();
  }
})();
