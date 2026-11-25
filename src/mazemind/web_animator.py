"""
Interactive HTML5 Canvas Animators (Dracula Theme) for MazeMind AI Web App.
Renders smooth, client-side, 60 FPS step-by-step visualizations directly from
precomputed SearchResult objects:
- Zero re-solving on Replay
- Every single step shown tile-by-tile without skipping
- Dynamic speed adjustment mid-playback without resetting state
- Single-viewport compact sizing
"""

import json
from typing import Dict, Optional, Tuple
from .environments import GridMaze
from .search_agents import SearchResult


def _safe_json(data: object) -> str:
    """Serialize data to JSON safely for inline <script> embedding."""
    return json.dumps(data).replace("</", "<\\/")


def build_maze_animator_html(
    maze: GridMaze,
    results: Optional[Dict[str, SearchResult]] = None,
    selected_algo: Optional[str] = None,
    autoplay: bool = False,
    instant: bool = False,
    canvas_px: int = 380,
    run_id: int = 0,
) -> str:
    """
    Generates a self-contained HTML5 Canvas widget to animate 2D GridMaze search
    results tile-by-tile using already solved data.
    """
    payload_results = {}
    algo_order = []
    if results:
        for name, res in results.items():
            algo_order.append(name)
            payload_results[name] = {
                "name": name,
                "found": bool(res.found),
                "explore": [[int(r), int(c)] for (r, c) in res.exploration_history],
                "path": [[int(r), int(c)] for (r, c) in res.path],
                "pathCost": int(res.path_cost),
                "nodesExpanded": int(res.nodes_expanded),
                "maxFrontier": int(res.max_frontier_size),
                "timeMs": round(float(res.execution_time_ms), 3),
            }

    default_algo = (
        selected_algo
        if selected_algo and selected_algo in payload_results
        else (algo_order[0] if algo_order else None)
    )

    data_json = _safe_json(
        {
            "rows": int(maze.rows),
            "cols": int(maze.cols),
            "grid": maze.grid.tolist(),
            "start": [int(maze.start[0]), int(maze.start[1])],
            "goal": [int(maze.goal[0]), int(maze.goal[1])],
            "algos": algo_order,
            "results": payload_results,
            "defaultAlgo": default_algo,
            "autoplay": bool(autoplay and payload_results),
            "instant": bool(instant and payload_results),
            "canvasPx": int(canvas_px),
            "runId": int(run_id),
        }
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    background: #282a36;
    color: #f8f8f2;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    overflow: hidden;
  }}
  .wrap {{
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 6px;
    width: 100%;
    padding: 2px 4px;
  }}
  .toolbar {{
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    justify-content: space-between;
    gap: 6px;
    width: 100%;
    background: #21222c;
    border: 1px solid #44475a;
    border-radius: 6px;
    padding: 5px 8px;
  }}
  .row {{
    display: flex;
    align-items: center;
    gap: 5px;
    flex-wrap: wrap;
  }}
  select {{
    background: #282a36;
    color: #f8f8f2;
    border: 1px solid #6272a4;
    border-radius: 4px;
    padding: 3px 6px;
    font-size: 12px;
    font-weight: 600;
    outline: none;
  }}
  .btn {{
    background: #bd93f9;
    color: #282a36;
    border: none;
    border-radius: 4px;
    padding: 4px 9px;
    font-size: 12px;
    font-weight: 700;
    cursor: pointer;
    transition: background 0.12s ease;
    line-height: 1.2;
  }}
  .btn:hover:not(:disabled) {{ background: #ff79c6; }}
  .btn:disabled {{ opacity: 0.45; cursor: not-allowed; }}
  .btn-secondary {{
    background: #44475a;
    color: #f8f8f2;
    border: 1px solid #6272a4;
  }}
  .btn-secondary:hover:not(:disabled) {{ background: #6272a4; color: #f8f8f2; }}
  .btn-active {{
    background: #8be9fd !important;
    color: #282a36 !important;
    border-color: #8be9fd !important;
  }}
  .speed-box {{
    display: flex;
    align-items: center;
    gap: 5px;
    font-size: 11.5px;
    color: #f1fa8c;
    font-weight: 600;
  }}
  input[type="range"] {{
    accent-color: #bd93f9;
    cursor: pointer;
  }}
  .status-bar {{
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 8px;
    width: 100%;
    background: #21222c;
    border: 1px solid #44475a;
    border-left: 4px solid #8be9fd;
    border-radius: 5px;
    padding: 4px 9px;
    font-size: 12px;
  }}
  #statusText {{ color: #8be9fd; font-weight: 600; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }}
  #metricsText {{ color: #50fa7b; font-size: 11.5px; font-weight: 600; white-space: nowrap; }}
  .scrubber-row {{
    display: flex;
    align-items: center;
    gap: 8px;
    width: 100%;
    padding: 0 2px;
  }}
  #scrubber {{ flex: 1; height: 8px; cursor: pointer; }}
  #stepCounter {{ font-size: 11.5px; color: #f8f8f2; min-width: 78px; text-align: right; font-variant-numeric: tabular-nums; }}
  canvas {{
    background: #282a36;
    border: 1px solid #6272a4;
    border-radius: 6px;
    display: block;
  }}
  .legend {{
    display: flex;
    flex-wrap: wrap;
    justify-content: center;
    gap: 10px;
    font-size: 11px;
    color: #f8f8f2;
    padding-top: 1px;
  }}
  .leg-item {{ display: flex; align-items: center; gap: 4px; }}
  .swatch {{ width: 11px; height: 11px; border-radius: 2px; border: 1px solid #6272a4; display: inline-block; }}
</style>
</head>
<body>
<div class="wrap">
  <div class="toolbar">
    <div class="row">
      <select id="algoSelect" style="display:none;"></select>
      <button class="btn" id="btnPlayPause" disabled>▶️ Play</button>
      <button class="btn btn-secondary" id="btnReplay" disabled title="Replay saved solution without re-solving">🔁 Replay</button>
      <button class="btn btn-secondary" id="btnReset" disabled>⏮️ Reset</button>
      <button class="btn btn-secondary" id="btnPrev" disabled title="Previous tile step">◀️</button>
      <button class="btn btn-secondary" id="btnNext" disabled title="Next tile step">▶️</button>
    </div>
    <div class="row">
      <button class="btn btn-secondary btn-active" id="btnModeBoth" disabled title="Animate explored tiles then final path">Explore + Path</button>
      <button class="btn btn-secondary" id="btnModePath" disabled title="Animate only final solution path tile by tile">Path Only</button>
      <div class="speed-box">
        <span>⚡</span>
        <input type="range" id="speedSlider" min="0" max="6" step="1" value="2" style="width:74px;">
        <span id="speedLabel" style="min-width:34px;">1.0x</span>
      </div>
    </div>
  </div>

  <div class="status-bar">
    <span id="statusText">Ready — click "🚀 Solve Maze" to begin</span>
    <span id="metricsText"></span>
  </div>

  <div class="scrubber-row">
    <input type="range" id="scrubber" min="0" max="0" value="0" disabled>
    <span id="stepCounter">0 / 0</span>
  </div>

  <canvas id="mazeCanvas"></canvas>

  <div class="legend">
    <span class="leg-item"><span class="swatch" style="background:#50fa7b;"></span>Start (S)</span>
    <span class="leg-item"><span class="swatch" style="background:#ff5555;"></span>Goal (G)</span>
    <span class="leg-item"><span class="swatch" style="background:#bd93f9;"></span>Explored</span>
    <span class="leg-item"><span class="swatch" style="background:#ff79c6;"></span>Active Tile</span>
    <span class="leg-item"><span class="swatch" style="background:#f1fa8c;"></span>Solution Path</span>
    <span class="leg-item"><span class="swatch" style="background:#191a21;"></span>Wall</span>
  </div>
</div>

<script>
(() => {{
  const DATA = {data_json};
  const DRACULA = {{
    bg: "#282a36",
    open: "#44475a",
    wall: "#191a21",
    border: "#6272a4",
    explored: "#bd93f9",
    active: "#ff79c6",
    pathCell: "#f1fa8c",
    pathLine: "#ffb86c",
    agent: "#8be9fd",
    start: "#50fa7b",
    goal: "#ff5555",
    fg: "#f8f8f2"
  }};

  const SPEED_LEVELS = [
    {{ label: "0.25x", delay: 200 }},
    {{ label: "0.5x",  delay: 100 }},
    {{ label: "1.0x",  delay: 42 }},
    {{ label: "1.5x",  delay: 26 }},
    {{ label: "2.0x",  delay: 16 }},
    {{ label: "3.5x",  delay: 8 }},
    {{ label: "5.0x",  delay: 4 }}
  ];

  const canvas = document.getElementById("mazeCanvas");
  const ctx = canvas.getContext("2d");
  const dpr = window.devicePixelRatio || 1;
  const pxSize = DATA.canvasPx;
  canvas.width = pxSize * dpr;
  canvas.height = pxSize * dpr;
  canvas.style.width = pxSize + "px";
  canvas.style.height = pxSize + "px";
  ctx.scale(dpr, dpr);

  const algoSelect = document.getElementById("algoSelect");
  const btnPlayPause = document.getElementById("btnPlayPause");
  const btnReplay = document.getElementById("btnReplay");
  const btnReset = document.getElementById("btnReset");
  const btnPrev = document.getElementById("btnPrev");
  const btnNext = document.getElementById("btnNext");
  const btnModeBoth = document.getElementById("btnModeBoth");
  const btnModePath = document.getElementById("btnModePath");
  const speedSlider = document.getElementById("speedSlider");
  const speedLabel = document.getElementById("speedLabel");
  const scrubber = document.getElementById("scrubber");
  const statusText = document.getElementById("statusText");
  const metricsText = document.getElementById("metricsText");
  const stepCounter = document.getElementById("stepCounter");

  let activeAlgo = DATA.defaultAlgo;
  let playbackMode = "both"; // "both" (explore + path) or "path" (solution path only)
  let currentStep = 0;
  let isPlaying = false;
  let timerId = null;

  function getActiveResult() {{
    if (!activeAlgo || !DATA.results[activeAlgo]) return null;
    return DATA.results[activeAlgo];
  }}

  function getTotalFrames() {{
    const res = getActiveResult();
    if (!res) return 0;
    const eLen = res.explore.length;
    const pLen = res.found ? res.path.length : 0;
    return playbackMode === "path" ? pLen : eLen + pLen;
  }}

  function getDelayForStep(step) {{
    const idx = parseInt(speedSlider.value, 10);
    const base = SPEED_LEVELS[idx].delay;
    const res = getActiveResult();
    if (!res) return base;
    if (playbackMode === "path" || step > res.explore.length) {{
      // Slight breathing room while tracing the optimal path tile-by-tile
      return Math.max(base, Math.min(220, Math.round(base * 1.35)));
    }}
    return base;
  }}

  function renderFrame() {{
    const rows = DATA.rows;
    const cols = DATA.cols;
    const grid = DATA.grid;
    const [sr, sc] = DATA.start;
    const [gr, gc] = DATA.goal;

    const pad = 4;
    const cellW = (pxSize - pad * 2) / cols;
    const cellH = (pxSize - pad * 2) / rows;
    const cell = Math.min(cellW, cellH);
    const ox = (pxSize - cell * cols) / 2;
    const oy = (pxSize - cell * rows) / 2;

    ctx.fillStyle = DRACULA.bg;
    ctx.fillRect(0, 0, pxSize, pxSize);

    const res = getActiveResult();
    const total = getTotalFrames();
    currentStep = Math.max(0, Math.min(total, currentStep));

    let exploreCount = 0;
    let pathCount = 0;
    let activeTile = null;
    let pathHead = null;

    if (res) {{
      const eLen = res.explore.length;
      if (playbackMode === "both") {{
        exploreCount = Math.min(currentStep, eLen);
        if (currentStep >= 1 && currentStep <= eLen) {{
          activeTile = res.explore[currentStep - 1];
        }}
        if (res.found && currentStep > eLen) {{
          pathCount = Math.min(res.path.length, currentStep - eLen);
          if (pathCount > 0) pathHead = res.path[pathCount - 1];
        }}
      }} else {{
        if (res.found && currentStep > 0) {{
          pathCount = Math.min(res.path.length, currentStep);
          pathHead = res.path[pathCount - 1];
        }}
      }}
    }}

    const exploredSet = new Set();
    if (res && exploreCount > 0) {{
      for (let i = 0; i < exploreCount; i++) {{
        exploredSet.add(res.explore[i][0] + "," + res.explore[i][1]);
      }}
    }}

    const pathSet = new Set();
    if (res && pathCount > 0) {{
      for (let i = 0; i < pathCount; i++) {{
        pathSet.add(res.path[i][0] + "," + res.path[i][1]);
      }}
    }}

    // Draw grid tiles
    for (let r = 0; r < rows; r++) {{
      for (let c = 0; c < cols; c++) {{
        const x = ox + c * cell;
        const y = oy + r * cell;
        const key = r + "," + c;
        let fill = grid[r][c] === 1 ? DRACULA.wall : DRACULA.open;
        if (exploredSet.has(key) && grid[r][c] === 0) fill = DRACULA.explored;
        if (pathSet.has(key)) fill = DRACULA.pathCell;
        if (activeTile && activeTile[0] === r && activeTile[1] === c) fill = DRACULA.active;
        if (r === sr && c === sc) fill = DRACULA.start;
        if (r === gr && c === gc) fill = DRACULA.goal;

        ctx.fillStyle = fill;
        ctx.fillRect(x, y, cell, cell);

        ctx.strokeStyle = DRACULA.border;
        ctx.lineWidth = Math.max(0.4, Math.min(1.0, cell * 0.05));
        ctx.strokeRect(x, y, cell, cell);
      }}
    }}

    // Highlight active expanding frontier cell
    if (activeTile) {{
      const ax = ox + activeTile[1] * cell;
      const ay = oy + activeTile[0] * cell;
      ctx.strokeStyle = DRACULA.pathCell;
      ctx.lineWidth = Math.max(1.5, cell * 0.14);
      ctx.strokeRect(ax + 1, ay + 1, cell - 2, cell - 2);
    }}

    // Draw progressive solution path polyline tile by tile
    if (res && pathCount > 1) {{
      ctx.beginPath();
      for (let i = 0; i < pathCount; i++) {{
        const [pr, pc] = res.path[i];
        const cx = ox + (pc + 0.5) * cell;
        const cy = oy + (pr + 0.5) * cell;
        if (i === 0) ctx.moveTo(cx, cy);
        else ctx.lineTo(cx, cy);
      }}
      ctx.strokeStyle = DRACULA.pathLine;
      ctx.lineWidth = Math.max(2, cell * 0.22);
      ctx.lineCap = "round";
      ctx.lineJoin = "round";
      ctx.stroke();
    }}

    // Draw cyan head cursor tracing the path tile-by-tile
    if (pathHead && currentStep < total) {{
      const [hr, hc] = pathHead;
      ctx.beginPath();
      ctx.arc(ox + (hc + 0.5) * cell, oy + (hr + 0.5) * cell, Math.max(3, cell * 0.28), 0, Math.PI * 2);
      ctx.fillStyle = DRACULA.agent;
      ctx.fill();
    }}

    // Crisp S and G labels
    const fontPx = Math.max(9, Math.floor(cell * 0.58));
    ctx.font = "bold " + fontPx + "px sans-serif";
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillStyle = DRACULA.bg;
    ctx.fillText("S", ox + (sc + 0.5) * cell, oy + (sr + 0.5) * cell);
    ctx.fillText("G", ox + (gc + 0.5) * cell, oy + (gr + 0.5) * cell);

    // Update scrubber & status text
    scrubber.max = total;
    scrubber.value = currentStep;
    stepCounter.textContent = currentStep + " / " + total;

    if (!res) {{
      statusText.textContent = "Ready (" + rows + "×" + cols + " Solvable Grid) — Click Solve Maze";
      statusText.style.color = DRACULA.agent;
      metricsText.textContent = "";
    }} else if (!res.found) {{
      statusText.textContent = "❌ " + res.name + ": No path exists from S to G";
      statusText.style.color = DRACULA.goal;
      metricsText.textContent = "Explored: " + res.nodesExpanded;
    }} else {{
      metricsText.textContent = "Path: " + res.pathCost + " | Explored: " + res.nodesExpanded + " | " + res.timeMs + "ms";
      if (currentStep === 0) {{
        statusText.textContent = "⏮️ " + res.name + ": Start State — Ready to play";
        statusText.style.color = DRACULA.agent;
      }} else if (currentStep >= total) {{
        statusText.textContent = "✅ " + res.name + ": Complete! Optimal Path = " + res.pathCost + " steps";
        statusText.style.color = DRACULA.start;
      }} else if (playbackMode === "both" && currentStep <= res.explore.length) {{
        const [er, ec] = activeTile || [0, 0];
        statusText.textContent = "🔍 Exploring tile " + currentStep + "/" + res.explore.length + " → (" + er + ", " + ec + ")";
        statusText.style.color = DRACULA.active;
      }} else {{
        const [pr, pc] = pathHead || [0, 0];
        statusText.textContent = "✨ Tracing Path step " + pathCount + "/" + res.path.length + " → (" + pr + ", " + pc + ")";
        statusText.style.color = DRACULA.pathCell;
      }}
    }}

    btnPlayPause.textContent = isPlaying ? "⏸️ Pause" : "▶️ Play";
  }}

  function stopPlayback() {{
    isPlaying = false;
    if (timerId !== null) {{
      clearTimeout(timerId);
      timerId = null;
    }}
    renderFrame();
  }}

  function scheduleNextTick() {{
    if (!isPlaying) return;
    const total = getTotalFrames();
    if (currentStep >= total) {{
      stopPlayback();
      return;
    }}
    const delay = getDelayForStep(currentStep + 1);
    timerId = setTimeout(() => {{
      if (!isPlaying) return;
      currentStep += 1;
      renderFrame();
      if (currentStep < getTotalFrames()) {{
        scheduleNextTick();
      }} else {{
        stopPlayback();
      }}
    }}, delay);
  }}

  function startPlayback(fromStart = false) {{
    const total = getTotalFrames();
    if (total === 0) return;
    if (timerId !== null) {{
      clearTimeout(timerId);
      timerId = null;
    }}
    if (fromStart || currentStep >= total) {{
      currentStep = 0;
    }}
    isPlaying = true;
    renderFrame();
    scheduleNextTick();
  }}

  function initControls() {{
    const hasResults = DATA.algos.length > 0;
    [btnPlayPause, btnReplay, btnReset, btnPrev, btnNext, btnModeBoth, btnModePath, scrubber].forEach(el => {{
      el.disabled = !hasResults;
    }});

    if (DATA.algos.length > 1) {{
      algoSelect.style.display = "inline-block";
      DATA.algos.forEach(name => {{
        const opt = document.createElement("option");
        opt.value = name;
        opt.textContent = name;
        if (name === activeAlgo) opt.selected = true;
        algoSelect.appendChild(opt);
      }});
    }}

    algoSelect.addEventListener("change", () => {{
      activeAlgo = algoSelect.value;
      // Switch cleanly between already solved results without re-solving!
      startPlayback(true);
    }});

    btnPlayPause.addEventListener("click", () => {{
      if (isPlaying) stopPlayback();
      else startPlayback(false);
    }});

    btnReplay.addEventListener("click", () => {{
      // Uses already solved data — never re-runs solver!
      startPlayback(true);
    }});

    btnReset.addEventListener("click", () => {{
      stopPlayback();
      currentStep = 0;
      renderFrame();
    }});

    btnPrev.addEventListener("click", () => {{
      stopPlayback();
      currentStep = Math.max(0, currentStep - 1);
      renderFrame();
    }});

    btnNext.addEventListener("click", () => {{
      stopPlayback();
      currentStep = Math.min(getTotalFrames(), currentStep + 1);
      renderFrame();
    }});

    btnModeBoth.addEventListener("click", () => {{
      playbackMode = "both";
      btnModeBoth.classList.add("btn-active");
      btnModePath.classList.remove("btn-active");
      startPlayback(true);
    }});

    btnModePath.addEventListener("click", () => {{
      playbackMode = "path";
      btnModePath.classList.add("btn-active");
      btnModeBoth.classList.remove("btn-active");
      startPlayback(true);
    }});

    speedSlider.addEventListener("input", () => {{
      const idx = parseInt(speedSlider.value, 10);
      speedLabel.textContent = SPEED_LEVELS[idx].label;
      // Live dynamic speed change mid-replay WITHOUT resetting currentStep!
      if (isPlaying) {{
        if (timerId !== null) clearTimeout(timerId);
        scheduleNextTick();
      }}
    }});

    const handleScrub = () => {{
      // Read target step BEFORE stopping playback, otherwise renderFrame() resets scrubber.value!
      const targetStep = parseInt(scrubber.value, 10);
      currentStep = isNaN(targetStep) ? currentStep : targetStep;
      stopPlayback();
    }};
    scrubber.addEventListener("input", handleScrub);
    scrubber.addEventListener("change", handleScrub);

    if (DATA.instant && hasResults) {{
      currentStep = getTotalFrames();
      renderFrame();
    }} else if (DATA.autoplay && hasResults) {{
      startPlayback(true);
    }} else {{
      renderFrame();
    }}
  }}

  initControls();
}})();
</script>
</body>
</html>"""


def build_puzzle_animator_html(
    start_state: Tuple[int, ...],
    goal_state: Tuple[int, ...],
    results: Optional[Dict[str, SearchResult]] = None,
    selected_algo: Optional[str] = None,
    autoplay: bool = False,
    instant: bool = False,
    canvas_px: int = 355,
    run_id: int = 0,
) -> str:
    """
    Generates a self-contained HTML5 Canvas widget to smoothly slide 8-puzzle
    tiles 1-by-1 using already solved SearchResult paths.
    """
    payload_results = {}
    algo_order = []
    if results:
        for name, res in results.items():
            algo_order.append(name)
            moves = []
            if res.found and res.path:
                moves.append(
                    {
                        "board": [int(x) for x in res.path[0]],
                        "movedTile": None,
                        "fromIdx": None,
                        "toIdx": None,
                        "desc": "Initial Board State",
                    }
                )
                for idx in range(1, len(res.path)):
                    prev_b = res.path[idx - 1]
                    curr_b = res.path[idx]
                    prev_zero = prev_b.index(0)
                    curr_zero = curr_b.index(0)
                    moved_tile = int(prev_b[curr_zero])

                    r0, c0 = divmod(curr_zero, 3)
                    r1, c1 = divmod(prev_zero, 3)
                    dr, dc = r1 - r0, c1 - c0
                    direction = (
                        "UP"
                        if dr == -1
                        else ("DOWN" if dr == 1 else ("LEFT" if dc == -1 else "RIGHT"))
                    )
                    moves.append(
                        {
                            "board": [int(x) for x in curr_b],
                            "movedTile": moved_tile,
                            "fromIdx": int(curr_zero),
                            "toIdx": int(prev_zero),
                            "desc": f"Step {idx}: Tile {moved_tile} slid {direction} into blank slot",
                        }
                    )

            payload_results[name] = {
                "name": name,
                "found": bool(res.found),
                "moves": moves,
                "pathCost": int(res.path_cost),
                "nodesExpanded": int(res.nodes_expanded),
                "timeMs": round(float(res.execution_time_ms), 3),
            }

    default_algo = (
        selected_algo
        if selected_algo and selected_algo in payload_results
        else (algo_order[0] if algo_order else None)
    )

    data_json = _safe_json(
        {
            "start": [int(x) for x in start_state],
            "goal": [int(x) for x in goal_state],
            "algos": algo_order,
            "results": payload_results,
            "defaultAlgo": default_algo,
            "autoplay": bool(autoplay and payload_results),
            "instant": bool(instant and payload_results),
            "canvasPx": int(canvas_px),
            "runId": int(run_id),
        }
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    background: #282a36;
    color: #f8f8f2;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    overflow: hidden;
  }}
  .wrap {{
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 6px;
    width: 100%;
    padding: 2px 4px;
  }}
  .toolbar {{
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    justify-content: space-between;
    gap: 6px;
    width: 100%;
    background: #21222c;
    border: 1px solid #44475a;
    border-radius: 6px;
    padding: 5px 8px;
  }}
  .row {{
    display: flex;
    align-items: center;
    gap: 5px;
    flex-wrap: wrap;
  }}
  select {{
    background: #282a36;
    color: #f8f8f2;
    border: 1px solid #6272a4;
    border-radius: 4px;
    padding: 3px 6px;
    font-size: 12px;
    font-weight: 600;
    outline: none;
  }}
  .btn {{
    background: #bd93f9;
    color: #282a36;
    border: none;
    border-radius: 4px;
    padding: 4px 9px;
    font-size: 12px;
    font-weight: 700;
    cursor: pointer;
    line-height: 1.2;
  }}
  .btn:hover:not(:disabled) {{ background: #ff79c6; }}
  .btn:disabled {{ opacity: 0.45; cursor: not-allowed; }}
  .btn-secondary {{
    background: #44475a;
    color: #f8f8f2;
    border: 1px solid #6272a4;
  }}
  .btn-secondary:hover:not(:disabled) {{ background: #6272a4; color: #f8f8f2; }}
  .speed-box {{
    display: flex;
    align-items: center;
    gap: 5px;
    font-size: 11.5px;
    color: #f1fa8c;
    font-weight: 600;
  }}
  input[type="range"] {{
    accent-color: #bd93f9;
    cursor: pointer;
  }}
  .status-bar {{
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 8px;
    width: 100%;
    background: #21222c;
    border: 1px solid #44475a;
    border-left: 4px solid #ff79c6;
    border-radius: 5px;
    padding: 4px 9px;
    font-size: 12px;
  }}
  #actionText {{ color: #ff79c6; font-weight: 700; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }}
  #metricsText {{ color: #50fa7b; font-size: 11.5px; font-weight: 600; white-space: nowrap; }}
  .scrubber-row {{
    display: flex;
    align-items: center;
    gap: 8px;
    width: 100%;
    padding: 0 2px;
  }}
  #scrubber {{ flex: 1; height: 8px; cursor: pointer; }}
  #stepCounter {{ font-size: 11.5px; color: #f8f8f2; min-width: 68px; text-align: right; font-variant-numeric: tabular-nums; }}
  canvas {{
    background: #282a36;
    border: 1px solid #6272a4;
    border-radius: 8px;
    display: block;
  }}
</style>
</head>
<body>
<div class="wrap">
  <div class="toolbar">
    <div class="row">
      <select id="algoSelect" style="display:none;"></select>
      <button class="btn" id="btnPlayPause" disabled>▶️ Play</button>
      <button class="btn btn-secondary" id="btnReplay" disabled>🔁 Replay</button>
      <button class="btn btn-secondary" id="btnReset" disabled>⏮️ Reset</button>
      <button class="btn btn-secondary" id="btnPrev" disabled>◀️</button>
      <button class="btn btn-secondary" id="btnNext" disabled>▶️</button>
    </div>
    <div class="speed-box">
      <span>⚡ Speed:</span>
      <input type="range" id="speedSlider" min="0" max="5" step="1" value="2" style="width:76px;">
      <span id="speedLabel" style="min-width:34px;">1.0x</span>
    </div>
  </div>

  <div class="status-bar">
    <span id="actionText">Initial Board State — Click "🚀 Solve 8-Puzzle"</span>
    <span id="metricsText"></span>
  </div>

  <div class="scrubber-row">
    <input type="range" id="scrubber" min="0" max="0" value="0" disabled>
    <span id="stepCounter">0 / 0</span>
  </div>

  <canvas id="puzzleCanvas"></canvas>
</div>

<script>
(() => {{
  const DATA = {data_json};
  const DRACULA = {{
    bg: "#282a36",
    tile: "#44475a",
    blank: "#191a21",
    border: "#bd93f9",
    comment: "#6272a4",
    pink: "#ff79c6",
    yellow: "#f1fa8c",
    cyan: "#8be9fd",
    green: "#50fa7b",
    red: "#ff5555",
    fg: "#f8f8f2"
  }};

  const SPEED_LEVELS = [
    {{ label: "0.25x", delay: 1100 }},
    {{ label: "0.5x",  delay: 650 }},
    {{ label: "1.0x",  delay: 380 }},
    {{ label: "1.5x",  delay: 240 }},
    {{ label: "2.0x",  delay: 140 }},
    {{ label: "4.0x",  delay: 65 }}
  ];

  const canvas = document.getElementById("puzzleCanvas");
  const ctx = canvas.getContext("2d");
  const dpr = window.devicePixelRatio || 1;
  const pxSize = DATA.canvasPx;
  canvas.width = pxSize * dpr;
  canvas.height = pxSize * dpr;
  canvas.style.width = pxSize + "px";
  canvas.style.height = pxSize + "px";
  ctx.scale(dpr, dpr);

  const algoSelect = document.getElementById("algoSelect");
  const btnPlayPause = document.getElementById("btnPlayPause");
  const btnReplay = document.getElementById("btnReplay");
  const btnReset = document.getElementById("btnReset");
  const btnPrev = document.getElementById("btnPrev");
  const btnNext = document.getElementById("btnNext");
  const speedSlider = document.getElementById("speedSlider");
  const speedLabel = document.getElementById("speedLabel");
  const scrubber = document.getElementById("scrubber");
  const actionText = document.getElementById("actionText");
  const metricsText = document.getElementById("metricsText");
  const stepCounter = document.getElementById("stepCounter");

  let activeAlgo = DATA.defaultAlgo;
  let currentStep = 0;
  let isPlaying = false;
  let timerId = null;
  let rafId = null;

  function getActiveResult() {{
    if (!activeAlgo || !DATA.results[activeAlgo]) return null;
    return DATA.results[activeAlgo];
  }}

  function getMaxStep() {{
    const res = getActiveResult();
    if (!res || !res.found || res.moves.length === 0) return 0;
    return res.moves.length - 1;
  }}

  function getDelay() {{
    return SPEED_LEVELS[parseInt(speedSlider.value, 10)].delay;
  }}

  function drawRoundedRect(x, y, w, h, r) {{
    ctx.beginPath();
    ctx.moveTo(x + r, y);
    ctx.arcTo(x + w, y, x + w, y + h, r);
    ctx.arcTo(x + w, y + h, x, y + h, r);
    ctx.arcTo(x, y + h, x, y, r);
    ctx.arcTo(x, y, x + w, y, r);
    ctx.closePath();
  }}

  function renderBoard(slideProgress = 1.0) {{
    const pad = 12;
    const cell = (pxSize - pad * 2) / 3;
    const inset = 6;
    const tileW = cell - inset * 2;

    ctx.fillStyle = DRACULA.bg;
    ctx.fillRect(0, 0, pxSize, pxSize);

    const res = getActiveResult();
    const maxStep = getMaxStep();
    currentStep = Math.max(0, Math.min(maxStep, currentStep));

    let board = DATA.start;
    let movedTile = null;
    let fromIdx = null;
    let toIdx = null;
    let desc = "Initial Board State";

    if (res && res.found && res.moves.length > 0) {{
      const mv = res.moves[currentStep];
      board = mv.board;
      movedTile = mv.movedTile;
      fromIdx = mv.fromIdx;
      toIdx = mv.toIdx;
      desc = mv.desc;
    }}

    // Draw blank slots underneath all tiles
    for (let idx = 0; idx < 9; idx++) {{
      const r = Math.floor(idx / 3);
      const c = idx % 3;
      const x = pad + c * cell + inset;
      const y = pad + r * cell + inset;
      if (board[idx] === 0 || (movedTile !== null && idx === toIdx && slideProgress < 1.0)) {{
        drawRoundedRect(x, y, tileW, tileW, 10);
        ctx.fillStyle = DRACULA.blank;
        ctx.fill();
        ctx.setLineDash([5, 4]);
        ctx.strokeStyle = DRACULA.comment;
        ctx.lineWidth = 2;
        ctx.stroke();
        ctx.setLineDash([]);
      }}
    }}

    // Draw numbered tiles
    for (let idx = 0; idx < 9; idx++) {{
      const val = board[idx];
      if (val === 0) continue;

      let r = Math.floor(idx / 3);
      let c = idx % 3;
      const isMoved = movedTile !== null && val === movedTile;

      if (isMoved && fromIdx !== null && toIdx !== null && slideProgress < 1.0) {{
        const r0 = Math.floor(fromIdx / 3), c0 = fromIdx % 3;
        const r1 = Math.floor(toIdx / 3), c1 = toIdx % 3;
        // Smooth ease-out slide interpolation
        const t = 1 - Math.pow(1 - slideProgress, 2);
        r = r0 + (r1 - r0) * t;
        c = c0 + (c1 - c0) * t;
      }}

      const x = pad + c * cell + inset;
      const y = pad + r * cell + inset;
      const isGoalTile = DATA.goal[idx] === val && (!isMoved || slideProgress >= 1.0);

      drawRoundedRect(x, y, tileW, tileW, 10);
      ctx.fillStyle = isMoved ? DRACULA.pink : DRACULA.tile;
      ctx.fill();
      ctx.strokeStyle = isMoved ? DRACULA.yellow : (isGoalTile ? DRACULA.green : DRACULA.border);
      ctx.lineWidth = isMoved ? 3.5 : 2.2;
      ctx.stroke();

      ctx.fillStyle = isMoved ? DRACULA.bg : DRACULA.cyan;
      ctx.font = "bold " + Math.floor(tileW * 0.44) + "px sans-serif";
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      ctx.fillText(String(val), x + tileW / 2, y + tileW / 2 + 2);
    }}

    // Update UI indicators
    scrubber.max = maxStep;
    scrubber.value = currentStep;
    stepCounter.textContent = currentStep + " / " + maxStep;

    if (!res) {{
      actionText.textContent = "Initial Board State — Click Solve 8-Puzzle";
      actionText.style.color = DRACULA.cyan;
      metricsText.textContent = "";
    }} else if (!res.found) {{
      actionText.textContent = "❌ " + res.name + ": No solution path found";
      actionText.style.color = DRACULA.red;
      metricsText.textContent = "Nodes: " + res.nodesExpanded;
    }} else {{
      metricsText.textContent = "Moves: " + res.pathCost + " | Nodes: " + res.nodesExpanded + " | " + res.timeMs + "ms";
      if (currentStep === maxStep) {{
        actionText.textContent = "🎉 Goal Reached! (" + desc + ")";
        actionText.style.color = DRACULA.green;
      }} else {{
        actionText.textContent = "📍 " + desc;
        actionText.style.color = DRACULA.pink;
      }}
    }}

    btnPlayPause.textContent = isPlaying ? "⏸️ Pause" : "▶️ Play";
  }}

  function stopPlayback() {{
    isPlaying = false;
    if (timerId !== null) {{ clearTimeout(timerId); timerId = null; }}
    if (rafId !== null) {{ cancelAnimationFrame(rafId); rafId = null; }}
    renderBoard(1.0);
  }}

  function animateStepTransition() {{
    const delay = getDelay();
    const slideDuration = Math.min(180, delay * 0.65);
    const startTs = performance.now();

    function stepRaf(now) {{
      if (!isPlaying) return;
      const elapsed = now - startTs;
      const p = slideDuration > 0 ? Math.min(1.0, elapsed / slideDuration) : 1.0;
      renderBoard(p);
      if (p < 1.0) {{
        rafId = requestAnimationFrame(stepRaf);
      }} else {{
        const waitMs = Math.max(10, getDelay() - slideDuration);
        timerId = setTimeout(() => {{
          if (!isPlaying) return;
          if (currentStep < getMaxStep()) {{
            currentStep += 1;
            animateStepTransition();
          }} else {{
            stopPlayback();
          }}
        }}, waitMs);
      }}
    }}
    rafId = requestAnimationFrame(stepRaf);
  }}

  function startPlayback(fromStart = false) {{
    const maxStep = getMaxStep();
    if (maxStep === 0) return;
    if (timerId !== null) {{ clearTimeout(timerId); timerId = null; }}
    if (rafId !== null) {{ cancelAnimationFrame(rafId); rafId = null; }}

    if (fromStart || currentStep >= maxStep) {{
      currentStep = 0;
    }}
    isPlaying = true;
    renderBoard(1.0);

    // Pause briefly on step 0 so user sees initial board, then advance tile 1
    timerId = setTimeout(() => {{
      if (!isPlaying) return;
      if (currentStep < maxStep) {{
        currentStep += 1;
        animateStepTransition();
      }} else {{
        stopPlayback();
      }}
    }}, Math.min(350, getDelay()));
  }}

  function init() {{
    const hasResults = DATA.algos.length > 0 && getMaxStep() > 0;
    [btnPlayPause, btnReplay, btnReset, btnPrev, btnNext, scrubber].forEach(el => {{
      el.disabled = !hasResults;
    }});

    if (DATA.algos.length > 1) {{
      algoSelect.style.display = "inline-block";
      DATA.algos.forEach(name => {{
        const opt = document.createElement("option");
        opt.value = name;
        opt.textContent = name;
        if (name === activeAlgo) opt.selected = true;
        algoSelect.appendChild(opt);
      }});
    }}

    algoSelect.addEventListener("change", () => {{
      activeAlgo = algoSelect.value;
      startPlayback(true);
    }});

    btnPlayPause.addEventListener("click", () => {{
      if (isPlaying) stopPlayback();
      else startPlayback(false);
    }});

    btnReplay.addEventListener("click", () => {{
      startPlayback(true);
    }});

    btnReset.addEventListener("click", () => {{
      stopPlayback();
      currentStep = 0;
      renderBoard(1.0);
    }});

    btnPrev.addEventListener("click", () => {{
      stopPlayback();
      currentStep = Math.max(0, currentStep - 1);
      renderBoard(1.0);
    }});

    btnNext.addEventListener("click", () => {{
      stopPlayback();
      currentStep = Math.min(getMaxStep(), currentStep + 1);
      renderBoard(1.0);
    }});

    speedSlider.addEventListener("input", () => {{
      const idx = parseInt(speedSlider.value, 10);
      speedLabel.textContent = SPEED_LEVELS[idx].label;
      // Dynamically adjust mid-playback WITHOUT resetting!
      if (isPlaying && timerId !== null) {{
        clearTimeout(timerId);
        timerId = setTimeout(() => {{
          if (!isPlaying) return;
          if (currentStep < getMaxStep()) {{
            currentStep += 1;
            animateStepTransition();
          }} else {{
            stopPlayback();
          }}
        }}, Math.max(15, Math.round(getDelay() * 0.4)));
      }}
    }});

    const handlePuzzleScrub = () => {{
      // Read target step BEFORE stopping playback, otherwise renderBoard() resets scrubber.value!
      const targetStep = parseInt(scrubber.value, 10);
      currentStep = isNaN(targetStep) ? currentStep : targetStep;
      stopPlayback();
    }};
    scrubber.addEventListener("input", handlePuzzleScrub);
    scrubber.addEventListener("change", handlePuzzleScrub);

    if (DATA.instant && hasResults) {{
      currentStep = getMaxStep();
      renderBoard(1.0);
    }} else if (DATA.autoplay && hasResults) {{
      startPlayback(true);
    }} else {{
      renderBoard(1.0);
    }}
  }}

  init();
}})();
</script>
</body>
</html>"""
