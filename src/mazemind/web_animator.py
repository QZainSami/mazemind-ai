"""
Interactive Visualizer & Animator Components (Dracula Theme) for MazeMind AI.
Provides:
1. Serialization helpers for GridMaze and EightPuzzle SearchResults
2. Streamlit v2 bidirectional components (canvas painting, click-to-slide, hover g/h/f inspector,
   open-frontier rendering, bidirectional dual-color, weighted terrain, and 2x2 Race Mode)
3. Self-contained HTML builders (`build_maze_animator_html`, `build_puzzle_animator_html`)
"""

import json
from typing import Any, Dict, List, Optional, Tuple
import streamlit as st

from .environments import GridMaze
from .search_agents import SearchResult


def _safe_json(data: object) -> str:
    """Serialize data to JSON safely for inline script embedding."""
    return json.dumps(data).replace("</", "<\\/")


def build_maze_payload(
    maze: GridMaze,
    results: Optional[Dict[str, SearchResult]] = None,
    selected_algo: Optional[str] = None,
    autoplay: bool = False,
    instant: bool = False,
    race_mode: bool = False,
    canvas_px: int = 365,
    run_id: int = 0,
) -> Dict[str, Any]:
    """Builds a compact JSON-serializable payload for the 2D GridMaze interactive component."""
    payload_results: Dict[str, Any] = {}
    algo_order: List[str] = []

    if results:
        for name, res in results.items():
            algo_order.append(name)
            frontier_list: List[List[int]] = []
            if res.frontier_spans:
                for state, (disc, exp) in res.frontier_spans.items():
                    frontier_list.append([int(state[0]), int(state[1]), int(disc), int(exp)])

            costs_map: Dict[str, List[float]] = {}
            if res.cell_costs:
                for state, (g_v, h_v, f_v) in res.cell_costs.items():
                    costs_map[f"{int(state[0])},{int(state[1])}"] = [
                        float(g_v),
                        float(h_v),
                        float(f_v),
                    ]

            payload_results[name] = {
                "name": name,
                "found": bool(res.found),
                "explore": [[int(r), int(c)] for (r, c) in res.exploration_history],
                "sides": [int(s) for s in res.exploration_sides]
                if res.exploration_sides is not None
                else None,
                "meet": [int(res.meet_state[0]), int(res.meet_state[1])]
                if res.meet_state is not None
                else None,
                "path": [[int(r), int(c)] for (r, c) in res.path],
                "pathCost": int(res.path_cost),
                "nodesExpanded": int(res.nodes_expanded),
                "maxFrontier": int(res.max_frontier_size),
                "timeMs": round(float(res.execution_time_ms), 3),
                "frontier": frontier_list,
                "costs": costs_map,
            }

    default_algo = (
        selected_algo
        if selected_algo and selected_algo in payload_results
        else (algo_order[0] if algo_order else None)
    )

    return {
        "rows": int(maze.rows),
        "cols": int(maze.cols),
        "grid": maze.grid.tolist(),
        "weights": maze.weights.tolist(),
        "start": [int(maze.start[0]), int(maze.start[1])],
        "goal": [int(maze.goal[0]), int(maze.goal[1])],
        "algos": algo_order,
        "results": payload_results,
        "defaultAlgo": default_algo,
        "raceMode": bool(race_mode and len(algo_order) > 1),
        "autoplay": bool(autoplay and payload_results),
        "instant": bool(instant and payload_results),
        "canvasPx": int(canvas_px),
        "runId": int(run_id),
    }


def build_puzzle_payload(
    start_state: Tuple[int, ...],
    goal_state: Tuple[int, ...],
    results: Optional[Dict[str, SearchResult]] = None,
    selected_algo: Optional[str] = None,
    autoplay: bool = False,
    instant: bool = False,
    race_mode: bool = False,
    canvas_px: int = 330,
    run_id: int = 0,
) -> Dict[str, Any]:
    """Builds JSON-serializable payload for 8-Puzzle (3x3) and 15-Puzzle (4x4)."""
    dim = int(round(len(start_state) ** 0.5))
    payload_results: Dict[str, Any] = {}
    algo_order: List[str] = []

    if results:
        for name, res in results.items():
            algo_order.append(name)
            moves: List[Dict[str, Any]] = []
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

                    r0, c0 = divmod(curr_zero, dim)
                    r1, c1 = divmod(prev_zero, dim)
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
                            "desc": f"Step {idx}: Tile {moved_tile} slid {direction}",
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

    return {
        "dim": dim,
        "start": [int(x) for x in start_state],
        "goal": [int(x) for x in goal_state],
        "algos": algo_order,
        "results": payload_results,
        "defaultAlgo": default_algo,
        "raceMode": bool(race_mode and len(algo_order) > 1),
        "autoplay": bool(autoplay and payload_results),
        "instant": bool(instant and payload_results),
        "canvasPx": int(canvas_px),
        "runId": int(run_id),
    }


# =========================================================
# Shared CSS & HTML Templates for Maze & Puzzle v2 Components
# =========================================================

_MAZE_CSS = """
* { box-sizing: border-box; margin: 0; padding: 0; }
.wrap {
  display: flex; flex-direction: column; align-items: center; gap: 5px;
  width: 100%; background: #282a36; color: #f8f8f2;
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
}
.toolbar {
  display: flex; flex-wrap: wrap; align-items: center; justify-content: space-between;
  gap: 5px; width: 100%; background: #21222c; border: 1px solid #44475a;
  border-radius: 6px; padding: 4px 7px;
}
.row { display: flex; align-items: center; gap: 4px; flex-wrap: wrap; }
select {
  background: #282a36; color: #f8f8f2; border: 1px solid #6272a4;
  border-radius: 4px; padding: 2px 5px; font-size: 11.5px; font-weight: 600; outline: none;
}
.btn {
  background: #bd93f9; color: #282a36; border: none; border-radius: 4px;
  padding: 3px 8px; font-size: 11.5px; font-weight: 700; cursor: pointer;
  line-height: 1.25; transition: background 0.12s ease;
}
.btn:hover:not(:disabled) { background: #ff79c6; }
.btn:disabled { opacity: 0.42; cursor: not-allowed; }
.btn-sec { background: #44475a; color: #f8f8f2; border: 1px solid #6272a4; }
.btn-sec:hover:not(:disabled) { background: #6272a4; color: #f8f8f2; }
.btn-active { background: #8be9fd !important; color: #282a36 !important; border-color: #8be9fd !important; }
.btn-paint-active { background: #ffb86c !important; color: #282a36 !important; border-color: #f1fa8c !important; }
.speed-box { display: flex; align-items: center; gap: 4px; font-size: 11px; color: #f1fa8c; font-weight: 600; }
input[type="range"] { accent-color: #bd93f9; cursor: pointer; }
.status-bar {
  display: flex; align-items: center; justify-content: space-between; gap: 8px;
  width: 100%; background: #21222c; border: 1px solid #44475a;
  border-left: 4px solid #8be9fd; border-radius: 5px; padding: 4px 8px; font-size: 11.5px;
}
.status-main { color: #8be9fd; font-weight: 600; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.status-right { color: #50fa7b; font-size: 11px; font-weight: 600; white-space: nowrap; }
.scrubber-row { display: flex; align-items: center; gap: 8px; width: 100%; padding: 0 2px; }
.scrubber { flex: 1; height: 8px; cursor: pointer; }
.step-counter { font-size: 11px; color: #f8f8f2; min-width: 72px; text-align: right; font-variant-numeric: tabular-nums; }
.stage { position: relative; display: flex; justify-content: center; width: 100%; }
.single-canvas { background: #282a36; border: 1px solid #6272a4; border-radius: 6px; display: block; touch-action: none; }
.race-grid {
  display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 6px;
  width: 100%; max-width: 520px;
}
.race-card {
  background: #21222c; border: 1px solid #44475a; border-radius: 6px; padding: 4px;
  display: flex; flex-direction: column; align-items: center; gap: 3px;
}
.race-card.winner { border-color: #50fa7b; box-shadow: 0 0 0 1px #50fa7b55; }
.race-head {
  display: flex; justify-content: space-between; align-items: center; width: 100%;
  font-size: 11px; font-weight: 700; padding: 0 2px;
}
.race-title { color: #8be9fd; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.race-stat { color: #f1fa8c; font-size: 10.5px; font-variant-numeric: tabular-nums; }
.race-canvas { background: #282a36; border: 1px solid #44475a; border-radius: 4px; display: block; cursor: pointer; }
.legend {
  display: flex; flex-wrap: wrap; justify-content: center; gap: 8px;
  font-size: 10.5px; color: #f8f8f2; padding-top: 1px;
}
.leg-item { display: flex; align-items: center; gap: 3px; }
.swatch { width: 10px; height: 10px; border-radius: 2px; border: 1px solid #6272a4; display: inline-block; }
"""

_MAZE_HTML = """
<div class="wrap">
  <div class="toolbar">
    <div class="row">
      <select id="algoSelect" style="display:none;"></select>
      <button class="btn btn-sec" id="btnRaceToggle" style="display:none;" title="Toggle 2x2 Algorithm Race View">🏁 Race 2×2</button>
      <button class="btn" id="btnPlayPause" disabled>▶️ Play</button>
      <button class="btn btn-sec" id="btnReplay" disabled>🔁 Replay</button>
      <button class="btn btn-sec" id="btnReset" disabled>⏮️ Reset</button>
      <button class="btn btn-sec" id="btnPrev" disabled title="Step Back">◀️</button>
      <button class="btn btn-sec" id="btnNext" disabled title="Step Forward">▶️</button>
    </div>
    <div class="row">
      <button class="btn btn-sec btn-active" id="btnModeBoth" disabled>Explore+Path</button>
      <button class="btn btn-sec" id="btnModePath" disabled>Path Only</button>
      <button class="btn btn-sec btn-active" id="btnFrontier" disabled title="Highlight Open Frontier">Frontier</button>
      <div class="speed-box">
        <span>⚡</span>
        <input type="range" id="speedSlider" min="0" max="6" step="1" value="2" style="width:66px;">
        <span id="speedLabel" style="min-width:30px;">1.0x</span>
      </div>
    </div>
  </div>

  <div class="toolbar" style="padding:3px 7px;">
    <div class="row">
      <span style="font-size:11px; color:#ffb86c; font-weight:700; margin-right:2px;">🖌️ Canvas Tool:</span>
      <button class="btn btn-sec btn-paint-active" data-tool="view">🖐️ Inspect</button>
      <button class="btn btn-sec" data-tool="wall">⬛ Wall</button>
      <button class="btn btn-sec" data-tool="mud">🟫 Mud(4)</button>
      <button class="btn btn-sec" data-tool="water">🟦 Water(8)</button>
      <button class="btn btn-sec" data-tool="open">⬜ Erase</button>
      <button class="btn btn-sec" data-tool="start">🟩 Move S</button>
      <button class="btn btn-sec" data-tool="goal">🟥 Move G</button>
    </div>
  </div>

  <div class="status-bar">
    <span class="status-main" id="statusText">Ready</span>
    <span class="status-right" id="metricsText"></span>
  </div>

  <div class="scrubber-row">
    <input type="range" class="scrubber" id="scrubber" min="0" max="0" value="0" disabled>
    <span class="step-counter" id="stepCounter">0 / 0</span>
  </div>

  <div class="stage" id="stageArea">
    <canvas id="mazeCanvas" class="single-canvas"></canvas>
    <div id="raceGrid" class="race-grid" style="display:none;"></div>
  </div>

  <div class="legend">
    <span class="leg-item"><span class="swatch" style="background:#50fa7b;"></span>Start</span>
    <span class="leg-item"><span class="swatch" style="background:#ff5555;"></span>Goal</span>
    <span class="leg-item"><span class="swatch" style="background:#6e473b;"></span>Mud(4)</span>
    <span class="leg-item"><span class="swatch" style="background:#1e4d6b;"></span>Water(8)</span>
    <span class="leg-item"><span class="swatch" style="background:#bd93f9;"></span>Explored</span>
    <span class="leg-item"><span class="swatch" style="background:#8be9fd;"></span>Bwd/Frontier</span>
    <span class="leg-item"><span class="swatch" style="background:#ff79c6;"></span>Active</span>
    <span class="leg-item"><span class="swatch" style="background:#f1fa8c;"></span>Path</span>
    <span class="leg-item"><span class="swatch" style="background:#191a21;"></span>Wall</span>
  </div>
</div>
"""

_MAZE_JS = """
export default function(component) {
  const { data: DATA, setTriggerValue, parentElement } = component;
  if (!DATA) return;

  const DRACULA = {
    bg: "#282a36",
    open: "#44475a",
    mud: "#6e473b",
    water: "#1e4d6b",
    wall: "#191a21",
    border: "#6272a4",
    exploredFwd: "#bd93f9",
    exploredBwd: "#38bdf8",
    frontierRing: "#8be9fd",
    active: "#ff79c6",
    pathCell: "#f1fa8c",
    pathLine: "#ffb86c",
    agent: "#8be9fd",
    start: "#50fa7b",
    goal: "#ff5555",
    fg: "#f8f8f2"
  };

  const SPEED_LEVELS = [
    { label: "0.25x", delay: 190 },
    { label: "0.5x",  delay: 95 },
    { label: "1.0x",  delay: 42 },
    { label: "1.5x",  delay: 26 },
    { label: "2.0x",  delay: 16 },
    { label: "3.5x",  delay: 8 },
    { label: "5.0x",  delay: 4 }
  ];

  const q = (sel) => parentElement.querySelector(sel);
  const algoSelect = q("#algoSelect");
  const btnRaceToggle = q("#btnRaceToggle");
  const btnPlayPause = q("#btnPlayPause");
  const btnReplay = q("#btnReplay");
  const btnReset = q("#btnReset");
  const btnPrev = q("#btnPrev");
  const btnNext = q("#btnNext");
  const btnModeBoth = q("#btnModeBoth");
  const btnModePath = q("#btnModePath");
  const btnFrontier = q("#btnFrontier");
  const speedSlider = q("#speedSlider");
  const speedLabel = q("#speedLabel");
  const scrubber = q("#scrubber");
  const statusText = q("#statusText");
  const metricsText = q("#metricsText");
  const stepCounter = q("#stepCounter");
  const singleCanvas = q("#mazeCanvas");
  const raceGrid = q("#raceGrid");
  const toolButtons = parentElement.querySelectorAll("[data-tool]");

  // Working local copies so painting stays smooth before syncing back to Python
  let grid = DATA.grid.map(row => row.slice());
  let weights = DATA.weights.map(row => row.slice());
  let startPos = DATA.start.slice();
  let goalPos = DATA.goal.slice();

  let activeAlgo = DATA.defaultAlgo;
  let isRaceView = Boolean(DATA.raceMode);
  let playbackMode = "both";
  let showFrontier = true;
  let activeTool = "view";
  let currentStep = 0;
  let isPlaying = false;
  let timerId = null;
  let hoverCell = null;
  let isPointerDown = false;
  let dirtyEdit = false;

  const raceAlgList = DATA.algos.slice(0, 4);
  const raceCanvases = [];

  function getActiveResult() {
    if (!activeAlgo || !DATA.results[activeAlgo]) return null;
    return DATA.results[activeAlgo];
  }

  function getFramesForResult(res) {
    if (!res) return 0;
    const eLen = res.explore.length;
    const pLen = res.found ? res.path.length : 0;
    return playbackMode === "path" ? pLen : eLen + pLen;
  }

  function getTotalFrames() {
    if (isRaceView && raceAlgList.length > 1) {
      let mx = 0;
      raceAlgList.forEach(n => { mx = Math.max(mx, getFramesForResult(DATA.results[n])); });
      return mx;
    }
    return getFramesForResult(getActiveResult());
  }

  function drawGridInstance(canvasEl, pxSize, res, stepForRes, showInspectHighlight) {
    const dpr = window.devicePixelRatio || 1;
    if (canvasEl.width !== pxSize * dpr) {
      canvasEl.width = pxSize * dpr;
      canvasEl.height = pxSize * dpr;
      canvasEl.style.width = pxSize + "px";
      canvasEl.style.height = pxSize + "px";
    }
    const ctx = canvasEl.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

    const rows = DATA.rows;
    const cols = DATA.cols;
    const [sr, sc] = startPos;
    const [gr, gc] = goalPos;
    const pad = 3;
    const cell = Math.min((pxSize - pad * 2) / cols, (pxSize - pad * 2) / rows);
    const ox = (pxSize - cell * cols) / 2;
    const oy = (pxSize - cell * rows) / 2;

    ctx.fillStyle = DRACULA.bg;
    ctx.fillRect(0, 0, pxSize, pxSize);

    const total = getFramesForResult(res);
    const step = Math.max(0, Math.min(total, stepForRes));

    let exploreCount = 0;
    let pathCount = 0;
    let activeTile = null;
    let pathHead = null;

    if (res) {
      const eLen = res.explore.length;
      if (playbackMode === "both") {
        exploreCount = Math.min(step, eLen);
        if (step >= 1 && step <= eLen) activeTile = res.explore[step - 1];
        if (res.found && step > eLen) {
          pathCount = Math.min(res.path.length, step - eLen);
          if (pathCount > 0) pathHead = res.path[pathCount - 1];
        }
      } else if (res.found && step > 0) {
        pathCount = Math.min(res.path.length, step);
        pathHead = res.path[pathCount - 1];
      }
    }

    const exploredSideMap = new Map();
    if (res && exploreCount > 0) {
      for (let i = 0; i < exploreCount; i++) {
        const key = res.explore[i][0] + "," + res.explore[i][1];
        const side = res.sides ? res.sides[i] : 0;
        exploredSideMap.set(key, side);
      }
    }

    const frontierSet = new Set();
    if (res && showFrontier && playbackMode === "both" && exploreCount < res.explore.length && res.frontier) {
      for (let i = 0; i < res.frontier.length; i++) {
        const [fr, fc, disc, exp] = res.frontier[i];
        if (disc <= exploreCount && exploreCount < exp) {
          frontierSet.add(fr + "," + fc);
        }
      }
    }

    const pathSet = new Set();
    if (res && pathCount > 0) {
      for (let i = 0; i < pathCount; i++) {
        pathSet.add(res.path[i][0] + "," + res.path[i][1]);
      }
    }

    for (let r = 0; r < rows; r++) {
      for (let c = 0; c < cols; c++) {
        const x = ox + c * cell;
        const y = oy + r * cell;
        const key = r + "," + c;
        const w = weights[r][c];
        let fill = DRACULA.open;
        if (grid[r][c] === 1) fill = DRACULA.wall;
        else if (w >= 8) fill = DRACULA.water;
        else if (w >= 4) fill = DRACULA.mud;

        if (exploredSideMap.has(key) && grid[r][c] === 0) {
          fill = exploredSideMap.get(key) === 1 ? DRACULA.exploredBwd : DRACULA.exploredFwd;
        }
        if (pathSet.has(key)) fill = DRACULA.pathCell;
        if (activeTile && activeTile[0] === r && activeTile[1] === c) fill = DRACULA.active;
        if (r === sr && c === sc) fill = DRACULA.start;
        if (r === gr && c === gc) fill = DRACULA.goal;

        ctx.fillStyle = fill;
        ctx.fillRect(x, y, cell, cell);

        if (frontierSet.has(key) && grid[r][c] === 0 && !pathSet.has(key)) {
          ctx.strokeStyle = DRACULA.frontierRing;
          ctx.lineWidth = Math.max(1.2, cell * 0.14);
          ctx.strokeRect(x + 1, y + 1, Math.max(1, cell - 2), Math.max(1, cell - 2));
        } else {
          ctx.strokeStyle = DRACULA.border;
          ctx.lineWidth = Math.max(0.35, Math.min(0.9, cell * 0.04));
          ctx.strokeRect(x, y, cell, cell);
        }
      }
    }

    // Highlight meet point when bidirectional frontiers collide
    if (res && res.meet && playbackMode === "both" && exploreCount >= res.explore.length) {
      const [mr, mc] = res.meet;
      ctx.strokeStyle = DRACULA.active;
      ctx.lineWidth = Math.max(2, cell * 0.18);
      ctx.strokeRect(ox + mc * cell + 1, oy + mr * cell + 1, cell - 2, cell - 2);
    }

    if (activeTile) {
      ctx.strokeStyle = DRACULA.pathCell;
      ctx.lineWidth = Math.max(1.5, cell * 0.14);
      ctx.strokeRect(ox + activeTile[1] * cell + 1, oy + activeTile[0] * cell + 1, cell - 2, cell - 2);
    }

    if (res && pathCount > 1) {
      ctx.beginPath();
      for (let i = 0; i < pathCount; i++) {
        const [pr, pc] = res.path[i];
        const cx = ox + (pc + 0.5) * cell;
        const cy = oy + (pr + 0.5) * cell;
        if (i === 0) ctx.moveTo(cx, cy);
        else ctx.lineTo(cx, cy);
      }
      ctx.strokeStyle = DRACULA.pathLine;
      ctx.lineWidth = Math.max(1.8, cell * 0.22);
      ctx.lineCap = "round";
      ctx.lineJoin = "round";
      ctx.stroke();
    }

    if (pathHead && step < total) {
      ctx.beginPath();
      ctx.arc(ox + (pathHead[1] + 0.5) * cell, oy + (pathHead[0] + 0.5) * cell, Math.max(2.5, cell * 0.28), 0, Math.PI * 2);
      ctx.fillStyle = DRACULA.agent;
      ctx.fill();
    }

    // Hover inspector highlight
    if (showInspectHighlight && hoverCell) {
      const [hr, hc] = hoverCell;
      ctx.strokeStyle = "#ffffff";
      ctx.lineWidth = 2;
      ctx.strokeRect(ox + hc * cell + 1, oy + hr * cell + 1, Math.max(2, cell - 2), Math.max(2, cell - 2));
    }

    const fontPx = Math.max(8, Math.floor(cell * 0.56));
    ctx.font = "bold " + fontPx + "px sans-serif";
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillStyle = DRACULA.bg;
    ctx.fillText("S", ox + (sc + 0.5) * cell, oy + (sr + 0.5) * cell);
    ctx.fillText("G", ox + (gc + 0.5) * cell, oy + (gr + 0.5) * cell);
  }

  function updateStatusBar() {
    const total = getTotalFrames();
    const res = getActiveResult();
    scrubber.max = total;
    scrubber.value = currentStep;
    stepCounter.textContent = currentStep + " / " + total;

    // Hover inspector overrides right metrics pill cleanly when hovering in single view
    if (!isRaceView && hoverCell) {
      const [hr, hc] = hoverCell;
      const isWall = grid[hr][hc] === 1;
      const w = weights[hr][hc];
      const tLabel = isWall ? "Wall" : (w >= 8 ? "Water(8)" : (w >= 4 ? "Mud(4)" : "Open(1)"));
      let extra = "";
      if (res && res.costs && res.costs[hr + "," + hc]) {
        const [g, h, f] = res.costs[hr + "," + hc];
        extra = " | g=" + g + ", h=" + h + ", f=" + f;
      }
      metricsText.textContent = "🔎 (" + hr + "," + hc + ") " + tLabel + extra;
      metricsText.style.color = DRACULA.agent;
    } else if (isRaceView && raceAlgList.length > 1) {
      metricsText.style.color = DRACULA.start;
      const done = raceAlgList.filter(n => DATA.results[n] && DATA.results[n].found && currentStep >= getFramesForResult(DATA.results[n]));
      metricsText.textContent = done.length > 0 ? ("Finished: " + done.length + " / " + raceAlgList.length) : "Racing " + raceAlgList.length + " solvers...";
    } else if (res) {
      metricsText.style.color = DRACULA.start;
      metricsText.textContent = res.found ? ("Cost: " + res.pathCost + " | Explored: " + res.nodesExpanded + " | " + res.timeMs + "ms") : ("Unsolvable | Explored: " + res.nodesExpanded);
    } else {
      metricsText.textContent = "Paint walls/terrain or click Solve Maze";
      metricsText.style.color = DRACULA.border;
    }

    if (activeTool !== "view") {
      statusText.textContent = "🖌️ Paint Tool Active (" + activeTool.toUpperCase() + ") — Click/drag on grid, then click Solve Maze";
      statusText.style.color = "#ffb86c";
    } else if (!res && !isRaceView) {
      statusText.textContent = "Ready (" + DATA.rows + "×" + DATA.cols + " Solvable Grid) — Hover cells or Paint";
      statusText.style.color = DRACULA.agent;
    } else if (isRaceView) {
      if (currentStep === 0) statusText.textContent = "🏁 2×2 Race Ready — Press Play!";
      else if (currentStep >= total) statusText.textContent = "🏆 Race Complete! Compare paths & explored area";
      else statusText.textContent = "🏁 Synchronized Race Step " + currentStep + " / " + total;
      statusText.style.color = currentStep >= total ? DRACULA.start : DRACULA.pathCell;
    } else if (!res.found) {
      statusText.textContent = "❌ " + res.name + ": No path exists";
      statusText.style.color = DRACULA.goal;
    } else if (currentStep === 0) {
      statusText.textContent = "⏮️ " + res.name + ": Start State";
      statusText.style.color = DRACULA.agent;
    } else if (currentStep >= total) {
      statusText.textContent = "✅ " + res.name + ": Complete! Optimal Cost = " + res.pathCost;
      statusText.style.color = DRACULA.start;
    } else if (playbackMode === "both" && currentStep <= res.explore.length) {
      const [er, ec] = res.explore[currentStep - 1];
      statusText.textContent = "🔍 Exploring " + currentStep + "/" + res.explore.length + " → (" + er + ", " + ec + ")";
      statusText.style.color = DRACULA.active;
    } else {
      const pIdx = playbackMode === "both" ? currentStep - res.explore.length : currentStep;
      const [pr, pc] = res.path[Math.max(0, Math.min(res.path.length - 1, pIdx - 1))];
      statusText.textContent = "✨ Path step " + pIdx + "/" + res.path.length + " → (" + pr + ", " + pc + ")";
      statusText.style.color = DRACULA.pathCell;
    }

    btnPlayPause.textContent = isPlaying ? "⏸️ Pause" : "▶️ Play";
  }

  function renderFrame() {
    const total = getTotalFrames();
    currentStep = Math.max(0, Math.min(total, currentStep));

    if (isRaceView && raceAlgList.length > 1) {
      singleCanvas.style.display = "none";
      raceGrid.style.display = "grid";

      // Rank finishers by fewest explore+path frames
      let minBestFrames = Infinity;
      let bestCost = Infinity;
      raceAlgList.forEach(n => {
        const r = DATA.results[n];
        if (r && r.found) {
          minBestFrames = Math.min(minBestFrames, getFramesForResult(r));
          bestCost = Math.min(bestCost, r.pathCost);
        }
      });

      raceCanvases.forEach(({ name, card, canvas, statEl }) => {
        const r = DATA.results[name];
        const fTotal = getFramesForResult(r);
        const stepForR = Math.min(currentStep, fTotal);
        drawGridInstance(canvas, 174, r, stepForR, false);
        const finished = r && r.found && stepForR >= fTotal;
        const isFirst = finished && fTotal === minBestFrames;
        card.classList.toggle("winner", isFirst);
        if (!r || !r.found) statEl.textContent = "No path";
        else if (finished) statEl.textContent = (isFirst ? "🏆 " : "✅ ") + "Cost:" + r.pathCost + " | " + r.nodesExpanded + "n";
        else statEl.textContent = stepForR + "/" + fTotal;
      });
    } else {
      raceGrid.style.display = "none";
      singleCanvas.style.display = "block";
      drawGridInstance(singleCanvas, DATA.canvasPx, getActiveResult(), currentStep, true);
    }
    updateStatusBar();
  }

  function stopPlayback() {
    isPlaying = false;
    if (timerId !== null) { clearTimeout(timerId); timerId = null; }
    renderFrame();
  }

  function getDelay() {
    const idx = parseInt(speedSlider.value, 10);
    const base = SPEED_LEVELS[idx].delay;
    const res = getActiveResult();
    if (!isRaceView && res && (playbackMode === "path" || currentStep >= res.explore.length)) {
      return Math.max(base, Math.min(200, Math.round(base * 1.3)));
    }
    return base;
  }

  function scheduleNextTick() {
    if (!isPlaying) return;
    const total = getTotalFrames();
    if (currentStep >= total) { stopPlayback(); return; }
    timerId = setTimeout(() => {
      if (!isPlaying) return;
      currentStep += 1;
      renderFrame();
      if (currentStep < getTotalFrames()) scheduleNextTick();
      else stopPlayback();
    }, getDelay());
  }

  function startPlayback(fromStart = false) {
    const total = getTotalFrames();
    if (total === 0) return;
    if (timerId !== null) { clearTimeout(timerId); timerId = null; }
    if (fromStart || currentStep >= total) currentStep = 0;
    isPlaying = true;
    renderFrame();
    scheduleNextTick();
  }

  // Paint & Hover Handlers on singleCanvas
  function eventToCell(e) {
    const rect = singleCanvas.getBoundingClientRect();
    const px = e.clientX - rect.left;
    const py = e.clientY - rect.top;
    const rows = DATA.rows, cols = DATA.cols;
    const pad = 3;
    const cell = Math.min((DATA.canvasPx - pad * 2) / cols, (DATA.canvasPx - pad * 2) / rows);
    const ox = (DATA.canvasPx - cell * cols) / 2;
    const oy = (DATA.canvasPx - cell * rows) / 2;
    const c = Math.floor((px - ox) / cell);
    const r = Math.floor((py - oy) / cell);
    if (r >= 0 && r < rows && c >= 0 && c < cols) return [r, c];
    return null;
  }

  function applyPaintAt(cell) {
    if (!cell || activeTool === "view") return;
    const [r, c] = cell;
    const isStart = r === startPos[0] && c === startPos[1];
    const isGoal = r === goalPos[0] && c === goalPos[1];

    if (activeTool === "start") {
      if (!isGoal) {
        startPos = [r, c];
        grid[r][c] = 0;
        weights[r][c] = 1;
        dirtyEdit = true;
      }
    } else if (activeTool === "goal") {
      if (!isStart) {
        goalPos = [r, c];
        grid[r][c] = 0;
        weights[r][c] = 1;
        dirtyEdit = true;
      }
    } else if (!isStart && !isGoal) {
      if (activeTool === "wall") { grid[r][c] = 1; weights[r][c] = 1; }
      else if (activeTool === "mud") { grid[r][c] = 0; weights[r][c] = 4; }
      else if (activeTool === "water") { grid[r][c] = 0; weights[r][c] = 8; }
      else if (activeTool === "open") { grid[r][c] = 0; weights[r][c] = 1; }
      dirtyEdit = true;
    }
    renderFrame();
  }

  const onPointerDown = (e) => {
    if (activeTool === "view") return;
    stopPlayback();
    isPointerDown = true;
    applyPaintAt(eventToCell(e));
  };
  const onPointerMove = (e) => {
    const cell = eventToCell(e);
    hoverCell = cell;
    if (isPointerDown && activeTool !== "view") {
      applyPaintAt(cell);
    } else {
      renderFrame();
    }
  };
  const onPointerUp = () => {
    if (isPointerDown && dirtyEdit) {
      dirtyEdit = false;
      isPointerDown = false;
      setTriggerValue("maze_edit", {
        grid: grid,
        weights: weights,
        start: startPos,
        goal: goalPos
      });
    }
    isPointerDown = false;
  };
  const onPointerLeave = () => {
    hoverCell = null;
    onPointerUp();
    renderFrame();
  };

  singleCanvas.addEventListener("pointerdown", onPointerDown);
  singleCanvas.addEventListener("pointermove", onPointerMove);
  singleCanvas.addEventListener("pointerup", onPointerUp);
  singleCanvas.addEventListener("pointerleave", onPointerLeave);

  // Build 2x2 Race Mini-Canvases
  raceGrid.innerHTML = "";
  if (raceAlgList.length > 1) {
    btnRaceToggle.style.display = "inline-block";
    btnRaceToggle.classList.toggle("btn-active", isRaceView);
    raceAlgList.forEach(name => {
      const card = document.createElement("div");
      card.className = "race-card";
      const head = document.createElement("div");
      head.className = "race-head";
      const title = document.createElement("span");
      title.className = "race-title";
      title.textContent = name;
      const statEl = document.createElement("span");
      statEl.className = "race-stat";
      head.appendChild(title);
      head.appendChild(statEl);

      const miniCanvas = document.createElement("canvas");
      miniCanvas.className = "race-canvas";
      miniCanvas.title = "Click to zoom into " + name;
      miniCanvas.onclick = () => {
        activeAlgo = name;
        algoSelect.value = name;
        isRaceView = false;
        btnRaceToggle.classList.remove("btn-active");
        algoSelect.style.display = "inline-block";
        renderFrame();
      };

      card.appendChild(head);
      card.appendChild(miniCanvas);
      raceGrid.appendChild(card);
      raceCanvases.push({ name, card, canvas: miniCanvas, statEl });
    });
  }

  const hasResults = DATA.algos.length > 0;
  [btnPlayPause, btnReplay, btnReset, btnPrev, btnNext, btnModeBoth, btnModePath, btnFrontier, scrubber].forEach(el => {
    el.disabled = !hasResults;
  });

  algoSelect.innerHTML = "";
  if (DATA.algos.length > 1) {
    algoSelect.style.display = isRaceView ? "none" : "inline-block";
    DATA.algos.forEach(name => {
      const opt = document.createElement("option");
      opt.value = name;
      opt.textContent = name;
      if (name === activeAlgo) opt.selected = true;
      algoSelect.appendChild(opt);
    });
  }

  algoSelect.onchange = () => { activeAlgo = algoSelect.value; startPlayback(true); };
  btnRaceToggle.onclick = () => {
    isRaceView = !isRaceView;
    btnRaceToggle.classList.toggle("btn-active", isRaceView);
    algoSelect.style.display = isRaceView ? "none" : "inline-block";
    if (isRaceView) {
      activeTool = "view";
      toolButtons.forEach(b => b.classList.toggle("btn-paint-active", b.dataset.tool === "view"));
    }
    renderFrame();
  };

  btnPlayPause.onclick = () => { if (isPlaying) stopPlayback(); else startPlayback(false); };
  btnReplay.onclick = () => startPlayback(true);
  btnReset.onclick = () => { stopPlayback(); currentStep = 0; renderFrame(); };
  btnPrev.onclick = () => { stopPlayback(); currentStep = Math.max(0, currentStep - 1); renderFrame(); };
  btnNext.onclick = () => { stopPlayback(); currentStep = Math.min(getTotalFrames(), currentStep + 1); renderFrame(); };

  btnModeBoth.onclick = () => {
    playbackMode = "both";
    btnModeBoth.classList.add("btn-active");
    btnModePath.classList.remove("btn-active");
    startPlayback(true);
  };
  btnModePath.onclick = () => {
    playbackMode = "path";
    btnModePath.classList.add("btn-active");
    btnModeBoth.classList.remove("btn-active");
    startPlayback(true);
  };
  btnFrontier.onclick = () => {
    showFrontier = !showFrontier;
    btnFrontier.classList.toggle("btn-active", showFrontier);
    renderFrame();
  };

  toolButtons.forEach(btn => {
    btn.onclick = () => {
      activeTool = btn.dataset.tool;
      toolButtons.forEach(b => b.classList.toggle("btn-paint-active", b === btn));
      if (activeTool !== "view" && isRaceView) {
        isRaceView = false;
        btnRaceToggle.classList.remove("btn-active");
        if (DATA.algos.length > 1) algoSelect.style.display = "inline-block";
      }
      singleCanvas.style.cursor = activeTool === "view" ? "crosshair" : "cell";
      renderFrame();
    };
  });

  speedSlider.oninput = () => {
    const idx = parseInt(speedSlider.value, 10);
    speedLabel.textContent = SPEED_LEVELS[idx].label;
    if (isPlaying) {
      if (timerId !== null) clearTimeout(timerId);
      scheduleNextTick();
    }
  };

  const handleScrub = () => {
    const targetStep = parseInt(scrubber.value, 10);
    currentStep = isNaN(targetStep) ? currentStep : targetStep;
    stopPlayback();
  };
  scrubber.oninput = handleScrub;
  scrubber.onchange = handleScrub;

  if (DATA.instant && hasResults) {
    currentStep = getTotalFrames();
    renderFrame();
  } else if (DATA.autoplay && hasResults) {
    startPlayback(true);
  } else {
    renderFrame();
  }

  return () => {
    if (timerId !== null) clearTimeout(timerId);
    singleCanvas.removeEventListener("pointerdown", onPointerDown);
    singleCanvas.removeEventListener("pointermove", onPointerMove);
    singleCanvas.removeEventListener("pointerup", onPointerUp);
    singleCanvas.removeEventListener("pointerleave", onPointerLeave);
  };
}
"""

_PUZZLE_CSS = """
* { box-sizing: border-box; margin: 0; padding: 0; }
.wrap {
  display: flex; flex-direction: column; align-items: center; gap: 6px;
  width: 100%; background: #282a36; color: #f8f8f2;
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
}
.toolbar {
  display: flex; flex-wrap: wrap; align-items: center; justify-content: space-between;
  gap: 5px; width: 100%; background: #21222c; border: 1px solid #44475a;
  border-radius: 6px; padding: 4px 8px;
}
.row { display: flex; align-items: center; gap: 4px; flex-wrap: wrap; }
select {
  background: #282a36; color: #f8f8f2; border: 1px solid #6272a4;
  border-radius: 4px; padding: 2px 6px; font-size: 11.5px; font-weight: 600; outline: none;
}
.btn {
  background: #bd93f9; color: #282a36; border: none; border-radius: 4px;
  padding: 4px 9px; font-size: 11.5px; font-weight: 700; cursor: pointer; line-height: 1.2;
}
.btn:hover:not(:disabled) { background: #ff79c6; }
.btn:disabled { opacity: 0.45; cursor: not-allowed; }
.btn-sec { background: #44475a; color: #f8f8f2; border: 1px solid #6272a4; }
.btn-sec:hover:not(:disabled) { background: #6272a4; color: #f8f8f2; }
.btn-active { background: #8be9fd !important; color: #282a36 !important; border-color: #8be9fd !important; }
.speed-box { display: flex; align-items: center; gap: 4px; font-size: 11.5px; color: #f1fa8c; font-weight: 600; }
input[type="range"] { accent-color: #bd93f9; cursor: pointer; }
.status-bar {
  display: flex; align-items: center; justify-content: space-between; gap: 8px;
  width: 100%; background: #21222c; border: 1px solid #44475a;
  border-left: 4px solid #ff79c6; border-radius: 5px; padding: 4px 9px; font-size: 11.5px;
}
.action-text { color: #ff79c6; font-weight: 700; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.metrics-text { color: #50fa7b; font-size: 11px; font-weight: 600; white-space: nowrap; }
.scrubber-row { display: flex; align-items: center; gap: 8px; width: 100%; padding: 0 2px; }
.scrubber { flex: 1; height: 8px; cursor: pointer; }
.step-counter { font-size: 11.5px; color: #f8f8f2; min-width: 68px; text-align: right; font-variant-numeric: tabular-nums; }
.single-canvas { background: #282a36; border: 1px solid #6272a4; border-radius: 8px; display: block; cursor: pointer; }
.race-grid {
  display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 6px;
  width: 100%; max-width: 480px;
}
.race-card {
  background: #21222c; border: 1px solid #44475a; border-radius: 6px; padding: 4px;
  display: flex; flex-direction: column; align-items: center; gap: 3px;
}
.race-card.winner { border-color: #50fa7b; box-shadow: 0 0 0 1px #50fa7b55; }
.race-head { display: flex; justify-content: space-between; width: 100%; font-size: 11px; font-weight: 700; }
.race-title { color: #8be9fd; }
.race-stat { color: #f1fa8c; font-size: 10.5px; }
.race-canvas { background: #282a36; border: 1px solid #44475a; border-radius: 5px; display: block; cursor: pointer; }
"""

_PUZZLE_HTML = """
<div class="wrap">
  <div class="toolbar">
    <div class="row">
      <select id="algoSelect" style="display:none;"></select>
      <button class="btn btn-sec" id="btnRaceToggle" style="display:none;">🏁 Race 2×2</button>
      <button class="btn" id="btnPlayPause" disabled>▶️ Play</button>
      <button class="btn btn-sec" id="btnReplay" disabled>🔁 Replay</button>
      <button class="btn btn-sec" id="btnReset" disabled>⏮️ Reset</button>
      <button class="btn btn-sec" id="btnPrev" disabled>◀️</button>
      <button class="btn btn-sec" id="btnNext" disabled>▶️</button>
    </div>
    <div class="speed-box">
      <span>⚡ Speed:</span>
      <input type="range" id="speedSlider" min="0" max="5" step="1" value="2" style="width:74px;">
      <span id="speedLabel" style="min-width:34px;">1.0x</span>
    </div>
  </div>

  <div class="status-bar">
    <span class="action-text" id="actionText">Click adjacent tiles to slide manually or click Solve!</span>
    <span class="metrics-text" id="metricsText"></span>
  </div>

  <div class="scrubber-row">
    <input type="range" class="scrubber" id="scrubber" min="0" max="0" value="0" disabled>
    <span class="step-counter" id="stepCounter">0 / 0</span>
  </div>

  <canvas id="puzzleCanvas" class="single-canvas"></canvas>
  <div id="raceGrid" class="race-grid" style="display:none;"></div>
</div>
"""

_PUZZLE_JS = """
export default function(component) {
  const { data: DATA, setTriggerValue, parentElement } = component;
  if (!DATA) return;

  const DRACULA = {
    bg: "#282a36",
    tile: "#44475a",
    blank: "#191a21",
    border: "#bd93f9",
    comment: "#6272a4",
    pink: "#ff79c6",
    yellow: "#f1fa8c",
    cyan: "#8be9fd",
    green: "#50fa7b",
    red: "#ff5555"
  };

  const SPEED_LEVELS = [
    { label: "0.25x", delay: 1100 },
    { label: "0.5x",  delay: 650 },
    { label: "1.0x",  delay: 360 },
    { label: "1.5x",  delay: 220 },
    { label: "2.0x",  delay: 130 },
    { label: "4.0x",  delay: 60 }
  ];

  const q = (sel) => parentElement.querySelector(sel);
  const algoSelect = q("#algoSelect");
  const btnRaceToggle = q("#btnRaceToggle");
  const btnPlayPause = q("#btnPlayPause");
  const btnReplay = q("#btnReplay");
  const btnReset = q("#btnReset");
  const btnPrev = q("#btnPrev");
  const btnNext = q("#btnNext");
  const speedSlider = q("#speedSlider");
  const speedLabel = q("#speedLabel");
  const scrubber = q("#scrubber");
  const actionText = q("#actionText");
  const metricsText = q("#metricsText");
  const stepCounter = q("#stepCounter");
  const singleCanvas = q("#puzzleCanvas");
  const raceGrid = q("#raceGrid");

  const dim = DATA.dim || 3;
  let activeAlgo = DATA.defaultAlgo;
  let isRaceView = Boolean(DATA.raceMode);
  let currentStep = 0;
  let isPlaying = false;
  let timerId = null;
  let rafId = null;

  const raceAlgList = DATA.algos.slice(0, 4);
  const raceCanvases = [];

  function getActiveResult() {
    if (!activeAlgo || !DATA.results[activeAlgo]) return null;
    return DATA.results[activeAlgo];
  }

  function getMaxForResult(res) {
    if (!res || !res.found || !res.moves || res.moves.length === 0) return 0;
    return res.moves.length - 1;
  }

  function getMaxStep() {
    if (isRaceView && raceAlgList.length > 1) {
      let mx = 0;
      raceAlgList.forEach(n => { mx = Math.max(mx, getMaxForResult(DATA.results[n])); });
      return mx;
    }
    return getMaxForResult(getActiveResult());
  }

  function getDelay() {
    return SPEED_LEVELS[parseInt(speedSlider.value, 10)].delay;
  }

  function drawRoundedRect(ctx, x, y, w, h, r) {
    ctx.beginPath();
    ctx.moveTo(x + r, y);
    ctx.arcTo(x + w, y, x + w, y + h, r);
    ctx.arcTo(x + w, y + h, x, y + h, r);
    ctx.arcTo(x, y + h, x, y, r);
    ctx.arcTo(x, y, x + w, y, r);
    ctx.closePath();
  }

  function drawBoardInstance(canvasEl, pxSize, res, stepForRes, slideProgress) {
    const dpr = window.devicePixelRatio || 1;
    if (canvasEl.width !== pxSize * dpr) {
      canvasEl.width = pxSize * dpr;
      canvasEl.height = pxSize * dpr;
      canvasEl.style.width = pxSize + "px";
      canvasEl.style.height = pxSize + "px";
    }
    const ctx = canvasEl.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

    const pad = Math.max(6, Math.round(pxSize * 0.03));
    const cell = (pxSize - pad * 2) / dim;
    const inset = Math.max(3, Math.round(cell * 0.06));
    const tileW = cell - inset * 2;

    ctx.fillStyle = DRACULA.bg;
    ctx.fillRect(0, 0, pxSize, pxSize);

    const maxS = getMaxForResult(res);
    const step = Math.max(0, Math.min(maxS, stepForRes));
    let board = DATA.start;
    let movedTile = null, fromIdx = null, toIdx = null;
    if (res && res.found && res.moves.length > 0) {
      const mv = res.moves[step];
      board = mv.board;
      movedTile = mv.movedTile;
      fromIdx = mv.fromIdx;
      toIdx = mv.toIdx;
    }

    const totalCells = dim * dim;
    for (let idx = 0; idx < totalCells; idx++) {
      const r = Math.floor(idx / dim), c = idx % dim;
      const x = pad + c * cell + inset, y = pad + r * cell + inset;
      if (board[idx] === 0 || (movedTile !== null && idx === toIdx && slideProgress < 1.0)) {
        drawRoundedRect(ctx, x, y, tileW, tileW, 8);
        ctx.fillStyle = DRACULA.blank;
        ctx.fill();
        ctx.setLineDash([4, 3]);
        ctx.strokeStyle = DRACULA.comment;
        ctx.lineWidth = 1.6;
        ctx.stroke();
        ctx.setLineDash([]);
      }
    }

    for (let idx = 0; idx < totalCells; idx++) {
      const val = board[idx];
      if (val === 0) continue;
      let r = Math.floor(idx / dim), c = idx % dim;
      const isMoved = movedTile !== null && val === movedTile;
      if (isMoved && fromIdx !== null && toIdx !== null && slideProgress < 1.0) {
        const r0 = Math.floor(fromIdx / dim), c0 = fromIdx % dim;
        const r1 = Math.floor(toIdx / dim), c1 = toIdx % dim;
        const t = 1 - Math.pow(1 - slideProgress, 2);
        r = r0 + (r1 - r0) * t;
        c = c0 + (c1 - c0) * t;
      }

      const x = pad + c * cell + inset, y = pad + r * cell + inset;
      const isGoalTile = DATA.goal[idx] === val && (!isMoved || slideProgress >= 1.0);
      drawRoundedRect(ctx, x, y, tileW, tileW, 8);
      ctx.fillStyle = isMoved ? DRACULA.pink : DRACULA.tile;
      ctx.fill();
      ctx.strokeStyle = isMoved ? DRACULA.yellow : (isGoalTile ? DRACULA.green : DRACULA.border);
      ctx.lineWidth = isMoved ? 3 : 1.8;
      ctx.stroke();

      ctx.fillStyle = isMoved ? DRACULA.bg : DRACULA.cyan;
      ctx.font = "bold " + Math.floor(tileW * 0.42) + "px sans-serif";
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      ctx.fillText(String(val), x + tileW / 2, y + tileW / 2 + 1);
    }
  }

  function renderBoard(slideProgress = 1.0) {
    const maxStep = getMaxStep();
    currentStep = Math.max(0, Math.min(maxStep, currentStep));

    if (isRaceView && raceAlgList.length > 1) {
      singleCanvas.style.display = "none";
      raceGrid.style.display = "grid";

      let minBest = Infinity;
      raceAlgList.forEach(n => {
        const r = DATA.results[n];
        if (r && r.found) minBest = Math.min(minBest, getMaxForResult(r));
      });

      raceCanvases.forEach(({ name, card, canvas, statEl }) => {
        const r = DATA.results[name];
        const mS = getMaxForResult(r);
        const sR = Math.min(currentStep, mS);
        drawBoardInstance(canvas, 168, r, sR, sR === currentStep ? slideProgress : 1.0);
        const done = r && r.found && sR >= mS;
        const win = done && mS === minBest;
        card.classList.toggle("winner", win);
        if (!r || !r.found) statEl.textContent = "Failed";
        else if (done) statEl.textContent = (win ? "🏆 " : "✅ ") + r.pathCost + "m | " + r.nodesExpanded + "n";
        else statEl.textContent = sR + "/" + mS;
      });
    } else {
      raceGrid.style.display = "none";
      singleCanvas.style.display = "block";
      drawBoardInstance(singleCanvas, DATA.canvasPx, getActiveResult(), currentStep, slideProgress);
    }

    scrubber.max = maxStep;
    scrubber.value = currentStep;
    stepCounter.textContent = currentStep + " / " + maxStep;

    const res = getActiveResult();
    if (isRaceView) {
      actionText.textContent = currentStep >= maxStep ? "🏆 Puzzle Race Complete!" : ("🏁 Racing Solvers — Move " + currentStep + " / " + maxStep);
      actionText.style.color = currentStep >= maxStep ? DRACULA.green : DRACULA.yellow;
      metricsText.textContent = raceAlgList.length + " solvers synced";
    } else if (!res) {
      actionText.textContent = "Click any tile next to blank slot to slide manually, or click Solve!";
      actionText.style.color = DRACULA.cyan;
      metricsText.textContent = "";
    } else if (!res.found) {
      actionText.textContent = "❌ " + res.name + ": No solution found";
      actionText.style.color = DRACULA.red;
      metricsText.textContent = "Nodes: " + res.nodesExpanded;
    } else {
      const desc = res.moves[currentStep] ? res.moves[currentStep].desc : "Ready";
      metricsText.textContent = "Moves: " + res.pathCost + " | Nodes: " + res.nodesExpanded + " | " + res.timeMs + "ms";
      if (currentStep === maxStep) {
        actionText.textContent = "🎉 Goal Reached! (" + desc + ")";
        actionText.style.color = DRACULA.green;
      } else {
        actionText.textContent = "📍 " + desc;
        actionText.style.color = DRACULA.pink;
      }
    }
    btnPlayPause.textContent = isPlaying ? "⏸️ Pause" : "▶️ Play";
  }

  function stopPlayback() {
    isPlaying = false;
    if (timerId !== null) { clearTimeout(timerId); timerId = null; }
    if (rafId !== null) { cancelAnimationFrame(rafId); rafId = null; }
    renderBoard(1.0);
  }

  function animateStepTransition() {
    const delay = getDelay();
    const slideDuration = Math.min(170, delay * 0.65);
    const startTs = performance.now();

    function stepRaf(now) {
      if (!isPlaying) return;
      const elapsed = now - startTs;
      const p = slideDuration > 0 ? Math.min(1.0, elapsed / slideDuration) : 1.0;
      renderBoard(p);
      if (p < 1.0) {
        rafId = requestAnimationFrame(stepRaf);
      } else {
        const waitMs = Math.max(10, getDelay() - slideDuration);
        timerId = setTimeout(() => {
          if (!isPlaying) return;
          if (currentStep < getMaxStep()) {
            currentStep += 1;
            animateStepTransition();
          } else {
            stopPlayback();
          }
        }, waitMs);
      }
    }
    rafId = requestAnimationFrame(stepRaf);
  }

  function startPlayback(fromStart = false) {
    const maxStep = getMaxStep();
    if (maxStep === 0) return;
    if (timerId !== null) { clearTimeout(timerId); timerId = null; }
    if (rafId !== null) { cancelAnimationFrame(rafId); rafId = null; }
    if (fromStart || currentStep >= maxStep) currentStep = 0;
    isPlaying = true;
    renderBoard(1.0);

    timerId = setTimeout(() => {
      if (!isPlaying) return;
      if (currentStep < maxStep) {
        currentStep += 1;
        animateStepTransition();
      } else {
        stopPlayback();
      }
    }, Math.min(300, getDelay()));
  }

  // Click-to-slide manual tile play when user clicks an adjacent tile on initial/unsolved board
  singleCanvas.onclick = (e) => {
    if (isPlaying || DATA.algos.length > 0) return;
    const rect = singleCanvas.getBoundingClientRect();
    const pad = Math.max(6, Math.round(DATA.canvasPx * 0.03));
    const cell = (DATA.canvasPx - pad * 2) / dim;
    const c = Math.floor((e.clientX - rect.left - pad) / cell);
    const r = Math.floor((e.clientY - rect.top - pad) / cell);
    if (r < 0 || r >= dim || c < 0 || c >= dim) return;
    const clickedIdx = r * dim + c;
    const blankIdx = DATA.start.indexOf(0);
    const br = Math.floor(blankIdx / dim), bc = blankIdx % dim;
    if (Math.abs(r - br) + Math.abs(c - bc) === 1) {
      const nextBoard = DATA.start.slice();
      nextBoard[blankIdx] = nextBoard[clickedIdx];
      nextBoard[clickedIdx] = 0;
      setTriggerValue("puzzle_edit", { board: nextBoard });
    }
  };

  raceGrid.innerHTML = "";
  if (raceAlgList.length > 1) {
    btnRaceToggle.style.display = "inline-block";
    btnRaceToggle.classList.toggle("btn-active", isRaceView);
    raceAlgList.forEach(name => {
      const card = document.createElement("div");
      card.className = "race-card";
      const head = document.createElement("div");
      head.className = "race-head";
      const title = document.createElement("span");
      title.className = "race-title";
      title.textContent = name;
      const statEl = document.createElement("span");
      statEl.className = "race-stat";
      head.appendChild(title);
      head.appendChild(statEl);
      const miniCanvas = document.createElement("canvas");
      miniCanvas.className = "race-canvas";
      miniCanvas.onclick = () => {
        activeAlgo = name;
        algoSelect.value = name;
        isRaceView = false;
        btnRaceToggle.classList.remove("btn-active");
        algoSelect.style.display = "inline-block";
        renderBoard(1.0);
      };
      card.appendChild(head);
      card.appendChild(miniCanvas);
      raceGrid.appendChild(card);
      raceCanvases.push({ name, card, canvas: miniCanvas, statEl });
    });
  }

  const hasResults = DATA.algos.length > 0 && getMaxStep() > 0;
  [btnPlayPause, btnReplay, btnReset, btnPrev, btnNext, scrubber].forEach(el => {
    el.disabled = !hasResults;
  });

  algoSelect.innerHTML = "";
  if (DATA.algos.length > 1) {
    algoSelect.style.display = isRaceView ? "none" : "inline-block";
    DATA.algos.forEach(name => {
      const opt = document.createElement("option");
      opt.value = name;
      opt.textContent = name;
      if (name === activeAlgo) opt.selected = true;
      algoSelect.appendChild(opt);
    });
  }

  algoSelect.onchange = () => { activeAlgo = algoSelect.value; startPlayback(true); };
  btnRaceToggle.onclick = () => {
    isRaceView = !isRaceView;
    btnRaceToggle.classList.toggle("btn-active", isRaceView);
    algoSelect.style.display = isRaceView ? "none" : "inline-block";
    renderBoard(1.0);
  };
  btnPlayPause.onclick = () => { if (isPlaying) stopPlayback(); else startPlayback(false); };
  btnReplay.onclick = () => startPlayback(true);
  btnReset.onclick = () => { stopPlayback(); currentStep = 0; renderBoard(1.0); };
  btnPrev.onclick = () => { stopPlayback(); currentStep = Math.max(0, currentStep - 1); renderBoard(1.0); };
  btnNext.onclick = () => { stopPlayback(); currentStep = Math.min(getMaxStep(), currentStep + 1); renderBoard(1.0); };

  speedSlider.oninput = () => {
    const idx = parseInt(speedSlider.value, 10);
    speedLabel.textContent = SPEED_LEVELS[idx].label;
    if (isPlaying && timerId !== null) {
      clearTimeout(timerId);
      timerId = setTimeout(() => {
        if (!isPlaying) return;
        if (currentStep < getMaxStep()) { currentStep += 1; animateStepTransition(); }
        else stopPlayback();
      }, Math.max(15, Math.round(getDelay() * 0.4)));
    }
  };

  const handleScrub = () => {
    const targetStep = parseInt(scrubber.value, 10);
    currentStep = isNaN(targetStep) ? currentStep : targetStep;
    stopPlayback();
  };
  scrubber.oninput = handleScrub;
  scrubber.onchange = handleScrub;

  if (DATA.instant && hasResults) {
    currentStep = getMaxStep();
    renderBoard(1.0);
  } else if (DATA.autoplay && hasResults) {
    startPlayback(true);
  } else {
    renderBoard(1.0);
  }

  return () => {
    if (timerId !== null) clearTimeout(timerId);
    if (rafId !== null) cancelAnimationFrame(rafId);
  };
}
"""


@st.cache_resource
def get_maze_component():
    """Registers and caches the bidirectional Streamlit v2 2D GridMaze component."""
    return st.components.v2.component(
        "mazemind_maze_v2",
        html=_MAZE_HTML,
        css=_MAZE_CSS,
        js=_MAZE_JS,
    )


@st.cache_resource
def get_puzzle_component():
    """Registers and caches the bidirectional Streamlit v2 Sliding Puzzle component."""
    return st.components.v2.component(
        "mazemind_puzzle_v2",
        html=_PUZZLE_HTML,
        css=_PUZZLE_CSS,
        js=_PUZZLE_JS,
    )


def build_maze_animator_html(
    maze: GridMaze,
    results: Optional[Dict[str, SearchResult]] = None,
    selected_algo: Optional[str] = None,
    autoplay: bool = False,
    instant: bool = False,
    race_mode: bool = False,
    canvas_px: int = 365,
    run_id: int = 0,
) -> str:
    """
    Standalone HTML document wrapper around the maze animator component.
    Useful for unit testing payload serialization and standalone embedding.
    """
    data_json = _safe_json(
        build_maze_payload(
            maze=maze,
            results=results,
            selected_algo=selected_algo,
            autoplay=autoplay,
            instant=instant,
            race_mode=race_mode,
            canvas_px=canvas_px,
            run_id=run_id,
        )
    )
    body_js = _MAZE_JS.replace("export default function(component) {", "function mount(component) {").rstrip()
    return f"""<!DOCTYPE html><html><head><meta charset="utf-8"><style>{_MAZE_CSS}</style></head>
<body style="background:#282a36; margin:0; padding:4px;">
<div id="root">{_MAZE_HTML}</div>
<script>
{body_js}
mount({{ data: {data_json}, setTriggerValue: () => {{}}, parentElement: document.getElementById("root") }});
</script></body></html>"""


def build_puzzle_animator_html(
    start_state: Tuple[int, ...],
    goal_state: Tuple[int, ...],
    results: Optional[Dict[str, SearchResult]] = None,
    selected_algo: Optional[str] = None,
    autoplay: bool = False,
    instant: bool = False,
    race_mode: bool = False,
    canvas_px: int = 330,
    run_id: int = 0,
) -> str:
    """
    Standalone HTML document wrapper around the sliding puzzle animator component.
    """
    data_json = _safe_json(
        build_puzzle_payload(
            start_state=start_state,
            goal_state=goal_state,
            results=results,
            selected_algo=selected_algo,
            autoplay=autoplay,
            instant=instant,
            race_mode=race_mode,
            canvas_px=canvas_px,
            run_id=run_id,
        )
    )
    body_js = _PUZZLE_JS.replace("export default function(component) {", "function mount(component) {").rstrip()
    return f"""<!DOCTYPE html><html><head><meta charset="utf-8"><style>{_PUZZLE_CSS}</style></head>
<body style="background:#282a36; margin:0; padding:4px;">
<div id="root">{_PUZZLE_HTML}</div>
<script>
{body_js}
mount({{ data: {data_json}, setTriggerValue: () => {{}}, parentElement: document.getElementById("root") }});
</script></body></html>"""
