/* ── Tree rendering (vertical orientation) ───────────────────────────────────── */
'use strict';

const Tree = (() => {
  const COL_W = 150, STEP_H = 64, NODE_W = 120, NODE_H = 44;
  const PAD_TOP = 8, PAD_LEFT = 110;
  const MMAP_W = 160, MMAP_H = 120;

  let _runName = null;
  let _records = [];
  let _inProgress = {};
  let _pendingAgents = new Set();
  let _failedAgents = {};   // agent_rank -> error string
  let _agentStatus  = {};   // agent_rank -> status string (live, in-progress only)
  let _agentLogs = {};      // agent_rank -> string[] (accumulated lines)
  let _logDialogRank = null; // which agent's log is currently shown
  let _pendingStepNum = null;
  let _pendingOpName  = null;
  let _stepFailed     = false;
  let _queue = [];          // queued operator names (not yet started)
  let _selectedUid = null;
  let _selectedAgent = null;
  let _onSelect = null;
  let _onDeleteStep = null;

  // ── Zoom / pan state ──────────────────────────────────────────────────────
  let _scale = 1;
  let _panX  = 16;
  let _panY  = 16;
  let _dragging  = false;
  let _dragLast  = null;
  let _interactionReady = false;

  // ── Build rows from records ───────────────────────────────────────────────
  function buildRows() {
    const rows = [];
    for (const rec of _records) {
      rows.push({ uid: rec.uid, opName: rec.operator_name, opType: rec.operator_type || 'base', agents: rec.post_agents, stepNum: rec.step_num });
    }
    if (_pendingAgents.size > 0 || Object.keys(_failedAgents).length > 0 || _pendingOpName !== null) {
      const prevAgents = rows.length > 0 ? rows[rows.length - 1].agents : [];
      // For the first step, prevAgents is empty; synthesise the list from
      // whatever ranks we've seen via agent_started / agent_completed / agent_failed.
      const effectiveAgents = prevAgents.length > 0 ? prevAgents : (() => {
        const seen = new Set([
          ..._pendingAgents,
          ...Object.keys(_inProgress).map(Number),
          ...Object.keys(_failedAgents).map(Number),
        ]);
        return [...seen].sort((a, b) => a - b).map(rank => ({ agent_rank: rank }));
      })();
      const inProgAgents = effectiveAgents.map(a => {
        if (_failedAgents[a.agent_rank] !== undefined)
          return { ...a, _status: 'failed', _error: _failedAgents[a.agent_rank] };
        return { ...a, _status: _inProgress[a.agent_rank] ? 'done' : 'running' };
      });
      rows.push({
        uid: null, opName: _pendingOpName || (_stepFailed ? '…failed…' : '…running…'),
        opType: '', agents: inProgAgents,
        inProgress: !_stepFailed, stepFailed: _stepFailed,
        stepNum: _pendingStepNum,
      });
    }
    // Queued rows — placeholder, agents inherited from last row
    if (_queue.length > 0) {
      const lastAgents = rows.length > 0 ? rows[rows.length - 1].agents : [];
      const queuedAgents = lastAgents.map(a => ({ ...a, _status: 'queued' }));
      for (const opName of _queue) {
        rows.push({ uid: null, opName, opType: '', agents: queuedAgents, queued: true, stepNum: null });
      }
    }
    return rows;
  }

  // ── Position helpers ──────────────────────────────────────────────────────
  function nodeX(rank)    { return PAD_LEFT + rank * COL_W + (COL_W - NODE_W) / 2; }
  function rowY(rowIdx)   { return PAD_TOP + rowIdx * STEP_H; }
  function nodeY(rowIdx)  { return rowY(rowIdx) + (STEP_H - NODE_H) / 2; }
  function nodeCx(rank)   { return PAD_LEFT + rank * COL_W + COL_W / 2; }
  function nodeCy(rowIdx) { return rowY(rowIdx) + STEP_H / 2; }

  // ── SVG helpers ───────────────────────────────────────────────────────────
  function svgEl(tag, attrs = {}, parent) {
    const el = document.createElementNS('http://www.w3.org/2000/svg', tag);
    for (const [k, v] of Object.entries(attrs)) el.setAttribute(k, v);
    if (parent) parent.appendChild(el);
    return el;
  }

  function makeNode(svg, rowIdx, agent, uid, inProgress) {
    const x  = nodeX(agent.agent_rank);
    const y  = nodeY(rowIdx);
    const cx = nodeCx(agent.agent_rank);
    const cy = nodeCy(rowIdx);

    const g = svgEl('g', { class: 'tree-node', 'data-uid': uid || '', 'data-rank': agent.agent_rank, cursor: 'pointer' }, svg);
    if (agent._status === 'failed') g.classList.add('agent-failed');
    else if (agent._status === 'running') g.classList.add('running-pending');
    else if (agent._status === 'queued') g.classList.add('queued');
    else if (inProgress && agent._status === 'done') g.classList.add('running');

    if (uid === _selectedUid && agent.agent_rank === _selectedAgent) g.classList.add('selected');
    if (uid === _selectedUid && _selectedAgent === null) g.classList.add('selected');

    const status = (agent._status === 'running' || agent._status === 'done')
      ? (_agentStatus[agent.agent_rank] || null) : null;

    svgEl('rect', { x, y, width: NODE_W, height: NODE_H, rx: 4 }, g);

    const nameY = status ? y + 11 : y + NODE_H / 2 - 6;
    svgEl('text', { x: cx, y: nameY, 'text-anchor': 'middle', 'font-size': 12, fill: '#9cdcfe' }, g)
      .textContent = `Agent ${agent.agent_rank}`;

    const uid8 = (agent.unique_id || '').slice(-8);
    const uidY = status ? y + 23 : y + NODE_H / 2 + 9;
    svgEl('text', { x: cx, y: uidY, 'text-anchor': 'middle', 'font-size': 10, fill: '#666' }, g)
      .textContent = uid8;

    if (status) {
      const maxLen = 17;
      const label  = status.length > maxLen ? status.slice(0, maxLen - 1) + '…' : status;
      svgEl('text', { x: cx, y: y + 36, 'text-anchor': 'middle', 'font-size': 9, fill: '#6a9f6a' }, g)
        .textContent = label;
    }

    // Single-click: select completed agents; no-op for in-progress/failed/queued
    g.addEventListener('click', e => {
      if (e.detail > 1) return;  // ignore clicks that are part of a dblclick
      if (inProgress || agent._status === 'failed' || agent._status === 'queued') return;
      _selectedUid = uid;
      _selectedAgent = agent.agent_rank;
      render();
      if (_onSelect) _onSelect(uid, agent.agent_rank);
    });

    // Double-click: open output log for any agent
    g.addEventListener('dblclick', e => {
      e.stopPropagation();
      _openLogDialog(agent.agent_rank);
    });

    return { cx, cy };
  }

  // ── Agent log dialog ─────────────────────────────────────────────────────
  function _openLogDialog(rank) {
    _logDialogRank = rank;
    const dlg   = document.getElementById('dlg-agent-log');
    const title = document.getElementById('dlg-agent-log-title');
    const pre   = document.getElementById('dlg-agent-log-text');
    title.textContent = `Agent ${rank} — Output`;

    const lines = _agentLogs[rank] || [];
    const err   = _failedAgents[rank];
    const stepErrAlreadyInLog = lines.some(l => l.includes('--- step failed ---'));
    if (lines.length > 0) {
      // Show captured stdout/stderr; append error only if not already written by onStepFailed
      let html = lines.map(l => _formatLogLine(l)).join('\n');
      if (err && !stepErrAlreadyInLog && err !== 'Agent did not complete (step failed).') {
        html += '\n' + err.split('\n').map(l => _formatLogLine('[stderr] ' + l)).join('\n');
      }
      pre.innerHTML = html;
    } else if (err && err !== 'Agent did not complete (step failed).') {
      // Fallback: show the error text captured in agent_failed (survives tab-switch replay)
      pre.innerHTML = err.split('\n').map(l => _formatLogLine('[stderr] ' + l)).join('\n');
    } else {
      pre.innerHTML = '<span style="color:#555;font-style:italic">No output captured for this agent.</span>';
    }

    pre.scrollTop = pre.scrollHeight;
    dlg.onclose = () => { _logDialogRank = null; };
    if (!dlg.open) dlg.showModal();
  }

  function _formatLogLine(line) {
    const esc = line.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
    if (line.startsWith('[stderr]'))  return `<span style="color:#f44747">${esc}</span>`;
    if (line.startsWith('[status]'))  return `<span style="color:#6a9f6a">${esc}</span>`;
    if (line.startsWith('[tool]'))    return `<span style="color:#ce9178">${esc}</span>`;
    if (line.startsWith('[tool result]')) return `<span style="color:#4ec9b0">${esc}</span>`;
    if (line.startsWith('[LLM call')) return `<span style="color:#569cd6">${esc}</span>`;
    return `<span style="color:#ccc">${esc}</span>`;
  }

  // ── Zoom / pan helpers ────────────────────────────────────────────────────
  function _applyTransform() {
    const svg = document.getElementById('tree-svg');
    if (!svg) return;
    svg.style.transform       = `translate(${_panX}px, ${_panY}px) scale(${_scale})`;
    svg.style.transformOrigin = '0 0';
    _updateMinimap();
  }

  function _updateMinimap() {
    const mainSvg   = document.getElementById('tree-svg');
    const container = document.getElementById('tree-svg-container');
    if (!mainSvg || !container) return;

    const svgW = parseFloat(mainSvg.getAttribute('width'))  || 0;
    const svgH = parseFloat(mainSvg.getAttribute('height')) || 0;
    if (!svgW || !svgH) return;

    // Create minimap SVG once
    let mm = document.getElementById('tree-minimap');
    if (!mm) {
      mm = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
      mm.id = 'tree-minimap';
      container.appendChild(mm);

      // Click-to-navigate
      mm.addEventListener('click', e => {
        e.stopPropagation();
        const r   = mm.getBoundingClientRect();
        const mmW = parseFloat(mm.getAttribute('width'));
        const mmH = parseFloat(mm.getAttribute('height'));
        const svW = parseFloat(mainSvg.getAttribute('width'))  || 1;
        const svH = parseFloat(mainSvg.getAttribute('height')) || 1;
        const sx  = (e.clientX - r.left) / mmW * svW;
        const sy  = (e.clientY - r.top)  / mmH * svH;
        _panX = container.clientWidth  / 2 - sx * _scale;
        _panY = container.clientHeight / 2 - sy * _scale;
        _applyTransform();
      });
    }

    mm.setAttribute('width',   MMAP_W);
    mm.setAttribute('height',  MMAP_H);
    mm.setAttribute('viewBox', `0 0 ${svgW} ${svgH}`);

    // Clone tree content (no listeners needed in minimap)
    mm.innerHTML = mainSvg.innerHTML;

    // Viewport indicator rect
    const cW  = container.clientWidth;
    const cH  = container.clientHeight;
    const vpX = -_panX / _scale;
    const vpY = -_panY / _scale;
    const vpW = cW / _scale;
    const vpH = cH / _scale;
    const strokeW = svgW / MMAP_W * 2;

    const vp = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
    vp.setAttribute('x',            vpX);
    vp.setAttribute('y',            vpY);
    vp.setAttribute('width',        Math.max(0, vpW));
    vp.setAttribute('height',       Math.max(0, vpH));
    vp.setAttribute('fill',         'rgba(86,156,214,0.12)');
    vp.setAttribute('stroke',       '#569cd6');
    vp.setAttribute('stroke-width', strokeW);
    vp.setAttribute('pointer-events', 'none');
    mm.appendChild(vp);
  }

  function _setupInteraction() {
    if (_interactionReady) return;
    _interactionReady = true;

    const container = document.getElementById('tree-svg-container');
    if (!container) return;

    // Wheel zoom
    container.addEventListener('wheel', e => {
      e.preventDefault();
      const rect   = container.getBoundingClientRect();
      const mx     = e.clientX - rect.left;
      const my     = e.clientY - rect.top;
      const factor = e.deltaY < 0 ? 1.12 : 1 / 1.12;
      const ns     = Math.max(0.15, Math.min(5, _scale * factor));
      _panX = mx - (mx - _panX) * (ns / _scale);
      _panY = my - (my - _panY) * (ns / _scale);
      _scale = ns;
      _applyTransform();
    }, { passive: false });

    // Drag pan
    container.addEventListener('mousedown', e => {
      if (e.button !== 0) return;
      // Don't start drag if clicking inside minimap
      if (e.target.closest('#tree-minimap')) return;
      _dragging = true;
      _dragLast = { x: e.clientX, y: e.clientY };
      container.classList.add('dragging');
    });

    window.addEventListener('mousemove', e => {
      if (!_dragging) return;
      _panX += e.clientX - _dragLast.x;
      _panY += e.clientY - _dragLast.y;
      _dragLast = { x: e.clientX, y: e.clientY };
      _applyTransform();
    });

    window.addEventListener('mouseup', () => {
      if (!_dragging) return;
      _dragging = false;
      _dragLast = null;
      const c = document.getElementById('tree-svg-container');
      if (c) c.classList.remove('dragging');
    });
  }

  // ── Main render ───────────────────────────────────────────────────────────
  function render(autoPanBottom = false) {
    const svg = document.getElementById('tree-svg');
    if (!svg) return;
    svg.innerHTML = '';

    const rows = buildRows();
    if (rows.length === 0) { _updateMinimap(); return; }

    const maxRank = rows.reduce((m, row) =>
      Math.max(m, ...row.agents.map(a => a.agent_rank), 0), 0);

    const W = PAD_LEFT + (maxRank + 1) * COL_W + 20;
    const H = PAD_TOP + rows.length * STEP_H + 20;
    svg.setAttribute('width',  W);
    svg.setAttribute('height', H);

    // Vertical separator
    svgEl('line', {
      x1: PAD_LEFT - 12, y1: PAD_TOP,
      x2: PAD_LEFT - 12, y2: H,
      stroke: '#333', 'stroke-width': 1,
    }, svg);

    const posMap = rows.map(() => ({}));

    // Row label buttons
    const LABEL_W    = PAD_LEFT - 12;
    const typeColors = { base: '#569cd6', fork: '#4ec9b0', kill: '#f44747', sort: '#ffd700', shuffle: '#c586c0' };

    rows.forEach((row, ri) => {
      const by  = rowY(ri);
      const bcy = by + STEP_H / 2;
      const lx  = LABEL_W / 2;
      const rowUid     = row.uid || (row.inProgress ? '__running__' : null);
      const isSelected = rowUid && _selectedUid === rowUid;
      const typeColor  = typeColors[row.opType] || '#888';

      const g = svgEl('g', { cursor: rowUid ? 'pointer' : 'default' }, svg);

      const btnFill   = isSelected ? '#1e3a5c' : row.stepFailed ? '#251515' : row.inProgress ? '#152515' : row.queued ? '#1a1a1a' : '#252526';
      const btnStroke = isSelected ? '#569cd6' : row.stepFailed ? '#f44747' : row.inProgress ? '#4ec9b0' : row.queued ? '#444' : '#3a3a3a';
      const btnDash   = (row.inProgress || row.stepFailed || row.queued) ? '4,3' : null;
      const btnRect   = svgEl('rect', {
        x: 0, y: by, width: LABEL_W, height: STEP_H,
        fill: btnFill, stroke: btnStroke, 'stroke-width': 1,
        ...(btnDash ? { 'stroke-dasharray': btnDash } : {}),
      }, g);

      svgEl('text', {
        x: lx, y: bcy - 14,
        class: 'tree-step-label', 'text-anchor': 'middle',
        style: isSelected ? 'fill:#ffd700' : row.stepFailed ? 'fill:#f44747' : row.inProgress ? 'fill:#4ec9b0' : row.queued ? 'fill:#555' : '',
      }, g).textContent = row.stepNum ? `Step ${row.stepNum}` : row.stepFailed ? 'failed' : row.inProgress ? 'running…' : row.queued ? 'queued' : '…';

      if (row.opType) {
        svgEl('text', {
          x: lx, y: bcy,
          'text-anchor': 'middle',
          style: `font-size:9px; font-weight:700; text-transform:uppercase; fill:${isSelected ? '#ffd700' : typeColor}`,
        }, g).textContent = row.opType;
      }

      svgEl('text', {
        x: lx, y: bcy + 14,
        class: 'tree-op-label', 'text-anchor': 'middle',
        style: isSelected ? 'fill:#ffd700' : '',
      }, g).textContent = row.opName.replace(/\.py$/, '').slice(0, 14);

      if (rowUid) {
        g.addEventListener('click', () => {
          _selectedUid = rowUid;
          _selectedAgent = null;
          render();
          if (_onSelect) _onSelect(rowUid, null);
        });
        g.addEventListener('mouseover', () => {
          if (_selectedUid !== rowUid) btnRect.setAttribute('fill', '#2a3a4a');
        });
        g.addEventListener('mouseout', () => {
          if (_selectedUid !== rowUid) btnRect.setAttribute('fill', row.inProgress ? '#152515' : '#252526');
        });
      }
    });

    // Horizontal dotted lines
    for (let ri = 0; ri <= rows.length; ri++) {
      svgEl('line', {
        x1: 0, y1: rowY(ri), x2: W, y2: rowY(ri),
        stroke: '#383838', 'stroke-width': 1, 'stroke-dasharray': '4,4',
      }, svg);
    }

    // Pre-compute posMap (needed before edges are drawn)
    rows.forEach((row, ri) => {
      row.agents.forEach(agent => {
        posMap[ri][agent.agent_rank] = { cx: nodeCx(agent.agent_rank), cy: nodeCy(ri) };
      });
    });

    // Draw edges (above dotted lines, below nodes)
    const EDGE_COLORS = { fork: '#4ec9b0', shuffle: '#ce9178', base: '#569cd6', kill: '#569cd6', sort: '#569cd6', default: '#569cd6' };
    for (let ri = 0; ri + 1 < rows.length; ri++) {
      const curRow      = rows[ri];
      const nextRow     = rows[ri + 1];

      // Queued rows: draw grey dashed same-rank continuation edges only
      if (nextRow.queued) {
        nextRow.agents.forEach(nextAgent => {
          const to   = posMap[ri + 1][nextAgent.agent_rank];
          const from = posMap[ri][nextAgent.agent_rank];
          if (!from || !to) return;
          svgEl('line', {
            x1: from.cx, y1: from.cy + NODE_H / 2,
            x2: to.cx,   y2: to.cy   - NODE_H / 2,
            stroke: '#3a3a3a', 'stroke-width': 1.5, 'stroke-dasharray': '4,3', fill: 'none',
          }, svg);
        });
        continue;
      }
      const opType      = nextRow.opType || 'default';
      const isForkStep    = nextRow.opType === 'fork';
      const isShuffleStep = nextRow.opType === 'shuffle';
      const consumedCurRanks = new Set();

      const drawEdge = (from, to, cls) => {
        svgEl('line', {
          x1: from.cx, y1: from.cy + NODE_H / 2,
          x2: to.cx,   y2: to.cy   - NODE_H / 2,
          stroke: EDGE_COLORS[cls] || EDGE_COLORS.default,
          'stroke-width': 2, fill: 'none',
          class: `tree-edge ${cls}`,
        }, svg);
      };

      nextRow.agents.forEach(nextAgent => {
        const to = posMap[ri + 1][nextAgent.agent_rank];
        if (!to) return;

        if (isForkStep && nextAgent.parent_id) {
          // Fork: connect each child to its parent
          const src = curRow.agents.find(a => a.unique_id === nextAgent.parent_id);
          if (!src) return;
          consumedCurRanks.add(src.agent_rank);
          const from = posMap[ri][src.agent_rank];
          if (from) drawEdge(from, to, 'fork');

        } else if (isShuffleStep && nextAgent.shuffle_sources) {
          // Shuffle: draw one orange edge from each source rank to this agent
          for (const srcRank of nextAgent.shuffle_sources) {
            const from = posMap[ri][srcRank];
            if (!from) continue;
            consumedCurRanks.add(srcRank);
            drawEdge(from, to, 'shuffle');
          }

        } else {
          // Default: match by unique_id, then fall back to same rank
          let src = (nextAgent.unique_id !== undefined)
            ? curRow.agents.find(a => a.unique_id === nextAgent.unique_id)
            : null;
          if (!src) src = curRow.agents.find(a => a.agent_rank === nextAgent.agent_rank);
          if (!src) return;
          consumedCurRanks.add(src.agent_rank);
          const from = posMap[ri][src.agent_rank];
          if (from) drawEdge(from, to, opType);
        }
      });

      // For shuffle: all agents persist, so mark all matching ranks as consumed
      if (isShuffleStep) {
        const nextRankSet = new Set(nextRow.agents.map(a => a.agent_rank));
        curRow.agents.forEach(a => {
          if (nextRankSet.has(a.agent_rank)) consumedCurRanks.add(a.agent_rank);
        });
      }

      // Dead-end (killed) agents: in curRow but not consumed by nextRow
      curRow.agents.forEach(curAgent => {
        if (consumedCurRanks.has(curAgent.agent_rank)) return;
        const from = posMap[ri][curAgent.agent_rank];
        if (!from) return;
        const x  = from.cx;
        const y1 = from.cy + NODE_H / 2;
        const y2 = y1 + 14;
        const xs = 5;
        svgEl('line', { x1: x, y1, x2: x, y2, stroke: '#f44747', 'stroke-width': 2 }, svg);
        svgEl('line', { x1: x - xs, y1: y2 - xs, x2: x + xs, y2: y2 + xs, stroke: '#f44747', 'stroke-width': 2 }, svg);
        svgEl('line', { x1: x + xs, y1: y2 - xs, x2: x - xs, y2: y2 + xs, stroke: '#f44747', 'stroke-width': 2 }, svg);
      });
    }

    // Draw nodes (above edges)
    rows.forEach((row, ri) => {
      row.agents.forEach(agent => {
        makeNode(svg, ri, agent, row.uid, row.inProgress);
      });
    });

    // Auto-pan to show bottom when new steps arrive
    if (autoPanBottom) {
      const container = document.getElementById('tree-svg-container');
      if (container) {
        const bottomInView = _panY + H * _scale;
        const containerH   = container.clientHeight;
        if (bottomInView > containerH + 10 || bottomInView < containerH - 80) {
          _panY = containerH - H * _scale - 16;
        }
      }
    }

    _applyTransform();
  }

  // ── Public API ────────────────────────────────────────────────────────────
  return {
    init(runName, records, onSelect, onDeleteStep) {
      _runName      = runName;
      _records      = [...records];  // own copy — don't share with BottomPanels
      _inProgress     = {};
      _pendingAgents  = new Set();
      _failedAgents   = {};
      _agentStatus    = {};
      _agentLogs      = {};
      _logDialogRank  = null;
      _pendingStepNum = null;
      _pendingOpName  = null;
      _stepFailed     = false;
      _queue          = [];
      _selectedUid    = null;
      _selectedAgent = null;
      _onSelect     = onSelect;
      _onDeleteStep = onDeleteStep;
      _scale = 1; _panX = 16; _panY = 16;
      _setupInteraction();
      render();
    },

    setRecords(records) { _records = records; render(); },

    onStepStarted(preAgents, stepNum, opName) {
      _stepFailed     = false;
      _queue          = [];
      _pendingAgents  = new Set(preAgents.map(a => a.agent_rank));
      _inProgress     = {};
      _failedAgents   = {};
      _agentStatus    = {};
      _agentLogs      = {};
      _pendingStepNum = stepNum || null;
      _pendingOpName  = opName  || null;
      render();
    },

    onAgentStarted(agentRank) {
      _pendingAgents.add(agentRank);
      render();
    },

    onAgentLog(agentRank, stream, text) {
      if (!_agentLogs[agentRank]) _agentLogs[agentRank] = [];
      const line = stream === 'stderr' ? `[stderr] ${text}` : text;
      _agentLogs[agentRank].push(line);
      // Live-update dialog if it's open for this agent
      if (_logDialogRank === agentRank) {
        const pre = document.getElementById('dlg-agent-log-text');
        if (pre) {
          const span = document.createElement('span');
          span.style.color = stream === 'stderr' ? '#f44747' : '#ccc';
          span.textContent = line;
          if (pre.innerHTML) pre.appendChild(document.createTextNode('\n'));
          pre.appendChild(span);
          pre.scrollTop = pre.scrollHeight;
        }
      }
    },

    onAgentStatus(agentRank, status) {
      _agentStatus[agentRank] = status;
      render();
    },

    onAgentCompleted(agentRank, state) {
      _inProgress[agentRank] = state;
      delete _agentStatus[agentRank];
      // Keep _pendingAgents populated until step_completed so the in-progress
      // row stays visible the whole time; _inProgress tracks per-agent done state
      render();
    },

    onAgentFailed(agentRank, error) {
      _failedAgents[agentRank] = error;
      delete _agentStatus[agentRank];
      // If the log dialog is open for this agent and has no streamed logs yet,
      // populate it now with the error text from agent_failed
      if (_logDialogRank === agentRank) {
        const pre = document.getElementById('dlg-agent-log-text');
        if (pre && !(_agentLogs[agentRank] || []).length && error) {
          pre.innerHTML = error.split('\n').map(l => _formatLogLine('[stderr] ' + l)).join('\n');
          pre.scrollTop = pre.scrollHeight;
        }
      }
      render();
    },

    onStepCompleted(record) {
      if (!_records.some(r => r.uid === record.uid)) _records.push(record);
      _inProgress     = {};
      _pendingAgents  = new Set();
      _failedAgents   = {};
      _agentStatus    = {};
      _stepFailed     = false;
      _pendingStepNum = null;
      _pendingOpName  = null;
      render(true);  // auto-pan to bottom
    },

    onStepFailed(error) {
      const fallback = error || 'Step failed.';
      for (const rank of _pendingAgents) {
        // Set error for any agent without its own specific error (covers cancelled
        // agents AND agents whose worker completed fine but post-processing failed)
        if (_failedAgents[rank] === undefined) {
          _failedAgents[rank] = fallback;
        }
        // Append step-level error to the log buffer so the dialog always shows it
        if (error) {
          if (!_agentLogs[rank]) _agentLogs[rank] = [];
          const already = _agentLogs[rank].some(l => l.includes('--- step failed ---'));
          if (!already) {
            _agentLogs[rank].push('[stderr] --- step failed ---');
            error.split('\n').forEach(l => { if (l.trim()) _agentLogs[rank].push('[stderr] ' + l); });
          }
          // Live-update dialog if it is open for this agent
          if (_logDialogRank === rank) {
            const pre = document.getElementById('dlg-agent-log-text');
            if (pre) {
              const lines = ['[stderr] --- step failed ---', ...error.split('\n').filter(l => l.trim()).map(l => '[stderr] ' + l)];
              lines.forEach(l => {
                if (pre.innerHTML || pre.children.length) pre.appendChild(document.createTextNode('\n'));
                const span = document.createElement('span');
                span.style.color = '#f44747';
                span.textContent = l;
                pre.appendChild(span);
              });
              pre.scrollTop = pre.scrollHeight;
            }
          }
        }
      }
      _stepFailed    = true;
      _inProgress    = {};
      _pendingAgents = new Set();
      _agentStatus   = {};
      render();
    },

    onQueueUpdated(queue) {
      _queue = queue || [];
      render();
    },

    getSelectedUid() { return _selectedUid; },
  };
})();
