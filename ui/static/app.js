/* ── BSA UI — Root coordinator ───────────────────────────────────────────────── */
'use strict';

// ── App state ─────────────────────────────────────────────────────────────────
const App = {
  runs: [],
  activeRun: null,
  sockets: {},   // run -> WebSocket
};

// ── Tree status label ─────────────────────────────────────────────────────────
function setTreeStatus(text, running) {
  const el = document.getElementById('tree-status');
  el.textContent = text;
  el.classList.toggle('running', !!running);
}
function clearTreeStatus() { setTreeStatus(''); }

// ── Fetch helpers ─────────────────────────────────────────────────────────────
async function api(method, path, body) {
  const opts = { method, headers: {} };
  if (body !== undefined) {
    opts.headers['Content-Type'] = 'application/json';
    opts.body = JSON.stringify(body);
  }
  const res = await fetch(path, opts);
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || res.statusText);
  }
  return res.json();
}

// ── Tab management ────────────────────────────────────────────────────────────
function renderTabs() {
  const container = document.getElementById('tabs');
  container.innerHTML = '';
  App.runs.forEach(name => {
    const tab = document.createElement('div');
    tab.className = 'tab' + (name === App.activeRun ? ' active' : '');
    tab.textContent = name;
    tab.addEventListener('click', () => switchRun(name));
    container.appendChild(tab);
  });
}

async function switchRun(name) {
  App.activeRun = name;
  renderTabs();
  await loadRunData(name);
}

async function loadRunData(runName) {
  clearTreeStatus();
  const [stateData, statusData] = await Promise.all([
    api('GET', `/api/runs/${runName}/engine_states`),
    api('GET', `/api/runs/${runName}/status`),
  ]);
  const records = stateData.engine_states;

  connectWS(runName);

  Tree.init(runName, records, (uid, agentRank) => { BottomPanels.selectState(uid, agentRank); });
  BottomPanels.init(runName, records);
  await OperatorPanel.load(runName);

  // Replay any in-progress step state (handles tab-switch without WS reconnect)
  for (const event of statusData.step_log) {
    handleWsEvent(runName, event);
  }
  // Replay per-agent stdout/stderr/status logs
  for (const [rank, lines] of Object.entries(statusData.agent_logs || {})) {
    for (const line of lines) {
      // Lines are already fully formatted (with [stderr] / [status] / [tool] prefixes).
      // Feed them as 'stdout' so _formatLogLine handles the colouring.
      Tree.onAgentLog(Number(rank), 'stdout', line);
    }
  }
  if (statusData.queue && statusData.queue.length > 0) {
    handleWsEvent(runName, { type: 'queue_updated', run: runName, queue: statusData.queue });
  }
}

// ── WebSocket ─────────────────────────────────────────────────────────────────
function connectWS(runName) {
  if (App.sockets[runName]) {
    const s = App.sockets[runName];
    if (s.readyState === WebSocket.OPEN || s.readyState === WebSocket.CONNECTING) return;
  }
  const ws = new WebSocket(`ws://${location.host}/ws/${runName}`);
  App.sockets[runName] = ws;

  ws.onmessage = e => {
    const msg = JSON.parse(e.data);
    if (msg.run !== runName) return;
    handleWsEvent(runName, msg);
  };
  ws.onclose = () => {
    delete App.sockets[runName];
    setTimeout(() => { if (App.activeRun === runName) connectWS(runName); }, 2000);
  };
}

function handleWsEvent(runName, msg) {
  if (runName !== App.activeRun) return;

  switch (msg.type) {
    case 'runs_updated':
      App.runs = msg.runs;
      renderTabs();
      break;

    case 'step_started':
      setTreeStatus('Running', true);
      OperatorPanel.onStepStarted(msg.operator_name);
      Tree.onStepStarted(msg.pre_agents || [], msg.step_num, msg.operator_name);
      break;

    case 'agent_started':
      Tree.onAgentStarted(msg.agent_rank);
      break;

    case 'agent_status':
      Tree.onAgentStatus(msg.agent_rank, msg.status);
      Tree.onAgentLog(msg.agent_rank, 'stdout', `[status] ${msg.status}`);
      break;

    case 'agent_completed':
      Tree.onAgentCompleted(msg.agent_rank, msg.state || {});
      BottomPanels.onAgentCompleted(msg.agent_rank, msg.state || {});
      break;

    case 'agent_log':
      Tree.onAgentLog(msg.agent_rank, msg.stream, msg.text);
      break;

    case 'agent_failed':
      Tree.onAgentFailed(msg.agent_rank, msg.error || '');
      break;

    case 'step_completed':
      clearTreeStatus();
      OperatorPanel.onStepDone();
      Tree.onStepCompleted(msg.record);
      BottomPanels.onRecordAdded(msg.record);
      break;

    case 'step_failed':
      clearTreeStatus();
      OperatorPanel.onStepDone();
      Tree.onStepFailed(msg.error || '');
      break;

    case 'queue_updated':
      OperatorPanel.onQueueUpdated(msg.queue);
      Tree.onQueueUpdated(msg.queue);
      break;

    case 'log_line':
      break;
  }
}


// ── Delete Step button ────────────────────────────────────────────────────────
document.getElementById('delete-states-btn').addEventListener('click', async () => {
  const uid = Tree.getSelectedUid();
  if (!uid) { alert('Select a step in the tree first.'); return; }
  const rn = App.activeRun;
  if (!rn) return;

  if (uid === '__running__') {
    if (!confirm('Kill the currently running step?')) return;
    await api('POST', `/api/runs/${rn}/kill`);
    return;
  }

  const sd  = await api('GET', `/api/runs/${rn}/engine_states`);
  const idx = sd.engine_states.findIndex(r => r.uid === uid);
  if (idx < 0) return;
  if (!confirm(`Remove ${sd.engine_states.length - idx} engine state(s) from step ${sd.engine_states[idx].step_num}?`)) return;
  await api('DELETE', `/api/runs/${rn}/engine_states/from/${idx}`);
  await loadRunData(rn);
});

// ── Menu ──────────────────────────────────────────────────────────────────────
const menuBtn = document.getElementById('menu-btn');
const menuDropdown = document.getElementById('menu-dropdown');

menuBtn.addEventListener('click', e => {
  e.stopPropagation();
  menuDropdown.classList.toggle('open');
});
document.addEventListener('click', () => menuDropdown.classList.remove('open'));

// Create run
document.getElementById('menu-create').addEventListener('click', () => {
  menuDropdown.classList.remove('open');
  document.getElementById('dlg-create-run-name').value = '';
  document.getElementById('dlg-create-run').showModal();
});
document.getElementById('dlg-create-run-ok').addEventListener('click', async () => {
  const name = document.getElementById('dlg-create-run-name').value.trim();
  if (!name) return;
  try {
    await api('POST', '/api/runs', { name });
    document.getElementById('dlg-create-run').close();
    App.runs = (await api('GET', '/api/runs')).runs;
    renderTabs();
    await switchRun(name);
  } catch (e) { alert(e.message); }
});

// Clear run
document.getElementById('menu-clear').addEventListener('click', () => {
  if (!App.activeRun) return;
  menuDropdown.classList.remove('open');
  document.getElementById('dlg-clear-msg').textContent =
    `Clear all engine states for run "${App.activeRun}"? This cannot be undone.`;
  document.getElementById('dlg-confirm-clear').showModal();
});
document.getElementById('dlg-clear-ok').addEventListener('click', async () => {
  if (!App.activeRun) return;
  try {
    await api('POST', `/api/runs/${App.activeRun}/clear`);
    document.getElementById('dlg-confirm-clear').close();
    await loadRunData(App.activeRun);
  } catch (e) { alert(e.message); }
});

// Delete run
document.getElementById('menu-delete').addEventListener('click', () => {
  if (!App.activeRun) return;
  menuDropdown.classList.remove('open');
  document.getElementById('dlg-delete-msg').textContent =
    `Permanently delete run "${App.activeRun}" and all its files?`;
  document.getElementById('dlg-confirm-delete').showModal();
});
document.getElementById('dlg-delete-ok').addEventListener('click', async () => {
  if (!App.activeRun) return;
  const name = App.activeRun;
  try {
    await api('DELETE', `/api/runs/${name}`);
    document.getElementById('dlg-confirm-delete').close();
    App.runs = App.runs.filter(r => r !== name);
    delete App.sockets[name];
    App.activeRun = App.runs[0] || null;
    renderTabs();
    if (App.activeRun) await loadRunData(App.activeRun);
    else {
      Tree.init(null, [], () => {});
      BottomPanels.init(null, []);
      document.getElementById('op-cells').innerHTML = '';
    }
  } catch (e) { alert(e.message); }
});

// ── Panel resizer ─────────────────────────────────────────────────────────────
((() => {
  const resizer   = document.getElementById('middle-resizer');
  const treePanel = document.getElementById('tree-panel');
  const middle    = document.getElementById('middle');
  let dragging = false;

  resizer.addEventListener('mousedown', e => {
    e.preventDefault();
    dragging = true;
    resizer.classList.add('active');
    document.body.style.cursor = 'col-resize';
    document.body.style.userSelect = 'none';
  });

  window.addEventListener('mousemove', e => {
    if (!dragging) return;
    const rect   = middle.getBoundingClientRect();
    const newW   = e.clientX - rect.left;
    const minW   = 150;
    const maxW   = rect.width - 150 - 4; // 4px for resizer
    treePanel.style.flex = `0 0 ${Math.max(minW, Math.min(maxW, newW))}px`;
  });

  window.addEventListener('mouseup', () => {
    if (!dragging) return;
    dragging = false;
    resizer.classList.remove('active');
    document.body.style.cursor = '';
    document.body.style.userSelect = '';
  });
})());

// ── Bootstrap ─────────────────────────────────────────────────────────────────
(async () => {
  const data = await api('GET', '/api/runs');
  App.runs = data.runs;
  renderTabs();
  if (App.runs.length > 0) {
    const last = localStorage.getItem('bsa_active_run');
    const initial = App.runs.includes(last) ? last : App.runs[0];
    await switchRun(initial);
  } else {
    Tree.init(null, [], () => {});
    BottomPanels.init(null, []);
  }
})();

window.addEventListener('beforeunload', () => {
  if (App.activeRun) localStorage.setItem('bsa_active_run', App.activeRun);
});
