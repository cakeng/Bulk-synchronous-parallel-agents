/* ── Bottom panels: engine state, agent state, chat context ─────────────────── */
'use strict';

const BottomPanels = (() => {
  let _runName = null;
  let _records = [];       // engine state summaries
  let _curFull = null;     // full engine state (globals + full_agents)
  let _selectedAgent = 0;

  const VAL_MAX = 16; // max chars to show inline for a variable value

  // ── Load a specific engine state by uid ──────────────────────────────────
  async function loadByUid(uid) {
    if (!_runName || !uid) return;
    const res = await fetch(`/api/runs/${_runName}/engine_states/${uid}`);
    if (!res.ok) return;
    _curFull = await res.json();
    renderEngineState();
    renderAgentState();
  }

  function renderEngineState() {
    if (!_curFull) return;
    const tbl = document.getElementById('globals-table');
    tbl.innerHTML = '';
    for (const [k, v] of Object.entries(_curFull.globals || {})) {
      const tr = document.createElement('tr');
      tr.innerHTML = `<td>${escHtml(k)}</td><td>${escHtml(JSON.stringify(v))}</td>`;
      tbl.appendChild(tr);
    }
    const codeEl = document.getElementById('op-code-display');
    codeEl.textContent = _curFull.operator_code || '';
  }

  function renderAgentState() {
    if (!_curFull) return;
    const agents = _curFull.agents || [];
    const agent = agents.find(a => (a.agent_rank ?? 0) === _selectedAgent) || agents[0];
    if (agent) renderAgent(agent);
  }

  function renderAgent(agent) {
    const tbl = document.getElementById('agent-vars-table');
    tbl.innerHTML = '';
    for (const [k, v] of Object.entries(agent)) {
      if (k === 'llm_state') continue;
      const tr = document.createElement('tr');
      const valStr = JSON.stringify(v);
      const short = valStr.length <= VAL_MAX;
      const displayVal = short
        ? escHtml(valStr)
        : `<span class="var-value expandable" title="${escHtml(valStr)}">${escHtml(valStr.slice(0, VAL_MAX))}… (click)</span>`;
      tr.innerHTML = `<td>${escHtml(k)}</td><td class="var-value">${short ? escHtml(valStr) : displayVal}</td>`;
      if (!short) {
        tr.querySelector('.expandable').addEventListener('click', function() {
          this.outerHTML = `<span class="var-value">${escHtml(valStr)}</span>`;
        });
      }
      tbl.appendChild(tr);
    }
    renderChat(agent.llm_state?.context || []);
  }

  function renderChat(messages) {
    const container = document.getElementById('chat-context');
    const panel     = document.getElementById('chat-panel');
    container.innerHTML = '';
    messages.forEach(msg => {
      const div = document.createElement('div');
      div.className = `chat-msg ${msg.role}`;
      const roleLabel = document.createElement('div');
      roleLabel.className = 'chat-msg-role';
      roleLabel.textContent = msg.role;
      div.appendChild(roleLabel);
      const content = document.createElement('div');
      content.textContent = (msg.content || '').trim();
      div.appendChild(content);
      container.appendChild(div);
    });
    requestAnimationFrame(() => { panel.scrollTop = panel.scrollHeight; });
  }

  function escHtml(s) {
    return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
  }

  function clearPanels() {
    document.getElementById('globals-table').innerHTML = '';
    document.getElementById('agent-vars-table').innerHTML = '';
    document.getElementById('chat-context').innerHTML = '';
    document.getElementById('op-code-display').textContent = '';
  }

  // ── Public API ────────────────────────────────────────────────────────────
  return {
    init(runName, records) {
      _runName = runName;
      _records = [...records];  // own copy — don't share with Tree
      _curFull = null;
      _selectedAgent = 0;
      clearPanels();
    },

    onRecordAdded(record) {
      _records.push(record);
    },

    selectState(uid, agentRank) {
      _selectedAgent = agentRank ?? 0;
      loadByUid(uid);
    },

    onAgentCompleted(agentRank, state) {
      _selectedAgent = agentRank;
      renderAgent(state);
    },
  };
})();
