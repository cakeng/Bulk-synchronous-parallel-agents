/* ── Bottom panels: engine state, agent state, chat context ─────────────────── */
'use strict';

const BottomPanels = (() => {
  let _runName = null;
  let _records = [];       // engine state summaries
  let _curFull = null;     // full engine state (globals + full_agents)
  let _lastAgent = null;   // last agent dict passed to renderAgent (for toggle re-render)
  let _selectedAgent = 0;
  let _showThinking  = false;
  let _showToolCalls = false;

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
    _lastAgent = agent;
    const tbl = document.getElementById('agent-vars-table');
    tbl.innerHTML = '';
    for (const [k, v] of Object.entries(agent)) {
      if (k === 'agent_config') continue;
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
    renderChat(agent.agent_config?.context || [], agent.agent_config?.call_log || []);
  }

  function renderChat(messages, callLog) {
    const container = document.getElementById('chat-context');
    const panel     = document.getElementById('chat-panel');
    container.innerHTML = '';

    // The agentic loop injects intermediate assistant messages (one per tool-call
    // round) into the context, but call_log has ONE entry per run_agent() call
    // (aggregating all inner rounds).  We attach thinking/tool-calls to the LAST
    // assistant message in each run_agent sequence — the one that has text content.
    // Pre-scan: for each assistant message index, decide which call_log entry it
    // maps to.  Strategy: find groups of [asst(tool), tool-result, …, asst(text)]
    // and assign the call_log entry only to the last assistant in each group.
    const assistantIndices = messages
      .map((m, i) => m.role === 'assistant' ? i : -1)
      .filter(i => i >= 0);

    // Map from message index → call_log index (-1 = none)
    const callLogMap = new Map();
    let callIdx = 0;
    let groupStart = 0;
    while (groupStart < assistantIndices.length) {
      // Walk forward while messages are intermediate (no text content)
      let groupEnd = groupStart;
      while (
        groupEnd + 1 < assistantIndices.length &&
        !(messages[assistantIndices[groupEnd]].content || '').trim()
      ) {
        groupEnd++;
      }
      // assistantIndices[groupEnd] is the last (or only) assistant in this group
      callLogMap.set(assistantIndices[groupEnd], callIdx);
      callIdx++;
      groupStart = groupEnd + 1;
    }

    messages.forEach((msg, msgIdx) => {
      // Tool response messages — only shown when showToolCalls is on
      if (msg.role === 'tool') {
        if (_showToolCalls) {
          container.appendChild(_makeBubble('tool-result', 'tool result', msg.content || ''));
        }
        return;
      }

      if (msg.role === 'assistant') {
        const logIdx = callLogMap.has(msgIdx) ? callLogMap.get(msgIdx) : -1;
        const entry  = logIdx >= 0 ? (callLog[logIdx] || null) : null;

        // Thinking bubbles — before the assistant reply
        if (_showThinking && entry && entry.thinking && entry.thinking.length > 0) {
          for (const thought of entry.thinking) {
            container.appendChild(_makeBubble('thinking', 'thinking', thought));
          }
        }

        // Tool call bubbles — before the assistant reply (LLM decided to call tools)
        if (_showToolCalls && entry && entry.tool_calls && entry.tool_calls.length > 0) {
          for (const tc of entry.tool_calls) {
            const name = tc.function?.name || tc.id || 'tool_call';
            const args = tc.function?.arguments || '';
            container.appendChild(_makeBubble('tool-call', `tool call: ${name}`, args));
          }
        }

        // The assistant message itself (skip empty tool-call intermediaries)
        const hasContent = (msg.content || '').trim();
        if (hasContent) {
          container.appendChild(_makeBubble('assistant', 'assistant', msg.content));
        }
        return;
      }

      // User / system messages
      container.appendChild(_makeBubble(msg.role, msg.role, msg.content || ''));
    });

    requestAnimationFrame(() => { panel.scrollTop = panel.scrollHeight; });
  }

  function _makeBubble(cssClass, roleLabel, content) {
    const div = document.createElement('div');
    div.className = `chat-msg ${cssClass}`;
    const roleEl = document.createElement('div');
    roleEl.className = 'chat-msg-role';
    roleEl.textContent = roleLabel;
    div.appendChild(roleEl);
    const contentEl = document.createElement('div');
    contentEl.textContent = String(content).trim();
    div.appendChild(contentEl);
    return div;
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

  // ── Toggle button wiring ─────────────────────────────────────────────────
  function _wireToggle(btnId, getter, setter) {
    const btn = document.getElementById(btnId);
    if (!btn) return;
    btn.addEventListener('click', () => {
      setter(!getter());
      btn.classList.toggle('active', getter());
      if (_curFull) renderAgentState();
      else if (_lastAgent) renderAgent(_lastAgent);
    });
  }

  _wireToggle('btn-show-thinking',  () => _showThinking,  v => { _showThinking  = v; });
  _wireToggle('btn-show-tool-calls',() => _showToolCalls, v => { _showToolCalls = v; });

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
