/* ── Operator panel (right side, Jupyter-style) ─────────────────────────────── */
'use strict';

const OperatorPanel = (() => {
  let _runName = null;
  let _operators = [];      // [{name, body}]
  let _editors = {};        // name -> monaco editor instance
  let _editorDivs = {};     // name -> persistent editor container div
  let _selectedNames = new Set();
  let _clipboard = null;    // {name, body}
  let _collapsed = {};      // name -> bool
  let _dragSrcIdx = null;
  let _runningOp = null;
  let _queuedOps = [];

  const saveDebounces = {};

  function container() { return document.getElementById('op-cells'); }

  // ── Fetch operators from server ───────────────────────────────────────────
  async function load(runName) {
    _runName = runName;
    _operators = [];
    _editors = {};
    _editorDivs = {};
    _selectedNames = new Set();
    _collapsed = {};
    const res = await fetch(`/api/runs/${runName}/operators`);
    if (!res.ok) return;
    const data = await res.json();
    _operators = data.operators;
    render();
  }

  // ── Render all cells ──────────────────────────────────────────────────────
  function render() {
    const c = container();
    if (!c) return;

    // Detach persistent editor divs before clearing so Monaco isn't destroyed
    for (const div of Object.values(_editorDivs)) {
      div.parentElement?.removeChild(div);
    }

    c.innerHTML = '';
    _operators.forEach((op, idx) => renderCell(op, idx, c));

    // Relayout any visible editors now that they're re-attached
    for (const [name, editor] of Object.entries(_editors)) {
      if (!(_collapsed[name] ?? true)) editor.layout();
    }
  }

  function renderCell(op, idx, parent) {
    const collapsed = _collapsed[op.name] ?? false;
    const cell = document.createElement('div');
    cell.className = 'op-cell' +
      (_selectedNames.has(op.name) ? ' selected-cell' : '') +
      (op.name === _runningOp ? ' running-cell' : '') +
      (_queuedOps.includes(op.name) ? ' queued-cell' : '');
    cell.dataset.name = op.name;
    cell.dataset.idx = idx;

    // Header
    const header = document.createElement('div');
    header.className = 'op-cell-header';
    header.draggable = true;

    const foldSpan = document.createElement('span');
    foldSpan.className = 'op-cell-fold';
    foldSpan.textContent = collapsed ? '▶' : '▼';

    const typeBadge = document.createElement('span');
    typeBadge.className = `op-type-badge ${op.op_type || 'base'}`;
    typeBadge.textContent = op.op_type || 'base';

    const nameSpan = document.createElement('span');
    nameSpan.className = 'op-cell-name';
    nameSpan.textContent = op.name.replace(/\.py$/, '');
    nameSpan.addEventListener('dblclick', e => {
      e.stopPropagation();
      _startRename(op, nameSpan);
    });

    const btnsDiv = document.createElement('div');
    btnsDiv.className = 'op-cell-btns';

    const kfLabel = document.createElement('label');
    kfLabel.className = 'kill-failed-label';
    kfLabel.title = 'Kill failed agents after this step';
    const kfCheck = document.createElement('input');
    kfCheck.type = 'checkbox';
    kfCheck.checked = op.kill_failed ?? true;
    kfCheck.addEventListener('change', async e => {
      e.stopPropagation();
      op.kill_failed = kfCheck.checked;
      await fetch(`/api/runs/${_runName}/operators/${op.name}`, {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ kill_failed: kfCheck.checked }),
      });
    });
    kfLabel.appendChild(kfCheck);
    kfLabel.appendChild(document.createTextNode(' kill failed'));
    btnsDiv.appendChild(kfLabel);

    const runBtn = document.createElement('button');
    runBtn.className = 'run-btn';
    runBtn.dataset.name = op.name;
    runBtn.textContent = '▶ Run';
    btnsDiv.appendChild(runBtn);

    header.appendChild(foldSpan);
    header.appendChild(typeBadge);
    header.appendChild(nameSpan);
    header.appendChild(btnsDiv);

    // Fold toggle
    foldSpan.addEventListener('click', e => {
      e.stopPropagation();
      _collapsed[op.name] = !(_collapsed[op.name] ?? true);
      render();
    });

    // Selection on header click — update classes directly to avoid DOM rebuild
    // (a full render() would destroy nameSpan before dblclick fires)
    header.addEventListener('click', e => {
      if (e.target.closest('button')) return;
      if (e.shiftKey) {
        if (_selectedNames.has(op.name)) _selectedNames.delete(op.name);
        else _selectedNames.add(op.name);
        cell.classList.toggle('selected-cell', _selectedNames.has(op.name));
      } else {
        document.querySelectorAll('#op-cells .op-cell').forEach(el => el.classList.remove('selected-cell'));
        _selectedNames = new Set([op.name]);
        cell.classList.add('selected-cell');
      }
    });

    // Run button
    runBtn.addEventListener('click', async e => {
      e.stopPropagation();
      await fetch(`/api/runs/${_runName}/operators/${op.name}/run`, { method: 'POST' });
    });

    // Drag reorder
    header.addEventListener('dragstart', e => {
      _dragSrcIdx = idx;
      cell.classList.add('dragging');
      e.dataTransfer.effectAllowed = 'move';
    });
    header.addEventListener('dragend', () => cell.classList.remove('dragging'));
    cell.addEventListener('dragover', e => { e.preventDefault(); cell.classList.add('drag-over'); });
    cell.addEventListener('dragleave', () => cell.classList.remove('drag-over'));
    cell.addEventListener('drop', async e => {
      e.preventDefault();
      cell.classList.remove('drag-over');
      if (_dragSrcIdx === null || _dragSrcIdx === idx) return;
      const moved = _operators.splice(_dragSrcIdx, 1)[0];
      _operators.splice(idx, 0, moved);
      _dragSrcIdx = null;
      render();
      await fetch(`/api/runs/${_runName}/operators/reorder`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ names: _operators.map(o => o.name) }),
      });
    });

    cell.appendChild(header);

    // Reuse the persistent editor div (so Monaco isn't destroyed on re-render)
    let editorDiv = _editorDivs[op.name];
    if (!editorDiv) {
      editorDiv = document.createElement('div');
      editorDiv.id = `editor-${op.name.replace(/\./g, '_')}`;
      _editorDivs[op.name] = editorDiv;
    }
    editorDiv.className = 'op-cell-editor' + (collapsed ? ' collapsed' : '');
    cell.appendChild(editorDiv);

    parent.appendChild(cell);

    if (!collapsed) {
      mountEditor(op, editorDiv);
    }
  }

  // ── Inline rename ─────────────────────────────────────────────────────────
  function _startRename(op, nameSpan) {
    const input = document.createElement('input');
    input.className = 'op-rename-input';
    input.value = op.name.replace(/\.py$/, '');
    nameSpan.replaceWith(input);
    input.focus();
    input.select();

    let committed = false;

    async function commit() {
      if (committed) return;
      committed = true;

      const raw = input.value.trim();
      const newName = raw.endsWith('.py') ? raw : raw + '.py';

      if (!raw || newName === op.name) {
        input.replaceWith(nameSpan);
        return;
      }

      try {
        const res = await fetch(`/api/runs/${_runName}/operators/${op.name}/rename`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ new_name: newName }),
        });
        if (!res.ok) {
          alert((await res.json()).detail);
          input.replaceWith(nameSpan);
          committed = false;
          return;
        }
        const updated = await res.json();  // {name, body}
        const oldName = op.name;

        // Migrate keyed state to new name (op object is mutated in place,
        // so Monaco closures that captured `op` will see the new name)
        if (_editors[oldName])    { _editors[updated.name]    = _editors[oldName];    delete _editors[oldName]; }
        if (_editorDivs[oldName]) { _editorDivs[updated.name] = _editorDivs[oldName]; delete _editorDivs[oldName]; }
        if (_collapsed[oldName] !== undefined) { _collapsed[updated.name] = _collapsed[oldName]; delete _collapsed[oldName]; }
        clearTimeout(saveDebounces[oldName]);
        delete saveDebounces[oldName];
        if (_selectedNames.has(oldName)) { _selectedNames.delete(oldName); _selectedNames.add(updated.name); }
        if (_runningOp === oldName) _runningOp = updated.name;

        op.name = updated.name;
        op.body = updated.body;
        render();
      } catch (e) {
        alert(e.message);
        input.replaceWith(nameSpan);
        committed = false;
      }
    }

    function cancel() {
      committed = true;
      input.replaceWith(nameSpan);
    }

    input.addEventListener('blur', commit);
    input.addEventListener('keydown', e => {
      if (e.key === 'Enter')  { e.preventDefault(); input.blur(); }
      if (e.key === 'Escape') { input.removeEventListener('blur', commit); cancel(); }
    });
  }

  function _fitEditorHeight(editor, container) {
    const h = editor.getContentHeight();
    container.style.height = h + 'px';
    editor.layout({ width: container.clientWidth, height: h });
  }

  async function mountEditor(op, editorContainer) {
    if (_editors[op.name]) {
      _fitEditorHeight(_editors[op.name], editorContainer);
      return;
    }
    await window.monacoReady;
    const editor = monaco.editor.create(editorContainer, {
      value: op.body,
      language: 'python',
      theme: 'vs-dark',
      minimap: { enabled: false },
      scrollBeyondLastLine: false,
      automaticLayout: false,
      lineNumbers: 'on',
      fontSize: 12,
      wordWrap: 'on',
      scrollbar: { vertical: 'hidden', alwaysConsumeMouseWheel: false },
      overviewRulerLanes: 0,
      padding: { top: 18, bottom: 18 },
    });
    _editors[op.name] = editor;

    editor.onDidContentSizeChange(() => _fitEditorHeight(editor, editorContainer));
    _fitEditorHeight(editor, editorContainer);

    editor.onDidChangeModelContent(() => {
      clearTimeout(saveDebounces[op.name]);
      saveDebounces[op.name] = setTimeout(async () => {
        const body = editor.getValue();
        const local = _operators.find(o => o.name === op.name);
        if (local) local.body = body;
        await fetch(`/api/runs/${_runName}/operators/${op.name}`, {
          method: 'PUT',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ body }),
        });
      }, 800);
    });
  }

  // ── Toolbar actions ───────────────────────────────────────────────────────
  async function createOp(name, opType, insertIdx) {
    const res = await fetch(`/api/runs/${_runName}/operators`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name, op_type: opType }),
    });
    if (!res.ok) { alert((await res.json()).detail); return; }
    const op = await res.json();  // {name, body}
    if (insertIdx !== undefined && insertIdx >= 0 && insertIdx < _operators.length) {
      _operators.splice(insertIdx, 0, op);
      await fetch(`/api/runs/${_runName}/operators/reorder`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ names: _operators.map(o => o.name) }),
      });
    } else {
      _operators.push(op);
    }
    render();
    const newCell = container()?.querySelector(`[data-name="${op.name}"]`);
    newCell?.scrollIntoView({ behavior: 'smooth' });
  }

  function copySelected() {
    const name = [..._selectedNames][0];
    if (!name) return null;
    const op = _operators.find(o => o.name === name);
    if (op) _clipboard = { ...op };
    return op ? op.name : null;
  }

  async function pasteOp() {
    if (!_clipboard) return;
    const newName = _clipboard.name.replace(/\.py$/, '') + '_copy.py';
    const res = await fetch(`/api/runs/${_runName}/operators/${_clipboard.name}/copy`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ new_name: newName }),
    });
    if (!res.ok) { alert((await res.json()).detail); return; }
    const op = await res.json();  // {name, body}
    _operators.push(op);
    render();
  }

  async function deleteSelected() {
    for (const name of [..._selectedNames]) {
      await fetch(`/api/runs/${_runName}/operators/${name}`, { method: 'DELETE' });
      _operators = _operators.filter(o => o.name !== name);
      delete _editors[name];
    }
    _selectedNames = new Set();
    render();
  }

  async function runAll() {
    for (const op of _operators) {
      await fetch(`/api/runs/${_runName}/operators/${op.name}/run`, { method: 'POST' });
    }
  }

  // ── Update running/queued CSS classes without rebuilding the DOM ─────────
  function _updateCellClasses() {
    const cells = container()?.querySelectorAll('.op-cell') || [];
    cells.forEach(cell => {
      const name = cell.dataset.name;
      cell.classList.toggle('running-cell', name === _runningOp);
      cell.classList.toggle('queued-cell',  _queuedOps.includes(name));
    });
  }

  // ── WebSocket event handlers ──────────────────────────────────────────────
  function onQueueUpdated(queue) {
    _queuedOps = queue;
    _updateCellClasses();
  }

  function onStepStarted(opName) {
    _runningOp = opName;
    _updateCellClasses();
  }

  function onStepDone() {
    _runningOp = null;
    _updateCellClasses();
  }

  function relayout() {
    for (const [name, editor] of Object.entries(_editors)) {
      if (!(_collapsed[name] ?? true)) {
        const div = _editorDivs[name];
        if (div) _fitEditorHeight(editor, div);
      }
    }
  }

  // ── Public API ────────────────────────────────────────────────────────────
  return {
    load,
    createOp,
    copySelected,
    pasteOp,
    deleteSelected,
    runAll,
    onQueueUpdated,
    onStepStarted,
    onStepDone,
    relayout,
    getSelected:      () => [..._selectedNames],
    getNames:         () => _operators.map(o => o.name),
    getSelectedIndex: () => {
      const name = [..._selectedNames][0];
      return name ? _operators.findIndex(o => o.name === name) : -1;
    },
  };
})();

// Wire up toolbar buttons
const _newOpBtn      = document.getElementById('btn-new-op');
const _newOpDropdown = document.getElementById('new-op-dropdown');

_newOpBtn.addEventListener('click', e => {
  e.stopPropagation();
  _newOpDropdown.classList.toggle('open');
});
document.addEventListener('click', () => _newOpDropdown.classList.remove('open'));

_newOpDropdown.querySelectorAll('.dropdown-item').forEach(item => {
  item.addEventListener('click', () => {
    _newOpDropdown.classList.remove('open');
    const opType = item.dataset.opType;
    const name   = _autoName(opType);
    const selIdx = OperatorPanel.getSelectedIndex();
    OperatorPanel.createOp(name, opType, selIdx >= 0 ? selIdx + 1 : undefined);
  });
});

function _autoName(opType) {
  const base     = opType === 'base' ? 'operator' : opType;
  const existing = new Set(OperatorPanel.getNames());
  if (!existing.has(`${base}.py`)) return `${base}.py`;
  for (let i = 2; ; i++) {
    const candidate = `${base}_${i}.py`;
    if (!existing.has(candidate)) return candidate;
  }
}

document.getElementById('btn-copy-op').addEventListener('click', () => {
  const name = OperatorPanel.copySelected();
  if (!name) alert('Select an operator first.');
});

document.getElementById('btn-paste-op').addEventListener('click', () => OperatorPanel.pasteOp());

document.getElementById('btn-delete-op-cell').addEventListener('click', async () => {
  const sel = OperatorPanel.getSelected();
  if (!sel.length) return;
  if (!confirm(`Delete operator(s): ${sel.map(n => n.replace(/\.py$/, '')).join(', ')}?`)) return;
  await OperatorPanel.deleteSelected();
});

document.getElementById('btn-run-all').addEventListener('click', () => OperatorPanel.runAll());

// Relayout Monaco editors whenever the operator panel changes size (handles both
// the drag-handle resizer and browser window resize).
new ResizeObserver(() => OperatorPanel.relayout())
  .observe(document.getElementById('editor-panel'));
