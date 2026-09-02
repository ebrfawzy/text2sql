/* Text2SQL Studio — the whole client, one Alpine component.
 *
 * Markup lives in index.html; repeated blocks are Jinja macros there, so a widget
 * (picker, stepper, event log, column table) exists exactly once in each language:
 * as markup when it needs Alpine reactivity, as a string helper when it doesn't. */

const STAGES = [
  { key: 'profiling', label: 'Profiling', icon: '🔬' },
  { key: 'schema_linking', label: 'Schema Linking', icon: '🔗' },
  { key: 'sql_generation', label: 'Generation', icon: '🤖' },
  { key: 'sql_repair', label: 'Repair', icon: '🛠️' },
  { key: 'selection', label: 'Selection', icon: '🗳️' },
  { key: 'pipeline', label: 'Done', icon: '🏁' },
];
// Stages a tab's stream can actually reach — /profile never generates SQL, so showing
// it four permanently grey circles would be noise.
const TAB_STAGES = {
  ask: STAGES, benchmark: STAGES,
  profile: STAGES.filter(s => ['profiling', 'pipeline'].includes(s.key)),
};
const idleStages = tab => Object.fromEntries(TAB_STAGES[tab].map(s => [s.key, 'idle']));
const scrollToBottom = el => { if (el) el.scrollTop = el.scrollHeight; };

function app() {
  return {
    theme: localStorage.getItem('t2s-theme') || (matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light'),
    tab: 'ask',
    version: '', health: 'unknown',
    schema: [], config: {}, openGroups: {},
    configYamlPath: '', configFiles: [], configLoading: false, configMsg: '', configError: false,
    settingsOpen: false, showTrace: false,
    stagesFor(tab) { return TAB_STAGES[tab]; },

    ask: {
      question: '', running: false, controller: null, events: [], stageStatus: idleStages('ask'),
      thinkingText: '', outputText: '', result: null,
      sel: {}, loadingSchema: false, schemaError: null, pickerOpen: false,
    },
    profile: {
      running: false, loadingSchema: false, events: [], result: null, error: null,
      sel: {}, schemaError: null, controller: null, stageStatus: idleStages('profile'),
      cached: null, showCached: false, cacheBusy: false,
    },
    benchmark: {
      running: false, previewing: false, error: null, started: null, controller: null,
      examples: null, previewTotal: 0, results: [], report: null, done: 0,
      current: null, usage: {}, stageStatus: idleStages('benchmark'),
    },

    // ── Boot & config form ──────────────────────────────────────

    // Keeps <html>.dark and localStorage in sync with `theme`; called by x-effect.
    applyTheme() {
      document.documentElement.classList.toggle('dark', this.theme === 'dark');
      localStorage.setItem('t2s-theme', this.theme);
    },

    async init() {
      await this.loadConfigSchema();
      const get = async (url, fallback) => { try { return await (await fetch(url)).json(); } catch (e) { return fallback; } };
      this.version = (await get('/version', {})).version || '';
      this.health = (await get('/health', { status: 'down' })).status;
      this.configFiles = (await get('/config/files', {})).files || [];
    },

    // Fetch the config-form schema and pre-fill the form from its defaults. With a
    // `configPath` the defaults come from that YAML file; without one they reflect the
    // server's startup config + env.
    async loadConfigSchema(configPath) {
      const q = configPath ? '?config=' + encodeURIComponent(configPath) : '';
      const r = await fetch('/config/schema' + q);
      if (!r.ok) throw new Error((await r.json().catch(() => ({}))).detail || 'HTTP ' + r.status);
      this.schema = (await r.json()).groups;
      this.fillDefaults();
    },

    // "Load" next to the Config YAML field: re-populate the form from that path
    // (or the server default when blank).
    async loadConfig() {
      this.configLoading = true; this.configMsg = ''; this.configError = false;
      const path = this.configYamlPath.trim();
      try {
        await this.loadConfigSchema(path || undefined);
        this.configMsg = path ? 'Loaded ' + path : 'Reset to server defaults';
      } catch (e) {
        this.configError = true; this.configMsg = String(e.message || e);
      } finally { this.configLoading = false; }
    },

    // Pre-fill every control with the effective default, so the form shows the values
    // the run will actually use.
    fillDefaults() {
      for (const g of this.schema) for (const f of g.fields) {
        if (f.control === 'toggle') this.config[f.name] = !!f.default;
        else if (f.control === 'multi') this.config[f.name] = [...(f.default || [])];
        else this.config[f.name] = f.default ?? '';
      }
    },

    // A field is inert when its `depends_on` controller holds an unlisted value (e.g. every
    // agent setting once sql_generation.mode is 'direct'). Hidden rather than greyed, so the
    // form only ever shows settings the run will read.
    // A multi controller must contain *all* the listed values (literal grounding needs both
    // 'reversed' and 'value' selected); a scalar one must be one of them. Dependencies chain,
    // so the whole chain has to hold: switching schema linking off hides the per-mode settings
    // too, even though the mode list still holds the values they name.
    isActive(f) {
      const byName = this.fieldsByName ??= Object.fromEntries(
        this.schema.flatMap(g => g.fields.map(x => [x.name, x])));
      // A <select> always yields a string, so num_candidates 3 arrives as '3' and an integer
      // `values` list never matched — selection_mode could not be revealed at all. And '' is
      // the "(default)" option, which submits nothing, so the server default is what runs.
      const held = n => this.config[n] === '' ? byName[n]?.default : this.config[n];
      for (let cur = f; cur?.depends_on; cur = byName[cur.depends_on.field]) {
        const v = held(cur.depends_on.field);
        const ok = Array.isArray(v) ? cur.depends_on.values.every(x => v.includes(x))
                                    : cur.depends_on.values.map(String).includes(String(v));
        if (!ok) return false;
      }
      return true;
    },

    // Sidebar groups for the current tab. 'benchmark' is the only relocation (its options
    // render in the centre pane), so any new setting reaches the UI with no change here.
    groups() {
      return this.schema
        .filter(g => g.key !== 'benchmark')
        .map(g => ({ ...g, fields: g.fields.filter(f => f.endpoints.includes(this.tab)) }))
        .filter(g => g.fields.length);
    },
    fields(key) { return this.schema.find(g => g.key === key)?.fields || []; },

    // Fields still at the server's effective default aren't sent, so an untouched form
    // reuses the configured engine instead of forcing a rebuild. Numbers are coerced
    // first, so the comparison can stay a plain string one.
    buildPayload(tab) {
      const payload = {};
      for (const g of this.schema) for (const f of g.fields) {
        let v = this.config[f.name];
        if (!f.endpoints.includes(tab) || v === null || v === undefined || v === '') continue;
        if (f.control === 'number') v = Number(v);
        // An empty multi-select is a real choice ("link nothing"), so it is always sent;
        // its comparison sorts first, since selection order is not meaningful.
        const key = x => Array.isArray(x) ? JSON.stringify([...x].sort()) : String(x);
        if (f.default == null || key(v) !== key(f.default)) payload[f.name] = v;
      }
      if (this.configYamlPath) payload.config = this.configYamlPath;
      if (tab === 'ask') payload.question = this.ask.question;
      if (tab === 'profile') payload.db_uri = this.config.db_uri;  // needs an explicit target
      return payload;
    },

    // ── Fragment rendering (x-html; Alpine initialises directives inside) ──

    // Escape for interpolation into generated markup (text or attribute value).
    esc(s) {
      return String(s ?? '').replace(/[&<>"']/g, c =>
        ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
    },

    // Inline stroke icons for the compact cache actions — no icon library, one helper.
    icon(name, size = 14) {
      const d = {
        reload: 'M21 12a9 9 0 1 1-2.64-6.36M21 3v6h-6',
        trash: 'M3 6h18M8 6V4h8v2M6 6l1 14h10l1-14M10 11v6M14 11v6',
        hide: 'M18 15l-6-6-6 6',
      }[name];
      return `<svg viewBox="0 0 24 24" width="${size}" height="${size}" fill="none" stroke="currentColor"
        stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="${d}"/></svg>`;
    },

    // Every config control (settings menu, sidebar, benchmark options) from one
    // /config/schema field descriptor.
    fieldHTML(f) {
      const cfg = `config['${f.name}']`;
      const box = 'mt-1 w-full rounded-md border border-slate-200 dark:border-slate-700 bg-transparent px-2 py-1.5'
        + ' text-sm focus:outline-none focus:ring-2 focus:ring-indigo-500';
      let ctrl;
      if (f.control === 'toggle') {
        ctrl = `<button @click="${cfg} = !${cfg}" class="mt-1 flex h-6 w-11 items-center rounded-full transition"
          :class="${cfg} ? 'bg-indigo-600 justify-end' : 'bg-slate-300 dark:bg-slate-700 justify-start'"
          ><span class="mx-0.5 h-5 w-5 rounded-full bg-white shadow"></span></button>`;
      } else if (f.control === 'multi') {
        // Each option toggles its own membership; the bound value stays an array, and an
        // empty one is a legitimate selection rather than "unset".
        ctrl = '<div class="mt-1 flex flex-wrap gap-x-3 gap-y-1">' + (f.options || []).map(o => {
          const q = `${cfg}.includes('${this.esc(o)}')`;
          return `<label class="flex items-center gap-1.5 text-sm cursor-pointer">
            <input type="checkbox" :checked="${q}" class="rounded border-slate-300 dark:border-slate-600
              text-indigo-600 focus:ring-indigo-500"
              @change="${q} ? ${cfg} = ${cfg}.filter(x => x !== '${this.esc(o)}')
                            : ${cfg} = [...${cfg}, '${this.esc(o)}']">${this.esc(o)}</label>`;
        }).join('') + '</div>';
      } else if (f.control === 'select') {
        // An empty-string member (stop_after's "run every stage") is what the leading
        // server-default option already submits, so it is never listed twice.
        const opts = (f.options || []).filter(o => o !== '')
          .map(o => `<option value="${this.esc(o)}">${this.esc(o)}</option>`).join('');
        ctrl = `<select x-model="${cfg}" class="${box}">`
          + `<option value="">(default: ${this.esc(f.default) || 'none'})</option>${opts}</select>`;
      } else {
        const num = f.control === 'number';
        const attr = (k, v) => v == null ? '' : ` ${k}="${v}"`;
        const bounds = num ? attr('min', f.min) + attr('max', f.max) + attr('step', f.step || 1) : '';
        const ph = this.esc(f.default === null || f.default === undefined || f.default === ''
          ? (f.help || '') : 'default: ' + f.default);
        ctrl = `<input type="${num ? 'number' : 'text'}" x-model="${cfg}"${bounds} placeholder="${ph}" class="${box}">`;
      }
      const tip = !f.help ? '' : `<span class="cursor-help select-none rounded-full border border-slate-300
          dark:border-slate-600 text-[9px] leading-none text-slate-400 w-3.5 h-3.5 inline-flex items-center
          justify-center">i</span>
        <span class="pointer-events-none absolute left-0 top-full z-50 mt-1 hidden w-56 max-w-full rounded-md
          bg-slate-900 px-2 py-1.5 text-[11px] leading-snug text-slate-100 shadow-lg ring-1 ring-black/10
          group-hover:block dark:bg-slate-700">${this.esc(f.help)}</span>`;
      return `<div class="relative group flex items-center gap-1">
        <label class="text-xs text-slate-500 dark:text-slate-400">${this.esc(f.label)}</label>${tip}</div>${ctrl}`;
    },

    // Knowledge-base panel for both the profile result and the cached view; an entry's
    // children_knowledge is -1 when it stands alone, else the ids it builds on.
    kbHTML(kb) {
      if (!kb || !kb.length) return '';
      const byId = Object.fromEntries(kb.map(e => [e.id, e]));
      const rows = kb.map(e => {
        const kids = (Array.isArray(e.children_knowledge) ? e.children_knowledge : [])
          .map(i => byId[i]?.knowledge).filter(Boolean);
        return `<div class="px-3 py-2 text-xs"><div class="flex items-baseline gap-2">
          <span class="font-semibold">${this.esc(e.knowledge)}</span>
          <span class="text-slate-400 mono">${this.esc(e.type)}</span></div>
          <p class="text-slate-500">${this.esc(e.description)}</p>
          <p class="mono text-slate-400 mt-0.5">${this.esc(e.definition)}</p>`
          + (kids.length ? `<p class="text-[10px] text-indigo-500 mt-0.5">builds on: ${this.esc(kids.join(', '))}</p>` : '')
          + '</div>';
      }).join('');
      return `<div class="rounded-md border border-slate-200 dark:border-slate-800">
        <div class="px-3 py-2 border-b border-slate-200 dark:border-slate-800"><h4 class="font-semibold">Knowledge base
        <span class="text-xs font-normal text-slate-400">· ${kb.length} entries</span></h4></div>
        <div class="overflow-auto max-h-72 divide-y divide-slate-100 dark:divide-slate-800">${rows}</div></div>`;
    },

    // Small static result table — shared by the Ask result and both benchmark SQL panes.
    resultTable(rows) {
      if (!rows || !rows.length) return '<p class="text-[11px] text-slate-400 italic">no rows</p>';
      const cols = Object.keys(rows[0]);
      const cells = (tag, cls, fn) => cols.map(c => `<${tag} class="${cls}">${this.esc(fn(c))}</${tag}>`).join('');
      const body = rows.map(r => `<tr class="odd:bg-slate-50 dark:odd:bg-slate-900/50">`
        + cells('td', 'px-2 py-1 mono', c => r[c]) + '</tr>').join('');
      return `<div class="overflow-auto max-h-72 rounded-md border border-slate-200 dark:border-slate-800">
        <table class="w-full text-xs"><thead class="bg-slate-100 dark:bg-slate-800 sticky top-0"><tr>`
        + cells('th', 'px-2 py-1 text-left font-semibold', c => c)
        + `</tr></thead><tbody>${body}</tbody></table></div>`;
    },

    // One benchmark row's verdict, in every form the markup needs. `execution_match: null`
    // means the official evaluator hasn't scored the run yet — never "wrong".
    // Class strings are written out in full so Tailwind's scanner can see them.
    verdict(r) {
      let v;
      return {
        error: { mark: '—', label: 'error', cls: 'bg-rose-100 text-rose-700 dark:bg-rose-950 dark:text-rose-300' },
        pending: { mark: '⋯', label: 'not scored', cls: 'bg-slate-100 text-slate-600 dark:bg-slate-800 dark:text-slate-300' },
        match: { mark: '✓', label: '✓ match', cls: 'bg-emerald-100 text-emerald-700 dark:bg-emerald-950 dark:text-emerald-300' },
        mismatch: { mark: '✗', label: '✗ mismatch', cls: 'bg-amber-100 text-amber-700 dark:bg-amber-950 dark:text-amber-300' },
      }[(v = r.verdict || {}).execution_match === true ? 'match'
        : v.execution_match === false ? 'mismatch' : v.error ? 'error' : 'pending'];
    },

    // Report tiles: percentages render '—' until the run has been scored.
    reportStats(rep) {
      const pct = v => v === null || v === undefined ? '—' : (v * 100).toFixed(1) + '%';
      const usd = v => v === null || v === undefined ? '—' : '$' + v.toFixed(4);
      const a = rep.accuracy || {}, c = rep.cost || {};
      const failed = a.correct === null || a.correct === undefined ? '—' : a.total - a.correct;
      const cost = c.total_cost_usd || 0;
      return [
        ['Execution Accuracy', pct(a.execution_accuracy)],
        ['Total Questions', a.total],
        ['Total Failed', failed],
        ['Total Cost', usd(cost)],
        ['Avg Cost / Question', usd(a.total ? cost / a.total : 0)],
        ['Avg Latency', c.avg_latency_seconds + 's'],
        ['Tables Recall', pct(rep.linking?.table.recall)],
        ['Columns Recall', pct(rep.linking?.column.recall)],
        ['Linking F1 (tables)', pct(rep.linking?.table.f1)],
        ['Linking F1 (columns)', pct(rep.linking?.column.f1)],
      ];
    },

    // Reusable LLM-usage status bar (Ask result + benchmark aggregate).
    usageStats(u = {}) {
      return [['Prompt', u.prompt_tokens || 0], ['Completion', u.completion_tokens || 0],
      ['Total', u.total_tokens || 0], ['Cost $', (u.total_cost_usd || 0).toFixed(4)], ['Calls', u.num_calls || 0]];
    },

    highlightSql(sql) {
      try { return hljs.highlight(sql || '', { language: 'sql' }).value; } catch (e) { return sql; }
    },

    // ── Requests ────────────────────────────────────────────────

    // POST JSON, raising the server's `detail` on failure — every non-streaming call.
    async postJSON(url, body) {
      const r = await fetch(url, {
        method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body),
      });
      if (!r.ok) throw new Error((await r.json().catch(() => ({}))).detail || 'HTTP ' + r.status);
      return r.json();
    },

    // Consume an SSE stream, dispatching each event to handlers[<event name>].
    async streamRequest(url, payload, handlers, controller = new AbortController()) {
      try {
        const resp = await fetch(url, {
          method: 'POST', headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload), signal: controller.signal,
        });
        if (!resp.ok || !resp.body) {
          const text = await resp.text().catch(() => resp.statusText);
          handlers.error?.({ error: `HTTP ${resp.status}: ${text}` });
          return controller;
        }
        const reader = resp.body.getReader(), decoder = new TextDecoder();
        let buf = '';
        for (; ;) {
          const { done, value } = await reader.read();
          if (done) break;
          buf += decoder.decode(value, { stream: true });
          const parts = buf.split('\n\n');
          buf = parts.pop();
          for (const part of parts.filter(p => p.trim())) {
            let name = 'message', data = '';
            for (const line of part.split('\n')) {
              if (line.startsWith('event:')) name = line.slice(6).trim();
              else if (line.startsWith('data:')) data += line.slice(5).trim();
            }
            try { data = JSON.parse(data); } catch (e) { }
            handlers[name]?.(data);
          }
        }
      } catch (e) {
        if (e.name !== 'AbortError') handlers.error?.({ error: String(e) });
      }
      return controller;
    },

    // Shared stepper update from a pipeline `progress` event.
    applyStage(stageStatus, e) {
      const state = { started: 'active', progress: 'active', completed: 'done', error: 'error' }[e.status];
      if (state && e.stage in stageStatus) stageStatus[e.stage] = state;
    },

    // Reset a tab's live state before a run and hand back its fresh AbortController.
    startRun(state, tab, extra = {}) {
      Object.assign(state, { running: true, error: null, stageStatus: idleStages(tab), ...extra });
      return state.controller = new AbortController();
    },
    cancel(state) { state.controller?.abort(); state.running = false; },

    // ── Ask ─────────────────────────────────────────────────────

    async runAsk() {
      const a = this.ask;
      const controller = this.startRun(a, 'ask', {
        events: [], result: null, thinkingText: '', outputText: '',
      });
      const payload = this.buildPayload('ask');
      const sel = this.buildSelectionFrom(a.sel);
      if (sel) payload.profile_selection = sel;
      await this.streamRequest('/ask', payload, {
        progress: (e) => {
          a.events.push({ type: 'progress', ...e });
          this.applyStage(a.stageStatus, e);
          this.$nextTick(() => scrollToBottom(this.$refs.askEvents));
        },
        token: (e) => {
          a.events.push({ type: 'token', ...e });
          if (e.is_thinking) a.thinkingText += e.text; else a.outputText += e.text;
        },
        result: (e) => {
          a.events.push({ type: 'result' });
          a.result = e; a.running = false;
          if (e.error) a.stageStatus.pipeline = 'error';
        },
        error: (e) => {
          a.events.push({ type: 'error', ...e });
          a.running = false;
          for (const k in a.stageStatus) if (a.stageStatus[k] === 'active') a.stageStatus[k] = 'error';
        },
      }, controller);
      a.running = false;
    },

    // ── Table/column selection widget (shared by Profile + Ask) ──
    // `target` is 'profile' (pick what to profile — every table) or 'ask' (pick which
    // *already-profiled* tables to include for a question).
    async loadSchema(target = 'profile') {
      const st = this[target];
      st.loadingSchema = true; st.schemaError = null;
      try {
        const j = await this.postJSON('/schema', { db_uri: this.config.db_uri });
        // Profile shows the full live schema with cached columns locked; Ask is restricted
        // to what's cached, freely toggleable.
        const isAsk = target === 'ask';
        st.sel = this.seedSelection(isAsk ? (j.cached_columns || {}) : j.tables,
          j.cached || {}, j.cached_columns || {}, !isAsk);
        if (isAsk) st.pickerOpen = true;
      } catch (e) { st.schemaError = String(e.message || e); }
      finally { st.loadingSchema = false; }
    },

    // lockCached=true (Profile): cached columns start checked but locked — shown for context,
    // never re-profiled by Run. Ask (false) leaves cached columns freely toggleable.
    seedSelection(tables, cached, cachedCols = {}, lockCached = false) {
      return Object.fromEntries(Object.entries(tables).map(([t, cols]) => {
        const on = new Set(cachedCols[t] || []);  // cached columns start checked
        const node = { open: false, search: '', cached: cached[t] || null, locked: {}, cols: {} };
        for (const c of cols) {
          node.cols[c] = on.has(c);
          if (lockCached && on.has(c)) node.locked[c] = true;
        }
        return [t, node];
      }));
    },
    colKeys(node) { return Object.keys(node.cols); },
    isLocked(node, c) { return !!node.locked?.[c]; },
    colCount(node) { return Object.values(node.cols).filter(Boolean).length; },
    selectedCols(node) { return this.colKeys(node).filter(c => node.cols[c]); },
    lockedCols(node) { return this.colKeys(node).filter(c => this.isLocked(node, c)); },
    // Columns Run will actually (re)profile: selected and not already cached/locked.
    newCols(node) { return this.selectedCols(node).filter(c => !this.isLocked(node, c)); },
    colBadge(node) {
      const [picked, cached] = [this.newCols(node).length, this.lockedCols(node).length];
      return [picked && picked + ' new', cached && cached + ' cached'].filter(Boolean).join(' · ') || '0 cols';
    },
    // Column names matching the per-table dropdown search box (case-insensitive).
    filteredCols(node) {
      const q = (node.search || '').toLowerCase();
      return this.colKeys(node).filter(c => c.toLowerCase().includes(q));
    },
    setTable(node, v) { for (const c in node.cols) if (!this.isLocked(node, c)) node.cols[c] = v; },
    selectAll(sel, v) { for (const n of Object.values(sel)) this.setTable(n, v); },
    selectionSummary(sel) {
      return Object.values(sel).filter(n => this.newCols(n).length).length + '/' + Object.keys(sel).length + ' tables';
    },
    agoBadge(iso) {
      if (!iso) return '';
      const secs = (Date.now() - Date.parse(iso)) / 1000;
      const [v, u] = secs < 3600 ? [secs / 60, 'm'] : secs < 86400 ? [secs / 3600, 'h'] : [secs / 86400, 'd'];
      return 'profiled ' + Math.max(1, Math.round(v)) + u + ' ago';
    },

    // {table:[cols]} of checked entries, or null for "no filter — use the whole cached
    // profile". Nothing checked means the picker was never used, so a naive Load + Run
    // still sees the full schema. Selecting everything sends everything, which the
    // server filters to the same thing.
    buildSelectionFrom(sel) {
      const out = Object.fromEntries(Object.entries(sel)
        .map(([t, n]) => [t, this.selectedCols(n)]).filter(([, cols]) => cols.length));
      return Object.keys(out).length ? out : null;
    },

    // Profile Run selection: only newly-picked (uncached) columns, so cached ones are never
    // needlessly re-profiled. {table:[newCols]} → build those; {} → nothing new on a cached
    // DB (no-op); null → nothing cached at all → profile the whole DB.
    buildProfileSelection(sel) {
      const out = {}; let anyLocked = false;
      for (const [t, n] of Object.entries(sel)) {
        if (this.lockedCols(n).length) anyLocked = true;
        if (this.newCols(n).length) out[t] = this.newCols(n);
      }
      return Object.keys(out).length ? out : (anyLocked ? {} : null);
    },

    // ── Profile ─────────────────────────────────────────────────

    async runProfile() {
      const p = this.profile;
      const sel = this.buildProfileSelection(p.sel);
      if (sel && !Object.keys(sel).length) {  // only cached selected — nothing new to do
        p.error = 'Nothing new to profile — cached columns are excluded. Edit them from “Show cached profile”.';
        return;
      }
      const controller = this.startRun(p, 'profile', { events: [], result: null });
      const payload = this.buildPayload('profile');
      if (sel) payload.profile_selection = sel;
      await this.streamRequest('/profile', payload, {
        ...this.profileProgress(),
        result: (e) => { p.result = this.shapeCache(e); p.running = false; p.stageStatus.pipeline = 'done'; },
        error: (e) => { p.error = e.error; p.running = false; },
      }, controller);
      p.running = false;
    },

    // Progress handler shared by a profile run and a single-table cache refresh.
    profileProgress() {
      const p = this.profile;
      return {
        progress: (e) => {
          p.events.push({ type: 'progress', ...e });
          this.applyStage(p.stageStatus, e);
          this.$nextTick(() => scrollToBottom(this.$refs.profileEvents));
        },
      };
    },

    // Mirrors profiler.stats.group(): flat "db|table|column" keys → {table: {column: value}}.
    // Only the last two segments matter, so any db prefix works.
    groupFlat(doc) {
      const out = {};
      for (const [k, v] of Object.entries(doc?.columns || {})) {
        const p = k.split('|');
        if (p.length >= 2) (out[p[p.length - 2]] ||= {})[p[p.length - 1]] = v;
      }
      return out;
    },
    // The four artifacts → the {tables, short, long, kb} shape the templates render.
    shapeCache(payload) {
      const meta = payload?.profile?.meta || {};
      const tables = Object.fromEntries(Object.entries(this.groupFlat(payload?.profile))
        .map(([t, columns]) => [t, { row_count: meta[t]?.row_count ?? 0, columns }]));
      return {
        tables, kb: Object.values(this.groupFlat(payload?.kb)).flatMap(e => Object.values(e)),
        short: this.groupFlat(payload?.meaning_base_short), long: this.groupFlat(payload?.meaning_base_long),
      };
    },
    // Mirrors DatabaseSummary.describe(): short first, else long — a dataset meaning base
    // ships long only, and profile_summary can generate just one kind.
    summaryFor(src, table, col) { return src?.short?.[table]?.[col] || src?.long?.[table]?.[col] || ''; },

    // ── Cached-profile management (view / delete / refresh) ──────

    async showCachedProfile() {
      this.profile.error = null;
      try {
        this.profile.cached = this.shapeCache(await this.postJSON('/cache', this.buildPayload('profile')));
        this.profile.showCached = true;
      } catch (e) { this.profile.error = String(e.message || e); }
    },
    async deleteCache(table, col = null) {
      this.profile.cacheBusy = true; this.profile.error = null;
      try {
        await this.postJSON('/cache/delete',
          { db_uri: this.config.db_uri, table, columns: col ? [col] : null });
        await this.showCachedProfile();
      } catch (e) { this.profile.error = String(e.message || e); }
      finally { this.profile.cacheBusy = false; }
    },
    // Re-profile a table (cols = all its cached columns) or a single column, overwriting the cache.
    async refreshCache(table, cols) {
      const p = this.profile;
      p.cacheBusy = true; p.error = null; p.events = []; p.stageStatus = idleStages('profile');
      const payload = { ...this.buildPayload('profile'), profile_selection: { [table]: cols } };
      await this.streamRequest('/profile', payload, {
        ...this.profileProgress(),
        result: async () => { await this.showCachedProfile(); },
        error: (e) => { p.error = e.error; },
      });
      p.cacheBusy = false;
    },

    // ── Benchmark ───────────────────────────────────────────────

    // Preview the filtered records a run would execute (no run).
    async previewBenchmark() {
      const b = this.benchmark;
      b.previewing = true; b.error = null;
      try {
        const j = await this.postJSON('/benchmark/preview', this.buildPayload('benchmark'));
        b.examples = j.examples; b.previewTotal = j.total;
      } catch (e) { b.error = String(e.message || e); }
      finally { b.previewing = false; }
    },

    async runBenchmark() {
      const b = this.benchmark;
      const controller = this.startRun(b, 'benchmark',
        { started: null, results: [], report: null, done: 0, current: null, usage: {} });
      await this.streamRequest('/benchmark', this.buildPayload('benchmark'), {
        progress: (e) => {
          if (e.stage === 'benchmark' && e.status === 'started') b.started = e;
          else this.applyStage(b.stageStatus, e);  // per-question pipeline stages
        },
        example_start: (e) => { b.current = e; b.stageStatus = idleStages('benchmark'); },
        example: (e) => {
          b.results.push(e); b.done++;
          const u = e.usage || {};
          for (const k of ['prompt_tokens', 'completion_tokens', 'total_tokens', 'total_cost_usd', 'num_calls']) {
            b.usage[k] = (b.usage[k] || 0) + (u[k] || 0);
          }
        },
        scores: (e) => {
          const by = Object.fromEntries(e.map(s => [s.id, s]));
          b.results = b.results.map(r => by[r.id]
            ? { ...r, verdict: { ...r.verdict, ...by[r.id].verdict } } : r);
        },
        result: (e) => { b.report = e; b.running = false; },
        error: (e) => { b.error = e.error; b.running = false; },
      }, controller);
      b.running = false;
    },
  };
}
