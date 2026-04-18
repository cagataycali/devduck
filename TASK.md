# TASK: Integrate strands-inspect into DevDuck

**Goal:** Deeply integrate [strands-inspect](https://github.com/cagataycali/strands-inspect) into DevDuck so every agent turn runs under observable, policy-enforced execution — with full session replay and a visual player.

**Status:** 📋 Planning → pick up any phase, any item

---

## 🎯 Outcomes When Done

1. `pip install devduck` pulls in `strands-inspect` automatically
2. Agent can call `inspect_tool` at runtime (scan packages, profile code, generate examples)
3. Every `DevDuck.__call__(query)` is wrapped in `@watch` — syscalls, memory, CPU captured per turn
4. `devduck service install --policy sandbox` persists a device agent under a kernel-enforced policy
5. Existing `--record` / `--resume` gains a per-turn inspect layer (syscall traces, memory timelines) inside the session ZIP
6. `docs/session-player.html` renders DevDuck events AND strands-inspect data side by side
7. Everything is opt-in, backwards compatible, and additive — no existing flag breaks

---

## 📐 Design Decisions (locked)

| Q | Decision | Why |
|---|----------|-----|
| Policy mutability mid-session? | **Lock at startup** | Agents shouldn't escalate own privileges |
| Kernel sandbox (`@lock`) for main loop? | **No** — only for new `execute_untrusted_code` tool | Fork breaks hot-reload, threads, servers |
| Dump storage location | **Inside DevDuck session ZIP** | One portable artifact |
| Policy sources & priority | CLI > identity DB > env var > config file > default | Matches existing devduck patterns |
| Default policy | `"allow_all"` (log-only) | Zero behavior change for current users |
| Wrap granularity | **Per-turn** (`DevDuck.__call__`), not `cli()` | Idle time shouldn't pollute traces |

---

## 📋 Phase Breakdown (4 layered commits)

Each phase is independently shippable. Finish one before starting the next.

---

## PHASE 1 — Dependency + Runtime Tool
**ETA:** ~1 hour · **Risk:** low · **Breakage surface:** zero

### 1.1 Add dependency
- [x] Add `"strands-inspect"` to `dependencies` in `/Users/cagatay/devduck/pyproject.toml`
- [x] ~~Bump version~~ — uses `setuptools_scm`, git tag handles it
- [x] Verify local install: `pip install -e .` from repo root

### 1.2 Create `devduck/tools/inspect.py`
- [x] New file that re-exports `strands_inspect.inspect_tool` as `inspect` (devduck naming)
- [x] Include docstring noting it's a passthrough
- [x] Keep file minimal (~15 lines) — all logic lives in strands-inspect
- [x] Handle `ImportError` gracefully (fall back to stub tool that tells user to install)

### 1.3 Wire into default tool config
- [x] In `devduck/__init__.py` `_select_default_tools()`, append `inspect` to the `devduck.tools:...` group
- [x] Verify via `devduck "use inspect_tool to scan json package"` — should work end-to-end

### 1.4 Docs
- [x] Add a "🔍 Code Inspection" section to README.md under existing tool list
- [x] Update the system prompt's "Available DevDuck Tools" list in `__init__.py`

### ✅ Phase 1 Acceptance
```bash
pip install -e .
devduck "inspect_tool action=scan target=pathlib"  # returns scan output
```

---

## PHASE 2 — Wrap `DevDuck.__call__` + Merge into SessionRecorder
**ETA:** ~3–4 hours · **Risk:** medium (touches hot path) · **Breakage surface:** session recording format

### 2.1 Resolve policy at `DevDuck.__init__`
- [x] Add `self.policy` field populated in this priority order:
  1. Constructor kwarg `policy=...`
  2. `DEVDUCK_POLICY` env var
  3. `.devduck.toml` → `[inspect] policy = "..."`
  4. Default: `"allow_all"`
- [x] Log the resolved policy at startup (`🦆 Policy: allow_all` etc)
- [ ] Store raw policy spec (for the ZIP) AND resolved dict

### 2.2 Wrap `DevDuck.__call__`
- [x] In `DevDuck.__call__(query)`, wrap the inner `self.agent(query_with_context)` call
- [x] Use a NEW thin wrapper that calls `strands_inspect.watch(...)` on an inner function — don't decorate the method directly (dynamic policy)
- [x] Wrap with `print_summary=False` and `dump=True` to our custom dir
- [x] Capture the `InspectSession` as return value; attach to `DevDuck` as `self._last_inspect_session`

### 2.3 Integrate with SessionRecorder
- [x] When recording active, write the `.dill` to `/tmp/devduck/recordings/{session_id}/inspect/turn_{N}.dill`
- [x] Add a new event type `inspect.turn_complete` to `EventBuffer` with payload: `{turn_idx, dill_path, syscalls_count, memory_peak_kb, denied_count, wall_time_ms}`
- [ ] Modify `SessionRecorder.export()` to pack `inspect/*.dill` into the ZIP (alongside `events.jsonl`, `snapshots.json`)

### 2.4 Graceful fallback when strands-inspect missing
- [x] Import strands-inspect inside `DevDuck.__init__` in a try/except; if unavailable, log warning + set `self._inspect_enabled = False`
- [x] `DevDuck.__call__` only wraps when `self._inspect_enabled` — otherwise runs as today

### 2.5 Test matrix
- [ ] Without `--record`: policy still enforced, no dills written ✔
- [ ] With `--record`: dills appear in `inspect/` subdir, get zipped on exit ✔
- [ ] Policy=`deny_network` blocks network tool calls ✔ (test with use_github tool)
- [ ] Hot-reload still works ✔
- [ ] Ambient mode still works ✔

### ✅ Phase 2 Acceptance
```bash
devduck --record --policy sandbox
> "write /tmp/test.txt"   # denied
> exit
unzip -l /tmp/devduck/recordings/*.zip | grep inspect/   # shows per-turn dills
```

---

## PHASE 3 — CLI + Service Integration
**ETA:** ~2 hours · **Risk:** low · **Breakage surface:** service env file format

### 3.1 Add `--policy` CLI flag
- [x] In `cli()` argparse, add `--policy NAME_OR_PATH` to main parser (not just subcommands)
- [x] Flag sets `DEVDUCK_POLICY` env var before `DevDuck()` is instantiated
- [x] Support both named presets (`sandbox`, `strict`, `deny_network`) AND path to `.toml` / `.json` policy file
- [ ] Show resolved policy in `devduck status` output

### 3.2 Service install integration
- [x] In `tools/service.py`, add `--policy` to the `install` subparser
- [x] `InstallPlan` gets a `policy` field
- [x] `env_file_content()` emits `DEVDUCK_POLICY=<value>` when set
- [x] Updated `show` subcommand displays policy in the summary

### 3.3 Identity DB integration
- [ ] Add `policy` column (TEXT, default `""`) to the `identities` table schema in `tools/identity.py`
- [ ] Add to `_TEXT_FIELDS` so `update` supports it
- [ ] Add to `_ENV_MAP` so `activate` exports it to `DEVDUCK_POLICY`
- [ ] Schema migration: in `_ensure_schema`, use `ALTER TABLE ... ADD COLUMN IF NOT EXISTS` pattern (SQLite-safe via `try/except`)
- [ ] Display policy in `list` output table

### 3.4 Config file support
- [ ] Support `.devduck.toml` in cwd or `~/.devduck/config.toml` with:
  ```toml
  [inspect]
  policy = "sandbox"
  # or custom policy:
  [inspect.policy]
  "file.write" = "deny"
  "network" = { action = "allow", hosts = ["api.openai.com"] }
  ```
- [ ] Load via `tomllib`/`tomli` just like strands-inspect does internally

### ✅ Phase 3 Acceptance
```bash
devduck service install --name telegram-bot --policy deny_network --env TELEGRAM_BOT_TOKEN=xxx
devduck service logs --name telegram-bot   # shows policy engaged
identity --action activate --name reviewer  # exports DEVDUCK_POLICY from DB
```

---

## PHASE 4 — Session Player HTML
**ETA:** ~3 hours · **Risk:** low (UI only) · **Breakage surface:** none

### 4.1 Base on existing `old-docs-plain-html/session-player.html`
- [ ] Copy → `docs/session-player.html` (or replace existing; remove "old-docs-plain-html" path)
- [ ] Keep all existing features: timeline, filters, snapshots, resume commands
- [ ] Load `dill` files using `pickle-to-js` or a lightweight parser (or embed as base64 + show metadata from events.jsonl instead — simpler path)

### 4.2 Simpler path — read inspect events from events.jsonl
- [ ] Each `inspect.turn_complete` event already has `{turn_idx, syscalls_count, memory_peak_kb, denied_count}` — render these inline without parsing dill
- [ ] Add a new "🔍 Inspect" tab alongside "Timeline"
- [ ] Per turn card shows:
  - Syscalls count (with icons by type: file/net/proc)
  - Memory peak
  - Denied calls (red badge)
  - Wall time

### 4.3 Policy panel
- [ ] If `metadata.json` contains `policy` key, render it at top of session header as a collapsible "🛡️ Policy" card
- [ ] Show which categories are `allow`/`deny`/`log`

### 4.4 Merged event timeline
- [ ] Existing event timeline gains a 4th layer color: `inspect` (red/orange)
- [ ] Clicking an inspect event opens a modal showing the full syscall list for that turn

### 4.5 Bonus: detail view using inspect data
- [ ] Optional: client-side dill parser (use `pyodide` lazy-loaded) to unpack actual `.dill` → show full syscall-by-syscall timeline, memory graph
- [ ] If too heavy, skip — the summary stats in events.jsonl are already useful

### 4.6 Ship
- [ ] Move into `devduck/docs/session-player.html`
- [ ] Update `pyproject.toml` `[tool.setuptools.package-data]` to include it
- [ ] Add `devduck player` CLI subcommand that opens the HTML with the chosen session pre-loaded

### ✅ Phase 4 Acceptance
```bash
devduck player /tmp/devduck/recordings/session-XXX.zip
# Opens browser, shows DevDuck events + Inspect tab + Policy panel
```

---

## PHASE 5 (stretch) — Kernel sandbox tool
**ETA:** ~1 hour · **Risk:** low (isolated tool)

### 5.1 New `devduck/tools/execute_sandboxed.py`
- [ ] Thin wrapper around `strands_inspect.lock(policy=...)`
- [ ] Input: arbitrary Python code + policy name
- [ ] Returns: stdout, stderr, return value, violations
- [ ] Agent can use this to run untrusted LLM-generated code safely

### 5.2 Docs
- [ ] Add example: "Ask the agent to solve a problem in Python, it writes code, runs via execute_sandboxed with `strict` policy"

---

## 🧪 Cross-cutting: tests

- [ ] `tests/test_inspect_integration.py` — unit tests for policy resolution, wrapping, dump location
- [ ] Manual smoke test checklist in this file under each phase
- [ ] CI: ensure existing tests pass, add a matrix entry that tests WITH and WITHOUT strands-inspect installed (via pip uninstall in CI step)

---

## 📦 Files that will be created/modified

### New files
- `devduck/tools/inspect.py`
- `devduck/tools/execute_sandboxed.py` (phase 5)
- `docs/session-player.html` (moved from `old-docs-plain-html/`)
- `tests/test_inspect_integration.py`
- `.devduck.toml.example`

### Modified files
- `pyproject.toml` — dependency + package_data
- `devduck/__init__.py` — wrap `__call__`, resolve policy, config loading, `--policy` flag
- `devduck/tools/service.py` — `--policy` in install
- `devduck/tools/identity.py` — add `policy` column + env mapping
- `devduck/callback_handler.py` — optional: show policy status in startup output
- `README.md` — new section
- `AGENTS.md` — mention inspect integration

### Unchanged (verified)
- All other tools
- Session-player for existing sessions (backwards compat: old sessions just don't have inspect data)

---

## 🚦 Ordering for safe rollout

1. **Phase 1** → merge, users can opt-in via `inspect_tool` calls
2. **Phase 2** with `allow_all` default → transparent; recorded sessions gain inspect data, nothing denied
3. **Phase 3** → power users configure strict policies for services
4. **Phase 4** → visualization layer
5. **Phase 5** → advanced sandboxing

Each phase independently releasable as a minor version bump.

---

## 📝 Notes / Open Items

- **strands-inspect version pin?** — start unpinned `"strands-inspect"`, pin later if breaking changes ship
- **dill vs pickle** — strands-inspect prefers dill; already a transitive dep here (we use it too). No new install weight.
- **Windows support** — strands-inspect `@lock` is Linux/macOS only. `@watch` is cross-platform. Document the limitation.
- **Performance** — `@watch` adds ~5-10% overhead per turn. Acceptable. Can bypass with `policy=None`.
- **Security note** — `@watch` is observational only; `@lock` is enforcement. DevDuck's main wrap uses `@watch`. This means a malicious tool *could* theoretically bypass hooks via C extensions. This is documented; kernel sandbox is opt-in per-code-snippet via `execute_sandboxed` tool.

---

## 🎯 Start here

Pick any checkbox. When you finish an item:
1. Check the box
2. Git commit with message `inspect: <item>`
3. Pass session to next item / next phase

**Recommended first touch:** Phase 1.1 → 1.2 → 1.3 (under 1 hour, proves the stack)
