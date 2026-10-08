# `lightx doctor` — Find Version Problems, Fix Them With Verified Patches

Library upgrades break agent projects all the time: a function moves, a package splits in two (`langchain` → `langchain_community`), an API is deprecated. `lightx doctor` finds these problems. With `--fix`, it asks an LLM for a patch, **proves the patch works in the sandbox**, shows you the diff, and changes your code only if you say yes.

```bash
lightx doctor                      # check the current project; changes nothing
lightx doctor path/to/project --fix --base-url http://localhost:11434/v1 --model qwen2.5:3b
```

---

## What it checks

| Check | Finds | How |
|---|---|---|
| **Dependencies** | Requirements not installed or at the wrong version (`pyproject.toml`, `requirements*.txt`, LightAgentX's own); conflicts between installed packages | Compares installed metadata with each requirement, plus `pip check` |
| **Imports** | Modules that no longer exist and names removed from a module, with **where they live now** in the installed library | Imports every external module of your project in the sandbox, then statically searches the installed package and its siblings (`foo_core`, `foo_community`, …) for the missing name |
| **Deprecations** | `DeprecationWarning` / `FutureWarning` raised **in your code**, with file and line | Runs your tests in the sandbox with warnings enabled, and keeps only warnings that point at project files |
| **Saved agents** | Snapshots in the old layout, archives in unsupported formats, registries without a format marker, corrupted registry history | Reads the files; re-hashes registry history |

**Checking never changes your project.** It works on a copy, and your project's code (imports, tests) runs only **inside the sandbox**: read-only system, no network, secrets hidden.

---

## How `--fix` works

```
for each problem:
  1. PROPOSE   the LLM gets: the error, your code around the line, the installed library
               version, and where the missing name/module lives now
  2. APPLY     the patch goes to a fresh COPY of the project (never your files)
  3. VERIFY    in the sandbox: the problem must be gone, no import that worked may break,
               and no test may fail that passed before
               (if it fails, the reason goes back to the LLM for another attempt)
  4. SHOW      you see the diff and the verification result
  5. APPLY?    only if you answer y; the original is backed up first
then: re-check, because fixing one problem can reveal another
      (e.g. a deprecation that only shows once the tests can run)
```

| Kind | Proposed by | Verified by | Applied by |
|---|---|---|---|
| **Code** (imports, deprecations) | The LLM: a JSON list of exact search/replace edits | Re-running the check and your tests on the patched copy, in the sandbox | Writing the files, after a backup to `~/.lightx/doctor/backups/<time>/` |
| **Dependencies** | The requirement itself (`pip install "pkg>=2"`) | Installing it into a throwaway overlay, then running imports and tests with it, in the sandbox | `pip install` in your environment |
| **Saved agents** | A deterministic migration (no LLM) | Restoring the migrated snapshot, or re-hashing every registry version | Rewriting the file (backed up), or writing the registry's `FORMAT` marker |

**Safety rules**
- **Every change asks you first.** With no terminal attached (CI, scripts), nothing is applied.
- **Patches may not touch tests.** A fix that edits tests is refused, so it can't "pass" by changing the tests.
- **Search/replace edits must match exactly once,** or the attempt is rejected.
- **Only patches that verify are offered.**
- **`--only code,migration`** limits what is attempted. For example, leave `dependency` out to never touch your Python environment.

---

## Example (from the test suite)

A library moved `load_data` into `fakelib.io`, split `fakelib.legacy` out into a separate package `fakelib_community`, and deprecated `compute()`:

```
Imports
  ✗ 'load_data' no longer exists in fakelib  app.py:2  [fixable]
      fakelib is installed; 'load_data' is now defined in: fakelib.io
  ✗ Can't import fakelib.legacy  app.py:3  [fixable]
      ModuleNotFoundError: No module named 'fakelib.legacy'
      Possibly moved to: fakelib_community.legacy
```

With a local `qwen2.5:3b`, `--fix` proposed and verified `from fakelib.io import load_data` and `from fakelib_community.legacy import helper`. Each was applied after approval. The re-check then found `DeprecationWarning ... app.py:8` (the tests could only run once the imports were fixed), and fixed it with `compute_v2()`. The project's tests pass afterwards, even with deprecation warnings turned into errors.

---

## Options

| Option | Meaning |
|---|---|
| `--fix` | Propose, verify and (with approval) apply fixes |
| `--only code,dependency,migration` | Which kinds of fixes to attempt |
| `--json` | Machine-readable findings; the exit code is 1 when errors are found |
| `--agents PATH` | Extra snapshot, archive or registry to check (repeatable). `~/.lightx/agents` is always checked |
| `--test-cmd "..."` / `--no-tests` / `--test-timeout` | Control how tests run (default: `pytest` if a `tests/` folder exists) |
| `--max-attempts N` | LLM attempts per problem (default 2) |
| `--provider / --model / --base-url` | Which LLM; it uses your API key from the environment, or a local server |
| `--backend` | Sandbox backend for running your code (`auto`, `bubblewrap`, `subprocess`) |

---

## Limitations (honest)

- **Verification is only as strong as your tests.** Without tests, a fix is verified only by re-running the check that found it.
- **Some API changes don't fail at import time** (a changed argument, different return values). The doctor sees them only through failing tests or deprecation warnings.
- **The "where it lives now" search is static and by name.** It finds moved and split code, but not *renamed* functions; the LLM has to work those out from the library version and context.
- **Dependency fixes need network access** to verify, and applying them changes your Python environment. That's why they can be excluded with `--only`.
- **Patches come from an LLM.** Verification catches patches that don't work, not patches that work but are inelegant. Read the diff before approving.
- **Tested on Linux** (bubblewrap sandbox). Elsewhere, code runs under the plain subprocess backend, which isn't isolated.
