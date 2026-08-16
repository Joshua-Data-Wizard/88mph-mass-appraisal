# A Practical Guide to `uv`

*You do not need to know Python packaging to use this repository. You need three commands.*

This guide is for anyone who just cloned this repo and wants it running — including
people who have never written Python. It also covers exactly what gets installed on
your machine and how to remove all of it, because that is usually the first question
your IT department will ask.

---

## The short version

```powershell
uv sync                                    # build the environment
uv run python scripts/check_setup.py       # confirm your machine can run the demos
uv run jupyter lab                         # open the notebooks
```

That's it. You do **not** need Python installed first. You do **not** need to
"activate" anything. If those three commands work, you're done.

---

## 1. What is `uv`, and why this repo uses it

`uv` is one tool that replaces four older ones:

| Old tool | What it did | Now |
|---|---|---|
| `pyenv` | install Python versions | `uv python install` |
| `venv` | make an isolated environment | `uv sync` |
| `pip` | install packages | `uv add` |
| `pip-tools` | lock exact versions | `uv.lock` (automatic) |

We chose it for one reason that matters to this project: **reproducibility.**
This repo ships a lockfile recording the exact version of all ~200 packages.
When you run `uv sync`, you get byte-identical packages to the ones the demos
were built and tested against. Not "close enough" — identical.

That's the difference between a demo that works on the presenter's laptop and a
demo that works on yours.

It's also very fast. A full environment build here takes under a minute.

---

## 2. Installing `uv`

Pick one. The standalone installer is the easiest and needs no admin rights.

**Windows (PowerShell):**
```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

**macOS / Linux:**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Or via a package manager:**
```bash
winget install --id=astral-sh.uv -e     # Windows
brew install uv                          # macOS
pipx install uv                          # anywhere with pipx
```

Close and reopen your terminal, then check it worked:

```powershell
uv --version
```

To update later: `uv self update`

---

## 3. Setting up this project

From the repository folder:

```powershell
uv sync
```

That single command does five things:

1. Reads `pyproject.toml` to see what the project needs
2. Downloads Python 3.12 if you don't have it *(it does not touch your system Python)*
3. Creates a `.venv/` folder inside the project
4. Installs all ~200 packages at the exact versions in `uv.lock`
5. Installs this project's own `delorean` package so notebooks can import it

Expect roughly **3 GB** the first time, most of it PyTorch. Later runs are seconds.

Then confirm your machine can actually run things:

```powershell
uv run python scripts/check_setup.py
```

This checks your GPU, tests 4-bit quantization, and runs two real models. If it
says all checks passed, every demo in this repo will work on your machine. If you
have no GPU, it will tell you to use the `_api` notebooks instead.

---

## 4. The mental model

Three files, three jobs. Understanding this is 90% of using `uv` well.

| File | Meaning | Who writes it |
|---|---|---|
| `pyproject.toml` | **What we want** — loose ranges like `transformers>=5.2` | You, by hand |
| `uv.lock` | **What we got** — exact versions and hashes for every package | `uv`, automatically |
| `.venv/` | **Where it lives** — the actual installed files | `uv`, automatically |

`pyproject.toml` says "any transformers newer than 5.2." `uv.lock` says
"transformers 5.15.0, and here is its checksum." The lockfile is what makes your
install match everyone else's.

### Which files belong in Git

| File | Commit it? | Why |
|---|---|---|
| `pyproject.toml` | **Yes** | It defines the project |
| `uv.lock` | **Yes** | This is what makes builds reproducible |
| `.python-version` | **Yes** | Records which Python this project uses |
| `.venv/` | **No** | Gigabytes of installed files. Disposable — `uv sync` rebuilds it |

People skip `uv.lock` and shouldn't. Without it, someone installing next month
gets whatever versions resolve *that* day, and your demo works while theirs doesn't.

---

## 5. Everyday commands

```powershell
uv sync                       # make the environment match pyproject.toml
uv add pandas                 # add a package (updates pyproject.toml AND uv.lock)
uv remove pandas              # remove one
uv run python myscript.py     # run a script inside the environment
uv run jupyter lab            # start Jupyter
uv sync --upgrade             # update packages within the ranges you declared
uv tree                       # see what depends on what
```

### `uv run` replaces "activate"

With older tools you'd run `activate` first, then `python`. **You don't do that here.**
`uv run` finds the environment itself, and re-syncs it first if it's out of date.

The upside: you can never accidentally run against the wrong Python. If you didn't
type `uv run`, you aren't in the environment.

---

## 6. The one rule

> ### Never run `pip install` in a uv project.

Use `uv add` instead.

`pip install` puts a package into `.venv/` that `pyproject.toml` doesn't know about.
It exists on your machine and nowhere else. Everything works for you, and then
fails for every colleague who clones the repo — which is the exact problem `uv`
exists to prevent.

```powershell
uv add requests        # correct
pip install requests   # do not
```

---

## 7. What `uv` puts on your computer

Worth knowing, and worth being able to tell your IT department.

| What | Where (Windows) | Size |
|---|---|---|
| The `uv` program | `C:\Users\<you>\.local\bin\uv.exe` | ~35 MB |
| Python interpreters | `C:\Users\<you>\AppData\Roaming\uv\python\` | ~63 MB per version |
| Download cache | `C:\Users\<you>\AppData\Local\uv\cache\` | grows over time |
| This project's environment | `<repo>\.venv\` | ~5 GB |
| Model weights | `C:\Users\<you>\.cache\huggingface\` | several GB |

**Everything is inside your user profile.** No admin rights are required, nothing
goes into `Program Files`, nothing is written to the Windows registry, and no other
user account on the machine is affected.

Notably, `uv` does **not** hijack the `python` command. If you type `python` in a
terminal, you get whatever you had before — system Python, Anaconda, or nothing.
`uv` only ever uses its own interpreters, and only when you ask it to.

### Removing all of it

```powershell
# 1. This project's environment (safe — uv sync rebuilds it)
Remove-Item -Recurse -Force .venv

# 2. The downloaded interpreters and cache
uv cache clean
Remove-Item -Recurse -Force "$(uv python dir)"
Remove-Item -Recurse -Force "$(uv tool dir)"

# 3. uv itself
Remove-Item "$HOME\.local\bin\uv.exe", "$HOME\.local\bin\uvx.exe", "$HOME\.local\bin\uvw.exe"

# 4. Downloaded model weights (several GB)
Remove-Item -Recurse -Force "$HOME\.cache\huggingface"
```

Nothing to uninstall from Control Panel. No registry cleanup. No leftover PATH entries.

---

## 8. Troubleshooting

These are real problems we hit while building this repo, not hypotheticals.

### "Python was not found" when you type `python`

**This is fine and expected.** `uv` never uses your system Python. Use `uv run python`
instead of `python`, and it will work.

### `uv sync` fails with `os error 396`

You're in a cloud-synced folder — OneDrive, Dropbox, or Google Drive. `uv` installs
packages using hardlinks, which those filesystems reject.

This repo already fixes it, via `pyproject.toml`:

```toml
[tool.uv]
link-mode = "copy"
```

If you hit it in a *different* project, either add those two lines or set
`UV_LINK_MODE=copy` in your environment.

### A package is installed but behaves impossibly

Symptoms: an import fails for a package `uv pip list` says is installed, or you get
a nonsensical error from a library that obviously works for everyone else.

Cloud-sync folders and aggressive antivirus can interrupt a package extraction
partway through. The metadata lands, so the package *looks* installed, but half its
files are missing. We hit this with `scikit-learn` — 40 of 1055 files.

```powershell
uv sync --reinstall-package <name>
```

If it keeps happening, move the repo somewhere that isn't cloud-synced, such as `C:\dev\`.

### VS Code activates the environment in every terminal

The Python extension does this by default. Since `uv run` doesn't need activation,
you can turn it off. In VS Code settings (JSON):

```json
"python.terminal.activateEnvironment": false
```

For terminals already open, type `deactivate` to get out of one.

If `deactivate` isn't recognized, VS Code activated the environment by setting
environment variables rather than running the activation script, so the command was
never defined. Use the setting above instead.

### Something is just wrong and you want a clean slate

```powershell
Remove-Item -Recurse -Force .venv
uv sync
```

Deleting `.venv` is always safe. It's fully rebuildable from `uv.lock`.

---

## 9. Cheat sheet

| I want to... | Command |
|---|---|
| Set up the project | `uv sync` |
| Check my machine can run it | `uv run python scripts/check_setup.py` |
| Open the notebooks | `uv run jupyter lab` |
| Run a script | `uv run python path/to/script.py` |
| Add a package | `uv add <name>` |
| Remove a package | `uv remove <name>` |
| Start over | `Remove-Item -Recurse -Force .venv` then `uv sync` |
| Update uv | `uv self update` |
| See where Python lives | `uv python dir` |
| See what's installed | `uv pip list` |

---

## Further reading

- Official docs: <https://docs.astral.sh/uv/>
- Why lockfiles matter: <https://docs.astral.sh/uv/concepts/projects/lock/>
- Using uv with PyTorch: <https://docs.astral.sh/uv/guides/integration/pytorch/>

---

*"Roads? Where we're going, we don't need roads."* — but we do need a reproducible
environment, and that is what this buys you.
