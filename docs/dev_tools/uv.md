# uv

uv 是一款的用 Rust 编写的极速 Python 包管理器，旨在统一 pip, pip-tools, poetry, virtualenv 的功能。uv 由 Astral 公司开发，它的另一款著名产品是 Ruff，一款用 Rust 编写的极速 Python 代码检查工具。

## Installation

=== "macOS"

    ```bash
    brew install uv
    # or:
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ```

=== "Linux"

    ```bash
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ```

=== "Windows"

    ```bash
    winget install uv
    # or:
    irm https://astral.sh/uv/install.ps1 | iex
    ```

You will see the output like these:

```text
downloading uv 0.9.17 x86_64-unknown-linux-gnu
no checksums to verify
installing to /home/xxx/.local/bin
  uv
  uvx
everything's installed!
```

Validate the installaion:

```bash
uv --version
# output: uv 0.9.17
uvx --version
# output: uvx 0.9.17
```

## 1. Managing projects

> Similar to `npm` in Node.js and `cargo` in Rust.

Step 1:

Initialize the project:

```bash
mkdir my-project
cd my-project
uv init
# output: Initialized project `my-project`
```

This will add:

1. `pyproject.toml` : the description file.
2. `.python-version` : lock the version of Python in the project.
3. `main.py` : an example script.
4. `README.md` .

---

Step 2:

Add dependencies(to replace `pip install` ):

```bash
# for example:
uv add requests
# output:
# Using CPython 3.12.3 interpreter at: /usr/bin/python3.12
# Creating virtual environment at: .venv
# Resolved 6 packages in 1.80s
# Prepared 5 packages in 450ms
# Installed 5 packages in 2ms
#  + certifi==2025.11.12
#  + charset-normalizer==3.4.4
#  + idna==3.11
#  + requests==2.32.5
#  + urllib3==2.6.2
```

uv will:

1. Create a virtual environment, defaultly `.venv` .
2. Download and install `requests` .
3. Update the `pyproject.toml` .
4. Generate or update the `uv.lock` .

---

Step 3:

You can the code *without manually activating the virtual environment*:

```bash
uv run main.py
```

---

Step 4:

When you pull the code from others and need to build the environment first, just run:

```bash
uv sync
```

uv will automatically install all the dependencies according to `pyproject.toml, uv.lock` or `requirements.txt`

!!! note
    `uv sync` is a dependency-management command that works like `pip install -r requirements.txt`, but faster, more powerful, and more reliable. It can install every third-party package (dependency) your project needs.

    If `uv sync` is too slow, you can point it to a Chinese mirror by adding the Tsinghua PyPI index. In your `pyproject.toml`, under the `[tool.uv]` section, add:

    ```toml
    [tool.uv]
    index-url = "https://pypi.tuna.tsinghua.edu.cn/simple"
    ```

## 2. Managing Python

> To replace `pyenv` .

Check the available versions of Python:

```bash
uv python list
# output:
# cpython-3.15.0a2-linux-x86_64-gnu                 <download available>
# cpython-3.15.0a2+freethreaded-linux-x86_64-gnu    <download available>
# cpython-3.14.2-linux-x86_64-gnu                   <download available>
# cpython-3.14.2+freethreaded-linux-x86_64-gnu      <download available>
# cpython-3.13.11-linux-x86_64-gnu                  <download available>
# cpython-3.13.11+freethreaded-linux-x86_64-gnu     <download available>
# cpython-3.12.12-linux-x86_64-gnu                  <download available>
# cpython-3.12.3-linux-x86_64-gnu                   /usr/bin/python3.12
# cpython-3.12.3-linux-x86_64-gnu                   /usr/bin/python3 -> python3.12
# cpython-3.12.3-linux-x86_64-gnu                   /usr/bin/python -> python3
# cpython-3.11.14-linux-x86_64-gnu                  <download available>
# cpython-3.10.19-linux-x86_64-gnu                  <download available>
# cpython-3.9.25-linux-x86_64-gnu                   <download available>
# cpython-3.8.20-linux-x86_64-gnu                   <download available>
# pypy-3.11.13-linux-x86_64-gnu                     <download available>
# pypy-3.10.16-linux-x86_64-gnu                     <download available>
# pypy-3.9.19-linux-x86_64-gnu                      <download available>
# pypy-3.8.16-linux-x86_64-gnu                      <download available>
# graalpy-3.12.0-linux-x86_64-gnu                   <download available>
# graalpy-3.11.0-linux-x86_64-gnu                   <download available>
# graalpy-3.10.0-linux-x86_64-gnu                   <download available>
# graalpy-3.8.5-linux-x86_64-gnu                    <download available>
```

Install the specified version of Python:

```bash
uv python install 3.12
uv python install pypy3.10
```

Set the global default version of Python:

```bash
uv python default 3.12
```

Set the version of Python in the project

```bash
uv pin 3.12
```

## 3. Running scripts

> To replace `pipx` .

If you have a single script `script.py` that depends on `pandas` ,there is no need to create a project. You can simply add this at the beginning of the script:

```py
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "pandas",
# ]
# ///
import pandas as pd
print("Pandas is ready!")
```

Then run:

```bash
uv run script.py
```

uv will create a temporary and isolated environment to run the code, and delete it automatically when finished. It is very convenient for quick tests and experiments.

## 4. Managing virtual environment

> To replace `pip` .

Create and activate a virtual environment:

```bash
# Create a virtual environment named `.venv`
uv venv

# macOS/Linux
source .venv/bin/activate

# Windows
.venv\Scripts\activate
```

Install packages:

```bash
uv pip install numpy pandas
```

Export the dependencies:

```bash
uv pip freeze > requirements.txt
```
