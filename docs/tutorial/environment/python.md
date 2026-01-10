# Python on Windows

## 1. Install Anaconda/Miniconda

Anaconda 和 Miniconda 均内置 python，搭配 conda 包管理。

Anaconda 自带科学计算常用库（如 numpy, scipy），有图形化管理界面。

Miniconda 相当于最小化安装，需要自己手动安装各种库（推荐使用）。

下载地址：

官网：<https://www.anaconda.com/download/>

也可在清华镜像站下载，速度更快，适用于没有科学上网时：
<https://mirrors.tuna.tsinghua.edu.cn/anaconda/miniconda/Miniconda3-latest-Windows-x86_64.exe>

安装过程可全默认，注意勾选“添加环境变量”。

以笔者为例，安装路径为：

```text
E:\miniconda3
```

安装完成之后注意打开“设置”——“系统”——“高级系统设置”——“环境变量”——“系统变量”——“PATH”，查看是否已经添加下面三个变量，若没有可手动添加：

```text
E:\miniconda3
E:\miniconda3\Scripts
E:\miniconda3\Library\bin
```

安装完成之后，Windows “开始”界面一般会出现 **Anaconda Prompt**，打开后自动进入 base 环境，请保持 base 环境纯净，不安装任何库，平时使用请创建不同的 conda 虚拟环境满足不同需要。

首先编辑conda配置文件，命令行输入（注意有点号）：

```bash
notepad .condarc
```

将以下内容复制粘贴进去（注意空格，格式不要乱）：

```text
channels:
  - defaults
show_channel_urls: true
default_channels:
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/r
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/msys2
custom_channels:
  conda-forge: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  msys2: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  bioconda: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  menpo: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  pytorch: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  pytorch-lts: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  simpleitk: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
```

然后命令行输入(换清华源)：

```bash
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
pip config set global.trusted-host pypi.tuna.tsinghua.edu.cn
pip config set global.timeout 6000
```

这个步骤也可以直接创建配置文件来实现。用户级的配置文件为 `$env:APPDATA\pip\pip.ini` 。因此运行命令 `notepad $env:APPDATA\pip\pip.ini` ，或者直接进入目录 `C:\Users\ASUS\AppData\Roaming\pip` ，然后创建 txt 文本文件，输入以下内容：

```text
[global]
timeout = 6000
index-url = https://pypi.tuna.tsinghua.edu.cn/simple
trusted-host = pypi.tuna.tsinghua.edu.cn
```

保存文件，重命名并修改扩展名，最后命名为 `pip.ini` 。

仍在上述 Anaconda Prompt，conda 常用命令如下：

```bash
conda create -n test python=3.9 # 指定python版本，-n等同于--name
conda activate test # 激活test虚拟环境
conda deactivate # 退出当前环境，回到base
conda env list
conda info -e
conda info --envs # 上面3个均为查看环境信息，输出中带星号*的为当前所在环境
conda remove -n test --all # 删除test环境
conda config --show # 显示conda配置
conda update conda # 更新conda
conda update Anaconda
```

注意：除了使用 conda 创建环境以外，后续建议使用 pip 安装和更新软件包，尽量避免 pip 和 conda 混用。

## 2. Configure VS Code

下载插件：python，jupyter（必需），ruff（可选）。

打开设置，转到 `setting.json` ，添加如下配置（注意修改路径为自己的 python 路径）：

```json
  //python解释器和设置
  "python.defaultInterpreterPath": "E:\\miniconda3\\envs\\env1\\python.exe",
  "python.condaPath": "E:\\miniconda3\\Scripts\\conda.exe",
  "python.venvPath": "E:\\miniconda3\\envs",
  "python.venvFolders": ["envs", ".pyenv", ".direnv", ".env"],

  "[python]": {
    "editor.tabSize": 4,
    // "editor.defaultFormatter": "ms-python.black-formatter", // 需要下载 black 插件
    "editor.defaultFormatter": "charliermarsh.ruff", // 需要下载 ruff 插件
    "editor.formatOnPaste": true,
    "editor.formatOnSaveMode": "file",
    "editor.formatOnSave": true,
    "editor.formatOnType": true,
    "editor.codeActionsOnSave": {
      "source.fixAll": "explicit",
      "source.organizeImports": "explicit"
    }
  },
  "notebook.formatOnSave.enabled": true,
  "notebook.codeActionsOnSave": {
    "notebook.source.fixAll": "explicit",
    "notebook.source.organizeImports": "explicit"
  },
```

## 3. Configure PyCharm

下载安装过程略（注意添加环境变量）。

新建 python 项目，选择添加本地解释器 Conda 环境，在 conda 可执行文件中选择：（对照自己的路径）

```text
E:\miniconda3\Scripts\conda.exe
```

然后加载，你会看到 base 环境和自己创建的各种环境，选择一个即可，然后就可以运行和调试。

## 4. venv

在当前目录下创建虚拟环境：

```bash
python -m venv [env_name]
python3 -m venv [env_name]

# activate
# linux bash
source [env_name]/bin/activate
# windows
[env_name]/Scripts/activate

# deactivate
deactivate
```

## 5. Pipenv

安装 pipenv：

```bash
pip install pipenv
```

更新 pipenv：

```bash
pip install --user --upgrade pipenv
```

使用：

```bash
# 创建虚拟环境
pipenv --python 3.12
pipenv install

# 进入虚拟环境
pipenv shell

# 安装、卸载依赖包
pipenv install xxx
pip uninstall xxx
pipenv graph

# 删除虚拟环境
pipenv -rm

# 显示虚拟环境安装路径
pipenv --venv

# 查看可更新的包、更新包
pipenv update --outdated
pipenv update
pipenv update xxx


# 从 requirements 导入环境
pipenv install -r path/to/requirements.txt

# 导出 requirements
pipenv lock -r
pipenv lock -r --dev # 只到处开发用的包

```

## 6. Attention

血泪教训：曾经遇到过无论如何包都会下载到 base 环境的问题，无法指向特定的虚拟环境，目前的解决办法是：

```bash
conda create -n test python=3.9 # 指定 python 版本，-n 等同于 --name
```

也就是一定要在创建虚拟环境时指定 Python 的版本号，否则创建好的环境会和 base 环境混在一起。

## 7. Else

进行文件、目录路径操作时，建议使用 `#!py from pathlib import Path` ，替代 `#!py import os.path` 可避免 Windows 和 Linux 路径分隔符不同的问题。

输出日志时，建议使用 `#!py from loguru import logger` 模块，替代 `#!py import logging` 或 `#!py print()` 函数，可方便控制日志级别和输出位置。

进度条，可使用 `#!py from tqdm import tqdm` 模块，或 `#!py import rich`，替代手动打印百分比进度。

获取日历日期、时刻时间，可使用 `#!py from datetime import datetime` 模块。获取时间戳、睡眠、计时，可使用 `#!py import time` 模块。
