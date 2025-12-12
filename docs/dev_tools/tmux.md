# tmux

tmux 是一款强大终端复用工具，允许用户在单一终端窗口中运行多个会话（session），支持分屏、后台运行和重新连接，最大的用处是可以保持程序运行不随 SSH 连接断开而中止。

Linux 下可直接通过 apt 安装：

```bash
sudo apt install tmux
```

## 基本命令

终端操作：这些命令是在普通 Shell中输入的，用于启动或管理 tmux 进程。

```bash
# 新建一个会话，默认名称为 0, 1, 2...
tmux

# 新建一个命名会话
tmux new -s <name>

# 列出所有会话，别名 tmux list-sessions
tmux ls

# 恢复/连接到最近的会话，别名 tmux attach
tmux a

# 接入 <name> 会话
tmux a -t <name>

# 杀死 <name> 会话
tmux kill-session -t <name>

# 重命名会话
tmux rename-session -t <name> <new_name>
```

会话管理：

| 快捷键                | 作用                                   |
| :-------------------- | :------------------------------------- |
| `Ctrl+B` + `D`        | 分离当前会话（Detach），使其在后台运行 |
| `Ctrl+B` + `$`        | 重命名当前会话                         |
| `Ctrl+B` + `S`        | 显示交互式会话列表并切换               |
| `Ctrl+B` + `%`        | 左右分割窗格（竖直切分）               |
| `Ctrl+B` + `"`        | 上下分割窗格（水平切分）               |
| `Ctrl+B` + `X`        | 关闭当前窗格                           |
| `Ctrl+B` + `Z`        | 最大化/还原当前窗格（Zoom）            |
| `Ctrl+B` + `方向键`   | 在窗格之间切换焦点                     |
| `Ctrl+B` + `{` 或 `}` | 左右交换窗格位置                       |
| `Ctrl+B` + `Q`        | 显示窗格编号（按数字跳转）             |
| `Ctrl+B` + `Space`    | 切换预设的窗格布局                     |
| `Ctrl+B` + `C`        | 新建一个窗口                           |
| `Ctrl+B` + `,`        | 重命名当前窗口                         |
| `Ctrl+B` + `W`        | 列出所有窗口并选择（交互式）           |
| `Ctrl+B` + `N`        | 切换到下一个窗口                       |
| `Ctrl+B` + `P`        | 切换到上一个窗口                       |
| `Ctrl+B` + `0-9`      | 切换到指定编号的窗口                   |
| `Ctrl+B` + `&`        | 关闭当前窗口                           |
| `Ctrl+B` + `[`        | 进入复制/滚动模式                      |
| `Q`                   | 退出复制模式（无需前缀）               |

## 配置

在 tmux 会话中。默认情况下鼠标滚轮滑动是选择历史命令，而非控制终端滑动来浏览终端输出历史。可以通过修改配置实现正常的鼠标滚动浏览。

（1）临时启用

依次按下 `Ctrl + B, :` 键，进入命令模式，输入 `set -g mouse on` 然后回车。或者直接在当前 tmux 会话中执行：

```bash
tmux set -g mouse on
```

（2）永久启用。创建配置文件然后写入配置：

```bash
touch ~/.tmux.conf
echo "set -g mouse on" >> ~/.tmux.conf
tmux source-file ~/.tmux.conf
```

---

总结常用配置：

```bash
# 鼠标支持
set -g mouse on

# 开启 256 色
set -g default-terminal "screen-256color"

# 状态栏默认 1 秒刷新
set -g status-interval 1

# 把默认延迟调小
set -sg escape-time 0
```

重新加载配置：

```bash
tmux source-file ~/.tmux.conf
```
