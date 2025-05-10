---
comments: true
---

# Ubuntu in Virtual Machine

!!! Tip
    Before installing any package or software, please think twice. It is recommended to keep the system clean and away from redundant packages.

## Change the mirror source

In China mainland, using the default Ubuntu source is very slow. It is recommended to change it to Tsinghua University source.

```bash
# For Ubuntu 24.04
sudo cp /etc/apt/sources.list.d/ubuntu.sources /etc/apt/sources.list.d/ubuntu.sources.bak
sudo vi /etc/apt/sources.list.d/ubuntu.sources

# For Ubuntu 22.04 or earlier
sudo cp /etc/apt/sources.list /etc/apt/sources.list.bak
sudo vi /etc/apt/sources.list
```

Go to <https://mirrors.tuna.tsinghua.edu.cn/help/ubuntu/> and choose your Ubuntu version. Take Ubuntu 20.04 for example, replace the content with the following:

```text
# 默认注释了源码镜像以提高 apt update 速度，如有需要可自行取消注释
deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ focal main restricted universe multiverse
# deb-src https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ focal main restricted universe multiverse
deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ focal-updates main restricted universe multiverse
# deb-src https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ focal-updates main restricted universe multiverse
deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ focal-backports main restricted universe multiverse
# deb-src https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ focal-backports main restricted universe multiverse

# 以下安全更新软件源包含了官方源与镜像站配置，如有需要可自行修改注释切换
deb http://security.ubuntu.com/ubuntu/ focal-security main restricted universe multiverse
# deb-src http://security.ubuntu.com/ubuntu/ focal-security main restricted universe multiverse

# 预发布软件源，不建议启用
# deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ focal-proposed main restricted universe multiverse
# # deb-src https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ focal-proposed main restricted universe multiverse
```

Then update the package list:

```bash
sudo apt update
sudo apt upgrade
```

## Install vmware tools

Run:

```bash
sudo apt install open-vm-tools
sudo apt install open-vm-tools-desktop
```

## Configure the proxy

### Using Clash Verge Rev

To visit some foreign websites, we need to use a proxy. Clash Verge Rev is a modified version of Clash-for-Windows, but with a core of Clash.Meta(mihomo). You can visit <https://www.clashverge.dev> for detailed information and installation guides.

Clash Verge Rev requires Ubuntu 24.04, 22.04 or 20.04. For Ubuntu 18.04 or earlier, please consider to use [Clash for Windows](https://archive.org/download/clash_for_windows_pkg), or [V2rayN](https://github.com/2dust/v2rayN).

!!! Warning
    If you use Ubuntu 20.04, you need to download and install the version of 1.7.7, because the latest version of 2.x requires additional dependencies, which is difficult to solve on Ubuntu 20.04.

```bash
# For Ubuntu 20.04
cd ~/Downloads
wget https://github.com/clash-verge-rev/clash-verge-rev/releases/download/v1.7.7/clash-verge_1.7.7_amd64.deb
sudo apt install ./clash-verge_1.7.7_amd64.deb

```

### Using Clash for Windows

Visit [Clash for Windows](https://archive.org/download/clash_for_windows_pkg) and select `Clash.for.Windows-0.20.39-x64-linux.tar.gz` . Or you can run:

```bash
cd ~/Downloads
wget https://github.com/clash-hub/clash_for_windows_pkg/releases/download/Latest/Clash.for.Windows-0.20.39-x64-linux.tar.gz
# Or:
wget https://github.com/clashdownload/Clash_for_Windows/releases/download/0.20.39/Clash.for.Windows-0.20.39-x64-linux.tar.gz

tar -zxvf Clash.for.Windows-0.20.39-x64-linux.tar.gz
mv Clash.for.Windows-0.20.39 clash
cd clash
./cfw
```

## Install Chinese input method

Run:

```bash
sudo apt install ibus-libpinyin
```

Then open Settings->Region & Language, add choose `Chinese-Intelligent Pinyin` .

## Install VS Code

Go to <https://code.visualstudio.com/> and download the latest version of `code_*.deb` . Then Run:

```bash
sudo apt install ~/Downloads/code_*.deb
```
