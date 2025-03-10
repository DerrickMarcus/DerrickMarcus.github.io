# JavaScript Envirment in Windows

## Install Node.js

访问官网：[Node.js — Download Node.js® (nodejs.org)](https://nodejs.org/en/download)。

或者中文版网站：[Node.js 中文网 (nodejs.com.cn)](https://www.nodejs.com.cn/download.html)。

或者清华大学开源镜像站：[Index of /nodejs-release/ | 清华大学开源软件镜像站 | Tsinghua Open Source Mirror](https://mirrors.tuna.tsinghua.edu.cn/nodejs-release/)。

建议下载最新版本。

下载安装包后双击安装，安装路径根据个人需求修改（不要包含中文），以笔者为例：

```text
E:\Program Files\nodejs
```

如果安装到 Program Files 这类文件夹，后续创建文件夹、更改权限等操作需要提供管理员权限。

剩余安装选项一路默认即可。

测试是否安装成功：打开 命令提示符cmd，命令行分别输入：

```cmd
node -v
npm -v
```

若能正确输出版本号，即为安装成功。

## Configure System Environment Variable

文件资源管理器中，打开刚才的安装目录：

```text
E:\Program Files\nodejs
```

新建两个文件夹： node_global 和 node_cache

创建完成后，以**管理员身份打开cmd**，输入下面两个命令：

```cmd
npm config set prefix E:\Program Files\nodejs\node_global

npm config set cache E:\Program Files\nodejs\node_cache
```

注意对照自己的安装路径进行修改。

设置完成后，可以输入下面两个命令检查是否配置成功：

```cmd
npm config get prefix

npm config get cache
```

然后进行**系统环境变量**设置，打开“查看高级系统设置——环境变量，在下方系统变量中新建变量**NODE_PATH**，变量值为：**`E:\Program Files\nodejs\node_global\node_modules`**

在 `PATH` 中，新建项 `%NODE_PATH%`，顺便把C盘下默认 `AppData\Roaming\npm` 修改为 `E:\Program Files\nodejs\node_global`（如果有）

创建好后，`E:\Program Files\nodejs\node_global` 目录下应该会出现文件夹 `node_modules`，如果没有，手动创建即可。

## Test

全局安装一个最常用的 express 模块进行测试

命令行输入：

```cmd
npm install -g express  // -g代表全局安装
```

若安装失败，可能是因为没有使用管理员身份打开 cmd，或者可以修改一下文件夹`node_global`和`node_cache`的权限：

右键点击对应文件夹——【属性】——【安全】——【编辑】——勾选所有权限——【确定】

## Install Taobao Mirror

```bash
npm config set registry https://registry.npmmirror.com/
```

查看是否安装成功：

```bash
npm config get registry
```

安装 cnpm（npm 是 node 官方的包管理器。cnpm 是个中国版的 npm，是淘宝定制的 cnpm (gzip 压缩支持) 命令行工具代替默认的 npm）。

```bash
npm install -g cnpm --registry=https://registry.npmmirror.com
```

查看是否安装成功：

```bash
cnpm -v
```

## Configure VS Code

下载插件 Code Runner（By Jun Han）

打开用户设置Open User Settings(JSON)，添加配置：

```json
"code-runner.executorMap": {
        "javascript": "node",
    },
```

即可直接运行 javascript 文件。
