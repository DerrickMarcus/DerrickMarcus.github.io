# 1 绪论(1)

!!! abstract "课程概述"
    本课程讲授重点为操作系统的原理，对于 Windows、Linux 的具体实现不做考察。

![202506071711366](https://cdn.jsdelivr.net/gh/DerrickMarcus/picgo-image/images/202506071711366.png)

![202506071712278](https://cdn.jsdelivr.net/gh/DerrickMarcus/picgo-image/images/202506071712278.png)

## 1.1 什么是操作系统

定义：定义：操作系统是计算机系统中的一个**系统软件**，它是这样一些**程序模块的集合**：它们能有效地**组织和管理**计算机的**软硬件资源**，合理地组织计算机的工作流程，控制程序的执行并向用户提供各种**服务功能**，使得用户能够方便地使用计算机，使整个计算机系统能高效的运行。

为了将硬件的复杂性与程序员分离开，在硬件裸机上加载一层软件来管理整个系统，同时给用户提供一个更容易理解和编程的接口，这层软件就是操作系统。

操作系统专指在核心态(kernel mode)下运行的软件，应用软件和操作系统以外的系统软件运行在用户态(user mode)下，它们并不是操作系统的组成部分。

计算机系统的层次（由底层到顶层）：计算机硬件——操作系统——系统工具——应用软件——用户。

操作系统的作用：

1. 作为扩展机器(extended machine)/虚拟机(virtual machine)的操作系统。为用户提供一台等价的扩展机器或虚拟机，它比低层硬件更容易编程和使用。在裸机上添加：设备管理、文件管理、存储管理(针对内存)、处理机管理(针对 CPU)。另外，为合理组织工作流程：作业管理。
2. 作为资源管理者的操作系统。管理对象包括：CPU、存储器、外部设备、信息(数据和软件)。管理的内容：资源的当前状态(数量和使用情况)、资源的分配、回收和访问操作，相应管理策略(包括用户权限)。从资源管理的观点看，操作系统的首要任务是跟踪资源的使用状况、满足资源请求、提高资源利用率，以及协调各程序和用户对资源的使用冲突。
3. 作为用户使用计算机软硬件的接口的操作系统。系统命令(命令行、菜单式、命令脚本式、图形用户接口 GUI)，系统调用(形式上类似于过程调用，在应用编程中使用)。

## 2. 操作系统的发展历史

第1代：电子管时代(1946年-1955年)。ENIAC。

第2代：晶体管时代(1955年-1965年)。程序设计语言诞生(Fortan, ALGOL, COBOL)，出现批处理操作系统。

第3代：集成电路时代(1965年-1980年)。多道程序设计(multiprogramming)，在单处理机上运行多道程序，宏观并行，微观串行。出现作业管理、处理机管理、存储管理、设备管理、文件管理/文件系统。出现分时系统(time-sharing system)。IBM 的 System/360 系统。UNIX 操作系统崛起与 C 语言的发明。

!!! note "概念辨析"
    单道批处理指每次运行一个作业，顺序执行，先进先出。多道批处理指在内存中同时存在多个作业，充分利用内存，但无确定次序。

    多道程序系统(multiprogramming system)指多个程序同时在内存中交替运行，多处理系统(multiprocessing system)指多个处理器同时运行多个程序。

第4代：大规模集成电路时代(1980年-)。代表 Windows, Linux， macOS 等操作系统。

当代操作系统向着大型和微型的两个不同的方向发展，大型系统的典型是**分布式操作系统**，微型系统的典型是**嵌入式操作系统**。

## 3. 操作系统的主要功能

操作系统的主要功能：处理机管理，存储管理，设备管理，文件管理，用户接口。

### 3.1 处理机管理

目的：完成处理机资源的分配调度等功能。处理机调度的单位为**进程或线程**。

进程控制：创建、撤销、挂起、改变运行优先级等——主动改变进程的状态。

进程同步与互斥：协调并发进程之间的推进步骤，以协调资源共享——交换信息能力弱。

进程间通信：进程之间传送数据，以协调进程间的协作——交换信息能力强，也可以用来协调进程之间的推进。

进程调度：进程的运行切换，以充分利用处理机资源和提高系统性能。

### 3.2 存储管理

目的：提高利用率，方便用户使用，提供足够的存储空间，方便进程并发运行。

存储分配与回收。

存储保护：保证进程间互不干扰、相互保密。例如：访问合法性检查，甚至要防止从“垃圾”中窃取其他进程的信。

地址映射：进程逻辑地址到内存物理地址的映射或变换。

内存扩充：覆盖、交换和虚拟存储——逻辑上的扩充，提高内存利用率、扩大进程的内存空间。

### 3.3 设备管理

目的：方便的设备使用，提高 CPU 与 I/O 设备利用率。

设备操作：利用设备驱动程序（通常在内核中）完成对设备的操作。

设备分配与回收：在多用户间共享 I/O 设备资源。

虚拟设备(virtual device)：设备由多个进程共享，每个进程如同独占该设备。

缓冲区管理：匹配 CPU 和外设的速度，提高两者的利用率。

### 3.4 文件管理

目的：解决信息资源的存储、共享、保密和保护，操作系统中负责这一功能的部分称为文件系统。

文件存储空间管理：解决如何存放信息，以提高空间利用率和读写性能。

目录管理：解决信息检索问题。

文件的读写管理和存取控制：解决信息安全问题

1. 系统设口令：哪个用户
2. 用户分类：哪个用户组
3. 文件权限：针对用户或用户组的读写权

### 3.5 用户接口

用户接口是操作系统提供给用户与计算机打交道的外部机制。用户能够借助用户接口来控制计算机系统。

操作系统向用户提供两种接口：

1. 命令接口：供用户用于组织和控制自己的作业运行
    1. 输入方式：命令行、菜单式、图形用户界面(GUI，Graphical User Interface)。
    2. 命令脚本。
2. 程序接口：即为**系统调用**，供用户程序和系统程序调用操作系统功能。
