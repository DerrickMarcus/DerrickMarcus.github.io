# CMake and Catkin

CMake 是 C/C++ 开发中常用的编译工具链，在 ROS 中我们可以使用 catkin_make 工具，它集成了 CMake 且功能更加适用于 ROS 开发，语法与 CMake 相同，配置文件均为 `CMakeLists.txt` 。

## CMake

注意：有一些预定义的变量， `PROJECT_SOURCE_DIR` 表示源代码目录（包含 `CMakeLists.txt` 文件的目录，一般也就是项目的根目录）， `PROJECT_BINARY_DIR` ，项目的构建目录，生成编译文件的位置，一般也就是 `build/` 。 `LIBRARY_OUTPUT_PATH` 是生成的库文件存放的位置。

`CMakeLists.txt` 文件示例：

```cmake
# 指定最低版本 cmake
cmake_minimum_required(VERSION <version>)
cmake_minimum_required(VERSION 3.10)

# 指定项目的名称和使用的语言
project(<project_name> [<language>...])
project(MyProject VERSION 1.0 LANGUAGES CXX)

# 指定生成的可执行文件和从那些源文件编译
add_executable(<target> <source_files>...)
add_executable(MyExecutable main.cpp other_file.cpp ...)

# 创建一个库（动态或静态）及其源文件
add_library(<target> <source_files>...)
add_library(MyLibrary STATIC library.cpp)

# 链接目标文件和其他库
target_link_libraries(<target> <libraries>...)
target_link_libraries(MyExecutable MyLibrary)

# 添加头文件搜索路径
target_include_directories(MyExecutable PRIVATE ${PROJECT_SOURCE_DIR}/include)
# 如果是库文件，那么可以设置为 PUBLIC ,这样库文件的头文件会传递给目标文件，不用再次添加

# 设置变量的值
set(<variable> <value>...)
set(CMAKE_CXX_STANDARD 11) # 比如指定C++标准

# 查找外部库和包
find_package(xxx REQUIRED)
find_package(catkin REQUIRED COMPONENTS roscpp rospy std_msgs genmsg)

# 编译选项
add_compile_options(-std=c++20 -Wall -O2)
# 使用变量存储源文件，适用于源文件特别多的时候
aux_source_directory(dir val)
aux_source_directory(. SRC_LIST) # 然后使用 add_executable(hello ${SRC_LIST})
```

一个示例：

```cmake
# %Tag(FULLTEXT)%
cmake_minimum_required(VERSION 2.8.3)
project(beginner_tutorials)

## Find catkin and any catkin packages
find_package(catkin REQUIRED COMPONENTS roscpp rospy std_msgs genmsg)

## Declare ROS messages and services
add_message_files(FILES Num.msg)
add_service_files(FILES AddTwoInts.srv)

## Generate added messages and services
generate_messages(DEPENDENCIES std_msgs)

## Declare a catkin package
catkin_package()

## Build talker and listener
include_directories(include ${catkin_INCLUDE_DIRS})

add_executable(talker src/talker.cpp)
target_link_libraries(talker ${catkin_LIBRARIES})

add_executable(listener src/listener.cpp)
target_link_libraries(listener ${catkin_LIBRARIES})

## Build service client and server
# %Tag(SRVCLIENT)%
add_executable(add_two_ints_server src/add_two_ints_server.cpp)
target_link_libraries(add_two_ints_server ${catkin_LIBRARIES})
add_dependencies(add_two_ints_server beginner_tutorials_gencpp)

add_executable(add_two_ints_client src/add_two_ints_client.cpp)
target_link_libraries(add_two_ints_client ${catkin_LIBRARIES})
add_dependencies(add_two_ints_client beginner_tutorials_gencpp)

# %EndTag(SRVCLIENT)%

# %EndTag(FULLTEXT)%
```

构建流程

Out-of-space 构建，将构建文件放于源代码之外的独立目录中，大致如下：

```text
MyProject/
├── CMakeLists.txt
├── src/
│   ├── main.cpp
│   └── mylib.cpp
└── include/
    └── mylib.h
```

构建步骤：

```bash
# 在项目的根目录下
mkdir build
cd build

# 此时已经在 build 目录下，生成构建文件，其中..指向包含 CMakeLists.txt 文件的目录（称为源代码目录，但非彼“源代码”）
cmake ..
cmake --build . # 指定将编译生成的文件放在指定目录，其中.表示恰好表示当前目录
cmake -G .. # 指定使用 MinGW 工具
# 也可指定构建类型，Debug / Release
cmake -DCMAKE_BUILD_TYPE=Release ..

# 使用构建文件进行编译
make
make MyExecutable # 指定构建目标名称

# 清理中间文件
make clean
rm -rf build/*
```

对于 catkin，有一些额外的需要注意：

预定义的变量：

`${catkin_INCLUDE_DIRS}` ，包含已声明依赖软件包的包含（头文件）目录。在 `find_package` 命令中指定了软件包的组件后，会自动查找这些软件包的头文件路径，并将它们收集在变量中。

```cmake
include_directories(
  ${catkin_INCLUDE_DIRS}
)
```

`${catkin_LIBRARIES}` ，包含已声明依赖软件包的目标（库文件）路径。在 `find_package` 命令中指定了软件包的组件后，catkin 会自动查找这些软件包的库文件路径，并将它们收集到变量中。

```cmake
target_link_libraries(your_executable
  ${catkin_LIBRARIES}
)
```

## catkin_make and catkin

`catkin_make` 是 ROS 的第一代编译构建工具，本质是对 cmake + make 的轻量封装，核心作用是简化 ROS 包的编译流程。它直接调用 CMake 和 Make 工具链，将编译过程封装成单个命令，是 ROS Kinect 以及更早版本的默认编译工具。

`catkin` 是 catkin_tools 工具集（核心命令为 catkin build），是 ROS 社区为解决 catkin_make 痛点开发的第二代编译工具，设计目标是提供更灵活、高效、模块化的编译体验，是 ROS Melodic/Noetic 及后续版本推荐使用的工具。

!!! note
    `catkin_make` 与 `catkin` 都是 ROS 1 的构建工具，而 ROS 2 的构建工具为 `colcon` 。

对比：

|          | catkin_make                                                  | catkin                                                       |
| :------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| 编译架构 | 单 CMake 上下文，所有包共享一个构建目录 `build` 和编译空间 `devel`，所有包编译配置耦合。 | 多 CMake 上下文，每个包独立构建，最终合并到统一的 `devel` / `install` 空间，包之间解耦。 |
| 并行编译 | 仅支持 `catkin_make -jN` ，但单上下文下并行效率低，易因包依赖冲突失败。 | 原生支持智能并行 `catkin build -jN` ，可自动分析包依赖，并行编译无依赖的包，效率大幅提升。 |
| 增量编译 | 单个包修改会触发整个工作空间的 CMake 重新配置，编译冗余。    | 仅重新编译修改的包及其直接依赖，增量编译效率极高。           |

`catkin_make` 的使用方式：

```bash
# 初始化工作空间
cd ~/catkin_ws
catkin_make

# 指定编译空间
catkin_make -DCMAKE_BUILD_TYPE=Release  # 编译发布版本
catkin_make install  # 安装到 install 目录

# 清除编译文件
catkin_make clean
```

`catkin` 的使用方式：

```bash
sudo apt-get install python3-catkin-tools

# 初始化工作空间
cd ~/catkin_ws
catkin init

# 编译所有包
catkin build

# 编译指定包
catkin build <package_name>

# 编译发布版本
catkin build --cmake-args -DCMAKE_BUILD_TYPE=Release

# 清除单个包的编译文件
catkin clean <package_name>

# 清除所有编译文件
catkin clean -y
```
