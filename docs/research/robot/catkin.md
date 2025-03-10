# CMake and Catkin

CMake 是 C/C++ 开发中常用的编译工具链，在 ROS 中我们可以使用 catkin_make 工具，它集成了 CMake 且功能更加适用于 ROS 开发，语法与 CMake 相同，配置文件均为 CMakeLists.txt

语法规则如下：

注意：有一些预定义的变量， `PROJECT_SOURCE_DIR` 表示源代码目录（包含 CMakeLists.txt 文件的目录，一般也就是项目的根目录），`PROJECT_BINARY_DIR` ，项目的构建目录，生成编译文件的位置，一般也就是 build/ 。`LIBRARY_OUTPUT_PATH` 是生成的库文件存放的位置。

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
find_package(xxx REQUIERD)
find_package(catkin REQUIRED COMPONENTS roscpp rospy std_msgs genmsg)

# else
# 编译选项
add_compile_options(-std=c++20 -Wall -O2)
# 使用变量存储源文件，适用于源文件特别多的时候
aux_source_directory(dir val)
aux_source_direcctory(. SRC_LIST) # 然后使用 add_executable(hello ${SRC_LIST})
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
make MyExecutable # 指定目标名称

# 清理中间文件
make clean
rm -rf build/* # 或者这个
```

对于 catkin，有一些额外的需要注意：

预定义的变量：

`${catkin_INCLUDE_DIRS}` ，包含已声明依赖软件包的包含（头文件）目录。在`find_package`命令中指定了软件包的组件后，会自动查找这些软件包的头文件路径，并将它们收集在变量中。

```cmake
include_directories(
  ${catkin_INCLUDE_DIRS}
)
```

${catkin_LIBRARIES} ，包含已声明依赖软件包的目标（库文件）路径。在`find_package`命令中指定了软件包的组件后，`catkin`会自动查找这些软件包的库文件路径，并将它们收集到变量中。

```cmake
target_link_libraries(your_executable
  ${catkin_LIBRARIES}
)
```
