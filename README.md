# metis
Metis - The Future GPU Renderer

需要工具：
CUDA 8.0 : https://developer.nvidia.com/cuda-downloads
Qt 5.6.0 : http://download.qt.io/official_releases/qt/5.6/5.6.0/qt-opensource-windows-x86-msvc2015_64-5.6.0.exe
VS 2010 SP1
CMake 3.2 以上

CMake设置 ：

CMAKE_INSTALL_PREFIX : E:/work/elara/metis
CUDA_SDK_ROOT_DIR : D:/CUDA/GTK
Qt5Widgets_DIR : D:/Qt/Qt5.6.0/5.6/msvc2015_64/lib/cmake/Qt5Widgets

VS 2010 设置 ：

右键Renderer项目 -> 生产自定义 -> CUDA 8.0
右键Renderer项目下的renderkernel.cu -> 属性 -> 项目类型 -> CUDA C/C++

设置Qt依赖库：

cmd D:\Qt\Qt5.6.0\5.6\msvc2015_64\bin\windeployqt.exe "E:\work\elara\metis\build\bin\Release\Renderer.exe"