# Tell CMake to run moc when necessary:
set(CMAKE_AUTOMOC ON)

find_package(OpenGL REQUIRED)
find_package(CUDA)

# As moc files are generated in the binary dir, tell CMake
# to always look for includes there:
set(CMAKE_INCLUDE_CURRENT_DIR ON)

SET(example_name Renderer)

SET(KIT_SRCS
  main.cpp
  mainwindow.cpp
  mainwindow.h
  qtcanvas.cpp
  qtcanvas.h
  CUDAWindow.cpp
  CUDAWindow.h
  renderkernel.cu
  Util.cpp
  SplitBVHBuilder.cpp
  Sort.cpp
  SceneLoader.cpp
  HDRloader.cpp
  CudaBVH.cpp
  Camera.cpp
  BVHNode.cpp
  BVH.cpp
  )

link_libraries("cudart.lib")
link_libraries("opengl32.lib")
link_libraries("glu32.lib")


ADD_EXECUTABLE(${example_name} ${KIT_SRCS})
TARGET_LINK_LIBRARIES(${example_name} ${PROJECT_NAME})
