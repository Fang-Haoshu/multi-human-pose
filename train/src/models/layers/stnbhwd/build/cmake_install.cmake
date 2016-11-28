# Install script for directory: /home/liuxhy237/Git/pose-hg-train/src/models/layers/stnbhwd

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "/home/liuxhy237/Git/torch/install")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "Release")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Install shared libraries without execute permission?
if(NOT DEFINED CMAKE_INSTALL_SO_NO_EXE)
  set(CMAKE_INSTALL_SO_NO_EXE "1")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/../../../../../lib/libstn.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/../../../../../lib/libstn.so")
    file(RPATH_CHECK
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/../../../../../lib/libstn.so"
         RPATH "$ORIGIN/../lib:/home/liuxhy237/Git/torch/install/lib")
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/../../../../../lib" TYPE MODULE FILES "/home/liuxhy237/Git/pose-hg-train/src/models/layers/stnbhwd/build/libstn.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/../../../../../lib/libstn.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/../../../../../lib/libstn.so")
    file(RPATH_CHANGE
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/../../../../../lib/libstn.so"
         OLD_RPATH "/home/liuxhy237/Git/torch/install/lib:::::::::::::::"
         NEW_RPATH "$ORIGIN/../lib:/home/liuxhy237/Git/torch/install/lib")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/../../../../../lib/libstn.so")
    endif()
  endif()
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/../../../../../lua/stn" TYPE FILE FILES
    "/home/liuxhy237/Git/pose-hg-train/src/models/layers/stnbhwd/test.lua"
    "/home/liuxhy237/Git/pose-hg-train/src/models/layers/stnbhwd/BilinearSamplerBHWD.lua"
    "/home/liuxhy237/Git/pose-hg-train/src/models/layers/stnbhwd/AffineTransformMatrixGenerator.lua"
    "/home/liuxhy237/Git/pose-hg-train/src/models/layers/stnbhwd/AffineGridGeneratorBHWD.lua"
    "/home/liuxhy237/Git/pose-hg-train/src/models/layers/stnbhwd/init.lua"
    )
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/../../../../../lib/libcustn.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/../../../../../lib/libcustn.so")
    file(RPATH_CHECK
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/../../../../../lib/libcustn.so"
         RPATH "$ORIGIN/../lib:/home/liuxhy237/Git/torch/install/lib:/usr/local/cuda/lib64")
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/../../../../../lib" TYPE MODULE FILES "/home/liuxhy237/Git/pose-hg-train/src/models/layers/stnbhwd/build/libcustn.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/../../../../../lib/libcustn.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/../../../../../lib/libcustn.so")
    file(RPATH_CHANGE
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/../../../../../lib/libcustn.so"
         OLD_RPATH "/home/liuxhy237/Git/torch/install/lib:/usr/local/cuda/lib64:::::::::::::::"
         NEW_RPATH "$ORIGIN/../lib:/home/liuxhy237/Git/torch/install/lib:/usr/local/cuda/lib64")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/../../../../../lib/libcustn.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT)
  set(CMAKE_INSTALL_MANIFEST "install_manifest_${CMAKE_INSTALL_COMPONENT}.txt")
else()
  set(CMAKE_INSTALL_MANIFEST "install_manifest.txt")
endif()

file(WRITE "/home/liuxhy237/Git/pose-hg-train/src/models/layers/stnbhwd/build/${CMAKE_INSTALL_MANIFEST}" "")
foreach(file ${CMAKE_INSTALL_MANIFEST_FILES})
  file(APPEND "/home/liuxhy237/Git/pose-hg-train/src/models/layers/stnbhwd/build/${CMAKE_INSTALL_MANIFEST}" "${file}\n")
endforeach()
