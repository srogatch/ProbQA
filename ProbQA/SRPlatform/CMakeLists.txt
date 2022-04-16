add_library(SRPlatform SHARED)

target_include_directories(SRPlatform PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
  $<INSTALL_INTERFACE:include>
)

target_sources(SRPlatform PRIVATE
  BucketerSubtaskSum.cpp
  DbgLogger.cpp
  dllmain.cpp
  FileLogger.cpp
  SRBaseTask.cpp
  SRConditionVariable.cpp
  SRCriticalSection.cpp
  SRDefaultLogger.cpp
  SRDoubleNumber.cpp
  SRException.cpp
  SRFastRandom.cpp
  SRGenericException.cpp
  SRLoggerFactory.cpp
  SRMemPool.cpp
  SRMultiException.cpp
  SRPlatform.cpp
  SRReaderWriterSync.cpp
  SRSimd.cpp
  SRSpinSync.cpp
  SRString.cpp
  SRThreadPool.cpp
  SRUtils.cpp
  SRVectMath.cpp
  stdafx.cpp
  SubtaskCompleter.cpp
)

target_precompile_headers(SRPlatform PUBLIC stdafx.h)

if(PROBQA_CUDA)
  target_compile_definitions(SRPlatform PRIVATE PROBQA_CUDA=1)

  set_target_properties(SRPlatform PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
    CUDA_STANDARD 17)
endif()
