// Probabilistic Question-Answering system
// @2017 Sarge Rogatch
// This software is distributed under GNU AGPLv3 license. See file LICENSE in repository root for details.

#pragma once

#include "../PqaCore/KBFileInfo.h"

namespace ProbQA {

class CudaPersistence {
  std::unique_ptr<uint8_t[]> _buffer;
  KBFileInfo *_pKbFi;
  size_t _bufSize;
  cudaStream_t _cuStr;
public:
  explicit CudaPersistence(const size_t bufSize, KBFileInfo *pKbFi, cudaStream_t cuStr);
  void ReadFile(void *d_p, size_t nBytes);
  void WriteFile(const void *d_p, size_t nBytes);
};

} // namespace ProbQA
