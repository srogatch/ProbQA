// Probabilistic Question-Answering system
// @2017 Sarge Rogatch
// This software is distributed under GNU AGPLv3 license. See file LICENSE in repository root for details.

#pragma once

#include "../SRPlatform/Interface/SRLogMacros.h"
#include "../SRPlatform/Interface/SRMessageBuilder.h"

namespace SRPlat {

class SRSmartFile {
  FILE *_fp;
public:
  SRSmartFile() : _fp(nullptr) { }

  explicit SRSmartFile(FILE *fp) : _fp(fp) { }

  FILE* Get() { return _fp; }

  bool Set(FILE *fp) {
    const bool bRet = EarlyClose();
    _fp = fp;
    return bRet;
  }

  ~SRSmartFile() {
    if (_fp != nullptr) {
      int retVal = fclose(_fp);
      if (retVal != 0) {
        SRMessageBuilder mb(SR_FILE_LINE "fclose() returned ");
        mb(retVal);
        SRDefaultLogger::Get()->Log(ISRLogger::Severity::Error, mb.GetUnownedSRString());
      }
    }
  }

  bool EarlyClose() {
    if (_fp == nullptr) {
      return true; // no error
    }
    const bool ans = (std::fclose(_fp) == 0);
    _fp = nullptr;
    return ans;
  }
};

} // namespace SRPlat
