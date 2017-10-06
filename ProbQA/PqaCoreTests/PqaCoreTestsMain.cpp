// Probabilistic Question-Answering system
// @2017 Sarge Rogatch
// This software is distributed under GNU AGPLv3 license. See file LICENSE in repository root for details.

#include "stdafx.h"

// Google test framework must be cloned to: $(SolutionDir)../../Imported/googletest
// Solution $(SolutionDir)../../Imported/googletest/googletest/msvc/2010/gtest-md.sln must be upgraded to the version
//   of MSVS used in the rest of ProbQA, then compiled for the configurations needed: x86/x64 + Debug/Release.

using namespace SRPlat;

GTEST_API_ int main(int argc, char **argv) {
  const char* baseName = "Logs\\PqaCoreTests";
  if (!CreateDirectoryA("Logs", nullptr)) {
    uint32_t le = GetLastError();
    if (le != ERROR_ALREADY_EXISTS) {
      baseName = "PqaCoreTests";
    }
  }
  SRDefaultLogger::Init(SRString::MakeUnowned(baseName));
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
