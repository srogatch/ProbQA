// Probabilistic Question-Answering system
// @2017 Sarge Rogatch
// This software is distributed under GNU AGPLv3 license. See file LICENSE in repository root for details.

#pragma once

namespace ProbQA {

struct KBFileInfo {
  SRPlat::SRSmartFile &_sf;
  const char* const _filePath;
  KBFileInfo(SRPlat::SRSmartFile &sf, const char* const filePath) : _sf(sf), _filePath(filePath) { }
};

} // namespace ProbQA
