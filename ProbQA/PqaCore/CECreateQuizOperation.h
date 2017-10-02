// Probabilistic Question-Answering system
// @2017 Sarge Rogatch
// This software is distributed under GNU AGPLv3 license. See file LICENSE in repository root for details.

#pragma once

#include "../PqaCore/CECreateQuizOperation.decl.h"

namespace ProbQA {

inline CECreateQuizOpBase::CECreateQuizOpBase(PqaError& err) : _err(err) { }
inline CECreateQuizOpBase::~CECreateQuizOpBase() { };

template<typename taNumber> inline CECreateQuizStart<taNumber>::CECreateQuizStart(PqaError& err)
  : CECreateQuizOpBase(err) { }

template<typename taNumber> inline CECreateQuizResume<taNumber>::CECreateQuizResume(
  PqaError& err, const TPqaId nAnswered, const AnsweredQuestion* const pAQs)
  : CECreateQuizOpBase(err), _nAnswered(nAnswered), _pAQs(pAQs)
{ }

template<typename taNumber> inline std::enable_if_t<SRPlat::SRSimd::_cNBytes % sizeof(taNumber) == 0, uint32_t>
CECreateQuizResume<taNumber>::CalcVectsInCache()
{
  if constexpr (SRPlat::SRCpuInfo::_l1DataCachePerPhysCoreBytes <= 
    _cScalarCacheUsageBytes * SRPlat::SRCpuInfo::_nLogicalCoresPerPhysCore)
  {
    return 0;
  }
  return ((SRPlat::SRCpuInfo::_l1DataCachePerPhysCoreBytes / SRPlat::SRCpuInfo::_nLogicalCoresPerPhysCore)
    - _cScalarCacheUsageBytes) >>  SRPlat::SRSimd::_cLogNBytes;
}

} // namespace ProbQA
