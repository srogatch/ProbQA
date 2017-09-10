// Probabilistic Question-Answering system
// @2017 Sarge Rogatch
// This software is distributed under GNU AGPLv3 license. See file LICENSE in repository root for details.

#include "stdafx.h"
#include "../SRPlatform/BucketerSubtaskSum.h"
#include "../SRPlatform/Interface/SRDoubleNumber.h"
#include "../SRPlatform/Interface/SRBucketSummator.h"
#include "../SRPlatform/BucketerTask.h"

namespace SRPlat {

template<> SRNumPack<SRDoubleNumber> __vectorcall BucketerSubtaskSum<SRDoubleNumber>::SumColumn(const size_t iVect) {
  __m256d sum = _pBs->GetVect(0, SRCast::ToInt32(iVect))._comps;
  for (SRThreadCount i = 1; i < _pBs->_nWorkers; i++) {
    sum = _mm256_add_pd(sum, _pBs->GetVect(i, SRCast::ToInt32(iVect))._comps);
  }
  return sum;
}

template<> void BucketerSubtaskSum<SRDoubleNumber>::Run() {
  auto const& task = static_cast<const BucketerTask<SRDoubleNumber>&>(*GetTask());
  _pBs = &task.GetBS();

  const bool isAtPartial = (_iLimit == task._iPartial + 1);
  __m256d total = _mm256_setzero_pd();
  for (size_t i = _iFirst, iEn = (isAtPartial ? task._iPartial : _iLimit); i < iEn; i++) {
    total = SRSimd::HorizAddStraight(total, SumColumn(i)._comps);
  }
  if (isAtPartial) {
    //TODO: refactor to a table + _mm_cvtsi32_si128()
    const __m256d mask = _mm256_castsi256_pd(SRSimd::SetLsb1(
      uint16_t(task._nValid) * SRNumTraits<double>::_cnTotalBits));
    total = SRSimd::HorizAddStraight(total, _mm256_and_pd(mask, SumColumn(task._iPartial)._comps));
  }
  _pBs->_pWorkerSums[_iWorker]._comps = total;
}


template class BucketerSubtaskSum<SRDoubleNumber>;

} // namespace SRPlat
