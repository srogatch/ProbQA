// Probabilistic Question-Answering system
// @2017 Sarge Rogatch
// This software is distributed under GNU AGPLv3 license. See file LICENSE in repository root for details.

#include "stdafx.h"
#include "../SRPlatform/BucketerSubtaskSum.h"
#include "../SRPlatform/Interface/SRDoubleNumber.h"
#include "../SRPlatform/Interface/SRBucketSummator.h"
#include "../SRPlatform/BucketerTask.h"

namespace SRPlat {

void BucketerSubtaskSum<SRDoubleNumber>::Run() {
  auto const& task = static_cast<const BucketerTask<SRDoubleNumber>&>(*GetTask());
  const SRBucketSummator<SRDoubleNumber> &bs = task.GetBS();

  const bool isAtPartial = (_iLimit == task._iPartial + 1);
  for (size_t i = _iFirst, iEn = (isAtPartial ? task._iPartial : _iLimit); i < iEn; i++) {
    for (size_t j = 0; j < bs._nWorkers; j++) {
      //const
    }
  }
  if (isAtPartial) {
    const __m256d mask = _mm256_castsi256_pd(SRSimd::SetLsb1(task._nValid * SRNumTraits<double>::_cnTotalBits));

  }
  //TODO: handle the remainder of items after division by vector size
  //TODO: implement
}


template class BucketerSubtaskSum<SRDoubleNumber>;

} // namespace SRPlat
