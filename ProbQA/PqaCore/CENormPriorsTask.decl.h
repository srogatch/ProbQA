// Probabilistic Question-Answering system
// @2017 Sarge Rogatch
// This software is distributed under GNU AGPLv3 license. See file LICENSE in repository root for details.

#pragma once

#include "../PqaCore/CENormPriorsTask.fwd.h"
#include "../PqaCore/CEQuiz.fwd.h"
#include "../PqaCore/CpuEngine.fwd.h"
#include "../PqaCore/CEBaseTask.h"

namespace ProbQA {

#pragma warning( push )
#pragma warning( disable : 4324 ) // structure was padded due to alignment specifier
template<typename taNumber> class CENormPriorsTask : public CEBaseTask {
public:
  const CEQuiz<taNumber> *const _pQuiz;
  SRPlat::SRBucketSummator<taNumber> *_pBs;
  // The number to add to the exponent so to get it within the representable range or to cut off if corrected exponent
  //   is too small. Repeated in each 64-bit component.
  __m256i _corrExp;

public:
  explicit inline CENormPriorsTask(CpuEngine<taNumber> &engine, CEQuiz<taNumber> &quiz,
    SRPlat::SRBucketSummator<taNumber> &bs);
};
#pragma warning( pop )

} // namespace ProbQA