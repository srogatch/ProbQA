// Probabilistic Question-Answering system
// @2017 Sarge Rogatch
// This software is distributed under GNU AGPLv3 license. See file LICENSE in repository root for details.

#pragma once

#include "../PqaCore/CENormPriorsTask.fwd.h"
#include "../PqaCore/CEQuiz.fwd.h"
#include "../PqaCore/CpuEngine.fwd.h"
#include "../PqaCore/CEBaseTask.h"

namespace ProbQA {

template<typename taNumber> class CENormPriorsTask : public CEBaseTask {
public:
  const CEQuiz<taNumber> *const _pQuiz;

public:
  explicit inline CENormPriorsTask(CpuEngine<taNumber> &engine, CEQuiz<taNumber> &quiz);
};

} // namespace ProbQA