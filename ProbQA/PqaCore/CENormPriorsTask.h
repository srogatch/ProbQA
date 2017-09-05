// Probabilistic Question-Answering system
// @2017 Sarge Rogatch
// This software is distributed under GNU AGPLv3 license. See file LICENSE in repository root for details.

#pragma once

#include "../PqaCore/CENormPriorsTask.decl.h"

namespace ProbQA {

template<typename taNumber> inline CENormPriorsTask<taNumber>::CENormPriorsTask(CpuEngine<taNumber> &engine,
  CEQuiz<taNumber> &quiz) : CEBaseTask(engine), _pQuiz(&quiz)
{ }

} // namespace ProbQA
