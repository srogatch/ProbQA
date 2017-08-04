// Probabilistic Question-Answering system
// @2017 Sarge Rogatch
// This software is distributed under GNU AGPLv3 license. See file LICENSE in repository root for details.

#pragma once

#include "../PqaCore/CETrainTask.fwd.h"
#include "../PqaCore/Interface/PqaCommon.h"

namespace ProbQA {

template<typename taNumber> class CETrainSubtaskDistrib : public SRPlat::SRBaseSubtask {
  const AnsweredQuestion *_pFirst;
  const AnsweredQuestion *_pLim;

public: // methods
  CETrainSubtaskDistrib(CETrainTask<taNumber> *pTask, const AnsweredQuestion *pFirst, const AnsweredQuestion *pLim);
  virtual void Run() override final;
};

} // namespace ProbQA
