// Probabilistic Question-Answering system
// @2017 Sarge Rogatch
// This software is distributed under GNU AGPLv3 license. See file LICENSE in repository root for details.

#pragma once

namespace ProbQA {

template<typename taNumber> class CETrainTask;

template<typename taNumber> class CETrainSubtaskDistrib : public SRBaseSubtask {
public: // variables
  CETrainSubtaskDistrib(CETrainTask<taNumber> *pTask, const AnsweredQuestion *pFirst, const AnsweredQuestion *pLim)
  : SRBaseSubtask(pTask), _pFirst(pFirst), _pLim(pLim) { }
  const AnsweredQuestion *_pFirst;
  const AnsweredQuestion *_pLim;
};

} // namespace ProbQA
