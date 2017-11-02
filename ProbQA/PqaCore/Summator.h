// Probabilistic Question-Answering system
// @2017 Sarge Rogatch
// This software is distributed under GNU AGPLv3 license. See file LICENSE in repository root for details.

#pragma once

namespace ProbQA {

template<typename taNumber> class Summator {
public:
  template<typename taSubtask> static void ForPriors(const SRPlat::SRPoolRunner::Keeper<taSubtask> &kp,
    typename taSubtask::TTask& task)
  {
    using namespace SRPlat;
    SRAccumulator<taNumber> acc(taNumber(0));
    //TODO: vectorize to multiple numbers at once
    for (SRSubtaskCount i = 0; i < kp.GetNSubtasks(); i++) {
      acc.Add(kp.GetSubtask(i)->_sumPriors);
    }
    task._sumPriors.Set1(acc.Get());
  }
};

} // namespace ProbQA
