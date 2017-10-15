// Probabilistic Question-Answering system
// @2017 Sarge Rogatch
// This software is distributed under GNU AGPLv3 license. See file LICENSE in repository root for details.

#pragma once

namespace ProbQA {

template<typename taNumber> struct AnswerMetrics {
  taNumber _weight;
  taNumber _entropy;
  taNumber _distance;
};

} // namespace ProbQA
