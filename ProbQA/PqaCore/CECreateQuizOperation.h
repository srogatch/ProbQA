// Probabilistic Question-Answering system
// @2017 Sarge Rogatch
// This software is distributed under GNU AGPLv3 license. See file LICENSE in repository root for details.

#pragma once

#include "../PqaCore/CECreateQuizOperation.decl.h"
#include "../PqaCore/CpuEngine.h"

namespace ProbQA {

inline CECreateQuizOpBase::CECreateQuizOpBase(PqaError& err) : _err(err) { }
inline CECreateQuizOpBase::~CECreateQuizOpBase() { };

inline CECreateQuizStart::CECreateQuizStart(PqaError& err) : CECreateQuizOpBase(err) { }
inline CECreateQuizOpBase::Operation CECreateQuizStart::GetCode() { return Operation::Start; }

template<typename taNumber> inline CECreateQuizResume<taNumber>::CECreateQuizResume(
  PqaError& err, const TPqaId nQuestions, const AnsweredQuestion* const pAQs)
  : CECreateQuizOpBase(err), _nQuestions(nQuestions), _pAQs(pAQs)
{ }

template<typename taNumber> inline CECreateQuizOpBase::Operation CECreateQuizResume<taNumber>::GetCode()
{ return Operation::Resume; }

template<typename taNumber> void CECreateQuizResume<taNumber>::UpdatePriorsWithAnsweredQuestions(
  CpuEngine<taNumber> *pCe, CEQuiz<taNumber> *pQuiz)
{
  //TODO: implement

  //// Sequential code (single-threaded) for reference
  //NOTE: it may be better to iterate by targets first instead, so to apply all multiplications for the first
  //  target and then move on to the next target. This involves 1 unsequential memory access per answered question
  //  application, while if we iterate first by questions, each question application involves 2 memory accesses: load
  //  and store.
  //const TPqaId nTargets = _dims._nTargets;
  //taNumber *pTargProb = resumeOp._pQuiz->GetTargProbs();
  //TPqaId i = 0;
  //for (; i + 1 < resumeOp._nQuestions; i++) {
  //  const AnsweredQuestion& aq = resumeOp._pAQs[i];
  //  for (TPqaId j = 0; j < nTargets; j++) {
  //    // Multiplier compensation is less robust than summation of logarithms, but it's substantially faster and is
  //    //   supported by AVX2. The idea is to make the multipliers equal to 1 in the average case p[j]=1/M, where M is
  //    //   the number of targets.
  // //FIXME: this will blow to infinity the top most likely targets, making them all equal, which is highly undesirable
  //    pTargProb[j] *= (nTargets * _sA[aq._iAnswer][aq._iQuestion][j] / _mD[aq._iQuestion][j]);
  //  }
  //}
  //taNumber sum(0); //TODO: instead, sort then sum
  //const AnsweredQuestion& aq = resumeOp._pAQs[i];
  //for (TPqaId j = 0; j < nTargets; j++) {
  //  taNumber product = pTargProb[j] * (nTargets * _sA[aq._iAnswer][aq._iQuestion][j] / _mD[aq._iQuestion][j]);
  //  pTargProb[j] = product;
  //  sum += product; //TODO: assign to a bucket instead
  //}
  //for (TPqaId j = 0; j < nTargets; j++) {
  //  pTargProb[j] /= sum;
  //}
}

} // namespace ProbQA
