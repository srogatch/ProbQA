// Probabilistic Question-Answering system
// @2017 Sarge Rogatch
// This software is distributed under GNU AGPLv3 license. See file LICENSE in repository root for details.

#include "stdafx.h"
#include "../PqaCore/CERecordAnswerSubtaskMul.h"
#include "../PqaCore/CEQuiz.h"

using namespace SRPlat;

namespace ProbQA {

template class CERecordAnswerSubtaskMul<SRDoubleNumber>;

template<typename taNumber> CERecordAnswerSubtaskMul<taNumber>::CERecordAnswerSubtaskMul(TTask *pTask)
  : SRStandardSubtask(pTask) { }

template<> void CERecordAnswerSubtaskMul<SRDoubleNumber>::Run() {
  auto &PTR_RESTRICT  task = static_cast<const TTask&>(*GetTask());
  auto &PTR_RESTRICT engine = static_cast<const CpuEngine<SRDoubleNumber>&>(task.GetBaseEngine());
  const CEQuiz<SRDoubleNumber> &PTR_RESTRICT quiz = task.GetQuiz();

  __m256d *PTR_RESTRICT pMants = SRCast::Ptr<__m256d>(quiz.GetTlhMants());
  static_assert(std::is_same<int64_t, CEQuiz<SRDoubleNumber>::TExponent>::value, "The code below assumes TExponent is"
    " 64-bit integer.");
  __m256i *PTR_RESTRICT pExps = SRCast::Ptr<__m256i>(quiz.GetTlhExps());

  const AnsweredQuestion &PTR_RESTRICT aq = task.GetAQ();
  const __m256d *PTR_RESTRICT pAdjMuls = SRCast::CPtr<__m256d>(&engine.GetA(aq._iQuestion, aq._iAnswer, 0));
  const __m256d *PTR_RESTRICT pAdjDivs = SRCast::CPtr<__m256d>(&engine.GetD(aq._iQuestion, 0));
  for (TPqaId i = _iFirst; i < _iLimit; i++) {
    const __m256d adjMuls = SRSimd::Load<false>(pAdjMuls + i);
    const __m256d adjDivs = SRSimd::Load<false>(pAdjDivs + i);
    // P(answer(aq._iQuestion)==aq._iAnswer GIVEN target==(j0,j1,j2,j3))
    const __m256d P_qa_given_t = _mm256_div_pd(adjMuls, adjDivs);

    const __m256d oldMants = SRSimd::Load<false>(pMants + i);
    const __m256d product = _mm256_mul_pd(oldMants, P_qa_given_t);
    //TODO: move separate summation of exponent to a common function (available to other subtasks etc.)?
    const __m256d newMants = SRSimd::MakeExponent0(product);
    SRSimd::Store<false>(pMants + i, newMants);

    //TODO: AND can be removed here if numbers are non-negative or we can assume a large exponent for negatives,
    // or use an arithmetic shift to assign such numbers a very small exponent. Unfortunately, there seems no
    // arithmetic shift for 64-bit components in AVX2 like _mm256_srai_epi64.
    const __m256i prodExps = SRSimd::ExtractExponents64<false>(product);
    const __m256i oldExps = SRSimd::Load<false>(pExps + i);
    const __m256i newExps = _mm256_add_epi64(prodExps, oldExps);
    SRSimd::Store<false>(pExps + i, newExps);
  }
}

} // namespace ProbQA
