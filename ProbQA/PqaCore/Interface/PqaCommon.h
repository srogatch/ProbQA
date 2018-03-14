// Probabilistic Question-Answering system
// @2017 Sarge Rogatch
// This software is distributed under GNU AGPLv3 license. See file LICENSE in repository root for details.

#pragma once

#include "../PqaCore/Interface/PqaCore.h"

namespace ProbQA {

// PQA - Probabilistic Question Answering
typedef int64_t TPqaId;
const TPqaId cInvalidPqaId = -1;

typedef SRPlat::SRAmount TPqaAmount;

enum TPqaPrecisionType : uint64_t {
  None = 0,
  Float = 1,
  FloatPair = 2, // May be more efficient than `double` on GPUs
  Double = 3,
  DoublePair = 4,
  Arbitrary = 5
};

struct PrecisionDefinition {
  TPqaPrecisionType _type : 4;
  //// Number of mantissa and exponent bytes, to be used in template instantiation.
  uint64_t _mantissa : 28;
  uint64_t _exponent : 16;
  uint64_t _reserved : 16;
};

struct EngineDimensions {
  TPqaId _nAnswers;
  TPqaId _nQuestions;
  TPqaId _nTargets;
};

struct EngineDefinition {
  static const size_t _cDefaultMemPoolMaxBytes = 512 * 1024 * 1024;
  EngineDimensions _dims;
  PrecisionDefinition _prec;
  TPqaAmount _initAmount = 1;
  size_t _memPoolMaxBytes = _cDefaultMemPoolMaxBytes;
};

struct AnsweredQuestion {
  TPqaId _iQuestion;
  TPqaId _iAnswer;
  explicit AnsweredQuestion(const TPqaId iQuestion, const TPqaId iAnswer) : _iQuestion(iQuestion), _iAnswer(iAnswer) { }
};

struct RatedTarget {
  TPqaId _iTarget;
  TPqaAmount _prob; // probability that this target is what the user needs

  bool operator<(const RatedTarget& fellow) const {
    return _prob < fellow._prob;
  }
};

struct CompactionResult {
  //// New counts of targets and questions
  TPqaId _nTargets;
  TPqaId _nQuestions;
  //// i-th item contains the old id for the new id=i
  TPqaId *_pOldTargets;
  TPqaId *_pOldQuestions;
};

struct AddQuestionParam {
  TPqaId _iQuestion; // filled in by the engine
  TPqaAmount _initialAmount; // filled in by the client before passing to the engine.

  static std::unique_ptr<AddQuestionParam[]> Uniform(const TPqaId nQuestions, const TPqaAmount initialAmount) {
    std::unique_ptr<AddQuestionParam[]> ans(new AddQuestionParam[nQuestions]);
    for (TPqaId i = 0; i < nQuestions; i++) {
      ans[i]._initialAmount = initialAmount;
    }
    return std::move(ans);
  }
};

struct AddTargetParam {
  TPqaId _iTarget; // filled in by the engine.
  TPqaAmount _initialAmount; // filled in by the client before passing to the engine.

  static std::unique_ptr<AddTargetParam[]> Uniform(const TPqaId nTargets, const TPqaAmount initialAmount) {
    std::unique_ptr<AddTargetParam[]> ans(new AddTargetParam[nTargets]);
    for (TPqaId i = 0; i < nTargets; i++) {
      ans[i]._initialAmount = initialAmount;
    }
    return std::move(ans);
  }
};

} // namespace ProbQA
