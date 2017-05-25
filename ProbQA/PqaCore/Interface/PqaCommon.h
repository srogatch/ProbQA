#pragma once

#include "../PqaCore/Interface/PqaCore.h"

namespace ProbQA {

// PQA - Probabilistic Question Answering
typedef int64_t TPqaId;
const TPqaId cInvalidPqaId = -1;

typedef double TPqaAmount;

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
  EngineDimensions _dims;
  PrecisionDefinition _prec;
};

struct AnsweredQuestion {
  TPqaId _iQuestion;
  TPqaId _iAnswer;
};

struct RatedTarget {
  TPqaId _iTarget;
  TPqaAmount _prob; // probability that this target is what the user needs
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
};

struct AddTargetParam {
  TPqaId _iTarget; // filled in by the engine.
  TPqaAmount _initialAmount; // filled in by the client before passing to the engine.
};

} // namespace ProbQA