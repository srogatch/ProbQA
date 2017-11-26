// Probabilistic Question-Answering system
// @2017 Sarge Rogatch
// This software is distributed under GNU AGPLv3 license. See file LICENSE in repository root for details.

#pragma once

#include "../PqaCore/Interface/PqaErrors.h"
#include "../SRPlatform/Interface/SRString.h"

namespace ProbQA {

class PQACORE_API NotImplementedErrorParams : public IPqaErrorParams {
  SRPlat::SRString _feature;
public:
  explicit NotImplementedErrorParams(SRPlat::SRString&& feature) : _feature(std::forward<SRPlat::SRString>(feature))
  { }

  const SRPlat::SRString& GetFeature() { return _feature; }

  virtual SRPlat::SRString ToString() override final {
    return SRPlat::SRMessageBuilder("Feature=")(_feature).GetOwnedSRString();
  }
};

class PQACORE_API CommonExceptionErrorParams : public IPqaErrorParams {
  SRPlat::SRString _etn; // exception type name
public:
  template <typename T> explicit CommonExceptionErrorParams(const T& ex) {
    _etn = SRPlat::SRString::MakeUnowned(typeid(ex).name());
  }
  const SRPlat::SRString& GetExceptionTypeName() { return _etn; }
  
  virtual SRPlat::SRString ToString() override final {
    return SRPlat::SRMessageBuilder("ExceptionType=")(_etn).GetOwnedSRString();
  }
};

class PQACORE_API InsufficientEngineDimensionsErrorParams : public IPqaErrorParams {
public: // variables
  const TPqaId _nAnswers;
  const TPqaId _minAnswers;
  const TPqaId _nQuestions;
  const TPqaId _minQuestions;
  const TPqaId _nTargets;
  const TPqaId _minTargets;
public: // methods
  InsufficientEngineDimensionsErrorParams(const TPqaId nAnswers, const TPqaId minAnswers,
    const TPqaId nQuestions, const TPqaId minQuestions, const TPqaId nTargets, const TPqaId minTargets)
    : _nAnswers(nAnswers), _minAnswers(minAnswers), _nQuestions(nQuestions), _minQuestions(minQuestions),
    _nTargets(nTargets), _minTargets(minTargets)
  { }

  virtual SRPlat::SRString ToString() override final {
    return SRPlat::SRMessageBuilder("[nAnswers=")(_nAnswers)(" of ")(_minAnswers)("] [nQuestions=")(_nQuestions)(" of ")
      (_minQuestions)("] [nTargets=")(_nTargets)(" of ")(_minTargets).AppendChar(']').GetOwnedSRString();
  }
};

class PQACORE_API MaintenanceModeErrorParams : public IPqaErrorParams {
  uint8_t _activeMode;
public:
  explicit MaintenanceModeErrorParams(const uint8_t activeMode) : _activeMode(activeMode) { }
  uint8_t GetActiveMode() const { return _activeMode; }

  virtual SRPlat::SRString ToString() override final {
    return SRPlat::SRMessageBuilder("ActiveMode=#")(int64_t(_activeMode)).GetOwnedSRString();
  }
};

class PQACORE_API ObjectShutDownErrorParams : public IPqaErrorParams {
  SRPlat::SRString _rejectedOp;
public:
  explicit ObjectShutDownErrorParams(SRPlat::SRString&& rejectedOp)
    : _rejectedOp(std::forward<SRPlat::SRString>(rejectedOp))
  { }
  const SRPlat::SRString& GetRejectedOp() { return _rejectedOp; }

  virtual SRPlat::SRString ToString() override final {
    return SRPlat::SRMessageBuilder("RejectedOperation=")(_rejectedOp).GetOwnedSRString();
  }
};

// Empty range should be represented by [0;-1] or [1;0] because we return inclusive bounds.
class PQACORE_API IndexOutOfRangeErrorParams : public IPqaErrorParams {
  TPqaId _subjIndex;
  TPqaId _minIndex;
  TPqaId _maxIndex;
public:
  explicit IndexOutOfRangeErrorParams(const TPqaId subjIndex, const TPqaId minIndex, const TPqaId maxIndex)
    : _subjIndex(subjIndex), _minIndex(minIndex), _maxIndex(maxIndex)
  { }
  TPqaId GetSubjIndex() const { return _subjIndex; }
  TPqaId GetMinIndex() const { return _minIndex; }
  TPqaId GetMaxIndex() const { return  _maxIndex; }

  virtual SRPlat::SRString ToString() override final {
    return SRPlat::SRMessageBuilder("subjIndex=")(_subjIndex)(" not in ")(_minIndex)("...")(_maxIndex)
      .GetOwnedSRString();
  }
};

class PQACORE_API InternalErrorParams : public IPqaErrorParams {
  const char *_sourceFN;
  int64_t _line;
public:
  explicit InternalErrorParams(const char* const sourceFN, const int64_t line) : _sourceFN(sourceFN), _line(line) { }
  virtual SRPlat::SRString ToString() override final {
    return SRPlat::SRMessageBuilder("Internal error at ")(_sourceFN)("(")(_line)(")").GetOwnedSRString();
  }
};

// Uses lazy initialization of _pImpl, so to minimize its overhead when no error happens.
class PQACORE_API AggregateErrorParams : public IPqaErrorParams {
  class Impl;
  Impl *_pImpl;

private: // methods
  // Takes ownership of the given Impl pointer.
  explicit AggregateErrorParams(Impl *pImpl);
  Impl* EnsureImpl();

public:
  explicit AggregateErrorParams();
  virtual ~AggregateErrorParams() override final;

  void Add(PqaError&& pe);
  size_t Count() const;

  AggregateErrorParams* Move();

  virtual SRPlat::SRString ToString() override final;
};

class PQACORE_API NegativeCountErrorParams : public IPqaErrorParams {
  int64_t _count;
public:
  explicit NegativeCountErrorParams(const int64_t count) : _count(count) { }
  int64_t GetCount() const { return _count; }
  virtual SRPlat::SRString ToString() override final {
    return SRPlat::SRMessageBuilder("count=")(_count).GetOwnedSRString();
  }
};

class PQACORE_API NonPositiveAmountErrorParams : public IPqaErrorParams {
  TPqaAmount _amount;
public:
  explicit NonPositiveAmountErrorParams(const TPqaAmount amount) : _amount(amount) { }
  TPqaAmount GetAmount() const { return _amount; }
  virtual SRPlat::SRString ToString() override final {
    return SRPlat::SRMessageBuilder("amount=")(_amount).GetOwnedSRString();
  }
};

class PQACORE_API AbsentIdErrorParams : public IPqaErrorParams {
  TPqaId _id;
public:
  explicit AbsentIdErrorParams(const TPqaId id) : _id(id) { }
  TPqaId GetId() const { return _id; }
  virtual SRPlat::SRString ToString() override final {
    return SRPlat::SRMessageBuilder("id=")(_id).GetOwnedSRString();
  }
};

class PQACORE_API I64UnderflowErrorParams : public IPqaErrorParams {
  int64_t _actual;
  int64_t _minAllowed;
public:
  explicit I64UnderflowErrorParams(const int64_t actual, const int64_t minAllowed) : _actual(actual),
    _minAllowed(minAllowed) { }
  int64_t GetActual() const { return _actual; }
  int64_t GetMinAllowed() const { return _minAllowed; }
  virtual SRPlat::SRString ToString() override final {
    return SRPlat::SRMessageBuilder("actual=")(_actual)(", minAllowed=")(_minAllowed).GetOwnedSRString();
  }
};

class PQACORE_API NoQuizActiveQuestionErrorParams : public IPqaErrorParams {
  TPqaId _iAnswer;
public:
  explicit NoQuizActiveQuestionErrorParams(const TPqaId iAnswer) : _iAnswer(iAnswer) { }
  TPqaId GetAnswerId() const { return _iAnswer; }
  virtual SRPlat::SRString ToString() override final {
    return SRPlat::SRMessageBuilder("answerId=")(_iAnswer).GetOwnedSRString();
  }
};

class PQACORE_API CantOpenFileErrorParams : public IPqaErrorParams {
  SRPlat::SRString _filePath;
public:
  explicit CantOpenFileErrorParams(const char *const filePath) : _filePath(SRPlat::SRString::MakeClone(filePath)) { }
  virtual SRPlat::SRString ToString() override final {
    return SRPlat::SRMessageBuilder("filePath=[")(_filePath)("]").GetOwnedSRString();
  }
};

class PQACORE_API FileOpErrorParams : public IPqaErrorParams {
  SRPlat::SRString _filePath;
public:
  explicit FileOpErrorParams(const char *const filePath) : _filePath(SRPlat::SRString::MakeClone(filePath)) { }
  virtual SRPlat::SRString ToString() override final {
    return SRPlat::SRMessageBuilder("filePath=[")(_filePath)("]").GetOwnedSRString();
  }
};

class PQACORE_API QuizzesActiveErrorParams : public IPqaErrorParams {
  TPqaId _nQuizzes;
public:
  explicit QuizzesActiveErrorParams(const TPqaId nQuizzes) : _nQuizzes(nQuizzes) { }
  virtual SRPlat::SRString ToString() override final {
    return SRPlat::SRMessageBuilder("nQuizzes=[")(_nQuizzes)("]").GetOwnedSRString();
  }
};

} // namespace ProbQA
