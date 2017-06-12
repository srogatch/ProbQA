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

  virtual SRPlat::SRString ToString() override {
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
  
  virtual SRPlat::SRString ToString() override {
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

  virtual SRPlat::SRString ToString() override {
    return SRPlat::SRMessageBuilder("[nAnswers=")(_nAnswers)(" of ")(_minAnswers)("] [nQuestions=")(_nQuestions)(" of ")
      (_minQuestions)("] [nTargets=")(_nTargets)(" of ")(_minTargets).AppendChar(']').GetOwnedSRString();
  }
};

class PQACORE_API MaintenanceModeErrorParams : public IPqaErrorParams {
  uint8_t _activeMode;
public:
  explicit MaintenanceModeErrorParams(const uint8_t activeMode) : _activeMode(activeMode) { }
  uint8_t GetActiveMode() const { return _activeMode; }

  virtual SRPlat::SRString ToString() override {
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

  virtual SRPlat::SRString ToString() override {
    return SRPlat::SRMessageBuilder("RejectedOperation=")(_rejectedOp).GetOwnedSRString();
  }
};

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

  virtual SRPlat::SRString ToString() override {
    return SRPlat::SRMessageBuilder("subjIndex=")(_subjIndex)(" not in ")(_minIndex)("...")(_maxIndex)
      .GetOwnedSRString();
  }
};

class PQACORE_API InternalErrorParams : public IPqaErrorParams {
  const char *_sourceFN;
  int64_t _line;
public:
  explicit InternalErrorParams(const char* const sourceFN, const int64_t line) : _sourceFN(sourceFN), _line(line) { }
  virtual SRPlat::SRString ToString() override {
    return SRPlat::SRMessageBuilder("Internal error at ")(_sourceFN)("(")(_line)(")").GetOwnedSRString();
  }
};

class PQACORE_API AggregateErrorParams : public IPqaErrorParams {
  class Impl;
  Impl *_pImpl;
public:
  explicit AggregateErrorParams();
  virtual ~AggregateErrorParams() override;

  void Add(PqaError&& pe);
  size_t Count();

  virtual SRPlat::SRString ToString() override;
};

} // namespace ProbQA
