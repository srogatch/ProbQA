#include "stdafx.h"
#include "../PqaCore/CpuEngine.h"
#include "../PqaCore/DoubleNumber.h"
#include "../PqaCore/PqaException.h"

using namespace SRPlat;

namespace ProbQA {

template<typename taNumber> CpuEngine<taNumber>::CpuEngine(const EngineDefinition& engDef)
  : _initAmount(engDef._initAmount), _dims(engDef._dims)
{
  if (_dims._nAnswers < cMinAnswers || _dims._nQuestions < cMinQuestions || _dims._nTargets < cMinTargets)
  {
    throw PqaException(PqaErrorCode::InsufficientEngineDimensions, new InsufficientEngineDimensionsErrorParams(
      _dims._nAnswers, cMinAnswers, _dims._nQuestions, cMinQuestions, _dims._nTargets, cMinTargets));
  }
  _mA.resize(_dims._nAnswers);
  for (TPqaId i = 0; i < _dims._nAnswers; i++) {
    _mA[i].resize(_dims._nQuestions);
    for (TPqaId j = 0; j < _dims._nQuestions; j++) {
      _mA[i][j].resize(_dims._nTargets, _initAmount);
    }
  }
  _vB.resize(_dims._nTargets, _initAmount);

  _questionGaps.GrowTo(_dims._nQuestions);
  _targetGaps.GrowTo(_dims._nTargets);

  throw PqaException(PqaErrorCode::NotImplemented, new NotImplementedErrorParams(SRString::MakeUnowned(
    "CpuEngine<taNumber>::CpuEngine(const EngineDefinition& engDef)")));
}

template<typename taNumber> PqaError CpuEngine<taNumber>::Train(const TPqaId nQuestions,
  const AnsweredQuestion* const pAQs, const TPqaId iTarget, const TPqaAmount amount) 
{
  // This method should increase the counter of questions asked by the number of questions in this training.
  return PqaError(PqaErrorCode::NotImplemented, new NotImplementedErrorParams(SRString::MakeUnowned(
    "CpuEngine<taNumber>::Train")));
}

template<typename taNumber> TPqaId CpuEngine<taNumber>::StartQuiz(PqaError& err) {
  err = PqaError(PqaErrorCode::NotImplemented, new NotImplementedErrorParams(SRString::MakeUnowned(
    "CpuEngine<taNumber>::StartQuiz")));
  return cInvalidPqaId;
}

template<typename taNumber> TPqaId CpuEngine<taNumber>::ResumeQuiz(PqaError& err, const TPqaId nQuestions,
  const AnsweredQuestion* const pAQs) 
{
  err = PqaError(PqaErrorCode::NotImplemented, new NotImplementedErrorParams(SRString::MakeUnowned(
    "CpuEngine<taNumber>::ResumeQuiz")));
  return cInvalidPqaId;
}

template<typename taNumber> TPqaId CpuEngine<taNumber>::NextQuestion(PqaError& err, const TPqaId iQuiz) {
  err = PqaError(PqaErrorCode::NotImplemented, new NotImplementedErrorParams(SRString::MakeUnowned(
    "CpuEngine<taNumber>::NextQuestion")));
  return cInvalidPqaId;
}

template<typename taNumber> PqaError CpuEngine<taNumber>::RecordAnswer(const TPqaId iQuiz, const TPqaId iAnswer) {
  return PqaError(PqaErrorCode::NotImplemented, new NotImplementedErrorParams(SRString::MakeUnowned(
    "CpuEngine<taNumber>::RecordAnswer")));
}

template<typename taNumber> TPqaId CpuEngine<taNumber>::ListTopTargets(PqaError& err, const TPqaId iQuiz,
  const TPqaId maxCount, RatedTarget *pDest) 
{
  err = PqaError(PqaErrorCode::NotImplemented, new NotImplementedErrorParams(SRString::MakeUnowned(
    "CpuEngine<taNumber>::ListTopTargets")));
  return cInvalidPqaId;
}

template<typename taNumber> PqaError CpuEngine<taNumber>::RecordQuizTarget(const TPqaId iQuiz, const TPqaId iTarget,
  const TPqaAmount amount) 
{
  return PqaError(PqaErrorCode::NotImplemented, new NotImplementedErrorParams(SRString::MakeUnowned(
    "CpuEngine<taNumber>::RecordQuizTarget")));
}

template<typename taNumber> PqaError CpuEngine<taNumber>::ReleaseQuiz(const TPqaId iQuiz) {
  return PqaError(PqaErrorCode::NotImplemented, new NotImplementedErrorParams(SRString::MakeUnowned(
    "CpuEngine<taNumber>::ReleaseQuiz")));
}


template<typename taNumber> PqaError CpuEngine<taNumber>::SaveKB(const char* const filePath, const bool bDoubleBuffer) {
  return PqaError(PqaErrorCode::NotImplemented, new NotImplementedErrorParams(SRString::MakeUnowned(
    "CpuEngine<taNumber>::SaveKB")));
}

template<typename taNumber> uint64_t CpuEngine<taNumber>::GetTotalQuestionsAsked(PqaError& err) {
  err.Release();
  return _nQuestionsAsked;
}

template<typename taNumber> PqaError CpuEngine<taNumber>::StartMaintenance(const bool forceQuizes) {
  return PqaError(PqaErrorCode::NotImplemented, new NotImplementedErrorParams(SRString::MakeUnowned(
    "CpuEngine<taNumber>::StartMaintenance")));
}

template<typename taNumber> PqaError CpuEngine<taNumber>::FinishMaintenance() {
  return PqaError(PqaErrorCode::NotImplemented, new NotImplementedErrorParams(SRString::MakeUnowned(
    "CpuEngine<taNumber>::FinishMaintenance")));
}

template<typename taNumber> TPqaId CpuEngine<taNumber>::AddQuestion(PqaError& err, const TPqaAmount initialAmount) {
  err = PqaError(PqaErrorCode::NotImplemented, new NotImplementedErrorParams(SRString::MakeUnowned(
    "CpuEngine<taNumber>::AddQuestion")));
  return cInvalidPqaId;
}

template<typename taNumber> PqaError CpuEngine<taNumber>::AddQuestions(TPqaId nQuestions, AddQuestionParam *pAqps) {
  return PqaError(PqaErrorCode::NotImplemented, new NotImplementedErrorParams(SRString::MakeUnowned(
    "CpuEngine<taNumber>::AddQuestions")));
}

template<typename taNumber> TPqaId CpuEngine<taNumber>::AddTarget(PqaError& err, const TPqaAmount initialAmount) {
  err = PqaError(PqaErrorCode::NotImplemented, new NotImplementedErrorParams(SRString::MakeUnowned(
    "CpuEngine<taNumber>::AddTarget")));
  return cInvalidPqaId;
}

template<typename taNumber> PqaError CpuEngine<taNumber>::AddTargets(TPqaId nTargets, AddTargetParam *pAtps) {
  return PqaError(PqaErrorCode::NotImplemented, new NotImplementedErrorParams(SRString::MakeUnowned(
    "CpuEngine<taNumber>::AddTargets")));
}

template<typename taNumber> PqaError CpuEngine<taNumber>::RemoveQuestion(const TPqaId iQuestion) {
  return PqaError(PqaErrorCode::NotImplemented, new NotImplementedErrorParams(SRString::MakeUnowned(
    "CpuEngine<taNumber>::RemoveQuestion")));
}

template<typename taNumber> PqaError CpuEngine<taNumber>::RemoveQuestions(const TPqaId nQuestions, const TPqaId *pQIds)
{
  return PqaError(PqaErrorCode::NotImplemented, new NotImplementedErrorParams(SRString::MakeUnowned(
    "CpuEngine<taNumber>::RemoveQuestions")));
}

template<typename taNumber> PqaError CpuEngine<taNumber>::RemoveTarget(const TPqaId iTarget) {
  return PqaError(PqaErrorCode::NotImplemented, new NotImplementedErrorParams(SRString::MakeUnowned(
    "CpuEngine<taNumber>::RemoveTarget")));
}

template<typename taNumber> PqaError CpuEngine<taNumber>::RemoveTargets(const TPqaId nTargets, const TPqaId *pTIds) {
  return PqaError(PqaErrorCode::NotImplemented, new NotImplementedErrorParams(SRString::MakeUnowned(
    "CpuEngine<taNumber>::RemoveTargets")));
}

template<typename taNumber> PqaError CpuEngine<taNumber>::Compact(CompactionResult &cr) {
  return PqaError(PqaErrorCode::NotImplemented, new NotImplementedErrorParams(SRString::MakeUnowned(
    "CpuEngine<taNumber>::Compact")));
}

template<typename taNumber> PqaError CpuEngine<taNumber>::ReleaseCompactionResult(CompactionResult &cr) {
  return PqaError(PqaErrorCode::NotImplemented, new NotImplementedErrorParams(SRString::MakeUnowned(
    "CpuEngine<taNumber>::ReleaseCompactionResult")));
}

//// Instantiations
template class CpuEngine<DoubleNumber>;

} // namespace ProbQA