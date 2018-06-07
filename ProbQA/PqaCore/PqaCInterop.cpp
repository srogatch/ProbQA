// Probabilistic Question-Answering system
// @2017 Sarge Rogatch
// This software is distributed under GNU AGPLv3 license. See file LICENSE in repository root for details.

#include "stdafx.h"
#include "../PqaCore/Interface/PqaCInterop.h"
#include "../PqaCore/Interface/IPqaEngineFactory.h"
#include "../PqaCore/Interface/IPqaEngine.h"

//TODO: catch exceptions
//TODO: handle null pointers
//TODO: return errors of the current file's logic

using namespace ProbQA;
using namespace SRPlat;

namespace {

char *PrepareSRString(const SRString &s) {
  const char *pSrc;
  size_t len = s.GetData(pSrc);
  char *pDest = new char[len + 1];
  memcpy(pDest, pSrc, len);
  pDest[len] = 0;
  return pDest;
}

void SendUnexpectedError(void **ppStrErr) {
  const char srcMsg[] = "Unexpected error.";
  constexpr size_t len = sizeof(srcMsg);
  char *pMsg = new char[len];
  memcpy(pMsg, srcMsg, len);
  *ppStrErr = pMsg;
}

} // anonymous namespace

PQACORE_API void* CiPqaGetEngineFactory() {
  return &(PqaGetEngineFactory());
}

PQACORE_API void* CiPqaEngineFactory_CreateCpuEngine(void* pvFactory, void **ppError, CiEngineDefinition *pEngDef) {
  IPqaEngineFactory *pEf = static_cast<IPqaEngineFactory *>(pvFactory);
  if (pEf == nullptr) {
    *ppError = new PqaError(PqaErrorCode::NullArgument, nullptr, SRString::MakeUnowned(
      SR_FILE_LINE "Nullptr is passed in place of IPqaEngineFactory."));
    return nullptr;
  }
  EngineDefinition engDef;
  engDef._dims._nAnswers = pEngDef->_nAnswers;
  engDef._dims._nQuestions = pEngDef->_nQuestions;
  engDef._dims._nTargets = pEngDef->_nTargets;
  engDef._prec._type = static_cast<TPqaPrecisionType>(pEngDef->_precType);
  engDef._prec._mantissa = pEngDef->_precMantissa;
  engDef._prec._exponent = pEngDef->_precExponent;
  engDef._initAmount = pEngDef->_initAmount;
  engDef._memPoolMaxBytes = pEngDef->_memPoolMaxBytes;
  PqaError err;
  IPqaEngine *pEngine = pEf->CreateCpuEngine(err, engDef);
  if (err.IsOk()) {
    *ppError = nullptr;
  }
  else
  {
    PqaError *pErr = new PqaError(std::move(err));
    *ppError = pErr;
  }
  return pEngine;
}

PQACORE_API void* CiqaEngineFactory_LoadCpuEngine(void *pvFactory, void **ppError, const char* filePath,
  uint64_t memPoolMaxBytes)
{
  IPqaEngineFactory *pEf = static_cast<IPqaEngineFactory *>(pvFactory);
  if (pEf == nullptr) {
    *ppError = new PqaError(PqaErrorCode::NullArgument, nullptr, SRString::MakeUnowned(
      SR_FILE_LINE "Nullptr is passed in place of IPqaEngineFactory."));
    return nullptr;
  }
  PqaError err;
  IPqaEngine *pEngine = pEf->LoadCpuEngine(err, filePath, memPoolMaxBytes);
  if (err.IsOk()) {
    *ppError = nullptr;
  }
  else
  {
    PqaError *pErr = new PqaError(std::move(err));
    *ppError = pErr;
  }
  return pEngine;
}

PQACORE_API void CiReleasePqaError(void *pvErr) {
  PqaError *pPe = static_cast<PqaError*>(pvErr);
  delete pPe;
}

PQACORE_API void CiReleaseString(void *pvString) {
  char *pS = static_cast<char*>(pvString);
  delete pS;
}

PQACORE_API void* CiPqaError_ToString(void *pvError, const bool withParams) {
  PqaError *pErr = static_cast<PqaError*>(pvError);
  SRString srStr = pErr->ToString(withParams);
  return PrepareSRString(srStr);
}

PQACORE_API bool CiLogger_Init(void **ppStrErr, const char* baseName) {
  try {
    SRDefaultLogger::Init(SRString::MakeUnowned(baseName));
    *ppStrErr = nullptr;
    return true;
  }
  catch (const std::exception& ex) {
    size_t len = strlen(ex.what());
    char *pMsg = new char[len+1];
    memcpy(pMsg, ex.what(), len + 1);
    *ppStrErr = pMsg;
    return false;
  }
  catch (...) {
    SendUnexpectedError(ppStrErr);
    return false;
  }
}

PQACORE_API void CiReleasePqaEngine(void *pvEngine) {
  IPqaEngine *pPe = static_cast<IPqaEngine*>(pvEngine);
  delete pPe;
}

PQACORE_API void* PqaEngine_Train(void *pvEngine, int64_t nQuestions, const CiAnsweredQuestion* const pAQs,
  const int64_t iTarget, const double amount)
{
  IPqaEngine *pEng = static_cast<IPqaEngine*>(pvEngine);
  if (pEng == nullptr) {
    return new PqaError(PqaErrorCode::NullArgument, nullptr, SRString::MakeUnowned(
      SR_FILE_LINE "Nullptr is passed in place of IPqaEngine."));
  }
  // Otherwise we have to do a translation here, placing AnsweredQuestion objects on the stack
  static_assert(sizeof(CiAnsweredQuestion) == sizeof(AnsweredQuestion)
    && offsetof(CiAnsweredQuestion, _iQuestion) == offsetof(AnsweredQuestion, _iQuestion)
    && offsetof(CiAnsweredQuestion, _iAnswer) == offsetof(AnsweredQuestion, _iAnswer));
  PqaError err = pEng->Train(nQuestions, reinterpret_cast<const AnsweredQuestion*>(pAQs), iTarget, amount);
  if (err.IsOk()) {
    return nullptr;
  }
  return new PqaError(std::move(err));
}
