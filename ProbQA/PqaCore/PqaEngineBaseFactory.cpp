// Probabilistic Question-Answering system
// @2017 Sarge Rogatch
// This software is distributed under GNU AGPLv3 license. See file LICENSE in repository root for details.

#include "stdafx.h"
#include "PqaEngineBaseFactory.h"
#include "../PqaCore/Interface/PqaErrorParams.h"
#include "../PqaCore/CpuEngine.h"
#include "../PqaCore/ErrorHelper.h"

using namespace SRPlat;

namespace ProbQA {

IPqaEngine* PqaEngineBaseFactory::MakeCpuEngine(PqaError& err, const EngineDefinition& engDef, KBFileInfo *pKbFi) {
  try {
    std::unique_ptr<IPqaEngine> pEngine;
    switch (engDef._prec._type) {
    case TPqaPrecisionType::Double:
      pEngine.reset(new CpuEngine<SRDoubleNumber>(engDef, pKbFi));
      break;
    default:
      //TODO: implement
      err = PqaError(PqaErrorCode::NotImplemented, new NotImplementedErrorParams(SRString::MakeUnowned(SR_FILE_LINE
        "ProbQA Engine on CPU for precision except double.")));
      return nullptr;
    }
    err.Release();
    return pEngine.release();
  }
  CATCH_TO_ERR_SET(err);
  return nullptr;
}

IPqaEngine* PqaEngineBaseFactory::CreateCpuEngine(PqaError& err, const EngineDefinition& engDef) {
  if (engDef._dims._nAnswers < _cMinAnswers || engDef._dims._nQuestions < _cMinQuestions
    || engDef._dims._nTargets < _cMinTargets)
  {
    err = PqaError(PqaErrorCode::InsufficientEngineDimensions, new InsufficientEngineDimensionsErrorParams(
      engDef._dims._nAnswers, _cMinAnswers, engDef._dims._nQuestions, _cMinQuestions, engDef._dims._nTargets,
      _cMinTargets));
    return nullptr;
  }
  return MakeCpuEngine(err, engDef, nullptr);
}

IPqaEngine* PqaEngineBaseFactory::LoadCpuEngine(PqaError& err, const char* const filePath, size_t memPoolMaxBytes) {
  SRSmartFile sf(std::fopen(filePath, "rb"));
  if (sf.Get() == nullptr) {
    err = PqaError(PqaErrorCode::CantOpenFile, new CantOpenFileErrorParams(filePath), SRString::MakeUnowned(
      SR_FILE_LINE "Can't open the KB file to read."));
    return nullptr;
  }
  if (std::setvbuf(sf.Get(), nullptr, _IOFBF, BaseCpuEngine::_cFileBufSize) != 0) {
    err = PqaError(PqaErrorCode::FileOp, new FileOpErrorParams(filePath), SRMessageBuilder(SR_FILE_LINE
      "Can't set file buffer size to ")(BaseCpuEngine::_cFileBufSize).GetOwnedSRString());
    return nullptr;
  }

  EngineDefinition engDef;
  engDef._memPoolMaxBytes = memPoolMaxBytes;
  if (std::fread(&engDef._prec, sizeof(engDef._prec), 1, sf.Get()) != 1) {
    err = PqaError(PqaErrorCode::FileOp, new FileOpErrorParams(filePath), SRString::MakeUnowned(SR_FILE_LINE
      "Can't read precision definition header."));
    return nullptr;
  }

  if (std::fread(&engDef._dims, sizeof(engDef._dims), 1, sf.Get()) != 1) {
    err = PqaError(PqaErrorCode::FileOp, new FileOpErrorParams(filePath), SRString::MakeUnowned(SR_FILE_LINE
      "Can't read engine dimensions header."));
    return nullptr;
  }

  KBFileInfo kbFi(sf, filePath);
  return MakeCpuEngine(err, engDef, &kbFi);
}

IPqaEngine* PqaEngineBaseFactory::CreateCudaEngine(PqaError& err, const EngineDefinition& engDef) {
  (void)engDef; //TODO: remove when implemented
  //TODO: implement
  err = PqaError(PqaErrorCode::NotImplemented, new NotImplementedErrorParams(SRString::MakeUnowned(SR_FILE_LINE
    "ProbQA Engine on CUDA.")));
  return nullptr;
}

IPqaEngine* PqaEngineBaseFactory::CreateGridEngine(PqaError& err, const EngineDefinition& engDef) {
  (void)engDef; //TODO: remove when implemented
  //TODO: implement
  err = PqaError(PqaErrorCode::NotImplemented, new NotImplementedErrorParams(SRString::MakeUnowned(SR_FILE_LINE
    "ProbQA Engine over a grid.")));
  return nullptr;
}

} // namespace ProbQA
