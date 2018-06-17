// Probabilistic Question-Answering system
// @2017 Sarge Rogatch
// This software is distributed under GNU AGPLv3 license. See file LICENSE in repository root for details.

#include "stdafx.h"
#include "PqaEngineBaseFactory.h"
#include "../PqaCore/Interface/PqaErrorParams.h"
#include "../PqaCore/CpuEngine.h"
#include "../PqaCore/ErrorHelper.h"
#include "../PqaCore/CudaEngine.h"

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
  err = CheckDimensions(engDef);
  if (!err.IsOk()) {
    return nullptr;
  }
  return MakeCpuEngine(err, engDef, nullptr);
}

IPqaEngine* PqaEngineBaseFactory::LoadCpuEngine(PqaError& err, const char* const filePath, size_t memPoolMaxBytes) {
  SRSmartFile sf;
  EngineDefinition engDef;
  err = LoadEngineDefinition(sf, filePath, engDef);
  if (!err.IsOk()) {
    return nullptr;
  }
  engDef._memPoolMaxBytes = memPoolMaxBytes;
  KBFileInfo kbFi(sf, filePath);
  return MakeCpuEngine(err, engDef, &kbFi);
}

PqaError PqaEngineBaseFactory::LoadEngineDefinition(SRSmartFile &sf, const char* const filePath,
  EngineDefinition& engDef)
{
  if (filePath == nullptr) {
    return PqaError(PqaErrorCode::NullArgument, nullptr, SRString::MakeUnowned(
      SR_FILE_LINE "Nullptr is passed in place of KB file name."));
  }
  sf.Set(std::fopen(filePath, "rb"));
  if (sf.Get() == nullptr) {
    return PqaError(PqaErrorCode::CantOpenFile, new CantOpenFileErrorParams(filePath), SRString::MakeUnowned(
      SR_FILE_LINE "Can't open the KB file to read."));
  }
  if (std::setvbuf(sf.Get(), nullptr, _IOFBF, BaseCpuEngine::_cFileBufSize) != 0) {
    return PqaError(PqaErrorCode::FileOp, new FileOpErrorParams(filePath), SRMessageBuilder(SR_FILE_LINE
      "Can't set file buffer size to ")(BaseCpuEngine::_cFileBufSize).GetOwnedSRString());
  }

  if (std::fread(&engDef._prec, sizeof(engDef._prec), 1, sf.Get()) != 1) {
    return PqaError(PqaErrorCode::FileOp, new FileOpErrorParams(filePath), SRString::MakeUnowned(SR_FILE_LINE
      "Can't read precision definition header."));
  }

  if (std::fread(&engDef._dims, sizeof(engDef._dims), 1, sf.Get()) != 1) {
    return PqaError(PqaErrorCode::FileOp, new FileOpErrorParams(filePath), SRString::MakeUnowned(SR_FILE_LINE
      "Can't read engine dimensions header."));
  }
  return PqaError();
}

IPqaEngine* PqaEngineBaseFactory::CreateGridEngine(PqaError& err, const EngineDefinition& engDef) {
  (void)engDef; //TODO: remove when implemented
  //TODO: implement
  err = PqaError(PqaErrorCode::NotImplemented, new NotImplementedErrorParams(SRString::MakeUnowned(SR_FILE_LINE
    "ProbQA Engine over a grid.")));
  return nullptr;
}

IPqaEngine* PqaEngineBaseFactory::CreateCudaEngine(PqaError& err, const EngineDefinition& engDef) {
  err = CheckDimensions(engDef);
  if (!err.IsOk()) {
    return nullptr;
  }
  return MakeCudaEngine(err, engDef, nullptr);
}

IPqaEngine* PqaEngineBaseFactory::LoadCudaEngine(PqaError& err, const char* const filePath, size_t memPoolMaxBytes) {
  SRSmartFile sf;
  EngineDefinition engDef;
  err = LoadEngineDefinition(sf, filePath, engDef);
  if (!err.IsOk()) {
    return nullptr;
  }
  engDef._memPoolMaxBytes = memPoolMaxBytes;
  KBFileInfo kbFi(sf, filePath);
  return MakeCudaEngine(err, engDef, &kbFi);
}

PqaError PqaEngineBaseFactory::CheckDimensions(const EngineDefinition& engDef) {
  if (engDef._dims._nAnswers < _cMinAnswers || engDef._dims._nQuestions < _cMinQuestions
    || engDef._dims._nTargets < _cMinTargets)
  {
    return PqaError(PqaErrorCode::InsufficientEngineDimensions, new InsufficientEngineDimensionsErrorParams(
      engDef._dims._nAnswers, _cMinAnswers, engDef._dims._nQuestions, _cMinQuestions, engDef._dims._nTargets,
      _cMinTargets));
  }
  return PqaError();
}

IPqaEngine* PqaEngineBaseFactory::MakeCudaEngine(PqaError& err, const EngineDefinition& engDef, KBFileInfo *pKbFi) {
  try {
    std::unique_ptr<IPqaEngine> pEngine;
    switch (engDef._prec._type) {
    case TPqaPrecisionType::Float:
      pEngine.reset(new CudaEngine<float>(engDef, pKbFi));
      break;
    default:
      //TODO: implement
      err = PqaError(PqaErrorCode::NotImplemented, new NotImplementedErrorParams(SRString::MakeUnowned(SR_FILE_LINE
        "ProbQA Engine on CUDA for precision except float.")));
      return nullptr;
    }
    err.Release();
    return pEngine.release();
  }
  CATCH_TO_ERR_SET(err);
  return nullptr;
}

} // namespace ProbQA
