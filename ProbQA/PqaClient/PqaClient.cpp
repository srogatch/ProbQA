// Probabilistic Question-Answering system
// @2017 Sarge Rogatch
// This software is distributed under GNU AGPLv3 license. See file LICENSE in repository root for details.

#include "stdafx.h"

using namespace ProbQA;
using namespace SRPlat;

int __cdecl main() {
  const char* baseName = "Logs\\PqaClient";
  if (!CreateDirectoryA("Logs", nullptr)) {
    uint32_t le = GetLastError();
    if (le != ERROR_ALREADY_EXISTS) {
      baseName = "PqaClient";
    }
  }
  SRDefaultLogger::Init(SRString::MakeUnowned(baseName));

  FILE *fpProgress = fopen("progress.txt", "wt");

  PqaError err;
  EngineDefinition ed;
  ed._dims._nAnswers = 5;
  ed._dims._nQuestions = 1000;
  ed._dims._nTargets = 1000;
  ed._initAmount = 0.1;
  ed._prec._type = TPqaPrecisionType::Double;
  IPqaEngine *pEngine = PqaGetEngineFactory().CreateCpuEngine(err, ed);
  if (!err.IsOk() || pEngine == nullptr) {
    fprintf(stderr, "Failed to instantiate a ProbQA engine.\n");
    return int(SRExitCode::UnspecifiedError);
  }

  SRFastRandom fr;
  SREntropyAdapter ea(fr);

  constexpr int64_t cnTrainings = 1000 * 1000;
  constexpr int64_t cMaxQuizLen = 100;
  constexpr int64_t cMaxTrialLen = 30;
  constexpr int64_t cnTopRated = 10;

  int64_t nCorrect = 0;
  for (int64_t i = 0; i < cnTrainings; i++) {
    if (((i & 255) == 0) && (i != 0)) {
      const uint64_t totQAsked = pEngine->GetTotalQuestionsAsked(err);
      if (!err.IsOk()) {
        fprintf(stderr, "Failed to query the total number of questions asked.\n");
        return int(SRExitCode::UnspecifiedError);
      }
      const double precision = nCorrect * 100.0 / 256;
      printf("\n*%" PRIu64 ";%.2lf%%*", totQAsked, precision);
      fprintf(fpProgress, "%" PRId64 "\t%" PRIu64 "\t%lf\n", i, totQAsked, precision);
      fflush(fpProgress);
      nCorrect = 0;
    }
    const TPqaId guess = ea.Generate<TPqaId>(ed._dims._nTargets);
    const TPqaId iQuiz = pEngine->StartQuiz(err);
    if (!err.IsOk() || iQuiz == cInvalidPqaId) {
      fprintf(stderr, "Failed to create a quiz.\n");
      return int(SRExitCode::UnspecifiedError);
    }
    int64_t j = 0;
    for (; j < cMaxQuizLen; j++) {
      const TPqaId iQuestion = pEngine->NextQuestion(err, iQuiz);
      if (!err.IsOk() || iQuestion == cInvalidPqaId) {
        fprintf(stderr, "Failed to query a next question.\n");
        return int(SRExitCode::UnspecifiedError);
      }
      TPqaId iAnswer;
      if (guess < iQuestion - 32) {
        iAnswer = 0;
      }
      else if (iQuestion - 32 <= guess && guess < iQuestion) {
        iAnswer = 1;
      }
      else if (iQuestion == guess) {
        iAnswer = 2;
      }
      else if (iQuestion < guess && guess <= iQuestion + 32) {
        iAnswer = 3;
      }
      else if (guess > iQuestion + 32) {
        iAnswer = 4;
      }
      else {
        fprintf(stderr, "Answering logic error.\n");
        return int(SRExitCode::UnspecifiedError);
      }
      err = pEngine->RecordAnswer(iQuiz, iAnswer);
      if (!err.IsOk()) {
        fprintf(stderr, "Failed to record answer: %s\n", err.ToString(true).ToString().c_str());
        return int(SRExitCode::UnspecifiedError);
      }

      RatedTarget rts[cnTopRated];
      const TPqaId nListed = pEngine->ListTopTargets(err, iQuiz, cnTopRated, rts);
      if (!err.IsOk() || nListed != cnTopRated) {
        fprintf(stderr, "Failed to list top targets.\n");
        return int(SRExitCode::UnspecifiedError);
      }

      //for (TPqaId k = 0; k < cnTopRated; k++) {
      //  printf(" [%g; %" PRId64 "] ", rts[k]._prob, rts[k]._iTarget);
      //}
      //printf("\n");

      TPqaId posInTop = cInvalidPqaId;
      for (TPqaId k = 0; k < cnTopRated; k++) {
        if (rts[k]._iTarget == guess) {
          posInTop = k;
          break;
        }
      }
      if (posInTop != cInvalidPqaId) {
        nCorrect++;
        printf("[guess=%" PRId64 ",top=%" PRId64 ",after=%" PRId64 "]", int64_t(guess), int64_t(posInTop), int64_t(j));
        break;
      }
    }
    if (j >= cMaxQuizLen) {
      printf("-");
    }
    err = pEngine->RecordQuizTarget(iQuiz, guess);
    if (!err.IsOk()) {
      fprintf(stderr, "Failed to record quiz target: %s\n", err.ToString(true).ToString().c_str());
      return int(SRExitCode::UnspecifiedError);
    }
    err = pEngine->ReleaseQuiz(iQuiz);
    if (!err.IsOk()) {
      fprintf(stderr, "Failed to release a quiz: %s\n", err.ToString(true).ToString().c_str());
      return int(SRExitCode::UnspecifiedError);
    }
  }
  return 0;
}
