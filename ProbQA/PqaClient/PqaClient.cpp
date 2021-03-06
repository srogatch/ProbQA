// Probabilistic Question-Answering system
// @2017 Sarge Rogatch
// This software is distributed under GNU AGPLv3 license. See file LICENSE in repository root for details.

#include "stdafx.h"

using namespace ProbQA;
using namespace SRPlat;

//////TODO: move these to common routines when needed by something else
const uint64_t gPerfCntFreq = []() {
  LARGE_INTEGER li;
  if (!QueryPerformanceFrequency(&li)) {
    printf("Can't get performance counter frequency: error %u.\n", GetLastError());
  }
  return li.QuadPart;
}();

uint64_t GetPerfCnt() {
  LARGE_INTEGER li;
  if (!QueryPerformanceCounter(&li)) {
    printf("Failed QueryPerformanceCounter(): error %u.\n", GetLastError());
  }
  return li.QuadPart;
}

bool ExistsDirectory(const char* const szPath) {
  DWORD dwAttrib = GetFileAttributesA(szPath);
  return (dwAttrib != INVALID_FILE_ATTRIBUTES &&
    (dwAttrib & FILE_ATTRIBUTE_DIRECTORY));
}

//////End of common routines

void InfLoopThread() {
  for (;;) {}
}

void InfLoopHead() {
  std::vector<std::thread> testLoad;
  for (int i = 0; i < 16; i++) {
    testLoad.emplace_back(&InfLoopThread);
  }
  for (int i = 0; i < 16; i++) {
    testLoad[i].join();
  }
}

namespace {

std::string gKbsDir;

constexpr int64_t cnTrainings = 4 * 1000 * 1000;
constexpr int64_t cMaxQuizLen = 25;
constexpr int64_t cnTopRated = 1;

SRCriticalSection gcsReport;
int64_t gnCorrect;
int64_t gnWrong;
int64_t gSumQuizLens;
double gTotCertainty;
uint64_t gpcStart;
uint64_t gPrevQAsked;
int64_t gNTrainings;
FILE *gFpProgress;

} // anonymous namespace

void LearnerThread(IPqaEngine *pEngine) {
  PqaError err;
  SRFastRandom fr;
  SREntropyAdapter ea(fr);
  for(;;) {
    {
      SRLock<SRCriticalSection> csl(gcsReport);
      gNTrainings++;
      if (gNTrainings > cnTrainings) {
        break;
      }
      if (((gNTrainings & 1023) == 0) && (gNTrainings != 0)) {
        const uint64_t totQAsked = pEngine->GetTotalQuestionsAsked(err);
        if (!err.IsOk()) {
          fprintf(stderr, SR_FILE_LINE "Failed to query the total number of questions asked.\n");
          return;
        }
        const double precision = gnCorrect * 100.0 / (gnCorrect + gnWrong);
        const double elapsedSec = double(GetPerfCnt() - gpcStart) / gPerfCntFreq;
        printf("\n*%" PRIu64 ";%.2lf%%*", totQAsked, precision);
        fprintf(gFpProgress, "%" PRId64 "\t%" PRIu64 "\t%lf\t%lf\t%lf\t%lf\n", gNTrainings, totQAsked, precision,
          double(gSumQuizLens) / gnCorrect, gTotCertainty / gnCorrect, (totQAsked - gPrevQAsked) / elapsedSec);
        fflush(gFpProgress);

        char kbFile[128];
        sprintf(kbFile, "%sdichotomy%.6" PRId64 ".kb", gKbsDir.c_str(), gNTrainings);
        err = pEngine->SaveKB(kbFile, false);
        if (!err.IsOk()) {
          fprintf(stderr, SR_FILE_LINE "Failed to save the KB.\n");
          return;
        }

        gnCorrect = 0;
        gnWrong = 0;
        gSumQuizLens = 0;
        gTotCertainty = 0;
        gPrevQAsked = totQAsked;
        gpcStart = GetPerfCnt();
      }
    }
    const TPqaId guess = ea.Generate<TPqaId>(pEngine->CopyDims()._nTargets);
    volatile TPqaId dbgGuess = guess;
    const TPqaId iQuiz = pEngine->StartQuiz(err);
    if (!err.IsOk() || iQuiz == cInvalidPqaId) {
      fprintf(stderr, "Failed to create a quiz.\n");
      return;
    }
    int64_t j = 0;
    for (; j < cMaxQuizLen; j++) {
      const TPqaId iQuestion = pEngine->NextQuestion(err, iQuiz);
      if (!err.IsOk() || iQuestion == cInvalidPqaId) {
        fprintf(stderr, "Failed to query a next question.\n");
        return;
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
        return;
      }
      err = pEngine->RecordAnswer(iQuiz, iAnswer);
      if (!err.IsOk()) {
        fprintf(stderr, "Failed to record answer: %s\n", err.ToString(true).ToStd().c_str());
        return;
      }

      RatedTarget rts[cnTopRated];
      const TPqaId nListed = pEngine->ListTopTargets(err, iQuiz, cnTopRated, rts);
      if (!err.IsOk() || nListed != cnTopRated) {
        fprintf(stderr, "Failed to list top targets.\n");
        return;
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
        const double certainty = rts[posInTop]._prob * 100;
        //printf("[guess=%" PRId64 ",top=%" PRId64 ",after=%" PRId64 "]", int64_t(guess), int64_t(posInTop),
        //  int64_t(j+1));
        printf("[G=%" PRId64 ",A=%" PRId64 ",P=%.2lf%%]", int64_t(guess), int64_t(j + 1), certainty);
        SRLock<SRCriticalSection> csl(gcsReport);
        gnCorrect++;
        gSumQuizLens += j + 1;
        gTotCertainty += certainty;
        break;
      }
    }
    if (j >= cMaxQuizLen) {
      printf("-");
      SRLock<SRCriticalSection> csl(gcsReport);
      gnWrong++;
    }
    err = pEngine->RecordQuizTarget(iQuiz, guess);
    if (!err.IsOk()) {
      fprintf(stderr, "Failed to record quiz target: %s\n", err.ToString(true).ToStd().c_str());
      return;
    }
    err = pEngine->ReleaseQuiz(iQuiz);
    if (!err.IsOk()) {
      fprintf(stderr, "Failed to release a quiz: %s\n", err.ToString(true).ToStd().c_str());
      return;
    }
  }
}

int LearnBinarySearch(const char* const initKbFp) {
  gFpProgress = fopen("progress.txt", "wt");

  PqaError err;
  IPqaEngine *pEngine;
  if (initKbFp == nullptr) {
    EngineDefinition ed;
    ed._dims._nAnswers = 5;
    ed._dims._nQuestions = 1000;
    ed._dims._nTargets = 1000;
    ed._initAmount = 0.1;
    //ed._prec._type = TPqaPrecisionType::Double;
    //pEngine = PqaGetEngineFactory().CreateCpuEngine(err, ed);
    ed._prec._type = TPqaPrecisionType::Float;
    pEngine = PqaGetEngineFactory().CreateCudaEngine(err, ed);
    if (!err.IsOk() || pEngine == nullptr) {
      fprintf(stderr, "Failed to instantiate a ProbQA engine: %s\n", err.ToString(true).ToStd().c_str());
      return int(SRExitCode::UnspecifiedError);
    }
  }
  else {
    //pEngine = PqaGetEngineFactory().LoadCpuEngine(err, initKbFp);
    pEngine = PqaGetEngineFactory().LoadCudaEngine(err, initKbFp);
    if (!err.IsOk() || pEngine == nullptr) {
      fprintf(stderr, "Failed to load a ProbQA engine: %s\n", err.ToString(true).ToStd().c_str());
      return int(SRExitCode::UnspecifiedError);
    }
  }

  gnCorrect = 0;
  gnWrong = 0;
  gSumQuizLens = 0;
  gTotCertainty = 0;
  gpcStart = GetPerfCnt();
  gPrevQAsked = pEngine->GetTotalQuestionsAsked(err);
  if (!err.IsOk()) {
    fprintf(stderr, SR_FILE_LINE "Failed to query the total number of questions asked.\n");
    return int(SRExitCode::UnspecifiedError);
  }
  gNTrainings = -1;
  std::vector<std::thread> learners;
  const int nLearners = std::thread::hardware_concurrency();
  for (int i = 0; i < nLearners; i++) {
    learners.push_back(std::thread(&LearnerThread, pEngine));
  }
  for (int i = 0; i < nLearners; i++) {
    learners[i].join();
  }
  delete pEngine;
  fclose(gFpProgress);
  return 0;
}

int __cdecl main() {
  const char* baseName = "Logs\\PqaClient";
  if (!CreateDirectoryA("Logs", nullptr)) {
    uint32_t le = GetLastError();
    if (le != ERROR_ALREADY_EXISTS) {
      baseName = "PqaClient";
    }
  }
  SRDefaultLogger::Init(SRString::MakeUnowned(baseName));

  gKbsDir = "E:\\Data\\Dev\\Engines\\ProbQA\\KBs\\";
  if (!ExistsDirectory(gKbsDir.c_str())) {
    const char* const sKBs = "KBs";
    gKbsDir = std::string(sKBs) + "\\";
    if (!CreateDirectoryA(sKBs, nullptr)) {
      uint32_t le = GetLastError();
      if (le != ERROR_ALREADY_EXISTS) {
        fprintf(stderr, "Failed to ensure that a directory for KBs exists.\n");
        return int(SRExitCode::UnspecifiedError);
      }
    }
  }

  //return LearnBinarySearch("KBs\\initial.kb"); // To load a saved KB
  return LearnBinarySearch(nullptr); // To create a KB from scratch by training
}
