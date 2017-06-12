#include "stdafx.h"
#include "../PqaCore/CETask.h"
#include "../PqaCore/DoubleNumber.h"
#include "../PqaCore/CpuEngine.h"

using namespace SRPlat;

namespace ProbQA {

#define CETLOG(severityVar) SRLogStream(ISRLogger::Severity::severityVar, _pCe->GetLogger())

template<typename taNumber> CETask<taNumber>::CETask(CpuEngine<taNumber> *pCe, const TPqaId nToDo)
  : _pCe(pCe), _nToDo(nToDo), _bCancel(false), _pAep(new AggregateErrorParams())
{ }

template<typename taNumber> void CETask<taNumber>::OnSubtaskComplete(CESubtask<taNumber> *pSubtask) {
  _pCe->ReleaseSubtask(pSubtask);
  const TPqaId nOld = _nToDo.fetch_sub(1, std::memory_order_release);
  if (nOld > 1) return;
  if (nOld <= 0) { // sanity check
    CETLOG(Critical) << "CESubtask to do counter in CETask has dropped to " << (nOld-1);
  }
  _isComplete.WakeAll();
}

template class CETask<DoubleNumber>;

} // namespace ProbQA