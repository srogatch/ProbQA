#pragma once

#include "../PqaCore/CETask.h"
#include "../PqaCore/CESubtask.h"

namespace ProbQA {

template<typename taNumber> class CESubtaskCompleter {
  CESubtask<taNumber> *_pSubtask;
public:
  void Set(CESubtask<taNumber> *pSubtask) { _pSubtask = pSubtask; }
  CESubtask<taNumber>* Get() const { return _pSubtask; }
  CESubtask<taNumber>* Detach() {
    CESubtask<taNumber> *answer = _pSubtask;
    _pSubtask = nullptr;
    return answer;
  }
  ~CESubtaskCompleter() {
    if (_pSubtask != nullptr) {
      _pSubtask->_pTask->OnSubtaskComplete(_pSubtask);
    }
  }
};

} // namespace ProbQA