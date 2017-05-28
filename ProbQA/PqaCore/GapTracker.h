#pragma once

namespace ProbQA {

// This class is not thread-safe.
template <typename taId> class GapTracker {
  std::vector<taId> _gaps;
  std::vector<bool> _isGap;
public:
  explicit GapTracker() { }

  bool IsGap(const taId at) const { return _isGap[at]; }

  void Release(const taId at) {
    assert(!_isGap[at]);
    _isGap[at] = true;
    _gaps.push_back(at); 
  }

  bool HasGaps() const { return _gaps.size() >= 1; }

  // If there is no gap to acquire, returns length+1 meaning the client must increase his arrays.
  taId Acquire() {
    if (_gaps.size() <= 0) {
      taId answer = _isGap.size();
      _isGap.push_back(false);
      return answer;
    }
    taId answer = _gaps.back();
    _gaps.pop_back();
    _isGap[answer] = false;
    return answer;
  }

  // Grow to the specified length setting the new items as not gaps.
  void GrowTo(const taId newLength) {
    assert(newLength >= taId(_isGap.size()));
    _isGap.resize(newLength, false);
  }

  // Client access to the gaps vector, e.g. to sort it and then make compaction.
  std::vector<taId>& Gaps() { return _gaps; }

  void Compact(const taId newLength) {
    _isGap.assign(newLength, false);
    _gaps.clear();
  }
};

} // namespace ProbQA