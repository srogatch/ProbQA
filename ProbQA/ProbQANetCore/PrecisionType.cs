using System;
using System.Collections.Generic;
using System.Text;

namespace ProbQANetCore
{
  public enum PrecisionType : byte
  {
    None = 0,
    Float = 1,
    FloatPair = 2, // May be more efficient than `double` on GPUs
    Double = 3, // Currently only this is implemented
    DoublePair = 4,
    Arbitrary = 5
  }
}
