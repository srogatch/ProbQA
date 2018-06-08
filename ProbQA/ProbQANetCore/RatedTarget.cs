using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Text;

namespace ProbQANetCore
{
  [StructLayout(LayoutKind.Sequential, Pack = 8)]
  public struct RatedTarget
  {
    Int64 _iTarget;
    double _prob; // probability that this target is what the user needs
  }
}
