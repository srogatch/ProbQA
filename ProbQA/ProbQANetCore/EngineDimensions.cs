using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Text;

namespace ProbQANetCore
{
  [StructLayout(LayoutKind.Sequential, Pack = 8)]
  public struct EngineDimensions
  {
    Int64 _nAnswers;
    Int64 _nQuestions;
    Int64 _nTargets;
  }
}
