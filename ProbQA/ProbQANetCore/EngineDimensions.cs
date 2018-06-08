using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Text;

namespace ProbQANetCore
{
  [StructLayout(LayoutKind.Sequential, Pack = 8)]
  public struct EngineDimensions
  {
    public Int64 _nAnswers;
    public Int64 _nQuestions;
    public Int64 _nTargets;
  }
}
