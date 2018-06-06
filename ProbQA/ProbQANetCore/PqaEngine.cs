using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Text;

namespace ProbQANetCore
{
  public class PqaEngine
  {
    [DllImport("PqaCore.dll", CallingConvention = CallingConvention.Cdecl)]
    private static extern void CiReleasePqaEngine(IntPtr pEngine);

    private IntPtr _native;

    internal PqaEngine(IntPtr native)
    {
      _native = native;
    }

    ~PqaEngine()
    {
      CiReleasePqaEngine(_native);
    }
  }
}
