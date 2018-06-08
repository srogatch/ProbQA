using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Text;

namespace ProbQANetCore
{
  public class PqaEngine
  {
    #region Compact-Permanent ID mapping
    [DllImport("PqaCore.dll", CallingConvention = CallingConvention.Cdecl)]
    private static extern byte PqaEngine_QuestionPermFromComp(IntPtr pEngine, Int64 count, IntPtr pIds);

    [DllImport("PqaCore.dll", CallingConvention = CallingConvention.Cdecl)]
    private static extern byte PqaEngine_QuestionCompFromPerm(IntPtr pEngine, Int64 count, IntPtr pIds);

    [DllImport("PqaCore.dll", CallingConvention = CallingConvention.Cdecl)]
    private static extern byte PqaEngine_TargetPermFromComp(IntPtr pEngine, Int64 count, IntPtr pIds);

    [DllImport("PqaCore.dll", CallingConvention = CallingConvention.Cdecl)]
    private static extern byte PqaEngine_TargetCompFromPerm(IntPtr pEngine, Int64 count, IntPtr pIds);

    public bool QuestionPermFromComp(Int64[] ids)
    {
      GCHandle pIds = GCHandle.Alloc(ids, GCHandleType.Pinned);
      try
      {
        return PqaEngine_QuestionPermFromComp(_nativeEngine, ids.LongLength, pIds.AddrOfPinnedObject()) != 0;
      }
      finally
      {
        pIds.Free();
      }
    }

    public bool QuestionCompFromPerm(Int64[] ids)
    {
      GCHandle pIds = GCHandle.Alloc(ids, GCHandleType.Pinned);
      try
      {
        return PqaEngine_QuestionCompFromPerm(_nativeEngine, ids.LongLength, pIds.AddrOfPinnedObject()) != 0;
      }
      finally
      {
        pIds.Free();
      }
    }

    public bool TargetPermFromComp(Int64[] ids)
    {
      GCHandle pIds = GCHandle.Alloc(ids, GCHandleType.Pinned);
      try
      {
        return PqaEngine_TargetPermFromComp(_nativeEngine, ids.LongLength, pIds.AddrOfPinnedObject()) != 0;
      }
      finally
      {
        pIds.Free();
      }
    }

    public bool TargetCompFromPerm(Int64[] ids)
    {
      GCHandle pIds = GCHandle.Alloc(ids, GCHandleType.Pinned);
      try
      {
        return PqaEngine_TargetCompFromPerm(_nativeEngine, ids.LongLength, pIds.AddrOfPinnedObject()) != 0;
      }
      finally
      {
        pIds.Free();
      }
    }
    #endregion

    private IntPtr _nativeEngine;

    internal IntPtr GetNativePtr()
    { return _nativeEngine; }

    internal PqaEngine(IntPtr native)
    {
      _nativeEngine = native;
    }

    [DllImport("PqaCore.dll", CallingConvention = CallingConvention.Cdecl)]
    private static extern void CiReleasePqaEngine(IntPtr pEngine);

    ~PqaEngine()
    {
      CiReleasePqaEngine(_nativeEngine);
    }

    [DllImport("PqaCore.dll", CallingConvention = CallingConvention.Cdecl)]
    private static extern IntPtr PqaEngine_Train(IntPtr pEngine, Int64 nQuestions, IntPtr pAQs, Int64 iTarget,
      double amount = 1.0);

    public PqaError Train(long nQuestions, AnsweredQuestion[] AQs, long iTarget, double amount = 1.0)
    {
      GCHandle pAQs = GCHandle.Alloc(AQs, GCHandleType.Pinned);
      try
      {
        return PqaError.Factor(PqaEngine_Train(_nativeEngine, nQuestions, pAQs.AddrOfPinnedObject(), iTarget, amount));
      }
      finally
      {
        pAQs.Free();
      }
    }

    [DllImport("PqaCore.dll", CallingConvention = CallingConvention.Cdecl)]
    private static extern UInt64 PqaEngine_GetTotalQuestionsAsked(IntPtr pEngine, out IntPtr ppError);

    public ulong GetTotalQuestionsAsked(out PqaError err)
    {
      IntPtr nativeErr = IntPtr.Zero;
      ulong res = PqaEngine_GetTotalQuestionsAsked(_nativeEngine, out nativeErr);
      err = PqaError.Factor(nativeErr);
      return res;
    }

    [DllImport("PqaCore.dll", CallingConvention = CallingConvention.Cdecl)]
    private static extern byte PqaEngine_CopyDims(IntPtr pEngine, out EngineDimensions pDims);

    public EngineDimensions CopyDims()
    {
      EngineDimensions ed;
      if(PqaEngine_CopyDims(_nativeEngine, out ed) == 0)
      {
        throw new PqaException("C++ Engine has failed to return dimensions.");
      }
      return ed;
    }

    [DllImport("PqaCore.dll", CallingConvention = CallingConvention.Cdecl)]
    private static extern Int64 PqaEngine_StartQuiz(IntPtr pEngine, out IntPtr ppError);

    //public Int64 StartQuiz(out PqaError err)
    //{

    //}
  }
}
