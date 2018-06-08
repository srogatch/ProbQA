﻿using System;
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

    public PqaError Train(Int64 nQuestions, AnsweredQuestion[] AQs, Int64 iTarget, double amount = 1.0)
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
    private static extern UInt64 PqaEngine_GetTotalQuestionsAsked(IntPtr pEngine, ref IntPtr ppError);

    public ulong GetTotalQuestionsAsked(out PqaError err)
    {
      IntPtr nativeErr = IntPtr.Zero;
      ulong res;
      try
      {
        res = PqaEngine_GetTotalQuestionsAsked(_nativeEngine, ref nativeErr);
      }
      finally
      {
        err = PqaError.Factor(nativeErr);
      }
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
    private static extern Int64 PqaEngine_StartQuiz(IntPtr pEngine, ref IntPtr ppError);

    public Int64 StartQuiz(out PqaError err)
    {
      Int64 iQuiz;
      IntPtr nativeErr = IntPtr.Zero;
      try
      {
        iQuiz = PqaEngine_StartQuiz(_nativeEngine, ref nativeErr);
      }
      finally
      {
        err = PqaError.Factor(nativeErr);
      }
      return iQuiz;
    }

    [DllImport("PqaCore.dll", CallingConvention = CallingConvention.Cdecl)]
    private static extern Int64 PqaEngine_ResumeQuiz(IntPtr pEngine, ref IntPtr ppError, Int64 nAnswered,
      IntPtr pAQs);

    public Int64 ResumeQuiz(out PqaError err, Int64 nAnswered, AnsweredQuestion[] AQs)
    {
      GCHandle pAQs = GCHandle.Alloc(AQs, GCHandleType.Pinned);
      Int64 iQuiz;
      try
      {
        IntPtr nativeErr = IntPtr.Zero;
        try
        {
          iQuiz = PqaEngine_ResumeQuiz(_nativeEngine, ref nativeErr, nAnswered, pAQs.AddrOfPinnedObject());
        }
        finally
        {
          err = PqaError.Factor(nativeErr);
        }
      }
      finally
      {
        pAQs.Free();
      }
      return iQuiz;
    }
  }
}
