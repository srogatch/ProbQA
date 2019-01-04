from __future__ import annotations
import os
import ctypes
from enum import Enum
from typing import List, Tuple, Callable

# Initialization and C wrapper follow: please, don't use them in Python code.
# Instead, please use OOP wrapper that follows later.

# Taken from https://stackoverflow.com/questions/7586504/python-accessing-dll-using-ctypes
if os.name == 'nt':
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL('kernel32', use_last_error=True)

    def check_bool(result, func, args):
        if not result:
            raise ctypes.WinError(ctypes.get_last_error())
        return args

    kernel32.LoadLibraryExW.errcheck = check_bool
    kernel32.LoadLibraryExW.restype = wintypes.HMODULE
    kernel32.LoadLibraryExW.argtypes = (wintypes.LPCWSTR,
                                        wintypes.HANDLE,
                                        wintypes.DWORD)


# CDLL vs WinDLL: https://ammous88.wordpress.com/2014/12/31/ctypes-cdll-vs-windll/
class CDLLEx(ctypes.CDLL):
    def __init__(self, name, mode=0, handle=None, 
                 use_errno=True, use_last_error=False):
        if os.name == 'nt' and handle is None:
            handle = kernel32.LoadLibraryExW(name, None, mode)
        super(CDLLEx, self).__init__(name, mode, handle,
                                     use_errno, use_last_error)


class WinDLLEx(ctypes.WinDLL):
    def __init__(self, name, mode=0, handle=None, 
                 use_errno=False, use_last_error=True):
        if os.name == 'nt' and handle is None:
            handle = kernel32.LoadLibraryExW(name, None, mode)
        super(WinDLLEx, self).__init__(name, mode, handle,
                                       use_errno, use_last_error)

DONT_RESOLVE_DLL_REFERENCES         = 0x00000001
LOAD_LIBRARY_AS_DATAFILE            = 0x00000002
LOAD_WITH_ALTERED_SEARCH_PATH       = 0x00000008
LOAD_IGNORE_CODE_AUTHZ_LEVEL        = 0x00000010  # NT 6.1
LOAD_LIBRARY_AS_IMAGE_RESOURCE      = 0x00000020  # NT 6.0
LOAD_LIBRARY_AS_DATAFILE_EXCLUSIVE  = 0x00000040  # NT 6.0

# These cannot be combined with LOAD_WITH_ALTERED_SEARCH_PATH.
# Install update KB2533623 for NT 6.0 & 6.1.
LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR    = 0x00000100
LOAD_LIBRARY_SEARCH_APPLICATION_DIR = 0x00000200
LOAD_LIBRARY_SEARCH_USER_DIRS       = 0x00000400
LOAD_LIBRARY_SEARCH_SYSTEM32        = 0x00000800
LOAD_LIBRARY_SEARCH_DEFAULT_DIRS    = 0x00001000

# TODO: on Linux that would be a .so, but the engine doesn't yet support Linux
pqa_engine_path = os.path.realpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'DLLs/PqaCore.dll'))
print('Initializing engine from:', pqa_engine_path)
pqa_core = CDLLEx(pqa_engine_path, LOAD_WITH_ALTERED_SEARCH_PATH)


# See https://github.com/srogatch/ProbQA/blob/master/ProbQA/PqaCore/Interface/PqaCInterop.h
class CiEngineDefinition(ctypes.Structure):
    _pack_ = 8
    _fields_ = [
        ('nAnswers', ctypes.c_int64),
        ('nQuestions', ctypes.c_int64),
        ('nTargets', ctypes.c_int64),
        ('precType', ctypes.c_uint8),
        ('precExponent', ctypes.c_uint16),
        ('precMantissa', ctypes.c_uint32),
        ('initAmount', ctypes.c_double),
        ('memPoolMaxBytes', ctypes.c_uint64),
    ]


class CiAnsweredQuestion(ctypes.Structure):
    _pack_ = 8
    _fields_ = [
        ('iQuestion', ctypes.c_int64),
        ('iAnswer', ctypes.c_int64),
    ]


class CiEngineDimensions(ctypes.Structure):
    _pack_ = 8
    _fields_ = [
        ('nAnswers', ctypes.c_int64),
        ('nQuestions', ctypes.c_int64),
        ('nTargets', ctypes.c_int64),
    ]


class CiRatedTarget(ctypes.Structure):
    _pack_ = 8
    _fields_ = [
        ('iTarget', ctypes.c_int64),
        ('prob', ctypes.c_double),
    ]

# PQACORE_API void CiDebugBreak(void);
pqa_core.CiDebugBreak.restype = None
pqa_core.CiDebugBreak.argtypes = None

# PQACORE_API uint8_t Logger_Init(void **ppStrErr, const char* baseName);
pqa_core.Logger_Init.restype = ctypes.c_uint8
pqa_core.Logger_Init.argtypes = (ctypes.POINTER(ctypes.c_char_p), ctypes.c_char_p)

# PQACORE_API void CiReleaseString(void *pvString);
pqa_core.CiReleaseString.restype = None
pqa_core.CiReleaseString.argtypes = (ctypes.c_char_p,)

# PQACORE_API void* CiGetPqaEngineFactory();
pqa_core.CiGetPqaEngineFactory.restype = ctypes.c_void_p
pqa_core.CiGetPqaEngineFactory.argtypes = None

# PQACORE_API void* PqaEngineFactory_CreateCpuEngine(void* pvFactory, void **ppError,
#     const CiEngineDefinition *pEngDef);
pqa_core.PqaEngineFactory_CreateCpuEngine.restype = ctypes.c_void_p
pqa_core.PqaEngineFactory_CreateCpuEngine.argtypes = (ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p),
    ctypes.POINTER(CiEngineDefinition))

# PQACORE_API void* PqaEngineFactory_LoadCpuEngine(void *pvFactory, void **ppError, const char* filePath,
#   uint64_t memPoolMaxBytes);
pqa_core.PqaEngineFactory_LoadCpuEngine.restype = ctypes.c_void_p  # C Engine
pqa_core.PqaEngineFactory_LoadCpuEngine.argtypes = (ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p),
    ctypes.c_char_p, ctypes.c_uint64)

# PQACORE_API void CiReleasePqaError(void *pvErr);
pqa_core.CiReleasePqaError.restype = None
pqa_core.CiReleasePqaError.argtypes = (ctypes.c_void_p,)

# PQACORE_API void* PqaError_ToString(void *pvError, const uint8_t withParams);
# restype has to be c_void_p to workaround this problem:
# https://stackoverflow.com/questions/53999442/inconsistent-c-char-p-behavior-between-returning-vs-pointer-assignment
pqa_core.PqaError_ToString.restype = ctypes.c_void_p
pqa_core.PqaError_ToString.argtypes = (ctypes.c_void_p, ctypes.c_bool)

# PQACORE_API void CiReleasePqaEngine(void *pvEngine);
pqa_core.CiReleasePqaEngine.restype = None
pqa_core.CiReleasePqaEngine.argtypes = (ctypes.c_void_p,)

# PQACORE_API void* PqaEngine_Train(void *pvEngine, int64_t nQuestions, const CiAnsweredQuestion* const pAQs,
#   const int64_t iTarget, const double amount = 1.0);
pqa_core.PqaEngine_Train.restype = ctypes.c_void_p # The error
pqa_core.PqaEngine_Train.argtypes = (ctypes.c_void_p, ctypes.c_int64, ctypes.POINTER(CiAnsweredQuestion),
    ctypes.c_int64, ctypes.c_double)

# PQACORE_API uint8_t PqaEngine_QuestionPermFromComp(void *pvEngine, const int64_t count, int64_t *pIds);
pqa_core.PqaEngine_QuestionPermFromComp.restype = ctypes.c_bool
pqa_core.PqaEngine_QuestionPermFromComp.argtypes = (ctypes.c_void_p, ctypes.c_int64, ctypes.POINTER(ctypes.c_int64))

# PQACORE_API uint8_t PqaEngine_QuestionCompFromPerm(void *pvEngine, const int64_t count, int64_t *pIds);
pqa_core.PqaEngine_QuestionCompFromPerm.restype = ctypes.c_bool
pqa_core.PqaEngine_QuestionCompFromPerm.argtypes = (ctypes.c_void_p, ctypes.c_int64, ctypes.POINTER(ctypes.c_int64))

# PQACORE_API uint8_t PqaEngine_TargetPermFromComp(void *pvEngine, const int64_t count, int64_t *pIds);
pqa_core.PqaEngine_TargetPermFromComp.restype = ctypes.c_bool
pqa_core.PqaEngine_TargetPermFromComp.argtypes = (ctypes.c_void_p, ctypes.c_int64, ctypes.POINTER(ctypes.c_int64))

# PQACORE_API uint8_t PqaEngine_TargetCompFromPerm(void *pvEngine, const int64_t count, int64_t *pIds);
pqa_core.PqaEngine_TargetCompFromPerm.restype = ctypes.c_bool
pqa_core.PqaEngine_TargetCompFromPerm.argtypes = (ctypes.c_void_p, ctypes.c_int64, ctypes.POINTER(ctypes.c_int64))

# PQACORE_API uint64_t PqaEngine_GetTotalQuestionsAsked(void *pvEngine, void **ppError);
pqa_core.PqaEngine_GetTotalQuestionsAsked.restype = ctypes.c_uint64
pqa_core.PqaEngine_GetTotalQuestionsAsked.argtypes = (ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p))

# PQACORE_API uint8_t PqaEngine_CopyDims(void *pvEngine, CiEngineDimensions *pDims);
pqa_core.PqaEngine_CopyDims.restype = ctypes.c_bool
pqa_core.PqaEngine_CopyDims.argtypes = (ctypes.c_void_p, ctypes.POINTER(CiEngineDimensions))

# PQACORE_API int64_t PqaEngine_StartQuiz(void *pvEngine, void **ppError);
pqa_core.PqaEngine_StartQuiz.restype = ctypes.c_int64
pqa_core.PqaEngine_StartQuiz.argtypes = (ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p))

# PQACORE_API int64_t PqaEngine_ResumeQuiz(void *pvEngine, void **ppError, const int64_t nAnswered,
#   const CiAnsweredQuestion* const pAQs);
pqa_core.PqaEngine_ResumeQuiz.restype = ctypes.c_int64
pqa_core.PqaEngine_ResumeQuiz.argtypes = (ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p), ctypes.c_int64,
    ctypes.POINTER(CiAnsweredQuestion))

# PQACORE_API int64_t PqaEngine_NextQuestion(void *pvEngine, void **ppError, const int64_t iQuiz);
pqa_core.PqaEngine_NextQuestion.restype = ctypes.c_int64
pqa_core.PqaEngine_NextQuestion.argtypes = (ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p), ctypes.c_int64)

# PQACORE_API void* PqaEngine_RecordAnswer(void *pvEngine, const int64_t iQuiz, const int64_t iAnswer);
pqa_core.PqaEngine_RecordAnswer.restype = ctypes.c_void_p # The error
pqa_core.PqaEngine_RecordAnswer.argtypes = (ctypes.c_void_p, ctypes.c_int64, ctypes.c_int64)

# PQACORE_API int64_t PqaEngine_GetActiveQuestionId(void *pvEngine, void **ppError, const int64_t iQuiz);
pqa_core.PqaEngine_GetActiveQuestionId.restype = ctypes.c_int64
pqa_core.PqaEngine_GetActiveQuestionId.argtypes = (ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p), ctypes.c_int64)

# PQACORE_API int64_t PqaEngine_ListTopTargets(void *pvEngine, void **ppError, const int64_t iQuiz,
#   const int64_t maxCount, CiRatedTarget *pDest);
pqa_core.PqaEngine_ListTopTargets.restype = ctypes.c_int64
pqa_core.PqaEngine_ListTopTargets.argtypes = (ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p), ctypes.c_int64,
    ctypes.c_int64, ctypes.POINTER(CiRatedTarget))

# PQACORE_API void* PqaEngine_RecordQuizTarget(void *pvEngine, const int64_t iQuiz, const int64_t iTarget,
#   const double amount = 1.0);
pqa_core.PqaEngine_RecordQuizTarget.restype = ctypes.c_void_p
pqa_core.PqaEngine_RecordQuizTarget.argtypes = (ctypes.c_void_p, ctypes.c_int64, ctypes.c_int64, ctypes.c_double)

# PQACORE_API void* PqaEngine_ReleaseQuiz(void *pvEngine, const int64_t iQuiz);
pqa_core.PqaEngine_ReleaseQuiz.restype = ctypes.c_void_p
pqa_core.PqaEngine_ReleaseQuiz.argtypes = (ctypes.c_void_p, ctypes.c_int64)

# PQACORE_API void* PqaEngine_SaveKB(void *pvEngine, const char* const filePath, const uint8_t bDoubleBuffer);
pqa_core.PqaEngine_SaveKB.restype = ctypes.c_void_p
pqa_core.PqaEngine_SaveKB.argtypes = (ctypes.c_void_p, ctypes.c_char_p, ctypes.c_bool)


# OOP wrapper follows - please, use these in Python code
class PqaException(Exception):
    pass


class Utils:
    @staticmethod
    def handle_native_string(c_str : ctypes.c_char_p) -> str:
        ans = c_str.value.decode('mbcs') # Windows-only
        pqa_core.CiReleaseString(c_str)
        return ans

    @staticmethod
    def str_to_c_char_p(s : str) -> ctypes.c_char_p:
        return ctypes.c_char_p(s.encode('mbcs')) # Windows-only


class SRLogger:
    @staticmethod
    def init(base_name : str) -> bool:
        str_err = ctypes.c_char_p()
        ans = pqa_core.Logger_Init(ctypes.byref(str_err), Utils.str_to_c_char_p(base_name))
        if str_err:
            raise PqaException(Utils.handle_native_string(str_err))
        return ans


class PrecisionType(Enum):
    NONE = 0
    FLOAT = 1
    FLOAT_PAIR = 2
    DOUBLE = 3
    DOUBLE_PAIR = 4
    ARBITRARY = 5


class AnsweredQuestion:
    def __init__(self, i_question, i_answer):
        self.i_question = i_question
        self.i_answer = i_answer

    def __repr__(self):
        return '[i_question=%d, i_answer=%d]' % (self.i_question, self.i_answer)

class RatedTarget:
    def __init__(self, i_target: int, prob: float):
        self.i_target = i_target
        self.prob = prob

    def __repr__(self):
        return '[i_target=%d, P=%f%%]' % (self.i_target, self.prob * 100)


class EngineDefinition:
    DEFAULT_MEM_POOL_MAX_BYTES = 512 * 1024 * 1024
    def __init__(self, n_answers : int, n_questions : int, n_targets : int, init_amount = 1.0,
                 prec_type = PrecisionType.DOUBLE, prec_exponent = 11, prec_mantissa = 53,
                 mem_pool_max_bytes = DEFAULT_MEM_POOL_MAX_BYTES):
        self.n_answers = n_answers
        self.n_questions = n_questions
        self.n_targets = n_targets
        self.init_amount = init_amount
        self.prec_type = prec_type
        self.prec_exponent = prec_exponent
        self.prec_mantissa = prec_mantissa
        self.mem_pool_max_bytes = mem_pool_max_bytes


class EngineDimensions:
    def __init__(self, n_answers:int, n_questions:int, n_targets:int):
        self.n_answers = n_answers
        self.n_questions = n_questions
        self.n_targets = n_targets

    def __repr__(self):
        return '[n_answers=%d, n_questions=%d, n_targets=%d]' % (self.n_answers, self.n_questions, self.n_targets)


class PqaError:
    @staticmethod
    def factor(c_err: ctypes.c_void_p) -> PqaError:
        if (c_err.value is None) or (c_err.value == 0):
            return None
        return PqaError(c_err)
    
    def __init__(self, c_err : ctypes.c_void_p):
        self.c_err = c_err
    
    def __del__(self):
        pqa_core.CiReleasePqaError(self.c_err)
        
    def __repr__(self):
        return self.to_string(True)
        
    def to_string(self, with_params : bool) -> str:
        if (self.c_err.value is None) or (self.c_err.value == 0):
            return 'Success'
        void_ptr = ctypes.c_void_p()
        void_ptr.value = pqa_core.PqaError_ToString(self.c_err, ctypes.c_bool(with_params))
        return Utils.handle_native_string(ctypes.cast(void_ptr, ctypes.c_char_p))
    

class PqaEngine:
    def __init__(self, c_engine : ctypes.c_void_p):
        self.c_engine = c_engine
    
    def __del__(self):
        pqa_core.CiReleasePqaEngine(self.c_engine)

    def __call_id_mapping(self,
            c_func : Callable[[ctypes.c_void_p, ctypes.c_int64, ctypes.POINTER(ctypes.c_int64)], ctypes.c_bool],
            ids: List[int]) -> List[int]:
        # https://stackoverflow.com/questions/37197631/ctypes-reading-modified-array
        n_ids = len(ids)
        array_type = ctypes.c_int64 * n_ids
        c_ids = array_type(*ids)
        b_ok = c_func(self.c_engine, ctypes.c_int64(n_ids), c_ids)
        if not b_ok:
            raise PqaException("Failed permanent<->compact ID conversion.")
        return list(c_ids)

    @staticmethod
    def to_c_answered_questions(answered_questions: List[AnsweredQuestion]) -> Tuple[ctypes.Array, int]:
        n_questions = len(answered_questions)
        array_type = CiAnsweredQuestion * n_questions
        c_aqs = array_type()
        for i in range(n_questions):
            c_aqs[i].iQuestion = answered_questions[i].i_question
            c_aqs[i].iAnswer = answered_questions[i].i_answer
        return c_aqs, n_questions

    # Permanent<->compact ID mappings follow. They raise on error.
    def question_perm_from_comp(self, ids: List[int]) -> List[int]:
        return self.__call_id_mapping(pqa_core.PqaEngine_QuestionPermFromComp, ids)

    def question_comp_from_perm(self, ids: List[int]) -> List[int]:
        return self.__call_id_mapping(pqa_core.PqaEngine_QuestionCompFromPerm, ids)

    def target_perm_from_comp(self, ids: List[int]) -> List[int]:
        return self.__call_id_mapping(pqa_core.PqaEngine_TargetPermFromComp, ids)

    def target_comp_from_perm(self, ids: List[int]) -> List[int]:
        return self.__call_id_mapping(pqa_core.PqaEngine_TargetCompFromPerm, ids)

    def train(self, answered_questions: List[AnsweredQuestion], i_target : int,
              amount: float = 1.0, throw: bool = True) -> PqaError:
        c_aqs, n_questions = PqaEngine.to_c_answered_questions(answered_questions)
        c_err = ctypes.c_void_p()
        c_err.value = pqa_core.PqaEngine_Train(
            self.c_engine, ctypes.c_int64(n_questions), c_aqs,
            ctypes.c_int64(i_target), ctypes.c_double(amount)
        )
        err = PqaError.factor(c_err)
        if err:
            if throw:
                raise PqaException('Failed to train() the engine: ' + str(err))
        return err

    def get_total_questions_asked(self) -> int:
        c_err = ctypes.c_void_p()
        ans = pqa_core.PqaEngine_GetTotalQuestionsAsked(self.c_engine, ctypes.byref(c_err))
        err = PqaError.factor(c_err)
        if err:
            raise PqaException('Failed to get the total number of questions asked: [%d, %s]' %
                (ans, str(err)))
        return ans

    def copy_dims(self) -> EngineDimensions:
        c_dims = CiEngineDimensions()
        if not pqa_core.PqaEngine_CopyDims(self.c_engine, ctypes.byref(c_dims)):
            raise PqaException('Failed to copy engine dimensions.')
        return EngineDimensions(c_dims.nAnswers, c_dims.nQuestions, c_dims.nTargets)

    def start_quiz(self) -> int:
        c_err = ctypes.c_void_p()
        try:
            i_quiz = pqa_core.PqaEngine_StartQuiz(self.c_engine, ctypes.byref(c_err))
        finally:
            err = PqaError.factor(c_err)
        if err:
            raise PqaException('Failed to start a quiz: [%d, %s]' % (i_quiz, str(err)))
        return i_quiz

    def resume_quiz(self, answered_questions: List[AnsweredQuestion]) -> int:
        c_aqs, n_questions = PqaEngine.to_c_answered_questions(answered_questions)
        c_err = ctypes.c_void_p()
        try:
            i_quiz = pqa_core.PqaEngine_ResumeQuiz(self.c_engine, ctypes.byref(c_err), ctypes.c_int64(n_questions),
                c_aqs)
        finally:
            err = PqaError.factor(c_err)
        if err:
            raise PqaException('Failed to resume a quiz: [%d, %s]' % (i_quiz, str(err)))
        return i_quiz

    def next_question(self, i_quiz: int) -> int:
        c_err = ctypes.c_void_p()
        try:
            i_question = pqa_core.PqaEngine_NextQuestion(self.c_engine, ctypes.byref(c_err), ctypes.c_int64(i_quiz))
        finally:
            err = PqaError.factor(c_err)
        if err:
            raise PqaException('Failed to compute the next question: [%d, %s]' % (i_question, str(err)))
        return i_question

    def record_answer(self, i_quiz: int, i_answer: int, throw: bool = True) -> PqaError:
        c_err = ctypes.c_void_p()
        c_err.value = pqa_core.PqaEngine_RecordAnswer(self.c_engine, ctypes.c_int64(i_quiz), ctypes.c_int64(i_answer))
        err = PqaError.factor(c_err)
        if err:
            if throw:
                raise PqaException('Failed to record_answer(): ' + str(err))
        return err

    def get_active_question_id(self, i_quiz: int) -> int:
        c_err = ctypes.c_void_p()
        try:
            i_question = pqa_core.PqaEngine_GetActiveQuestionId(self.c_engine, ctypes.byref(c_err),
                ctypes.c_int64(i_quiz))
        finally:
            err = PqaError.factor(c_err)
        if err:
            raise PqaException('Failed to get_active_question_id(): [%d, %s]' % (i_question, str(err)))
        return i_question

    def list_top_targets(self, i_quiz: int, max_count: int) -> List[RatedTarget]:
        c_err = ctypes.c_void_p()
        try:
            array_type = CiRatedTarget * max_count
            c_rated_targets = array_type()
            n_top = pqa_core.PqaEngine_ListTopTargets(self.c_engine, ctypes.byref(c_err), ctypes.c_int64(i_quiz),
                ctypes.c_int64(max_count), c_rated_targets)
        finally:
            err = PqaError.factor(c_err)
        if err:
            raise PqaException('Failed to list_top_targets(): [%d, %s]' % (n_top, str(err)))
        ans = []
        for i in range(n_top):
            ans.append(RatedTarget(c_rated_targets[i].iTarget, c_rated_targets[i].prob))
        return ans

    def record_quiz_target(self, i_quiz: int, i_target: int, amount: float = 1.0, throw: bool = True) -> PqaError:
        c_err = ctypes.c_void_p()
        c_err.value = pqa_core.PqaEngine_RecordQuizTarget(self.c_engine, ctypes.c_int64(i_quiz),
            ctypes.c_int64(i_target), ctypes.c_double(amount))
        err = PqaError.factor(c_err)
        if err:
            if throw:
                raise PqaException('Failed to record_quiz_target(): ' + str(err))
        return err

    def release_quiz(self, i_quiz: int, throw: bool = True) -> PqaError:
        c_err = ctypes.c_void_p()
        c_err.value = pqa_core.PqaEngine_ReleaseQuiz(self.c_engine, ctypes.c_int64(i_quiz))
        err = PqaError.factor(c_err)
        if err:
            if throw:
                raise PqaException('Failed to release_quiz(): ' + str(err))
        return err

    def save_kb(self, file_path: str, b_double_buffer: bool, throw: bool = True) -> PqaError:
        c_err = ctypes.c_void_p()
        c_err.value = pqa_core.PqaEngine_SaveKB(self.c_engine, Utils.str_to_c_char_p(file_path),
            ctypes.c_bool(b_double_buffer))
        err = PqaError.factor(c_err)
        if err:
            if throw:
                raise PqaException('Failed to save_kb(): ' + str(err))
        return err


class PqaEngineFactory:
    instance = None
    
    def __init__(self):
        self.c_factory = pqa_core.CiGetPqaEngineFactory()

    def create_cpu_engine(self, eng_def : EngineDefinition) -> Tuple[PqaEngine, PqaError]:
        c_err = ctypes.c_void_p()
        c_eng_def = CiEngineDefinition()
        c_eng_def.nAnswers = eng_def.n_answers
        c_eng_def.nQuestions = eng_def.n_questions
        c_eng_def.nTargets = eng_def.n_targets
        c_eng_def.precType = eng_def.prec_type.value
        c_eng_def.precExponent = eng_def.prec_exponent
        c_eng_def.precMantissa = eng_def.prec_mantissa
        c_eng_def.initAmount = eng_def.init_amount
        c_eng_def.mem_pool_max_bytes = eng_def.mem_pool_max_bytes
        c_engine = ctypes.c_void_p()
        try:
            # If an exception is thrown from C++ code, then we can safely assume no engine is returned
            c_engine.value = pqa_core.PqaEngineFactory_CreateCpuEngine(self.c_factory, ctypes.byref(c_err),
                ctypes.byref(c_eng_def))
        finally:
            err = PqaError.factor(c_err)
        if (c_engine.value is None) or (c_engine.value == 0):
            raise PqaException('Couldn\'t create a CPU Engine due to a native error: ' + str(err))
        return (PqaEngine(c_engine), err)

    def load_cpu_engine(self, file_path:str, mem_pool_max_bytes:int = EngineDefinition.DEFAULT_MEM_POOL_MAX_BYTES
                        ) -> Tuple[PqaEngine, PqaError]:
        c_err = ctypes.c_void_p()
        c_engine = ctypes.c_void_p()
        try:
            c_engine.value = pqa_core.PqaEngineFactory_LoadCpuEngine(self.c_factory, ctypes.byref(c_err),
                Utils.str_to_c_char_p(file_path), ctypes.c_uint64(mem_pool_max_bytes))
        finally:
            err = PqaError.factor(c_err)
        if (c_engine.value is None) or (c_engine.value == 0):
            raise PqaException('Couldn\'t load a CPU Engine due to a native error: ' + str(err))
        return (PqaEngine(c_engine), err)

PqaEngineFactory.instance = PqaEngineFactory()


def debug_break():
    pqa_core.CiDebugBreak()
