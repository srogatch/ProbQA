import ProbQAInterop.ProbQA as probqa
import os

MODULE_DIR = os.path.dirname(os.path.abspath(__file__))

# input('Attach the debugger and press ENTER')

probqa.SRLogger.init(os.path.join(MODULE_DIR, '../../../logs/PqaWebTest'))

# Test creation of the engine
engine, err = probqa.PqaEngineFactory.instance.create_cpu_engine(probqa.EngineDefinition(
    n_answers = 5, # Use 0 or 1 to trigger an error
    n_questions = 10,
    n_targets = 10,
))
if err:
    print(err.to_string(True))
if not engine:
    print('Got no engine')
    exit(1)

comp_ids = list(range(2, 10, 2))
print('Compact IDs:', comp_ids)
perm_ids = engine.question_perm_from_comp(comp_ids)
print('Permanent IDs:', perm_ids)

engine.train(
    [probqa.AnsweredQuestion(0, 0),
     probqa.AnsweredQuestion(1, 1),
     probqa.AnsweredQuestion(2, 2),
     probqa.AnsweredQuestion(3, 3),
     probqa.AnsweredQuestion(4, 4),
     ], 0)

i_quiz1 = engine.start_quiz()
i_quiz2 = engine.resume_quiz([probqa.AnsweredQuestion(0, 1), probqa.AnsweredQuestion(1, 2)])
print('Quizzes:', i_quiz1, i_quiz2)
print('Next questions:', engine.next_question(i_quiz1), engine.next_question(i_quiz2))
print('Active questions:', engine.get_active_question_id(i_quiz1), engine.get_active_question_id(i_quiz2))
engine.record_answer(i_quiz1, 0)
engine.record_answer(i_quiz2, 1)
print('Top targets:', engine.list_top_targets(i_quiz1, 3), engine.list_top_targets(i_quiz2, 3))
engine.record_quiz_target(i_quiz1, 3, 1.1)
engine.record_quiz_target(i_quiz2, 4, 0.9)
engine.release_quiz(i_quiz1)
engine.release_quiz(i_quiz2)
engine.save_kb(os.path.join(MODULE_DIR, '../../../Data/KBs/Привет.kb'), True)

engine, err = probqa.PqaEngineFactory.instance.load_cpu_engine(os.path.join(MODULE_DIR, '../../../Data/KBs/current.kb'))
if err:
    print(err.to_string(True))
if not engine:
    print('Loaded no engine')
    exit(1)

print('Total number of questions asked:', engine.get_total_questions_asked())

dims = engine.copy_dims()
print(dims)