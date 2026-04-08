from priority_hire_env.graders import (
    grade_easy_critical_backend,
    grade_medium_scarce_ml_specialist,
    grade_hard_multi_tradeoff,
    grade_medium_deadline_pressure,
    grade_hard_conflicting_priorities,
)

result = grade_easy_critical_backend(score=0.5, task_name='easy_critical_backend')
print('easy_critical_backend:', result)

result = grade_medium_scarce_ml_specialist(score=0.5, task_name='medium_scarce_ml_specialist')
print('medium_scarce_ml_specialist:', result)

result = grade_hard_multi_tradeoff(score=0.5, task_name='hard_multi_tradeoff')
print('hard_multi_tradeoff:', result)

result = grade_medium_deadline_pressure(score=0.5, task_name='medium_deadline_pressure')
print('medium_deadline_pressure:', result)

result = grade_hard_conflicting_priorities(score=0.5, task_name='hard_conflicting_priorities')
print('hard_conflicting_priorities:', result)