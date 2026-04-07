from __future__ import annotations

from typing import Dict, List, TypedDict


class CandidateSeed(TypedDict):
    candidate_id: str
    role: str
    priority: int
    urgency: int
    fit_score: float
    deadline: int
    available_slots: List[str]
    required_specialization: str


class InterviewerSeed(TypedDict):
    interviewer_id: str
    specialization: str
    available_slots: List[str]
    max_capacity: int


class TaskDefinition(TypedDict):
    name: str
    difficulty: str
    description: str
    max_steps: int
    candidates: List[CandidateSeed]
    interviewers: List[InterviewerSeed]


TASKS: Dict[str, TaskDefinition] = {
    "easy_critical_backend": {
        "name": "easy_critical_backend",
        "difficulty": "easy",
        "description": "Enough capacity exists, but a single backend candidate is clearly the most urgent.",
        "max_steps": 8,
        "candidates": [
            {
                "candidate_id": "cand_backend_1",
                "role": "Senior Backend Engineer",
                "priority": 5,
                "urgency": 5,
                "fit_score": 0.93,
                "deadline": 2,
                "available_slots": ["s1", "s2"],
                "required_specialization": "backend",
            },
            {
                "candidate_id": "cand_frontend_1",
                "role": "Frontend Engineer",
                "priority": 2,
                "urgency": 2,
                "fit_score": 0.74,
                "deadline": 4,
                "available_slots": ["s2", "s3"],
                "required_specialization": "frontend",
            },
            {
                "candidate_id": "cand_data_1",
                "role": "Data Analyst",
                "priority": 3,
                "urgency": 2,
                "fit_score": 0.81,
                "deadline": 3,
                "available_slots": ["s1", "s3"],
                "required_specialization": "data",
            },
        ],
        "interviewers": [
            {
                "interviewer_id": "int_backend_a",
                "specialization": "backend",
                "available_slots": ["s1", "s2"],
                "max_capacity": 2,
            },
            {
                "interviewer_id": "int_frontend_a",
                "specialization": "frontend",
                "available_slots": ["s2", "s3"],
                "max_capacity": 1,
            },
            {
                "interviewer_id": "int_data_a",
                "specialization": "data",
                "available_slots": ["s1", "s3"],
                "max_capacity": 1,
            },
        ],
    },
    "medium_scarce_ml_specialist": {
        "name": "medium_scarce_ml_specialist",
        "difficulty": "medium",
        "description": "A scarce ML specialist must be preserved for the urgent candidate rather than a lower-priority high-fit profile.",
        "max_steps": 9,
        "candidates": [
            {
                "candidate_id": "cand_ml_urgent",
                "role": "ML Platform Engineer",
                "priority": 5,
                "urgency": 4,
                "fit_score": 0.82,
                "deadline": 1,
                "available_slots": ["s1"],
                "required_specialization": "ml",
            },
            {
                "candidate_id": "cand_ml_nice",
                "role": "ML Research Engineer",
                "priority": 2,
                "urgency": 2,
                "fit_score": 0.94,
                "deadline": 3,
                "available_slots": ["s1", "s2"],
                "required_specialization": "ml",
            },
            {
                "candidate_id": "cand_general_backend",
                "role": "Backend Engineer",
                "priority": 4,
                "urgency": 3,
                "fit_score": 0.86,
                "deadline": 2,
                "available_slots": ["s2", "s3"],
                "required_specialization": "backend",
            },
        ],
        "interviewers": [
            {
                "interviewer_id": "int_ml_only",
                "specialization": "ml",
                "available_slots": ["s1"],
                "max_capacity": 1,
            },
            {
                "interviewer_id": "int_backend_b",
                "specialization": "backend",
                "available_slots": ["s2", "s3"],
                "max_capacity": 2,
            },
        ],
    },
    "hard_multi_tradeoff": {
        "name": "hard_multi_tradeoff",
        "difficulty": "hard",
        "description": "Competing deadlines, scarce security slots, and conflicting fit-vs-urgency decisions require careful scheduling.",
        "max_steps": 12,
        "candidates": [
            {
                "candidate_id": "cand_sec_critical",
                "role": "Security Architect",
                "priority": 5,
                "urgency": 5,
                "fit_score": 0.79,
                "deadline": 1,
                "available_slots": ["s1", "s2"],
                "required_specialization": "security",
            },
            {
                "candidate_id": "cand_sec_fit",
                "role": "Security Engineer",
                "priority": 3,
                "urgency": 2,
                "fit_score": 0.97,
                "deadline": 3,
                "available_slots": ["s2"],
                "required_specialization": "security",
            },
            {
                "candidate_id": "cand_backend_urgent",
                "role": "Staff Backend Engineer",
                "priority": 4,
                "urgency": 5,
                "fit_score": 0.88,
                "deadline": 1,
                "available_slots": ["s1", "s3"],
                "required_specialization": "backend",
            },
            {
                "candidate_id": "cand_frontend_fit",
                "role": "Senior Frontend Engineer",
                "priority": 2,
                "urgency": 2,
                "fit_score": 0.95,
                "deadline": 4,
                "available_slots": ["s3", "s4"],
                "required_specialization": "frontend",
            },
            {
                "candidate_id": "cand_data_urgent",
                "role": "Analytics Engineer",
                "priority": 4,
                "urgency": 4,
                "fit_score": 0.84,
                "deadline": 2,
                "available_slots": ["s2", "s4"],
                "required_specialization": "data",
            },
            {
                "candidate_id": "cand_backend_low",
                "role": "Backend Engineer",
                "priority": 1,
                "urgency": 1,
                "fit_score": 0.77,
                "deadline": 4,
                "available_slots": ["s1", "s4"],
                "required_specialization": "backend",
            },
        ],
        "interviewers": [
            {
                "interviewer_id": "int_security_only",
                "specialization": "security",
                "available_slots": ["s1", "s2"],
                "max_capacity": 1,
            },
            {
                "interviewer_id": "int_backend_c",
                "specialization": "backend",
                "available_slots": ["s1", "s3", "s4"],
                "max_capacity": 2,
            },
            {
                "interviewer_id": "int_frontend_b",
                "specialization": "frontend",
                "available_slots": ["s3", "s4"],
                "max_capacity": 1,
            },
            {
                "interviewer_id": "int_data_b",
                "specialization": "data",
                "available_slots": ["s2", "s4"],
                "max_capacity": 1,
            },
        ],
    },
"medium_deadline_pressure": {
        "name": "medium_deadline_pressure",
        "difficulty": "medium",
        "description": "All candidates have tight deadlines; urgency must drive scheduling order.",
        "max_steps": 10,
        "candidates": [
            {
                "candidate_id": "cand_be_urgent1",
                "role": "Backend Engineer",
                "priority": 5,
                "urgency": 5,
                "fit_score": 0.88,
                "deadline": 1,
                "available_slots": ["s1", "s2"],
                "required_specialization": "backend",
            },
            {
                "candidate_id": "cand_fe_urgent1",
                "role": "Frontend Engineer",
                "priority": 4,
                "urgency": 4,
                "fit_score": 0.82,
                "deadline": 1,
                "available_slots": ["s1", "s3"],
                "required_specialization": "frontend",
            },
            {
                "candidate_id": "cand_ml_urgent1",
                "role": "ML Engineer",
                "priority": 3,
                "urgency": 4,
                "fit_score": 0.79,
                "deadline": 2,
                "available_slots": ["s2", "s3"],
                "required_specialization": "ml",
            },
            {
                "candidate_id": "cand_data_urgent1",
                "role": "Data Engineer",
                "priority": 3,
                "urgency": 3,
                "fit_score": 0.75,
                "deadline": 2,
                "available_slots": ["s1", "s2"],
                "required_specialization": "data",
            },
        ],
        "interviewers": [
            {
                "interviewer_id": "int_backend_d",
                "specialization": "backend",
                "available_slots": ["s1", "s2"],
                "max_capacity": 1,
            },
            {
                "interviewer_id": "int_frontend_c",
                "specialization": "frontend",
                "available_slots": ["s1", "s3"],
                "max_capacity": 1,
            },
            {
                "interviewer_id": "int_ml_b",
                "specialization": "ml",
                "available_slots": ["s2", "s3"],
                "max_capacity": 1,
            },
            {
                "interviewer_id": "int_data_c",
                "specialization": "data",
                "available_slots": ["s1", "s2"],
                "max_capacity": 1,
            },
        ],
    },
    "hard_conflicting_priorities": {
        "name": "hard_conflicting_priorities",
        "difficulty": "hard",
        "description": "Multiple high-priority candidates compete for a single scarce interviewer; trade-offs are unavoidable.",
        "max_steps": 14,
        "candidates": [
            {
                "candidate_id": "cand_sec_high1",
                "role": "Security Engineer",
                "priority": 5,
                "urgency": 5,
                "fit_score": 0.91,
                "deadline": 1,
                "available_slots": ["s1"],
                "required_specialization": "security",
            },
            {
                "candidate_id": "cand_sec_high2",
                "role": "Security Architect",
                "priority": 5,
                "urgency": 4,
                "fit_score": 0.87,
                "deadline": 2,
                "available_slots": ["s1", "s2"],
                "required_specialization": "security",
            },
            {
                "candidate_id": "cand_sec_mid1",
                "role": "Security Analyst",
                "priority": 3,
                "urgency": 3,
                "fit_score": 0.95,
                "deadline": 3,
                "available_slots": ["s1", "s2"],
                "required_specialization": "security",
            },
            {
                "candidate_id": "cand_be_conf1",
                "role": "Staff Backend Engineer",
                "priority": 4,
                "urgency": 4,
                "fit_score": 0.84,
                "deadline": 2,
                "available_slots": ["s2", "s3"],
                "required_specialization": "backend",
            },
            {
                "candidate_id": "cand_fe_conf1",
                "role": "Senior Frontend Engineer",
                "priority": 3,
                "urgency": 3,
                "fit_score": 0.78,
                "deadline": 3,
                "available_slots": ["s3", "s4"],
                "required_specialization": "frontend",
            },
        ],
        "interviewers": [
            {
                "interviewer_id": "int_security_scarce",
                "specialization": "security",
                "available_slots": ["s1", "s2"],
                "max_capacity": 1,
            },
            {
                "interviewer_id": "int_backend_e",
                "specialization": "backend",
                "available_slots": ["s2", "s3"],
                "max_capacity": 2,
            },
            {
                "interviewer_id": "int_frontend_d",
                "specialization": "frontend",
                "available_slots": ["s3", "s4"],
                "max_capacity": 1,
            },
        ],
    },
}


def list_task_names() -> List[str]:
    return list(TASKS.keys())
