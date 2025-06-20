from .task import Task
from .io_example import I, O, IOExample
from .program import Program, value
from .library import Library

from typing import Sequence
from abc import ABC, abstractmethod
from random import choice, choices


class Model(ABC):
    def __init__(self, library: Library, tasks: Sequence[Task[I, O]]):
        self.library = library
        self.tasks = list(tasks)

    @abstractmethod
    def infer_program(self, task: Task[I, O]) -> Program[O]:
        """
        Infer a program that solves the given sequence of IOExamples.
        """
        ...

    @abstractmethod
    def train(self, tasks: Sequence[Task[I, O]]):
        """
        Train the program inference model with the given tasks.
        """
        ...

    def wake(self, trial: int = 100):
        for task in self.tasks:
            inferred_programs = [self.infer_program(task) for _ in range(trial)]
            solutions = [
                program
                for program in inferred_programs
                if task.validate_program(program)
            ]
            if not solutions:
                continue
            ranked_solutions = sorted(solutions, key=lambda program: program.cost)
            best_solution = ranked_solutions[0]
            task.solution_program = best_solution

    def abstract(self):
        self.library.compress(
            *[task.solution_program for task in self.tasks if task.solution_program]
        )

    def generate_fantasy(self, example_nums: int = 3) -> Task:
        random_program = self.library.sample()
        exmaples = []
        for _ in range(example_nums):
            random_input = choice([i for i in choice(self.tasks).inputs()])
            exmaples.append(
                IOExample(
                    random_input, random_program.apply(value(random_input)).eval()
                )
            )
        return Task(exmaples)

    @abstractmethod
    def dream(self, replay_fantasy_mix: tuple[int, int] = (50, 30)):
        replays = choices(
            [task for task in self.tasks if task.solution_program],
            k=replay_fantasy_mix[0],
        )
        fantasies = [self.generate_fantasy() for _ in range(replay_fantasy_mix[1])]
        self.train(replays + fantasies)

    def helmholtz_enumerate(self, iteration: int, trial: int = 100):
        for i in range(iteration):
            self.wake(trial)
            self.abstract()
            self.dream()
