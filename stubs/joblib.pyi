from typing import Any, Callable, Iterable, ParamSpec, TypeVar

T = TypeVar("T")
P = ParamSpec("P")
Q = ParamSpec("Q")

def delayed(
    func: Callable[P, T]
) -> Callable[..., tuple[Callable[P, T], Q.args, Q.kwargs]]: ...

class Parallel:
    def __init__(self, *args: Any, n_jobs: int, backend: str, **kwargs: Any): ...
    def __call__(
        self,
        iterable: Iterable[
            tuple[
                Callable[P, T],
                P.args,
                P.kwargs,
            ]
        ],
    ) -> Iterable[T]: ...
