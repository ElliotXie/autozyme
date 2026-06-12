"""Worker-side target module for multiprocess_override_synthetic."""


def worker_payload(n: int) -> int:
    acc = 0
    for i in range(n):
        acc += (i % 97) * (i % 89)
    return acc
