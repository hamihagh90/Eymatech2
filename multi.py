import math
import time
from multiprocessing import Pool
if __name__ == '__main__':
    start = time.perf_counter()
    results1 = [math.factorial(x) for x in range(12000)]
    end = time.perf_counter()

    print(end - start)

    start = time.perf_counter()
    with Pool(5) as p:
        results2 = p.map(math.factorial, list(range(12000)))
    end = time.perf_counter()

    print(end - start)

#              ┌───────────────┐
#              │ Main Process   │
#              └──────┬────────┘
#           sends tasks│
#                      ▼
#         ┌──────────────────────────┐
#         │      Task Queue          │
#         └──────────────────────────┘
#           ▲        ▲        ▲
#           │        │        │
#  ┌────────┴───┐ ┌──┴────────┐ ┌────────┴───┐
#  │ Worker-1   │ │ Worker-2   │ │ Worker-3   │ ... (تا 5)
#  └─────┬──────┘ └─────┬──────┘ └─────┬──────┘
#        │ انجام کار      │ انجام کار      │ انجام کار
#        ▼                ▼                ▼
#    Pickle           Pickle           Pickle
#        │                │                │
#        ▼                ▼                ▼
#       Pipe            Pipe            Pipe
#        │                │                │
#        └───────► Main Process ◄──────────┘
#                      │
#                 Unpickle
#                      │
#                      ▼
#                  results2[]
