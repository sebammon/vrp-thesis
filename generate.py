import argparse
from pathlib import Path

from joblib import Parallel, delayed

from utils.common import save_pickle
from utils.data import generate_and_solve

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--nodes", type=int, required=True)
    parser.add_argument("--instances", type=int, required=True)
    parser.add_argument("--time_limit", type=int, required=True)
    parser.add_argument("--jobs", type=int, default=-1)

    args = parser.parse_args()

    results = Parallel(n_jobs=args.jobs, verbose=10, backend="multiprocessing")(
        delayed(generate_and_solve)(args.nodes, time_limit=args.time_limit)
        for _ in range(args.instances)
    )

    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)

    file_name = f"vrp_{args.nodes}_{args.time_limit}s"

    save_pickle(results, data_dir / f"{file_name}.pkl")
