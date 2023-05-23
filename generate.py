import argparse
import pickle
from pathlib import Path

from joblib import Parallel, delayed

from utils.data import generate_and_solve

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--n_nodes', type=int, default=20)
    parser.add_argument('--n_instances', type=int, default=5000)
    parser.add_argument('--time_limit', type=int, default=3)
    parser.add_argument('--n_jobs', type=int, default=-1)

    args = parser.parse_args()

    results = Parallel(n_jobs=args.n_jobs, verbose=10, backend='multiprocessing')(
        delayed(generate_and_solve)(args.n_nodes, time_limit=args.time_limit) for _ in range(args.n_instances))

    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)

    file_name = f"vrp_{args.n_nodes}_{args.n_instances}_{args.time_limit}s"
    if (data_dir / f"{file_name}.pkl").exists():
        file_name += "_1"

    with open(data_dir / f"{file_name}.pkl", "wb") as f:
        pickle.dump(results, f)
