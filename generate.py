import argparse
import pickle
from pathlib import Path

from joblib import Parallel, delayed

from utils.data import generate_and_solve

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--n_nodes', type=int, default=20)
    parser.add_argument('--n_instances', type=int, default=5_000)
    parser.add_argument('--n_jobs', type=int, default=5)

    args = parser.parse_args()

    results = Parallel(n_jobs=args.n_jobs, verbose=10, backend='multiprocessing')(
        delayed(generate_and_solve)(args.n_nodes) for _ in range(args.n_instances))

    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)

    # TODO: ensure that the file is not overwritten
    with open(data_dir / f"results_{args.n_instances}x{args.n_nodes}_nodes.pkl", "wb") as f:
        pickle.dump(results, f)
