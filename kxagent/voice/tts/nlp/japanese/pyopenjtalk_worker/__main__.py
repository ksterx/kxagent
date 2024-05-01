import argparse

from .worker_common import WORKER_PORT
from .worker_server import WorkerServer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=WORKER_PORT)
    args = parser.parse_args()
    server = WorkerServer()
    server.start_server(port=args.port)


if __name__ == "__main__":
    main()
