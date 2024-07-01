if __name__ == "__main__":
    from huggingface_hub import snapshot_download
    import subprocess
    import os

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("repo", type=str)
    parser.add_argument("-r", "--revision", type=str, required=True)
    args = parser.parse_args()

    path = snapshot_download(repo_id=args.repo, revision=args.revision)

    os.chdir("../../../llama.cpp/")  # move to llama.cpp
    subprocess.run(
        [
            "python3",
            "convert-hf-to-gguf.py",
            "--outpath",
            os.path.join(path, "model.gguf"),
        ]
    )
