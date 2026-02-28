import json
import os
import argparse


def get_repos_list():
    with open(os.path.join(os.getcwd(), "data", "repos_fim.json"), "r") as f:
        repos = json.load(f)
    with open(os.path.join(os.getcwd(), "data", "discard_fim.json"), "r") as f:
        discard_repos = json.load(f)
    merged_repos = repos + discard_repos
    return merged_repos


def clone_repos(repos, save_dir):
    for repo in repos:
        print(repo["repostory_name"])

        os.system(
            f"git clone {repo['repostory_url']} {save_dir}/{repo['repostory_name']}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clone a subset of repositories.")
    parser.add_argument("--subset", type=int, help="Number of repositories to clone.")
    args = parser.parse_args()
    l = get_repos_list()
    if args.subset:
        l = l[: args.subset]
    clone_repos(
        l,
        os.path.join(
            os.getcwd(),
            "data",
            "repos",
        ),
    )
    print("hi")
