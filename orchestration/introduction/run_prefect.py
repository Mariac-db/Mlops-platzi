import httpx
from prefect import task, flow, get_run_logger
from typing import List


@task(retries=3)
def get_start(repo: str) -> None:
    url = f"https://api.github.com/repos/{repo}"
    response = httpx.get(url)
    repo = response.json()
    print(f"Stars 🌠 : {repo['stargazers_count']}")


@flow(name="Github Starts", log_prints=True)
def github_starts(repos: List[str]) -> None:
    for repo in repos:
        print(f"Info url para {repo}")
        get_start(repo)

if __name__ == "__main__":
    github_starts(["PrefectHQ/Prefect", "PrefectHQ/miter-design"])

# ejecutamos prefect server start en la terminal para abrir el server de prefect