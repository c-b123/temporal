import json
import pandas as pd
import requests

from pathlib import Path
from io import StringIO


class GitHubDataloader:
    def __init__(self, colab: bool = True):
        self.__colab = colab
        self.__file_path = None
        self.__file_type = None
        self.__data = None

    def __read_web_data(self):
        if self.__colab:
            from google.colab import userdata
            pat = userdata.get("pat")
        else:
            from credentials import git
            pat = git["pat"]
        headers = {
            'Authorization': f'token {pat}',
            'Accept': 'application/vnd.github.v3.raw',
            'User-Agent': 'Python'
        }
        url = f'https://api.github.com/repos/c-b123/masterThesisPlay/contents/{self.__file_path.as_posix()}'
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            if self.__file_type == ".json":
                self.__data = json.loads(response.text)
                print(f"SUCCESS Dataset \"{self.__file_path.as_posix().split('/')[-1]}\" loaded from GitHub")
            elif self.__file_type == ".csv":
                self.__data = StringIO(response.text)
                print(f"SUCCESS Dataset \"{self.__file_path.as_posix().split('/')[-1]}\" loaded from GitHub")
            elif self.__file_type == ".txt":
                self.__data = response.text
                print(f"SUCCESS Dataset \"{self.__file_path.as_posix().split('/')[-1]}\" loaded from GitHub")
        else:
            print(f"Failed to retrieve file: {response.status_code}")
            return None

    def get_data(self, file_path: Path):
        self.__file_path = file_path
        self.__file_type = file_path.suffix
        self.__read_web_data()
        return self.__data


if __name__ == "__main__":
    gitLoader = GitHubDataloader(colab=False)
    csv_file = gitLoader.get_data(Path("Resources/aquaculture_registry.csv"))
    df = pd.read_csv(csv_file)
