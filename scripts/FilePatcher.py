import os
from pathlib import Path


def patch(patchListIn):
    for entry in patchListIn:
        fileName = entry.get("file_name")
        try:
            # read file
            with open(fileName, "r") as file:
                data = file.read()

            # replace strings
            for replacement in entry.get("replace"):
                data = data.replace(replacement.get("from"), replacement.get("to"))

            # overwrite file
            with open(fileName, "w") as file:
                file.write(data)

            print(f"File patched: {fileName}")
        except Exception as e:
            print(f"Failed to patch file: {fileName}")
            print(str(e))


def patch_wildcard(rootDirectoryIn, targetTypeIn, patchListIn):
    result = list(Path(rootDirectoryIn).rglob(targetTypeIn))
    for path in result:
        fileName = path.as_posix()
        for entry in patchListIn:
            patch([{"file_name": fileName, "replace": entry.get("replace")}])

