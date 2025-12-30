import json
from pathlib import Path


def getAllVerses():
    res = []
    data = getDataFromJson()
    res = []
    for verses in data.values():
        for text in verses.values():
            res.append(text)
    return res


def getDataFromJson():
    with open("./cleaned_esv.json", "r") as json_file:
        data = json.load(json_file)
        return data


def saveGraph(title: str, plt):
    outdir = Path("visualizations")
    plt.savefig(outdir / title)
