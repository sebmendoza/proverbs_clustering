from os import listdir
import re
import json


def moreCleanup(all_verses):
    for chapter, verses in all_verses.items():
        for verse, text in verses.items():
            text = text.replace(';', '; ').replace('  ', ' ')
            text = text.strip()
            all_verses[chapter][verse] = text
    return all_verses


def getAllVerses():
    files = listdir('./raw_proverbs_esv')
    all_verses = {}

    for f in files:
        chapter: str = f.split("proverbs")[1].split(".")[0]
        read_file = open(f'./raw_proverbs_esv/{f}', 'r+')
        verse = None
        text = ""
        lines = read_file.readlines()
        for l in lines:
            l = l.strip()
            #  Remove empty lines
            if len(l) == 0:
                continue
            #  Remove the [a] references
            if re.search(r"\[[a-z]\]", l):
                l = re.sub(r"\[[a-z]\]", "", l)

            #  Handle ner verse
            if re.search("^[1-9]", l):
                if verse and text:
                    if chapter not in all_verses:
                        all_verses[chapter] = {}
                    all_verses[chapter][verse] = text
                v, rest = l.split(" ", 1)
                verse = v
                text = rest
            elif verse != None:  # add to current verse
                text += " " + l + " "
            else:
                raise Exception("Not possible")

                # segments = l.split(" ", 1)
                # print(segments)
                #         if chapter not in all_verses:
                #             all_verses[chapter] = {}
                #         all_verses[chapter][verse] = text

        if chapter not in all_verses:
            all_verses[chapter] = {}
        all_verses[chapter][verse] = text

    return all_verses


if __name__ == "__main__":
    all_verses = getAllVerses()
    all_verses = moreCleanup(all_verses)

    output_file = "cleaned_esv.json"
    with open(output_file, 'w') as json_file:
        json.dump(all_verses, json_file, indent=4)
