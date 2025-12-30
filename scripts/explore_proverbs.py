import pandas as pd
from pathlib import Path

root_path = Path()
PROVERBS_BOOK_NUMBER = 20

asv_bible_path = root_path / "resources" / "BibleCorpus" / "t_asv.csv"
bible = pd.read_csv(asv_bible_path)
# Filter only proverbs
proverbs = bible[bible['b'] == PROVERBS_BOOK_NUMBER]
pd.set_option('display.max_colwidth', None)
# Get summary stats
print(proverbs.head())
print(proverbs.shape)

verse7 = proverbs.loc[
    (proverbs["v"] == 7) &
    (proverbs["c"] == 1)
]["t"]
print(verse7)

value_counts = proverbs[proverbs["t"] ==
                        "So shall thy poverty come as a robber, And thy want as an armed man."]


all_proverb_text = proverbs["t"]
# print(all_proverb_text.head())
