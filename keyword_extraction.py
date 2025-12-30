from keybert import KeyBERT
kw_model = KeyBERT(model='all-MiniLM-L6-v2')


cluster_example = [
    "The sluggard does not plow in the autumn; he will seek at harvest and have nothing.",
    "The desire of the sluggard kills him, for his hands refuse to labor.", ""
    "The sluggard says, “There is a lion outside! I shall be killed in the streets!”",
    "The sluggard says, 'There is a lion in the road! There is a lion in the streets!'",
    "As a door turns on its hinges, so does a sluggard on his bed.",
    "The sluggard buries his hand in the dish; it wears him out to bring it back to his mouth.",
    "The sluggard is wiser in his own eyes than seven men who can answer sensibly.",
    "When the grass is gone and the new growth appears and the vegetation of the mountains is gathered,",
    "The sluggard buries his hand in the dish and will not even bring it back to his mouth.",
    "the rock badgers are a people not mighty, yet they make their homes in the cliffs;",
    "the lion, which is mightiest among beasts and does not turn back before any;",
    "I passed by the field of a sluggard, by the vineyard of a man lacking sense,",
    "The way of a sluggard is like a hedge of thorns, but the path of the upright is a level highway.",
    "Where there are no oxen, the manger is clean, but abundant crops come by the strength of the ox.",
    "The soul of the sluggard craves and gets nothing, while the soul of the diligent is richly supplied.",
    "Like vinegar to the teeth and smoke to the eyes, so is the sluggard to those who send him.",
    "When he established the heavens, I was there; when he drew a circle on the face of the deep,",
]

keywords = kw_model.extract_keywords(
    cluster_example, keyphrase_ngram_range=(1, 5), stop_words=None)

print(keywords)
