import pandas as pd

df = pd.read_csv("yelp2022/yelp.inter", sep="\t")  # or the correct delimiter
df = df.sort_values(by="timestamp:float")           # replace with actual column name

size = len(df)
subset = df.iloc[-size//8:]                   # last one-eighth

subset.to_csv("yelp.inter", sep="\t", index=False)
