from sklearn.datasets import load_wine
import pandas as pd

wine = load_wine()
df = pd.DataFrame(wine.data, columns=wine.feature_names)
df["target"] = wine.target
df.to_csv("data/raw/wine.csv", index=False)
print("Lignes :", len(df))