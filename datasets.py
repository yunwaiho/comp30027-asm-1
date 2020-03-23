import pandas as pd


# Datasets

# Nominal

# Predict cancer [11]
# [1] not to be used as attributes
cancer = pd.read_csv("datasets/breast-cancer-wisconsin.data", header=None, index_col=False,
                     names=("Sample code number", "Clump Thickness",
                            "Uniformity of Cell Size", "Uniformity of Cell Shape",
                            "Marginal Adhesion", "Single Epithelial Cell Size",
                            "Bare Nuclei", "Bland Chromatin", "Normal Nucleoli",
                            "Mitoses", "Class"))

mushroom = pd.read_csv("datasets/mushroom.data", header=None, index_col=False,
                       names=("cap - shape", "cap - surface", "cap - color",
                              "bruises", "odor", "gill - attachment", "gill - spacing",
                              "gill - size", "gill - color", "stalk - shape",
                              "stalk - root", "stalk - surface - above - ring",
                              "stalk - surface - below - ring", "stalk - color - above - ring",
                              "stalk - color - below - ring", "veil - type", "veil - color",
                              "ring - number", "ring - type", "spore - print - color",
                              "population", "habitat"))

lymphography = pd.read_csv("datasets/lymphography.data", header=None, index_col=False,
                           names=("class", "lymphatics", "block of affere", "bl. of lymph. c",
                                  "bl. of lymph. s", "by pass", "extravasates", "regeneration of",
                                  "early uptake in", "lym.nodes dimin", "lym.nodes enlar",
                                  "changes in lym.", "defect in node", "changes in node",
                                  "changes in stru", "special forms", "dislocation of",
                                  "exclusion of no", "no. of nodes in"))

# Numeric

wdbc = pd.read_csv("datasets/wdbc.data", header=None)
wine = pd.read_csv("datasets/wine.data", header=None)

# Ordinal

car = pd.read_csv("datasets/car.data", header=None)
nursery = pd.read_csv("datasets/nursery.data", header=None)
somerville = pd.read_csv("datasets/somerville.data", header=None)

# Mixed

# Predict >$50k [15]
adult = pd.read_csv("datasets/adult.data", header=None, index_col=False,
                    names=("age", "workclass", "fnlwgt", "education",
                           "education-num", "marital-status", "occupation",
                           "relationship", "race", "sex", "capital-gain",
                           "capital-loss", "hours-per-week", "native-country", "class"))

# Predict term deposit purchase [15]
bank = pd.read_csv("datasets/bank.data", header=None, index_col=False,
                   names=("age", "job", "marital", "education", "default", "balance",
                          "housing", "loan", "contact", "day", "month", "duration",
                          "campaign", "pdays", "previous", "poutcome", "purchase"))

university = pd.read_csv("datasets/university.data", header=None)


