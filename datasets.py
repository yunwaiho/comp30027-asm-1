import pandas as pd


# Datasets

# Nominal

# Predict cancer [11]
# [1] not to be used as attributes
# Missing vals: ?
cancer = (pd.read_csv("datasets/breast-cancer-wisconsin.data", header=None, index_col=False,
                     names=("Sample code number", "Clump Thickness",
                            "Uniformity of Cell Size", "Uniformity of Cell Shape",
                            "Marginal Adhesion", "Single Epithelial Cell Size",
                            "Bare Nuclei", "Bland Chromatin", "Normal Nucleoli",
                            "Mitoses", "Class")),
          11, [1], ["?"])

# Predict type of mushroom [1]
# Missing vals: ?
mushroom = (pd.read_csv("datasets/mushroom.data", header=None, index_col=False,
                        names=("edible", "cap - shape", "cap - surface", "cap - color",
                               "bruises", "odor", "gill - attachment", "gill - spacing",
                               "gill - size", "gill - color", "stalk - shape",
                               "stalk - root", "stalk - surface - above - ring",
                               "stalk - surface - below - ring", "stalk - color - above - ring",
                               "stalk - color - below - ring", "veil - type", "veil - color",
                               "ring - number", "ring - type", "spore - print - color",
                               "population", "habitat")),
            1, None, ["?"])


# Predict diagnosis [1]
lymphography = (pd.read_csv("datasets/lymphography.data", header=None, index_col=False,
                           names=("class", "lymphatics", "block of affere", "bl. of lymph. c",
                                  "bl. of lymph. s", "by pass", "extravasates", "regeneration of",
                                  "early uptake in", "lym.nodes dimin", "lym.nodes enlar",
                                  "changes in lym.", "defect in node", "changes in node",
                                  "changes in stru", "special forms", "dislocation of",
                                  "exclusion of no", "no. of nodes in")),
                1, None, None)

# Numeric

wdbc = (pd.read_csv("datasets/wdbc.data", header=None,index_col=False,
                    names=("id", "diagnosis", "mean-radius", "mean-texture", "mean-perimeter",
                           "mean-area", "mean-smoothness", "mean-compactness", "mean-concavity",
                           "mean-concave points", "mean-symmetry", "mean-fractal dimension",
                           "se-radius", "se-texture", "se-perimeter", "se-area", "se-smoothness",
                           "se-compactness", "se-concavity", "se-concave points", "se-symmetry",
                           "se-fractal dimension", "worst-radius", "worst-texture", "worst-perimeter",
                           "worst-area", "worst-smoothness", "worst-compactness", "worst-concavity",
                           "worst-concave points", "worst-symmetry", "worst-fractal dimension")),
        2, [1], None)
       
wine = pd.read_csv("datasets/wine.data", header=None, index_col=False)

# Ordinal

car = pd.read_csv("datasets/car.data", header=None, index_col=False)
nursery = pd.read_csv("datasets/nursery.data", header=None, index_col=False)
somerville = pd.read_csv("datasets/somerville.data", header=None, index_col=False)

# Mixed

# Predict >$50k [15]
adult = (pd.read_csv("datasets/adult.data", header=None, index_col=False,
                     names=("age", "workclass", "fnlwgt", "education",
                            "education-num", "marital-status", "occupation",
                            "relationship", "race", "sex", "capital-gain",
                            "capital-loss", "hours-per-week", "native-country", "class")),
         15, None, ["?"])

# Predict term deposit purchase [15]
bank = (pd.read_csv("datasets/bank.data", header=None, index_col=False,
                    names=("age", "job", "marital", "education", "default", "balance",
                           "housing", "loan", "contact", "day","campaign", "pdays",
                           "previous", "poutcome", "purchase")),
        15, None, ["?"])


university = (pd.read_csv("datasets/university.data", header=None, index_col=False,
                          names=("University - name", "State", "Control",
                                 "number - of - students", "male: female(ratio)",
                                 "student: faculty(ratio)", "sat - verbal", "sat - math", 
                                 "expenses", "percent - financial - aid", 
                                 "number - of - applicants", "percent - admittance",
                                 "percent - enrolled", "academics", "social",
                                 "quality - of - life", "academic - emphasis")),
              (9, 14, 15), [1], [0])



