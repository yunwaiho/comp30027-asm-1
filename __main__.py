import datasets as ds

from NB_learner import NaiveBayes
from sklearn.naive_bayes import GaussianNB, MultinomialNB


def main():

    data = ds.mushroom
    df = data[0]
    class_col = data[1]
    exclude = data[2]
    missing = data[3]

    alpha = 1

    # Mine
    classifier = NaiveBayes(df, alpha=alpha, missing=missing)
    classifier.preprocess(class_col=class_col, exclude=exclude)
    classifier.train()
    predictions = classifier.predict()
    accuracy = classifier.evaluate(predictions)

    # Sci-kit
    skl = MultinomialNB(alpha=alpha)
    skl.fit(classifier.return_attributes_no_text(), classifier.return_classes())
    comparison = skl.score(classifier.return_attributes_no_text(), classifier.return_classes())

    print(accuracy, comparison)


if __name__ == "__main__":
    main()
