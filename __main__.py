import datasets as ds

from NB_learner import NaiveBayes
from sklearn.naive_bayes import GaussianNB, MultinomialNB


def main():

    data = ds.university

    df = data[0]
    class_col = data[1]
    exclude = data[2]
    missing = data[3]
    datatypes = data[4]
    multiple = False

    alpha = 1

    if type(class_col) is tuple:
        multiple = True

    class_col = tuple(class_col)

    for col_num in class_col:
        if multiple:
            data_types = datatypes[:col_num-1] + datatypes[col_num:]
        else:
            data_types = datatypes

        # Mine
        classifier = NaiveBayes(df, alpha=alpha, missing=missing)
        classifier.preprocess(class_col=col_num, datatypes=data_types, exclude=exclude)
        classifier.train()
        predictions = classifier.predict()
        accuracy = classifier.evaluate(predictions)


        # Sci-kit
        if type(datatypes) is not list or datatypes in (0, 1):
            skl = MultinomialNB(alpha=alpha)
            skl.fit(classifier.return_attributes_no_text(), classifier.return_class())
            comparison = skl.score(classifier.return_attributes_no_text(), classifier.return_class())
        else:
            skl = GaussianNB()
            skl.fit(classifier.return_attributes_no_text(), classifier.return_class())
            comparison = skl.score(classifier.return_attributes_no_text(), classifier.return_class())

        print(accuracy, comparison)


if __name__ == "__main__":
    main()
