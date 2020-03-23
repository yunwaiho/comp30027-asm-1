from NB_learner import NaiveBayes
import datasets as ds


def main():

    df = ds.lymphography

    classifier = NaiveBayes(df)
    classifier.preprocess(class_col=1)
    classifier.train()
    predictions = classifier.predict()
    accuracy = classifier.evaluate(predictions)

    print(accuracy)


if __name__ == "__main__":
    main()
