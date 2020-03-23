

class NaiveBayes:

    def __init__(self, df, missing=None, alpha=1):
        """

        :param df: pandas dataframe of data
        """
        self.df = df
        self.class_col = None
        self.class_s = None
        self.missing = missing
        self.alpha = alpha
        self.priors = None
        self.likelihoods = None

    def preprocess(self, class_col, exclude=None):
        """

        :param class_col: column number for classes
        :param exclude: tuple of column numbers for columns that are excluded as attributes
        :return: None; Just prepares the dataframe for work
        """
        self.class_col = self.df.columns[class_col-1]
        self.class_s = self.df[self.class_col]

        if exclude:
            self.df = self.df.drop(self.df.columns[[col-1 for col in exclude]], axis=1)

        if self.missing:
            self.df = self.df.replace(self.missing, None)

    def train(self):
        """
        Finds the priors and likelihoods based off data
        :return: None; just assigns likelihood to class
        """

        self.priors = self.get_priors()
        self.likelihoods = self.get_likelihoods(self.alpha)

    def predict(self):
        """
        Predicts class based off attributes
        :return: Series of predictions
        """
        from pandas import Series
        from math import log

        features = self.df.drop([self.class_col], axis=1)
        predictions = {}

        for index, row in features.iterrows():
            max_prob = float("-inf")
            max_class = None

            for c in self.class_s.unique():
                prob = log(self.priors[c])

                for attribute in row.index:
                    if not row[attribute]:
                        continue
                    prob += self.likelihoods[attribute][c][row[attribute]]

                if prob > max_prob:
                    max_prob = prob
                    max_class = c

            predictions[index] = max_class

        return Series(predictions)

    def evaluate(self, predictions):
        return (self.class_s == predictions).sum() / self.class_s.count()

    def get_priors(self):
        """
        :return: Empirical priors for each class c_j
        """

        s = self.class_s
        classes = s.unique()
        priors = {}

        for c in classes:
            priors[c] = s[s == c].count()/s.count()

        return priors

    def get_likelihoods(self, alpha):
        """
        Gets the likelihood for each attribute-class pair
        Also implements Laplace smoothing with parameter alpha

        :return: likelihoods
        """

        likelihoods = {}

        for attribute in self.df.columns:
            if attribute != self.class_col:
                classes = {}
                unique_instances = self.df[attribute].unique()

                for c in self.class_s.unique():
                    instances = {}
                    d = len(unique_instances)

                    if self.missing:
                        d = len(set(unique_instances).difference({None}))

                    total = self.class_s[self.class_s == c].count() + alpha * d

                    for instance in unique_instances:
                        if not instance:
                            continue

                        subtotal = ((self.df[attribute] == instance) & (self.class_s == c)).sum() + alpha

                        loglikelihood = subtotal/total
                        instances[instance] = loglikelihood

                    classes[c] = instances

                likelihoods[attribute] = classes

        return likelihoods

    def return_df(self):
        return self.df

    def return_attributes(self):
        return self.df.drop([self.class_col], axis=1)

    def return_attributes_no_text(self):
        from sklearn import preprocessing as pp

        enc = pp.OrdinalEncoder()
        return enc.fit_transform(self.return_attributes())

    def return_classes(self):
        return self.class_s
