

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
        self.datatypes = None
        self.mixed = False

    def preprocess(self, class_col, datatypes, exclude=None):
        """

        :param class_col: column number for classes
        :param class_col: data types of attributes,
            single number if all attributes same type, list if attributes different types
        :param exclude: tuple of column numbers for columns that are excluded as attributes
        :return: None; Just prepares the dataframe for work
        """
        self.class_col = self.df.columns[class_col-1]
        self.class_s = self.df[self.class_col]

        if exclude:
            self.df = self.df.drop(self.df.columns[[col-1 for col in exclude]], axis=1)

        if self.missing:
            self.df = self.df.replace(self.missing, None)

        if type(datatypes) is list:
            assert(len(datatypes) == len(self.return_attributes().columns))
            self.mixed = True
        else:
            datatypes = [datatypes for x in range(len(self.df.columns) - 1)]

        self.datatypes = datatypes

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

        for row_index, row in features.iterrows():
            max_prob = float("-inf")
            max_class = None

            for c in self.class_s.unique():
                prob = log(self.priors[c])

                for col_index, attribute in enumerate(row.index):
                    if not row[attribute]:
                        continue

                    # Numeric
                    if self.datatypes[col_index] == 2:
                        prob_instance = self.get_gaussianpdf(attribute, c, row[attribute])

                        # Most likely outlier???
                        if prob_instance == 0:
                            continue
                    # Ordinal or Nominal
                    else:
                        prob_instance = self.likelihoods[attribute][c][row[attribute]]

                    prob += log(prob_instance)

                if prob > max_prob:
                    max_prob = prob
                    max_class = c

                predictions[row_index] = max_class

        return Series(predictions)

    def evaluate(self, predictions):
        """
        Evaluates predictions based off classes

        :param predictions: Series of predictions
        :return: Accuracy
        """

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
            if attribute == self.class_col:
                continue

            classes = {}
            index = self.return_attributes().columns.get_loc(attribute)

            for c in self.class_s.unique():
                # Ordinal or Numeric
                if self.datatypes[index] == 2:
                    classes[c] = self.get_numeric_likelihood(attribute, c)

                # Nominal
                else:
                    classes[c] = self.get_nominal_likelihood(alpha, attribute, c)

            likelihoods[attribute] = classes

        return likelihoods

    def get_nominal_likelihood(self, alpha, attribute, c):
        """
        Returns a dictionary of P(a_i = a|c_j)

        :param alpha: parameter for Laplace smoothing
        :param attribute: attribute/feature being checked
        :param c: class being checked
        :return: A dictionary of feature:P(instance) pairs
        """
        unique_instances = self.df[attribute].unique()
        instances = {}
        d = len(unique_instances)

        if self.missing:
            d = len(set(unique_instances).difference({None}))

        total = self.class_s[self.class_s == c].count() + alpha * d

        for instance in unique_instances:
            if not instance:
                continue

            subtotal = ((self.df[attribute] == instance) & (self.class_s == c)).sum() + alpha

            likelihood = subtotal / total
            instances[instance] = likelihood

        return instances

    def get_numeric_likelihood(self, attribute, c):
        """
        Returns the mean, sd pair of P(attribute| c_j)

        :param attribute: attribute to be assessed
        :param c: class to be assessed
        :return: mean, sd pair
        """
        attribute_c = self.df[self.df[self.class_col] == c][attribute]
        mean = attribute_c.mean()
        sd = attribute_c.std()

        return mean, sd

    def get_gaussianpdf(self, attribute, c, instances):
        from scipy import stats

        mean = self.likelihoods[attribute][c][0]
        sd = self.likelihoods[attribute][c][1]

        return stats.norm.pdf(x=instances, loc=mean, scale=sd)

    def return_df(self):
        return self.df

    def return_attributes(self):
        return self.df.drop([self.class_col], axis=1)

    def return_attributes_no_text(self):
        from sklearn import preprocessing as pp

        enc = pp.OrdinalEncoder()
        return enc.fit_transform(self.return_attributes())

    def return_class(self):
        return self.class_s
