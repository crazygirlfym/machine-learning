# -*-- coding:utf-8 -*--
from sklearn.tree import DecisionTreeRegressor
from copy import deepcopy
import pickle
from model_lambdamart.metrics import *
import warnings
warnings.filterwarnings('ignore')

class LambdaMART:
    def __init__(self, n_estimators, base_estimator=DecisionTreeRegressor(), step_estimator=DecisionTreeRegressor(),
                 alpha=1., beta=1., adaptive_step=True, feature_subset=False, feature_fraction=1., stochastic=False):
        """
            n_estimators :
            base_estimator :
            step_estimator :
            alpha :
            beta :
            adaptive_step :
            feature_subset :
        """
        self.estimators = []
        self.step_estimators = []
        self.feature_idx = []
        self.n_estimators = n_estimators
        self.base_estimator = base_estimator
        self.step_estimator = step_estimator
        self.alpha = alpha
        self.beta = beta
        self.adaptive_step = adaptive_step
        self.feature_subset = feature_subset
        self.feature_fraction = feature_fraction
        self.stochastic = stochastic

    def fit(self, X_train, y_train, qids_train, queries_train, X_test, y_test, qids_test, queries_test, verbose=False):

        X = X_train
        y = y_train
        qids = qids_train
        queries = queries_train
        self.estimators = []
        self.step_estimators = []
        self.last_iteration = 0

        # id_y = np.argsort(np.argsort(y_train)[::-1])
        for i in range(self.n_estimators):
            predicted = self.predict(X_train, qids_train, queries_train, verbose)
            y_pred = self.predict(X_test, qids_test, queries_test)
            train_result = ndcgl(predicted, y_train, qids_train)
            test_result = ndcgl(y_pred, y_test, qids_test)
            print ("Iteration {}, train_result {}, test_result {}".format(i, train_result, test_result))
            # if self.feature_subset:
            #     results_buf1.append((i, train_result, test_result))
            # else:
            #     results_buf2.append((i, train_result, test_result))
                # print "Iteration ", i, "train_result", ndcgl(predicted) "" test_result ",
            sub_idx = []
            if self.stochastic:
                sub_idx = np.random.choice(np.arange(X_train.shape[0]), int(0.6 * X_train.shape[0]), replace=False)
                # grad, h = self.loss_grad(predicted[sub_idx], y_train[sub_idx], id_y[sub_idx])
                grad, h = self.loss_grad(predicted[sub_idx], y_train[sub_idx], qids_train[sub_idx])
                X = X_train[sub_idx]
                y = y_train[sub_idx]
                qids = qids_train[sub_idx]
                queries = queries_train[sub_idx]
            else:
                # grad, h = self.loss_grad(predicted, y, id_y)
                grad, h = self.loss_grad(predicted, y, qids)
            if verbose:
                print ("Grad:")
                print(grad)
                print("H:")
                print (h)
            estimator = deepcopy(self.base_estimator)
            if self.feature_subset:

                feature_idx = np.random.choice(np.arange(X.shape[1]), int(self.feature_fraction * X.shape[1]),
                                               replace=False)
                self.feature_idx.append(feature_idx)
                estimator.fit(X.T[feature_idx].T, -grad / (self.alpha + h))
            else:
                estimator.fit(X, -grad / (self.alpha + h))

            self.estimators.append(estimator)
            if verbose:
                print("Predict:")
                print(predicted)

            if self.adaptive_step:
                step_estimator = deepcopy(self.step_estimator)

                qs = np.unique(qids)

                pred_values = np.zeros(len(qs))
                q_list = np.zeros((len(qs), queries.shape[1]))
                for idx, q in enumerate(qs):

                    pred_values[idx] = 0.
                    for j in range(len(X)):
                        if qids[j] == q:
                            q_list[idx] = queries[j]
                            pred_values[idx] += 1. / (1. + self.beta * h[j])
                            # q_list[idx] = q_dict[q]
                sum_pred_values = np.sum(pred_values)
                step_estimator.fit(q_list, pred_values * len(qs) / sum_pred_values)
                self.step_estimators.append(step_estimator)
            self.last_iteration = i
        return self

    def predict(self, X, qids, queries, verbose=False):
        # y_pred = np.zeros(X.shape[0])
        y_pred = np.zeros(len(X))

        for est_i, estimator in enumerate(self.estimators):
            if self.feature_subset:
                feature_idx = self.feature_idx[est_i]
                est_result = estimator.predict(X.T[feature_idx].T)
            else:
                est_result = estimator.predict(X)
            for i in range(len(X)):
                step = 1. / (i + 1.)
                if self.adaptive_step:
                    step = self.step_estimators[est_i].predict(queries[i].reshape(1, -1)[0])
                if verbose:
                    print("Step: ", step, " for qid=", qids[i])
                y_pred[i] += est_result[i] * step

        return y_pred

    def continue_fit(self, n_estimators, X_train, y_train, qids_train, queries_train, X_test, y_test, qids_test,
                     queries_test):
        X = X_train
        y = y_train
        qids = qids_train
        queries = queries_train
        prev_n_estimators = self.n_estimators
        self.n_estimators = n_estimators

        # id_y = np.argsort(np.argsort(y_train)[::-1])
        for i in range(prev_n_estimators, self.n_estimators):
            predicted = self.predict(X_train, qids_train, queries_train, False)
            y_pred = self.predict(X_test, qids_test, queries_test)
            train_result = ndcgl(predicted, y_train, qids_train)
            test_result = ndcgl(y_pred, y_test, qids_test)
            print("Iteration {}, train_result {}, test_result {}".format(i, train_result, test_result))
            # if self.feature_subset:
            #     results_buf1.append((i, train_result, test_result))
            # else:
            #     results_buf2.append((i, train_result, test_result))
            sub_idx = []
            if self.stochastic:

                sub_idx = np.random.choice(np.arange(X_train.shape[0]), int(0.6 * X_train.shape[0]), replace=False)
                grad, h = self.loss_grad(predicted[sub_idx], y_train[sub_idx], qids_train[sub_idx])
                X = X_train[sub_idx]
                y = y_train[sub_idx]
                qids = qids_train[sub_idx]
                queries = queries_train[sub_idx]
            else:
                grad, h = self.loss_grad(predicted, y, qids)
            estimator = deepcopy(self.base_estimator)
            if self.feature_subset:

                feature_idx = np.random.choice(np.arange(X.shape[1]), int(self.feature_fraction * X.shape[1]),
                                               replace=False)
                self.feature_idx.append(feature_idx)
                estimator.fit(X.T[feature_idx].T, -grad / (self.alpha + h))
            else:
                estimator.fit(X, -grad / (self.alpha + h))
            self.estimators.append(estimator)

            if self.adaptive_step:
                step_estimator = deepcopy(self.step_estimator)

                qs = np.unique(qids)
                pred_values = np.zeros(len(qs))
                q_list = np.zeros((len(qs), queries.shape[1]))
                for idx, q in enumerate(qs):

                    pred_values[idx] = 0.
                    for j in range(X.shape[0]):
                        if qids[j] == q:
                            q_list[idx] = queries[j]
                            pred_values[idx] += 1. / (1. + self.beta * h[j])
                            # q_list[idx] = q_dict[q]
                sum_pred_values = np.sum(pred_values)
                step_estimator.fit(q_list, pred_values * len(qs) / sum_pred_values)
                self.step_estimators.append(step_estimator)
            self.last_iteration = i

        return self

    def loss_grad(self, pred_y, y, qids):
        """
        :param pred_y: score for the predicted label
        :param y: true label
        :param qids:
        :return: first derivative,  second derivation
        """
        n_elems = y.shape[0]
        grad = np.zeros(n_elems)
        h = np.zeros(n_elems)

        id_y = get_dict_by_qid(qids, pred_y)
        for i in range(n_elems):
            for j in range(n_elems):
                if qids[i] == qids[j]:
                    buf = 0.
                    delta_ndcg = np.abs(self.ndcg_replace(y[i], y[j], id_y[qids[i]][i], id_y[qids[j]][j]))
                    if y[i] > y[j]:
                        buf = -self.rho(pred_y[i], pred_y[j])
                    if y[i] < y[j]:
                        buf = self.rho(pred_y[j], pred_y[i])
                    if buf != 0.:
                        grad[i] += delta_ndcg * buf
                    h[i] += delta_ndcg * self.rho(pred_y[i], pred_y[j]) * (1 - self.rho(pred_y[i], pred_y[j]))
        return grad, h

    def ndcg_replace(self, value_a, value_b, id_a, id_b):
        return (2. ** value_b - 2. ** value_a) * (1. / np.log2(2 + id_a) - 1. / np.log2(2 + id_b))

    def rho(self, a, b):
        return 1. / (1. + np.exp(a - b))