from model_lambdamart.data_utils import *
from model_lambdamart.lambda_MART import *
if __name__ == "__main__":
    ## baseline without rank
    X_train, y_train, Query_train = readDataset('../model_ranknet/Data/train.txt')
    X_test, y_test, Query_test = readDataset('../model_ranknet/Data/test.txt')
    clf_temp = DecisionTreeRegressor().fit(X_train, y_train)
    print(ndcgl(clf_temp.predict(X_train), y_train, Query_train))
    print(ndcgl(clf_temp.predict(X_test), y_test, Query_test))


    ## lambda_MART
    query_d = 20
    q_dict = defaultdict(lambda: np.zeros(query_d))
    unique_q, counts = np.unique(Query_train, return_counts=True)

    for i, q in enumerate(unique_q):
        for j in range(len(X_train)):
            if Query_train[j] == q:
                q_dict[q] += X_train[j][11:31] / counts[i]
    q_train = np.zeros((len(X_train), query_d))

    for i in range(len(X_train)):
        # print(Query_train[i])
        q_train[i] = q_dict[Query_train[i]]

    q_dict = defaultdict(lambda: np.zeros(query_d))
    unique_q, counts = np.unique(Query_test, return_counts=True)
    for i, q in enumerate(unique_q):
        for j in range(len(X_test)):
            if Query_test[j] == q:
                q_dict[q] += X_test[j][11:31] / counts[i]
    q_test = np.zeros((len(X_test), query_d))

    for i in range(len(X_test)):
        q_test[i] = q_dict[Query_test[i]]

    # print((X_train[0]))
    clf_lambdaMart = LambdaMART(200, alpha=0.5, beta=1., feature_subset=False, feature_fraction=0.3)
    clf_lambdaMart.fit(X_train, y_train, Query_train, q_train, X_test, y_test, Query_test, q_test)

