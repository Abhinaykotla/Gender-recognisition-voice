from sklearn.decomposition import PCA
from sklearn.preprocessing import Normalizer


def process_data(x_train, y_train, x_test):

    norm = Normalizer(norm='l2')
    X_train_l2 = norm.transform(x_train)
    X_test_l2 = norm.transform(x_test)

    pca = PCA()
    pca.fit(X_train_l2)
    X_train_pca = pca.transform(X_train_l2)
    X_test_pca = pca.transform(X_test_l2)
    
    return X_train_pca, y_train, X_test_pca, pca