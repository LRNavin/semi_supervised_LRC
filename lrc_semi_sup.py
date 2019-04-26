import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.semi_supervised import LabelPropagation
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

plt.rc('text', usetex=True)
plt.rc('font', family='serif')


np.random.seed()


def shuffle_ordered_arrays(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]


def prepare_labeled_unlabeled(X, y, num_labeled, num_unlabeled):
    perclass_labelled   = num_labeled/2;
    perclass_unlabelled = num_unlabeled/2;

    # 11 - Dimensional data
    class_positive = X[:4000, :]
    class_negative = X[4000:-1, :]

    # class label
    label_positive = y[:4000]
    label_negative = y[4000:-1]

    class_positive, label_positive = shuffle_ordered_arrays(class_positive, label_positive)
    class_negative, label_negative = shuffle_ordered_arrays(class_negative, label_negative)

    X_label = np.concatenate( (class_positive[:perclass_labelled] ,
                             class_negative[:perclass_labelled]), axis=0)
    X_un_label = np.concatenate( (class_positive[num_labeled:num_labeled + perclass_unlabelled] ,
                                class_negative[num_labeled:num_labeled + perclass_unlabelled]), axis=0)

    y_label = np.concatenate( (label_positive[:perclass_labelled] ,
                             label_negative[:perclass_labelled]), axis=0)
    y_un_label = np.concatenate( (label_positive[num_labeled:num_labeled + perclass_unlabelled] ,
                                label_negative[num_labeled:num_labeled + perclass_unlabelled]), axis=0)

    return (
        X_label,
        X_un_label,
        y_label,
        y_un_label
    )


def self_training(clf, X_l, y_l, X_u, top_k=5):

    # Initial fit on labeled data
    clf.fit(X_l, y_l)

    # Predict Unlabeled data and create new Decision Boundary on new data (labelled + unlabelled)
    if X_u.shape[0] > 1:
        y_u_pred = [1 if  i >= 0.5 else 0 for i in clf.predict(X_u)]

        X_l = np.concatenate((X_l, X_u), axis=0)
        y_l = np.concatenate((y_l, y_u_pred), axis=0)

        clf.fit(X_l, y_l)

    return clf


def propagate_labels(X_u, y_u, X_l, y_l, num_unlabeled):
    # unlabeled samples are represented by -1 in labelprop
    y_u_placeholder = np.zeros(num_unlabeled) - 1

    X_train_prop = np.concatenate((X_l, X_u), axis=0)
    y_train_prop = np.concatenate((y_l, y_u_placeholder), axis=0)

    prop = LabelPropagation()
    prop.fit(X_train_prop, y_train_prop)

    y_train_lrc = prop.transduction_

    X_train_lrc = np.concatenate((X_l, X_u), axis=0)

    return X_train_lrc, y_train_lrc


def split_classwise_train_test(X, y, percent):

    # 11 - Dimensional data
    class_positive = X[:5000, :]
    class_negative = X[5000:-1, :]

    # class label
    label_positive = y[:5000]
    label_negative = y[5000:-1]

    X_train1, X_test1, y_train1, y_test1 = train_test_split(class_positive, label_positive, test_size=percent)
    X_train2, X_test2, y_train2, y_test2 = train_test_split(class_negative, label_negative, test_size=percent)

    return np.concatenate((X_train1, X_train2), axis=0) ,\
           np.concatenate((X_test1, X_test2), axis=0),\
           np.concatenate((y_train1, y_train2), axis=0),\
           np.concatenate((y_test1, y_test2), axis=0)

gauss_data = pd.read_csv('twoGaussians.csv', header=None)

# Get data in a format that fits sklearn
gauss_data[11] = pd.Categorical(gauss_data[11])
gauss_data[11] = gauss_data[11].cat.codes
X_raw = gauss_data.values

y = X_raw[:, -1]
X = X_raw[:, :-1]


# Train a model supervised on all data
print("===\tNow training on all data with labels\t===\n")
clf = LinearRegression()
clf.fit(X, y)
tt = clf.predict(X)
y_pred = [1 if  i >= 0.5 else 0 for i in clf.predict(X)]
accuracy_all_data = accuracy_score(y_pred=y_pred, y_true=y)
print("Accuracy from training lda on all data: {:.4f}\n".format(
        accuracy_all_data
    ))

# Reserve 20% of data for testing classifiers
X_train, X_test, y_train, y_test = split_classwise_train_test(X, y, percent=.2)

# Select data for supervised and unsupervised training
num_labeled = 16
nums_unlabeled = [0, 10, 20, 40, 80, 160, 320, 640]

N_iterations = 10  #average error rates over (Curve Smoothening)

# Allocate array to hold error_rates from all classifiers
error_rates = np.ndarray(shape=(N_iterations, len(nums_unlabeled), 2))
sq_loss = np.ndarray(shape=(N_iterations, len(nums_unlabeled), 2))


for i, num_unlabeled in enumerate(nums_unlabeled):
    print("****\t Training on {} labelled data, {} unlabelled data\t****\n".format(num_labeled, num_unlabeled))

    for j in range(N_iterations):
        if (j + 1) % 10 == 0:
            print('Iteration: {}'.format(j + 1))

        # ######################## --Semi-supervised 1-- ########################
        # 1 - Train on labeled data first,
        # 2 - predict labels for unlabeled data,
        # 3 - and train classifier further with these predicted labels

        # Set up data for Same Decision Boundary Train
        X_l, X_u, y_l, y_u = prepare_labeled_unlabeled(
            X_train, y_train, num_labeled, num_unlabeled
        )

        clf = LinearRegression()
        clf = self_training(clf, X_l, y_l, X_u)

        # Do predictions for test set and evaluate
        y_pred = [1 if buff >= 0.5 else 0 for buff in clf.predict(X_test)]
        y_error = mean_squared_error(y_test, clf.predict(X_test)) #np.sum(np.max(clf.predict_log_proba(X_test), axis=1))
        accuracy = accuracy_score(y_true=y_test, y_pred=y_pred)
        error_rates[j, i, 0] = 1 - accuracy
        sq_loss[j, i, 0] = y_error


        # ######################## --Semi-supervised 2-- ########################
        # Find labels for unlabeled data with label propagation

        # Set up data for LabelPropagation
        X_l, X_u, y_l, y_u = prepare_labeled_unlabeled(
            X_train, y_train, num_labeled, num_unlabeled
        )

        if num_unlabeled != 0:
            # Semi-Supervised learning using Label Propogation
            X_train_lrc, y_train_lrc = propagate_labels(X_u, y_u, X_l, y_l, num_unlabeled)

            clf = LinearRegression()
            clf.fit(X_train_lrc, y_train_lrc)

            # Do predictions for test set and evaluate
            y_pred = [1 if  buff >= 0.5 else 0 for buff in clf.predict(X_test)]
            y_error = mean_squared_error(y_test, clf.predict(X_test))
            accuracy = accuracy_score(y_true=y_test, y_pred=y_pred)

        else:
            #0 - unlabelled data
            clf = LinearRegression()
            clf.fit(X_l, y_l)

            # Do predictions for test set and evaluate
            y_pred = [1 if buff >= 0.5 else 0 for buff in clf.predict(X_test)]
            y_error = mean_squared_error(y_test,
                                         clf.predict(X_test)) 
            accuracy = accuracy_score(y_true=y_test, y_pred=y_pred)

        error_rates[j, i, 1] = 1 - accuracy
        sq_loss[j, i, 1] = y_error

    print('\n')

avg_error_rates = np.mean(error_rates, axis=0)

avg_sq_loss = np.mean(sq_loss, axis=0)

print(avg_error_rates)
colors = ['blue', 'red']

fig, [ax1, ax2] = plt.subplots(nrows=2, sharex=True)
ax1.plot(nums_unlabeled, avg_error_rates[:, 0], color=colors[0])
ax1.plot(nums_unlabeled, avg_error_rates[:, 1], color=colors[1])
ax1.legend(['Self-training', 'Labelpropagation'])

# ax1.set_xlabel('\# unlabeled data points')
ax1.set_ylabel('Error rate')
ax1.grid()

ax2.plot(nums_unlabeled, avg_sq_loss[:, 0], color=colors[0])
ax2.plot(nums_unlabeled, avg_sq_loss[:, 1], color=colors[1])
ax2.legend(['Self-training', 'Labelpropagation'])

ax2.set_xlabel('\# unlabeled data points')
ax2.set_ylabel('Mean Squared Loss')
ax2.grid()

plt.tight_layout()

plt.savefig('figures/learning_curves.eps', dpi=300)
plt.show()