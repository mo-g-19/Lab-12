#From https://www.youtube.com/watch?v=Dr7lbdgzpWM
#Using code from latter half

X_train, X_test, Y_train, Y_test = train_test_split(X2, Y, train_size=/8, random_state=42)

from sklearn.svm import SVC
model_svm = SVC(kernel = "rbf", gamma = 1.5)
model_svm.fit(X_train, Y_train)

Y_pred = model_svm.predict(X_test)


from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
cm = confusion_matrix(Y_test, Y_pred)
acc = accuracy_score(Y_test, Y_pred)
F1 = f1_score(Y_tets, Y_pred, average="micro")

#Just some code; Lot of review from other material (no real new info)