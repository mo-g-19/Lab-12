#This is my comprehensive notes from https://www.geeksforgeeks.org/machine-learning/comprehensive-guide-to-classification-models-in-scikit-learn/
#Not my code

"""Classification Metrics
        Process categorize data/objects based traits/properties into specified group/categories
        Type supervised ML -> trained labelled dataset -> predict clas or category fresh, unseen data
            Primary goal: capable properly assign new observation on properties
Confusion Matrix - table summarize performance into 4 metrics
    True Positives (TP)
        Cases model correct predict positive class (ex: disease present when indeed present actual data)
            medical diagnostics -> correct id individuals with disease
            Correctly predicted positive instances
    True Negatives (TN)
        Model correct predict negative class (ex: no disease when not present actual data)
            context email: correct identify non-spam
            Correctly predicted negative instances
    False Positives (FP)
        Model incorrectly predict positive when not present in actual data
            context: diagnose disease when not there (unnecessary stress + cost)
            Incorctly predicted positive instances
    False Negatives (FN)
        Model incorrectly predict negative class when actually in positive class
            context: mistake classify spam email as non-spam
            Incorrectly predicted negative instances
    Matrix Example:
                        Predicted Negatives (0)     Predicted Positive (1)
        Actual Neg (0)          TN                              FP
        Actual Pos (1)          FN                              TP          """

"""
Accuracy
    fundamental metric eval performance classification models. Measures proportion of
        correctly predicted instances (both TN and TP) / all instances in dataset
        Accuracy: (TP + TN) / (TP + TN + FP + FN)
    Strongly misleading -> imbalanced datasets (one class significantly outweigh other)

    Strengths:
        Easy Interpretation: expressed as percentage (technical + non-technical)
        Suitable Balanced Datasets: where each class roughly equal representation
    Limitations:
        Imbalanced Datasets: misleading -> model predicts majority all instances = high accuracy (predict dominant class)
        Misleading in Critical Applications: cost FP and FN -> vary
            Accuracy => all errors equally (not case all time)
            Ex: FN -> life-threatening, FP -> might less sever treatment
    When Use:
        Valuable: class balance not concern + cost misclassification relatively equal all classes
        Common: starting point eval models (complemented other metric: precision, recall, F1-scoe, and analysis confusion matrix)
            especially imbalanced/critical applications"""

"""
Precision 
        Asses quality positive predictions made by classification model.
        Quantifies proportion TP among all positive predictions
            Precision = TP / (TP + FP)
        Ability make accurate positive prediction (valuable cost/consequences FP high)
    Significance:
        Medical Diagnoses: utmost importance (ensure positive diagnosis reliable); lower uncessecary
        Spam Detection: FP -> important messages missed, precision crucial"""

"""
Recall (Sensitivity)
        Model ability correctly id all positive instances
        Recall = TP / (TP + FN)
    Significance:
        Medical: high recall -> id all cases disease (min risk miss critical diagnoses)
        Security and Anomaly Detect: detect security threat/unusual behavior (1 threat -> sig security breach)"""

"""
F1-Score
        Combo precision and recall -> single value.
        Balanced assessment of performance (especially where imbalance between classes predicted)
        Caluculated using harmonic mean of precision and recall
            *Harmonic Mean: used find average rate of change
                One of 3 Pythagorean mean (other = arithmetic + geometric mean) *always lower compared to other two (and all 3 are related)
                HM = n / (1/x1) + (1/x2) + (1/x3)...+ (1/xn)
                Important (https://www.picsellia.com/post/understanding-the-f1-score-in-machine-learning-the-harmonic-mean-of-precision-and-recall#:~:text=In%20the%20context%20of%20evaluating%20the%20F1%20score%2C,balance%20these%20two%20metrics%20by%20considering%20their%20reciprocals.)"
                    Concept calculate average value way gives equal weight each value being averaged (regardless magnitude)
                        Common rates, ratios, and other quantities involve reciprocals
                            calculated by taking reciprocal (ie x -> 1/x), find the average of that reciprocal (sum 1/xi * 1/n), then find the reciprocal of the average (ie 1 divided by last num)
                    Equal Weighting: smaller values -> more impact HM than larger val
                    Influence Extreme Values: harmonic mean strongly influenced small values (1 val -> tend to that val)
                    Use Rates and Ratios: particularly useful averaging rates, ratios, and other quantities involve reciprocals (average speeds or rates of work)

                    Adv F1 Score -> balance Precision and Recall + Robustness to Imbalanced Datasets
                        High score: well with precision and recall
                        Low score: room improvement (either precision or recall)
        F1 = 2* (Precison * Recall) / (Precison + Recall)
            Multiply by 2 because if both 100%, would return a .50 because 1/2
        Significance:
            Handling Class Imbalance: where one class significantly outnumber other
                Ex: model = high accuracy (predict majority class most time, F1 -> FP and FN => more accurate overall)
            Balancing Precision and Recall: crucial make decisions applications cost or consequences FP and FN differ
            Single Metric Model Eval: two important aspect performance -> one value, convinent model selection, hyperparameter tuning, + compare models
        Threshold Consideration:
            Changing threshold -> both precision and recall
                Essential specific context + priorities problem
        Use Cases:
            Information Retrieval: search engines (both precision (relevance) and recall (comprehensiveness)) essential
            Medical Testing: diagnosis diseases/medical conditions (id positive casses + minimize false alarms)"""

"""
ROC Curve
        Reciever Operating Characteristic -> graphical representation classification model -> positive and negative classes at various classification threshold
            Plot TP Rate (recall/sensitivity) against FP Rate (calculated as 1-Specificity)
        Illustrates model performance change as threshold classifying instance as positive
        In the Graph:
            x-axis (FPR) - proportion Negative instances id incorrectly as Positive
            y-axis (TPR) - proportion Positive instances id correctly
            Typical -> ascending curve, bottom-left corner -> top-right
                ideal -> right-angle curve from bottom-left to top-left (perfect discrimination at all thresholds)
        Aread Under ROC Curve (AUC)
            overall performance classification (0-1):
                AUC 0.5 -> performance = random guessing
                AUC 1.0 -> perfect discrimination -> perfectly distinguish positive and negative instances all thresholds
                Single scalar val summarize model ability rank positive instances higher negative instances (regardless specific threshold, Higher -> better perform)
        Significance:
            Model Comparison: ROC-AUC -> multiple classification models -> one perform better (higher AUC -> generally more effective distinguishing between classes)
            Threhold Selection: ROC curves help choose appropriate classification threshold based specific application requirements
                Select threshold balances TPR and FPR -> desired trade-off true positives and false positives
            Imbalanced Datasets: provide more comprehensive eval of model performance beyond accuracy
        Limitations:
            not insight specific consequences/costs FP or FN
            Often conjunction with precision, recall, and F1-Score -> complete understanding model performance"""

"""Implementation classification Metrics
Import Necessary libraries"""