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