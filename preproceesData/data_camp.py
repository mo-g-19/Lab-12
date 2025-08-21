#Taking notes from https://www.datacamp.com/blog/data-preprocessing
#I'm putting Best Practices first because that's the bigger takeaway to remember

"""What is Data Preprocessing
    refers any processing applied raw data -> ready future analysis or processing tasks
    prelininary step in data analysis + recently -> train ML and AI models -> inferences
"""

"""Best Practices for Data Preprocessing
    Understand the data
        Conduct exploratory data anaylsis to identify strutucture data at hand.
            Specifically:
                Key features
                Potential anomalies
                Relationships
    Automate repetitive steps
        Often involves repetitive steps.
            Automate:
                build pipelines -> consistency + efficiency (reduce manual errors)
                Use scikit-learn or cloud-based
    Document preprocessing steps
        Two objectives: Reproducibility + Understanding (later date or others)
    Iterative improvements
        Should be iterative process => as models evolve + provie feedback performance => revisit + refine -> better results
            Instance, feature engineeing => new useful features or tuning outlier handling -> improve model accuracy
"""

"""Steps in Data Prepocessing
Step 1: Data cleaning
    Correct errors/inconsistencies => accurate and complete
        Ex:
            