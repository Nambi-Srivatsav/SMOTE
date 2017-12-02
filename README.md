# SMOTE
SMOTE: Synthetic Minority Over-sampling Technique - Implementation and experiments on datasets

Please note that SMOTE percentage should be 100% or more. The program will exit if lesser values are passed
The datasets have been included in the folder itself.

To run the program for three datasets-
Navigate to 'smote' folder
run the following command from main.py file

Pima dataset - 		generate_smote_and_compare(filename='pima-indians-diabetes.csv', smote_percentage=100)
Phoneme dataset-	generate_smote_and_compare(filename='phoneme.csv', smote_percentage=200)
Forest cover -   	generate_smote_and_compare(filename='covtype.csv', smote_percentage=300)