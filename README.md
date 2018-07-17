SMOTE
==========

SMOTE: Synthetic Minority Over-sampling Technique - Implementation and experiments on datasets

Precondition
------------

Please note that SMOTE percentage should be 100% or more. The program will exit if lesser values are passed
The datasets have been included in the folder itself.

Usage
------------

To run the program for three datasets-
Navigate to 'smote' folder and please run the following command.

```python
python main.py
```


Configuration
------------

Datasets are already present in the folder. To select respective dataset, please have one of the following lines in main function.

```python
generate_smote_and_compare(filename='pima-indians-diabetes.csv', smote_percentage=100) ## For Pima dataset 		
generate_smote_and_compare(filename='phoneme.csv', smote_percentage=200) ## For Phoneme dataset
generate_smote_and_compare(filename='covtype.csv', smote_percentage=300) ## For Forest Cover dataset
```