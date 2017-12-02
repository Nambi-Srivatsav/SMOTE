
import numpy as np
import matplotlib.pyplot as plt
from numpy import genfromtxt
import pdb
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import random
from sklearn.model_selection import KFold
import collections, numpy
from scipy.spatial import ConvexHull
from sklearn.neighbors import NearestNeighbors
import pdb
#generates synthetic samples
#input samples- minority class samples 
#smote_percentage - >100   
#k= nearest neighbors for SMOTE
def generate_samples(samples, smote_percentage=100, k=5):
    if smote_percentage < 100:
        print('min SMOTE percentage should be 100')
        exit(0)
    number_of_samples, number_of_features = samples.shape
    synthetic_points = np.zeros((number_of_samples*int(smote_percentage/100), number_of_features))
    nearest_k = NearestNeighbors(n_neighbors=k).fit(samples)
    N_indexes = np.random.permutation(number_of_samples*int(smote_percentage/100))%number_of_samples
    syn_sample_index = 0
    
    for i in N_indexes:
        neighbors_set = nearest_k.kneighbors([samples[i]])[1].reshape(-1)
        picked_neighbor = np.random.randint(0, k)
        while neighbors_set[picked_neighbor] == i: picked_neighbor = np.random.randint(0, k)
        difference = samples[neighbors_set[picked_neighbor]] - samples[i]
        noise = np.random.rand(1,number_of_features)
        synthetic_points[syn_sample_index] = samples[i] +  difference.reshape(1,number_of_features) * noise
        syn_sample_index += 1            

    return synthetic_points
#Reads input file name and calls generate_samples() to generate synthetic samples
#input filename- dataset filename present in the folder
#smote_percentage - >100           
def generate_smote_and_compare(filename, smote_percentage=100):
    Total_samples = genfromtxt(filename, delimiter=',')
    
    if(filename=='covtype.csv'):                      # modifications for forest cover dataset
    
        test_samples_zero = Total_samples[ Total_samples[:,-1] == 3 ]
        test_samples_ones = Total_samples[ Total_samples[:,-1] == 4 ]
        
        test_samples = np.concatenate((test_samples_ones,test_samples_zero))
        
        Total_samples = test_samples
        
        class_count_dict =  collections.Counter(Total_samples[:,-1])
        minor_class =  min(class_count_dict, key=class_count_dict.get)
        
        Total_samples[ Total_samples[:,-1] != minor_class,-1 ] = 0
        Total_samples[ Total_samples[:,-1] == minor_class,-1 ] = 1
    
    all_samples = Total_samples[Total_samples[:,-1] > 0.5]
    k = 5
    synth = generate_samples(all_samples,smote_percentage,k=k)
    np.random.shuffle(Total_samples)
    Total_samples = np.round(Total_samples)

    Total_X = Total_samples[:,:-2]
    Total_Y = Total_samples[:,-1]


    train_size =  int(np.round(Total_X.shape[0]*0.7)) 
    test_size = Total_X.shape[0] - train_size

    train_X = Total_X[:train_size]
    train_Y = Total_Y[:train_size]

    test_X = Total_X[train_size:]
    test_Y = Total_Y[train_size:]


    ####################### Guassian Naive Bayes#######################################
    # 10 fold validation of Naive Bayes classifier
    kf = KFold(n_splits=10)
    kf.get_n_splits(Total_X)

    fprs = []
    tprs = []

    fprs_means = []
    tprs_means = []

    prior_values = []
    #priors for Naive Bayes classifier
    for i in range(1,50):
        prior_values.append([1/(i+1),1-1/(i+1)])
        

    for i in range(len(prior_values)):
        for train_index, test_index in kf.split(Total_X):
           
            X_train, X_test = Total_X[train_index], Total_X[test_index]
            Y_train, Y_test = Total_Y[train_index], Total_Y[test_index]
            clf = GaussianNB(priors=prior_values[i])
            clf.fit(train_X, train_Y)
            predicted_Ys = clf.predict(X_test)
            false_positive_rate, true_positive_rate, thresholds = roc_curve(Y_test, predicted_Ys)
            false_positive_rate *= 100
            true_positive_rate *= 100
            fprs.append(false_positive_rate[1])
            tprs.append(true_positive_rate[1])
        fprs_mean = np.mean(fprs)
        fprs_means.append(fprs_mean)
        tprs_mean = np.mean(tprs)
        tprs_means.append(tprs_mean)



    # C4.5 Decision Tree 
    #
    ctotal_samples = Total_samples
    all_positive_samples = ctotal_samples[ctotal_samples[:,-1] > 0.5]
    all_negative_samples = ctotal_samples[ctotal_samples[:,-1] < 0.5]

    number_of_positive_samples = all_positive_samples.shape[0]
    number_of_negative_samples = all_negative_samples.shape[0]
    #under sampling of majority class with various percentages 
    all_percentages = [10,15,25,50,75,100,125,150,175,200,300,400,500,600,700,800,1000,2000]

    samples_to_be_taken = []

    for i in range(len(all_percentages)):
        samples_number = int(number_of_positive_samples*100/all_percentages[i])
        if(samples_number > 0 and samples_number < number_of_negative_samples):
            samples_to_be_taken.append(samples_number)


    fprs_undersampling = []
    tprs_undersampling = []

    undersampling_values = []
    auc_array_0 = []
    for i in range(len(samples_to_be_taken)):
        zero_samples_p = all_negative_samples[:samples_to_be_taken[i]]
        data_with_some_percent = np.concatenate((zero_samples_p,all_positive_samples))
        np.random.shuffle(data_with_some_percent)
        data_with_some_percent = np.round(data_with_some_percent)


        Total_X = data_with_some_percent[:,:-2]
        Total_Y = data_with_some_percent[:,-1]


        train_size =  int(np.round(Total_X.shape[0]*0.7)) # 70% training data
        test_size = Total_X.shape[0] - train_size

        train_X = Total_X[:train_size]
        train_Y = Total_Y[:train_size]

        test_X = Total_X[train_size:]
        test_Y = Total_Y[train_size:]

        clf = DecisionTreeClassifier(random_state=9)
        clf.fit(train_X, train_Y)

        predicted_Ys = clf.predict(test_X)
        false_positive_rate, true_positive_rate, thresholds = roc_curve(test_Y, predicted_Ys)
        false_positive_rate *= 100
        true_positive_rate *= 100
        roc_auc = auc(false_positive_rate, true_positive_rate)
        auc_array_0.append(roc_auc)
        fprs_undersampling.append(false_positive_rate[1])
        tprs_undersampling.append(true_positive_rate[1])
    
    print("Maximum AUC for undersampling:",round(max(auc_array_0)))
    tprs_np = np.resize(np.array(tprs_undersampling),(13,1))
    fprs_np = np.resize(np.array(fprs_undersampling),(13,1))
    only_undersampling = np.concatenate((tprs_np,fprs_np),axis=1)
    only_undersampling = only_undersampling[np.argsort(only_undersampling[:, 1])]

    print("\n\n\n")
    print('No of synthetic Samples added: ', synth.shape[0])
    print("\n\n\n")

    ########################### After adding synthetic Data ######################

    Total_samples = np.concatenate((synth, Total_samples), axis=0)

    ctotal_samples = Total_samples

    all_positive_samples = ctotal_samples[ctotal_samples[:,-1] > 0.5]
    all_negative_samples = ctotal_samples[ctotal_samples[:,-1] < 0.5]

    number_of_positive_samples = all_positive_samples.shape[0]
    number_of_negative_samples = all_negative_samples.shape[0]

    #under sampling of majority class with various percentages for SMOTEd dataset
    all_percentages = [10,15,25,50,75,100,125,150,175,200,300,400,500,600,700,800,1000,2000]

    samples_to_be_taken = []

    for i in range(len(all_percentages)):
        samples_number = int(number_of_positive_samples*100/all_percentages[i])
        if(samples_number > 0 and samples_number < number_of_negative_samples):
            samples_to_be_taken.append(samples_number)

          
    fprs_smote_under = []
    tprs_smote_under = []
    
    auc_array = []
    for i in range(len(samples_to_be_taken)):
        zero_samples_p = all_negative_samples[:samples_to_be_taken[i]]
        data_with_some_percent = np.concatenate((zero_samples_p,all_positive_samples))
        data_with_some_percent = np.concatenate((data_with_some_percent,synth))
        np.random.shuffle(data_with_some_percent)
        data_with_some_percent = np.round(data_with_some_percent)


        Total_X = data_with_some_percent[:,:-2]
        Total_Y = data_with_some_percent[:,-1]


        train_size =  int(np.round(Total_X.shape[0]*0.7)) # 70% training data
        test_size = Total_X.shape[0] - train_size

        train_X = Total_X[:train_size]
        train_Y = Total_Y[:train_size]

        test_X = Total_X[train_size:]
        test_Y = Total_Y[train_size:]

        clf = DecisionTreeClassifier(random_state=9)
        clf.fit(train_X, train_Y)

        predicted_Ys = clf.predict(test_X)
        false_positive_rate, true_positive_rate, thresholds = roc_curve(test_Y, predicted_Ys)
        false_positive_rate *= 100
        true_positive_rate *= 100
        roc_auc = auc(false_positive_rate, true_positive_rate)
        auc_array.append(roc_auc)
        fprs_smote_under.append(false_positive_rate[1])
        tprs_smote_under.append(true_positive_rate[1])
    
    
    tprs_np = np.resize(np.array(tprs_smote_under),(13,1))
    fprs_np = np.resize(np.array(fprs_smote_under),(13,1))
    smote_undersampling = np.concatenate((tprs_np,fprs_np),axis=1)
    smote_undersampling = smote_undersampling[np.argsort(smote_undersampling[:, 1])]
    
    print("Maximum AUC for SMOTE:",round(max(auc_array)))

    fprs_np_means = np.resize(fprs_means,(len(fprs_means),1))
    tprs_np_means = np.resize(tprs_means,(len(tprs_means),1))
    
    all_points = np.concatenate((tprs_np_means,fprs_np_means),axis=1)
    all_points = np.concatenate((all_points,only_undersampling))
    all_points = np.concatenate((all_points,smote_undersampling))
    
    hull = ConvexHull(all_points)
    
    
   
    plt.plot(only_undersampling[:,1],only_undersampling[:,0],label="Under−C4.5")
    plt.plot(np.array(fprs_means),np.array(tprs_means),label="Naive Bayes")
    plt.plot(smote_undersampling[:,1],smote_undersampling[:,0],label=str(smote_percentage)+" SMOTE−C4")
    plt.plot(all_points[hull.vertices,1],all_points[hull.vertices,0],label="ConvexHull",linestyle='dotted',color='0.35')
    plt.ylabel('TP%')
    plt.xlabel('FP%')
    plt.ylim( (50, 100) )
    plt.xlim((0,100))
    
    plt.legend(numpoints=1)
    plt.show()
    
