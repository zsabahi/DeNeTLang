# -*- coding: utf-8 -*-
"""
Created on Sun Oct 27 11:34:07 2019

@author: Zeynab Sabahi
"""

# =============================================================================
# Import Packages
# =============================================================================

import sys,os,time
import pandas as pd
import datetime
import numpy as np
import matplotlib.pyplot as plt
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense
import itertools
from sklearn.metrics import confusion_matrix , classification_report
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical

from configurations import tracesdir,DSName
from sklearn.metrics import accuracy_score

import seaborn as sn


def MLPNetwork(prefix,filenameArr):
    # =============================================================================
    # Config Params
    # =============================================================================
    # EPOCH_NUMS = 5
    # BATCH_SIZE = 15

    # =============================================================================
    # Load Data
    # =============================================================================
    
    #from keras.datasets import mnist
    #(X_train, y_train), (X_test, y_test) = myLoadCSV(,0.70,10)
    # path='C:/Users/Lenovo/Desktop/PHD/Bs.CPrjs/Miss.AliMadadi/formallab/DataSets/recon-arff/recon99ed.csv'
    # X = pd.read_csv(path)
    # path2='C:/Users/Lenovo/Desktop/PHD/Bs.CPrjs/Miss.AliMadadi/formallab/DataSets/recon-arff/labels99.csv'
    # y = pd.read_csv(path2)
    
    #configggg
    # prefix = tracesdir + "/Application/S500_F5_D5_Stats1_Cl1/100/DL/"
    train_path=prefix+'/train.csv'
    train_label_path=prefix+'/trainLabel.csv'
   
    test_path=prefix+'/test.csv'
    test_label_path=prefix+'/testLabel.csv'
    # test_path=prefix+'/./mix/mixed.csv'
    # test_label_path=prefix+'/mix/mixed2Label.csv'
    
    # mix_path=tracesdir+'/mixed.csv'
    # mix_label_path=tracesdir+'/mixed2Label.csv'
    
    # merged_path=tracesdir+'/merged.csv'
    # merged_label_path=tracesdir+'/mergedLabel.csv'
    
    # with open(test_path) as fp:
    #     data = fp.read()
  
    # # Reading data from file2
    # with open(mix_path) as fp:
    #     data2 = fp.read()
      
    # data += data2
      
    # with open (merged_path, 'w') as fp:
    #     fp.write(data)
    
    # with open(test_label_path) as fp:
    #     data = fp.read()
  
    # # Reading data from file2
    # with open(mix_label_path) as fp:
    #     data2 = fp.read()
      
    # data += data2
      
    # with open (merged_label_path, 'w') as fp:
    #     fp.write(data)
    
    
    x_train = pd.read_csv(train_path)
    y_train = pd.read_csv(train_label_path)
   
    # x_test = pd.read_csv(merged_path)
    # y_test = pd.read_csv(merged_label_path)
    x_test = pd.read_csv(test_path)
    y_test = pd.read_csv(test_label_path)
   
   
    
    # x_test =x_test.append(x_mix)
    # y_test =y_test.append(y_mix)
    
    # Checkout the Data
    print('Training data shape : ', x_train.shape, y_train.shape)
    print('Testing data shape : ', x_test.shape, y_test.shape)
    
    
    # Find the unique numbers from the train labels
    classes = np.unique(y_train)
    nClasses = len(classes)
    # print('Total number of outputs : ', nClasses)
    #print('Output classes : ', classes)
    
    #y = to_categorical(y)
    
    # from sklearn.decomposition import PCA
    # pca = PCA(n_components=49)
    # X = pca.fit_transform(X)
    
    
    # X_train = x_train.to_numpy()
    # X_test = x_test.to_numpy()
    
    # from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
    
    # #configggg
    # lda = LDA(n_components=nClasses-1)
    # X = np.concatenate((x_train.to_numpy(),x_test.to_numpy()))
    # y = np.concatenate((y_train.to_numpy(),y_test.to_numpy())) 
    
    X = x_train.to_numpy()
    y = y_train.to_numpy() 
    
    
    
    # Checkout the Data
    # print('Training data shape : ', x_train.shape, y_train.shape)
    # print('Testing data shape : ', x_test.shape, y_test.shape)
    # print('X,y data shape : ', X.shape, y.shape)
    # print(X,y) 
    # start_time = time.time()
    # X = lda.fit_transform(X,y.ravel())
    # elapsed_time = time.time() - start_time
    # print("time for lda: "+time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))
    
    # print('X,y data shape : ', X.shape, y.shape)
    # [X_train,X_test] = np.split(X,[x_train.shape[0]],axis=0)
    X_train = X
    # X_test = lda.transform(x_test.to_numpy())
    X_test = x_test.to_numpy()

    # print('X_train,X_test data shape : ', X_train.shape, X_test.shape)
    
    # #configggg
    # resultfile = open('C:/Users/Lenovo/Desktop/lda.csv', 'a')
    # resultfile.write(str(X))
    # resultfile.close()
    
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42,stratify=y)
    
    # print("...............",np.unique(y_train))
    # print("...............",np.unique(X_train))
    
     #eliminate the weight => all weight should be 1
    # X_train,X_test = elmWeight(X_train,X_test)
    # print(X_train)
    # print(X_test)
    
    # X_train, X_val, Y_train, y_val = train_test_split(X_train, y_train, test_size=0.3, random_state=42,stratify=y_train)
    X_train, X_val, Y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42,stratify=y_train)
    
    
    y_train = Y_train
    

    # =============================================================================
    # preprocess
    # =============================================================================
    
    plt.figure(figsize=[10,5])
      
    # Noramlization from [0;255] to [0;1], Scale the data to lie between 0 to 1
    #configggg & authomatic find after LDA
    # X_train /= 84
    # X_test /= 7
    
    # convert labels to one-hot vectors
    Y_train = np_utils.to_categorical(y_train,num_classes=nClasses)
    Y_test = np_utils.to_categorical(y_test,num_classes=nClasses)
    Y_val = np_utils.to_categorical(y_val,num_classes=nClasses)
    
    		
    
    # print('Training data shape : ', X_train.shape, y_train.shape,Y_train.shape)
    # print('Testing data shape : ', X_test.shape, y_test.shape,Y_test.shape)
    # print('Testing data shape : ', X_val.shape, y_val.shape, Y_val.shape)
    
    
    # =============================================================================
    # Create Model
    # =============================================================================
    
    # print(X_train.shape[1])
    # print(X_test.shape[0])
          
    NNmodel = "MLP"#"MLP"#"CNN" #"MLP" "EndtoEnd"

    if ( NNmodel == "CNN"):
        
        #======= CNN Model  
        X_train=X_train.reshape(X_train.shape[0],X_train.shape[1],1,1)
        X_test=X_test.reshape(X_test.shape[0],X_test.shape[1],1,1)
        
        from keras.models import Sequential
        # from keras.utils import np_utils
        from keras import layers
        
        model = Sequential()
        model.add(layers.Conv2D(filters=64, kernel_size=2, padding='same', activation='relu', input_shape=(X_train.shape[1],1,1))) 
        model.add(layers.MaxPooling2D(pool_size=1))
        model.add(layers.Dropout(0.3))
        
        model.add(layers.Conv2D(filters=32, kernel_size=2, padding='same', activation='relu'))
        model.add(layers.MaxPooling2D(pool_size=1))
        model.add(layers.Dropout(0.3))
        
        model.add(layers.Flatten())
        model.add(layers.Dense(256, activation='relu'))
        model.add(layers.Dropout(0.5))
        model.add(layers.Dense(nClasses, activation='softmax'))
        
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        # optimizer='rmsprop'
        import tensorflow as tf
        checkpoint=tf.keras.callbacks.ModelCheckpoint(filepath="bestRes.txt", monitor='val_loss', verbose=0, save_best_only=True,
            save_weights_only=False, mode='auto', save_freq='epoch', options=None)
        lr_adjust=tf.keras.callbacks.ReduceLROnPlateau( monitor="val_loss", factor=0.5, patience=1, verbose=0, mode="auto",
            min_delta=0.00001,  cooldown=0,  min_lr=0) 
        callbacks=[checkpoint, lr_adjust]
    
        # history = model.fit(X_train, Y_train, epochs=16, batch_size=15) 	
        # history = model.fit(X_train, Y_train, batch_size=16, epochs=10, validation_split=0.2,verbose=0)

        history = model.fit(X_train, Y_train, batch_size=16, epochs=12, validation_split=0.2)


    else: 
        if ( NNmodel == "MLP"):#MLP
            #====== MLP Functional API Model
            from keras.layers.core import Dense
            from keras.models import Model
            from keras.layers import Input
            
            inputs  = Input(shape=(X_train.shape[1],))
            first   = Dense(256,activation="relu")(inputs)
            second  = Dense(128,activation="relu")(first)
            outputs = Dense(nClasses,activation="softmax")(second)
            model = Model(inputs,outputs)
            
            # #========= Batch Normalization
            # from keras.layers import BatchNormalization
            
            # inputs  = Input(shape=(X_train.shape[1],))
            # first   = Dense(256,activation="relu")(inputs)
            # batch1 =  BatchNormalization()(first)
            # second  = Dense(128,activation="relu")(batch1)
            # batch2 =  BatchNormalization()(second)
            # outputs = Dense(nClasses,activation="softmax")(batch2)
            # model = Model(inputs,outputs)
            
            #====== MLP with Sequentioal Model
            # from keras.layers.core import Dense
            # from keras.models import Model
            # from keras.layers import Input
            
            # model = Sequential()
            # model.add(Dense(512, activation='relu', input_shape=(X_train.shape[1],)))	#Hidden Layer 1
            # model.add(Dense(256, activation='relu'))	#Hidden Layer 2
            # model.add(Dense(nClasses, activation='softmax')) #Last layer with one output per class
            
            
            # model.summary()
            # Configure the Network
            model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
            # optimizer='rmsprop'
            import tensorflow as tf
            checkpoint=tf.keras.callbacks.ModelCheckpoint(filepath="bestRes.txt", monitor='val_loss', verbose=0, save_best_only=True,
                save_weights_only=False, mode='auto', save_freq='epoch', options=None)
            lr_adjust=tf.keras.callbacks.ReduceLROnPlateau( monitor="val_loss", factor=0.5, patience=1, verbose=0, mode="auto",
                min_delta=0.00001,  cooldown=0,  min_lr=0) 
            callbacks=[checkpoint, lr_adjust]
            
            
            # # ========= Early Stopping
            # from tensorflow.keras.callbacks import EarlyStopping
            # # early_stopping = EarlyStopping()
            # custom_early_stopping = EarlyStopping(
            #     monitor='val_accuracy', 
            #     patience=8, 
            #     min_delta=0.001, 
            #     mode='max'
            # )
            
            # =============================================================================
            # Train Model (Fitting)
            # =============================================================================
            
            # history = model.fit(X_train, Y_train, epochs=16, batch_size=15) 	
            # history = model.fit(X_train, Y_train, epochs=16, batch_size=15, validation_split=0.2,  callbacks=[custom_early_stopping])
            # history = model.fit(X_train, Y_train, epochs=EPOCH_NUMS, batch_size=BATCH_SIZE, validation_data=(X_test, Y_test)) 	
            # history = model.fit(X_train, Y_train, epochs=16, batch_size=30, validation_data=(X_val, Y_val),callbacks=[custom_early_stopping]) 	
            
            start_time = time.time()
        
            history = model.fit(X_train, Y_train, epochs=16, batch_size=16, validation_data=(X_val, Y_val), verbose=0) 	
        
            elapsed_time = time.time() - start_time
            print("time for train dl: "+time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))
       
        else:
            if ( NNmodel == "EndtoEnd"):
                 #======= CNN Model  
                X_train=X_train.reshape(X_train.shape[0],X_train.shape[1],1)
                X_test=X_test.reshape(X_test.shape[0],X_test.shape[1],1)
                
                from keras.models import Sequential
                # from keras.utils import np_utils
                from keras import layers
                
                model = Sequential()
                model.add(layers.Conv1D(filters=25,  kernel_size=2, strides=1, padding='same', activation='relu', input_shape=(X_train.shape[1],1))) 
                model.add(layers.MaxPooling1D(  strides=3))
                
                model.add(layers.Conv1D(filters=25,  kernel_size=2, strides=1, padding='same', activation='relu')) 
                model.add(layers.MaxPooling1D(  strides=3))
                model.add(layers.Flatten())
                model.add(layers.Dense(64))
                model.add(layers.Dense(1024))
                
                model.add(layers.Dense(nClasses, activation='softmax'))
                
                model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
                # optimizer='rmsprop'
                import tensorflow as tf
                checkpoint=tf.keras.callbacks.ModelCheckpoint(filepath="bestRes.txt", monitor='val_loss', verbose=0, save_best_only=True,
                    save_weights_only=False, mode='auto', save_freq='epoch', options=None)
                lr_adjust=tf.keras.callbacks.ReduceLROnPlateau( monitor="val_loss", factor=0.5, patience=1, verbose=0, mode="auto",
                    min_delta=0.00001,  cooldown=0,  min_lr=0) 
                callbacks=[checkpoint, lr_adjust]
            
                # history = model.fit(X_train, Y_train, epochs=16, batch_size=15) 	
                # history = model.fit(X_train, Y_train, batch_size=16, epochs=10, validation_split=0.2,verbose=0)
        
                history = model.fit(X_train, Y_train, batch_size=16, epochs=12, validation_split=0.2)
    #after if else nn model
    # np.save("C:\\Users\\Lenovo\\Desktop\\history_"+DSName+".npy",history.history)    
    np.save("C:\\Users\\Lenovo\\Desktop\\history_ISCX_TrC_npy",history.history)    

    
    diag_name = "UT (AI)"
    # Plotting Metrics
    # Plot the Accuracy Curves
    fig = plt.figure()
    plt.plot(history.history['accuracy'],'r')
    plt.plot(history.history['val_accuracy'],'b')
    plt.title(diag_name)
    plt.ylabel('model accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='lower right')
    plt.grid()
    
    # Plot the Loss Curves
    fig = plt.figure()
    plt.plot(history.history['loss'],'r')
    plt.plot(history.history['val_loss'],'b')
    plt.title(diag_name)
    plt.ylabel('model loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper right')
    plt.grid()
    
    # =============================================================================
    # Evaluation
    # =============================================================================
    
    # Prediction Labels
    # start_time = time.time()

    Y_pred = model.predict(X_test)
    # elapsed_time = time.time() - start_time
    # print("time for dl test: "+time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))
    
    y_pred = np.argmax(Y_pred, axis=1)
    y_test = y_test.to_numpy()
    # print( y_pred.shape)
    y_pred = y_pred.reshape(X_test.shape[0],1)
    # print( y_pred.shape)
    # print( y_test.shape)
    #print(Y_pred, Y_pred.shape)
    
    
    # Evaluate the trained model
    [test_loss, test_acc] = model.evaluate(X_test, Y_test)
    print("Test Loss", test_loss)
    print("Test Accuracy", test_acc)
    
    
    # print("y_pred == y_test",y_pred == y_test)
    # print(np.nonzero(y_pred == y_test))
    
    correct_indices = np.nonzero(y_pred == y_test)[0]
    incorrect_indices = np.nonzero(y_pred != y_test)[0]
    
    # print(" classified correctly", len(correct_indices))
    # print(" classified incorrectly", len(incorrect_indices))
    
    #figure = plt.figure(figsize=(20, 8))
    #for i, index in enumerate(np.random.choice(X_test.shape[0], size=15, replace=False)):
        #ax = figure.add_subplot(3, 5, i + 1, xticks=[], yticks=[])
        ## Display each image
        #ax.imshow(X_test[index].reshape(28,28), cmap='gray')
        #predict_index = y_pred[index]
        #true_index = y_test[index]
        ## Set the title for each image
        #ax.set_title("{} ({})".format(predict_index, 
         #                             true_index),
         #                             color=("black" if predict_index == true_index else "red"))
    
    
    
    confusion_mtx = confusion_matrix(y_test, y_pred)
    # print("confusion matrix=\n",confusion_mtx)

    # from scipy.cluster import hierarchy
    # Y = hierarchy.distance.pdist(np.asmatrix(confusion_mtx), metric='euclidean')
    # Z = hierarchy.linkage(Y, method='single')
    # ax = hierarchy.dendrogram(Z, show_contracted=True, labels=filenameArr, leaf_rotation=0, orientation="left", color_threshold='default', above_threshold_color='grey', leaf_font_size=9)
    # plt.title("test")
    # plt.show()
    
    # df_cm = pd.DataFrame(confusion_mtx, index = filenameArr,  columns = filenameArr)
    # df_conf_norm = df_cm / df_cm.sum(axis=1)
    # plt.figure(figsize = (10,7))
    # sn.heatmap(df_conf_norm, cmap=plt.cm.Blues)#, annot=True)#,
    # plt.savefig(tracesdir+'/CM_'+DSName+'.png')    # CM_Class
    # plt.close("all")  
    
    
    # plot_confusion_matrix(confusion_mtx, filenameArr) 
    # plt.savefig(tracesdir+'cm.png', bbox_inches='tight')
    plt.savefig(tracesdir+'/cm'+DSName+'.png')

    print(classification_report(y_test,y_pred))

    cm_dict = classification_report(y_test,y_pred,output_dict=True)
    df,acc = buildResultsTable(cm_dict,classes,filenameArr)
    return df,acc,nClasses

  
def plot_confusion_matrix(cm, classes):
    df_cm = pd.DataFrame(cm, index = classes, columns = classes)
    df_conf_norm = df_cm / df_cm.sum(axis=1)
    plt.figure(figsize = (10,7))
    plt.xticks(fontsize=18) 
    plt.yticks(fontsize=18) 

    sn.heatmap(df_conf_norm, cmap=plt.cm.Blues)#, annot=True)#,
    # plt.savefig(diagramdir+"/CM_App.png")    # CM_Class
    # plt.close("all")  
    
    
# def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix'):
    
#     """
#     This function prints and plots the confusion matrix.
#     Normalization can be applied by setting `normalize=True`.
#     """
    
#     cmap=plt.cm.Blues
#     plt.figure(figsize = (5,5))
#     plt.matshow(cm, interpolation='nearest', cmap=cmap)
#     # plt.title(title)
#     plt.colorbar()
#     tick_marks = np.arange(len(classes))
#     plt.xticks(tick_marks, classes,rotation=90, fontsize = 12)
#     plt.yticks(tick_marks, classes, fontsize = 12)
#     if normalize:
#         cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

#     thresh = cm.max() / 2.
#     for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
#         plt.text(j, i, cm[i, j],
#                  horizontalalignment="center",
#                  color="white" if cm[i, j] > thresh else "black")
#     plt.tight_layout()
#     # plt.ylabel('True label')
#     # plt.xlabel('Predicted label')
    

def buildResultsTable(cm_dict,classes,filenameArr):
    
    # os.makedirs (outputdir,exist_ok=True)
    # resultfile = open(outputdir+"/all_k_results_"+date+".txt", 'a')
    # resultfile.write("Network Settings = "+networksettings+", K = "+str(k_window)+"\n")              

    acc= round(cm_dict.get('accuracy'),2)
    result =[]
    for app in cm_dict.keys():    
        if app == 'accuracy':
            break
        else:
            # newRow = [filename,avgRecalls,avgPr,avgF1s,avgAcc,predatasetLengths[filename],datasetLengths[filename][0],datasetLengths[filename][1],len(testArr[filename])]
            newRow = [filenameArr[int(app)],round(cm_dict.get(app)['recall'],2),round(cm_dict.get(app)['precision'],2),round(cm_dict.get(app)['f1-score'],2),acc,cm_dict.get(app)['support']]
            # newRow = [app,round(cm_dict.get(app)['recall'],2),round(cm_dict.get(app)['precision'],2),round(cm_dict.get(app)['f1-score'],2),round(cm_dict.get('accuracy'),2),cm_dict.get(app)['support']]
            result.append(newRow)

    
    # resultfile.write("-----------------------------------------------------------------------------------------------------\n")
    # resultfile.close()
    
    # conflictfile = open(outputdir+'/conflicts_'+date+'.txt', 'a')
    # conflictfile.write("Network Settings = "+networksettings+", K = "+str(k_window)+"\n")
    # conflictfile.write("-----------------------------------------------------------------------------------------------------\n")
    # conflictfile.close()

    # df = pd.DataFrame(data=result,columns=['appName','recall','pr','f1','acc','tr','newTr','train-chunks','test-chunks'])
    df = pd.DataFrame(data=result,columns=['appName','recall','pr','f1','acc','test-chunks'])

    # print('***************************')
    # print(df)
    return df,acc

def elmWeight(x_train,x_test):
    
    i=0
    j=0
    for i in range(len(x_train)):
        for j in range (len(x_train[i])):
            # print("---",j)

            if x_train[i][j] > 0:
                x_train[i][j] = 1
            else:
                x_train[i][j] = 0
            j+=1
        i+=1
        
    i=0
    j=0
    for i in range(len(x_test)):
        for j in range (len(x_test[i])):
            # print("--9999-",i)

            if x_test[i][j] > 0:
                x_test[i][j] = 1
            else:
                x_test[i][j] = 0
            j+=1
        i+=1
      
    print("-------------------")

    return x_train, x_test
        

def main(argv):
    

    # pathdir = tracesdir + "/Application/S100001_F4.6_D5.5/100_98" #unb3
    # pathdir = tracesdir + "/Application/S100002_F4.6_D5.5/100_93" #unb2_class
    # pathdir = tracesdir + "/Application/S100001_F4.6_D5.5/100_99"  #UT
    # pathdir = tracesdir + "/Application/S100001_F4.6_D5.5/100_99" #UT_class
    pathdir = tracesdir + "/Application/S100001_F0.1_D0.1/100" #UT_class

    
    traindir = pathdir + "/Train"

    filenameArr = []
    for file in os.listdir(traindir):
        filename = file.split(".txt")[0] 
        print(filename)
        filenameArr.append(filename)
    # print("len: "+str(len(filenameArr))+"\n")

    csvdir = pathdir + "/DL_sp3_k3"
    # MLPNetwork(csvdir,{"Chat","Email","FileTransfer","Streaming","Torrent","VoIP","VPNchat","VPNemail","VPNfile","VPNstreaming","VPNtorrent","VPNvoIP"})
    MLPNetwork(csvdir,filenameArr)


    # efficeincy, acc,nClasses = MLPNetwork(csvdir,filenameArr)
    # efficeincy.to_excel(writer, sheet_name="R_%s_sp%s_k%s"%(networksettings, sp,k_window), index=False)


# {"AIMchat","Email","Facebook","FTPS","Gmail","Hangouts","ICQ","Netflix","SCP","SFTP","Skype","Spotify","Tor","Torrent","Vimeo","Voipbuster","Youtube"}
    

if __name__ == "__main__":
    main(sys.argv[1:])
    
