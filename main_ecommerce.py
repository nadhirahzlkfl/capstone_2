#%% import packages
from keras import layers,losses,callbacks
from sklearn import preprocessing,model_selection,metrics
from nltk.corpus import stopwords
import pickle,mlflow,keras,nltk,os
import pandas as pd 
import numpy as np 

os.environ['KERAS_BACKEND']='tensorflow'
print(keras.backend.backend())
nltk.download('stopwords')

#%% data loading
csv_path=os.path.join(os.getcwd(),'dataset','ecommerceDataset.csv')
df=pd.read_csv(csv_path,header=None)
#%% EDA
print(df.head()) # 1st column: category, 2nd column: text
#%%
print(df.info()) # 50425 entries in category column, 50424 entries in text column
#%%
print(df[0][0]) # category
print(df[1][0]) # text
nClass=len(df[0].unique()) # 4 categories
print('Categories: ',df[0].unique()) # category: 'Household', 'Books', 'Clothing & Accessories', 'Electronics'
print('Number of categories: ',nClass)

#%% inspect data
# check missing value
print('Missing values:\n',df.isna().sum()) # missing value in text/2nd column
# fill missing value
df[1]=df[1].fillna('none') 
# check missing value again
print('Missing values:\n',df.isna().sum()) 
print(df.info()) # both column now has same count of entries

#%% check duplicated value
print('Duplicates: ',df.duplicated().sum()) # there are 22622 duplicated values
# test by creatig a new copy of df
df_copy=df.copy()
df_copy=df_copy.drop_duplicates() # remove duplicates of df copy
print(df_copy.info()) # 27803 entries after remove duplicates
# check count after remove duplicates
print('Label distribution from original:\n',df[0].value_counts()) 
print('Label distribution after removing duplicates:\n',df_copy[0].value_counts())

#%% feature and label selection
label=df[0].values # category
feature=df[1].values # text

#%% encode the label
label_encoder=preprocessing.LabelEncoder()
label_encoded=label_encoder.fit_transform(label)
print(np.unique(label_encoded))
print(label_encoder.inverse_transform(np.unique(label_encoded)))

#%% data splitting
seed=42
X_train,X_split,y_train,y_split=model_selection.train_test_split(feature,label_encoded,train_size=0.7,random_state=seed)
X_val,X_test,y_val,y_test=model_selection.train_test_split(X_split,y_split,train_size=0.5,random_state=seed)

#%% perform NLP processes
# tokenization
vocab_size=5000
tokenizer=layers.TextVectorization(max_tokens=5000,output_sequence_length=250)
tokenizer.adapt(X_train) # fit with X_train
# test out tokenizer
print(tokenizer.get_vocabulary())
sample_texts=X_train[:2]
sample_tokens=tokenizer(sample_texts)
print(sample_texts[0])
print(sample_tokens[0])
# embedding
embedding=layers.Embedding(input_dim=vocab_size,output_dim=64)
# test the embedding layer
sample_embedding=embedding(sample_tokens)
print(sample_embedding[0])

# create the keras model 
model=keras.Sequential()
# add NLP layers
model.add(tokenizer)
model.add(embedding)
# add RNN layers 
model.add(layers.Bidirectional(layers.LSTM(16,return_sequences=False)))
model.add(layers.Dense(nClass,activation='softmax'))

# model compilation
loss=losses.SparseCategoricalCrossentropy()
model.compile(optimizer='adam',loss=loss,metrics=['accuracy'])

print(X_test)

# setup mlflow
# setup experiement
exp=mlflow.set_experiment('Ecommerce Text Classification')
# train model
with mlflow.start_run() as run:
    mlflow_callback=mlflow.keras.MlflowCallback(run)
    es=callbacks.EarlyStopping(patience=2,verbose=1)
    run_id=run.info.run_id
    log_path=f'logs/{run_id}'
    tb=callbacks.TensorBoard(log_dir=log_path)
    history=model.fit(X_train,y_train,validation_data=(X_val,y_val),epochs=10,batch_size=16,callbacks=[mlflow_callback,es,tb])
    mlflow.keras.save.log_model(model,artifact_path='model')

# evaluate the model with test data
print(model.evaluate(X_test,y_test))

# use the model to make prediction
y_pred=model.predict(X_test)
print(y_pred[0])
y_pred=np.argmax(y_pred,axis=1) # to find maximum value
print(y_pred[0])
print(metrics.classification_report(y_test,y_pred))
y_pred=label_encoder.inverse_transform(y_pred)
print(y_pred[0])

#%% save encoder 
with open('encoder.pkl','wb') as f:
    pickle.dump(label_encoder,f)
# save as artifact in mlflow run
mlflow.log_artifact('encoder.pkl',run_id='59a65599a8554a63a871ef19d5641bf9')
# %%
