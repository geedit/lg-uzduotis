import pandas as pd
import numpy as np
from sklearn import preprocessing
import xgboost as xgb
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
from datetime import date
from dateutil.relativedelta import relativedelta
from sklearn.multioutput import MultiOutputRegressor


##### helper functions

def data_load(file):
    if file.endswith(".xlsx"):
        data = pd.read_excel(file)
        data['Date'] = pd.to_datetime(data['Date'], format="%Y-%m-%d")
        data = data.drop('Unnamed: 0', axis=1)
        data.dropna(inplace = True)
        return data
    else:
        raise Exception("Bad file format. Try using .xlsx")

def sequences_generator(sequence, seq_len = 13, step=1):
    sequences = []
    next_sequences = []
    for i in range(0, len(sequence) - seq_len, step):
        sequences.append(sequence[i: i + seq_len])
        next_sequences.append(sequence[i + seq_len])
    return np.array(sequences), np.array(next_sequences)

def sequences_generator_future(sequence, seq_len = 13, step=1):
    sequences = []
    for i in range(0, len(sequence) - seq_len, step):
        sequences.append(sequence[i+1: i + seq_len+1])
    return np.array(sequences)

####
    
class Predict_model():
    
    def __init__(self):
        self.model = xgb.XGBRegressor(n_estimators=1000, seed=1234, booster = 'gblinear')
        
    def train_predict_one(self, file):
        
       # if file == 'data\Total-All.xlsx':
       #     self.columns = ['Augalines kilmes produktai', 'Kietasis mineralinis kuras',
       #                    'Maisto pramones produktai', 'Mediena, kametiena',
       #                    'Nafta ir naftos produktai']  
       #     self.model = MultiOutputRegressor(self.model)
       # else:
        
        self.columns = 'Total'     
        self.file = file
        self.data = data_load(file)
        print(f'Using - {self.file}')
        
        x, y = sequences_generator(self.data[self.columns].values)
        
        self.split_size = 0.11
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(x, y, 
                                                                                test_size = self.split_size, 
                                                                                shuffle=False, 
                                                                                random_state=1234)
        
        
        self.model.fit(self.x_train,self.y_train)
        
        model_results = self.model.predict(self.x_test)
        
        self.starting_date = (self.data.Date.iloc[ - len(self.y_test)]).date()
        self.end_date = (self.data.Date.iloc[ - 1]).date()
        
        print(f'Test data range from {self.starting_date} to {self.end_date}')
        print(f'Testing using {self.split_size*100} % of total dataset')
        print(f'Mean absolute error - {mean_absolute_error(self.y_test, model_results)}')
        print(f'Average real price during this period - {np.mean(self.y_test)}')
        
        plt.plot(model_results, label="Predicted")
        plt.plot(self.y_test, label="Real")
        plt.title('Testing')
        plt.legend(loc="upper left")
        plt.show()
        
        print('#####')
        
    def future_predict(self):
        all_data = self.data[self.columns].values
        
        self.prediction = []
        self.prediction_date = []
        idx = 1
        
        for i in range(12):
            x = sequences_generator_future(all_data)
            pred = np.round_(self.model.predict(np.array([x[-1]])))
            date_pred = (self.end_date + relativedelta(months=+idx))
            
            idx += 1
            
            all_data = np.append(all_data, pred)
            self.prediction.append(pred[0])
            self.prediction_date.append(date_pred)
            
            print(f'{date_pred} predicted - {pred[0]}')
            
        plt.plot(all_data, label="Predicted")
        plt.plot(self.data['Total'].values, label="Real")
        plt.title('Prediction')
        plt.legend(loc="upper left")
        plt.show()
        print('#####')
        
    def save_future_prediction(self):
        print(f'Using - {self.file}')
        name = input("Enter file name: ") 
        
        file_to_save = pd.DataFrame({'Date': self.prediction_date,
                                     'Total': self.prediction})
        file_to_save.to_excel(f'data\ {name}.xlsx')
        
        print(f'File {name}.xlsx was saved in data folder')
        print('#####')
        
if __name__ == "__main__":
    file = 'data\\Total-Augalines kilmes produktai.xlsx'
 #   file = 'data\\Total-Kietasis mineralinis kuras.xlsx'
 #   file = 'data\\Total-Maisto pramones produktai.xlsx'
 #   file = 'data\\Total-Mediena, kametiena.xlsx'
 #   file = 'data\\Total-Nafta ir naftos produktai.xlsx'
    
    model = Predict_model()
    model.train_predict_one(file)
    model.future_predict()
 #   model.save_future_prediction()