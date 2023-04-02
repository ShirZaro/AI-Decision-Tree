import copy
import math
from scipy.stats import chi2
import numpy as np
import pandas as pd
from datetime import datetime,date
import random
import pptree


class node:
    def __init__(self,data,index,lable,father,children,answere,finalans,val):
        self.data= data
        self.index = index
        self.lable = lable
        self.father = father
        self.answere = answere
        self.children = children
        self.finalans = finalans
        self.val = val

    def set_ans (self,ans):
        self.finalans = ans

    def add_boy(self,boy):
        self.children.append(boy)

    def set_ans (self,val):
        self.val = val
        myans = allOptions(self)
        self.answere = "{} {}".format(self.father.lable, myans)

    def proning(self):
        for child in self.children:
            if not child.finalans == '':
                child.proning()
        if self.father != None:
            toCut = chi2test(self.data,self.father.data,self)
            if not toCut:
                self.father.finalans = self.finalans
                self.father.children.remove(self)



    def __str__(self):
        if len(self.children) == 0:
            t= self.answere +"--->"+ self.finalans
        elif self.children is str:
            t = self.answere +"--->"+  self.finalans
        elif self.finalans == '':
            if self.answere =='':
                t = self.lable
            else:
                t= self.answere+ "--->"+  self.lable
        return t



def returnVal(myData):
    col = myData[:, -1]
    a1 = myData[col == 1]
    a2 = myData[col == 0]
    if len(a1)>len(a2):
        return 'True'
    if len(a1) < len(a2):
        return 'False'
    else:
        rand = random.randint(0, 1)
        if rand ==0:
            return 'False'
        if rand ==1 :
            return 'True'




def decision_tree(df,parent, attributes,counter = 0):
    if counter == 0:
        global column_headers
        column_headers = df.columns
        data= df.values
        parent = None
    else:
        data = df
    num_rows, num_cols = data.shape
    if attributes == []: # if I run out of questions
        return str(returnVal(data))
    elif num_rows ==0: # if I run out of datta
        classification = classify_data(parent.data)
        return str(classification)
    elif check_purity(data): # if all the data have the same answere
        classification = classify_data(data)
        return str(classification)
    elif num_cols == 1:
        classification = classify_data(data)
        return str(classification)
    #recirsive path
    else:
        counter += 1
        splite_cullom,split_val = entropy(data,attributes)
        if split_val == 0:
            return str(returnVal(data))
        feature_name = column_headers[splite_cullom]
        children = []
        tree = node(data,splite_cullom,feature_name,parent,children,'','',0)
        split_col_vals = data[:, splite_cullom]
        a = attributes.copy()
        a.remove(splite_cullom) # remove from attribute the chosen question
        if splite_cullom == 0 or splite_cullom == 2: # finding the children of the tree acording to the question that asked (diffrent ranges for each one)
            for i in range(0,2):
                add_to_tree(data,a,counter,i,splite_cullom,feature_name,tree,split_col_vals)
        elif splite_cullom == 4 or splite_cullom == 5 or splite_cullom == 6 or splite_cullom == 8 or splite_cullom == 9 or splite_cullom == 10 or splite_cullom == 3 or splite_cullom == 7:
            i = 100
            while i < 301:
                add_to_tree(data,a,counter,i,splite_cullom,feature_name,tree,split_col_vals)
                i = i +100
        elif splite_cullom == 11:
            all = ['Winter','Autumn','Spring','Summer']
            for i in all:
                add_to_tree(data,a,counter,i,splite_cullom,feature_name,tree,split_col_vals)
        elif splite_cullom == 12:
            all = ['Holiday','No Holiday']
            for i in all:
                add_to_tree(data,a,counter,i,splite_cullom,feature_name,tree,split_col_vals)
        elif splite_cullom == 13:
            all = ['Yes','No']
            for i in all:
                add_to_tree(data,a,counter,i,splite_cullom,feature_name,tree,split_col_vals)
        return tree


def add_to_tree(data,a,counter,i,splite_cullom,feature_name,tree,split_col_vals):
    answer = decision_tree(data[split_col_vals == i], tree, a, counter)
    if type(answer) is str: # if it string - there is a final answere
        t = node(data[split_col_vals == i], splite_cullom, feature_name, tree, [], '', answer, 0)
        t.set_ans(i)
        tree.add_boy(t)
    else:
        tree.add_boy(answer)
        answer.set_ans(i)



def chi2test(data, parentData, subTree):
    busy_col = parentData[:,-1]
    a, counts = np.unique(busy_col, return_counts= True)
    not_busy_parent = counts[0]
    busy_parent = counts[1]
    free = busy_parent + not_busy_parent
    crit = chi2.ppf(q= 0.05, df = free)
    delta = 0
    for a in subTree.children:
        new_data = a.data
        if not new_data.empty:
            a, counts1 = np.unique(busy_col, return_counts=True)
            not_child_busy = counts1[0]
            child_busy = counts1[1]
            pk = busy_parent*(not_child_busy+child_busy)/(not_busy_parent+busy_parent)
            nk = not_busy_parent*(not_child_busy+child_busy)/(not_busy_parent+busy_parent)
            delta = delta + ((child_busy - pk)**2/pk)+((child_busy- nk)**2/nk)
    if delta < crit:
        return False
    else:
        return True


def test(test_df,tree):  # test the test set
    sum_true = 0
    sum_false = 0
    data = test_df.values
    num_rows, num_cols = data.shape
    for i in range(0,num_rows):
        ans = str(find_ans(data[i],tree,False))
        if ans == 'False' and data[i][-1] == 0:
            sum_true +=1
        elif ans == 'True' and data[i][-1] == 1:
            sum_true += 1
        else:
            sum_false += 1
    p_true = sum_true/(sum_true+sum_false)
    p_flase = sum_false/(sum_true+sum_false)
    return p_flase


def find_ans(num1,tree,bool): # return the prediction of the row
    index = tree.index
    if bool == True:
        if index >1 :
            index  = index -1
    for i in tree.children:
        if i.val == num1[index]:
            nextTree = i
            break
    if nextTree.finalans == '':
        return find_ans(num1, nextTree,bool)
    else:
        an = str(nextTree.finalans)
        return an



def entropy(data,attributes): # entropy for all the questions I have in the attribute
    all_gains = []
    Date_gain,Hour_gain,Temperature_gain,Humidity_gain,WindSpeed_gain,Dew_point_gain=(0,0,0,0,0,0)
    Visibility_gain,Solar_gain,Rainfall_gain,Snowfall_gain,Seasons_gain,Holiday_gain,Functioning_gain = (0,0,0,0,0,0,0)
    x10,x11 = clacul_entropy(data)
    myentropy = (x10+x11)
    max, index = (0,0)
    if 0 in attributes:
        Date_gain = myentropy - towentropy(data,0)
    if 2 in attributes:
        Hour_gain =myentropy- towentropy(data,2)
    if 3 in attributes:
        Temperature_gain= myentropy- clac_entropy(data,3)
    if 4 in attributes:
        Humidity_gain =myentropy- clac_entropy(data,4)
    if 5 in attributes:
        WindSpeed_gain =myentropy- clac_entropy(data,5)
    if 6 in attributes:
        Visibility_gain =myentropy- clac_entropy(data,6)
    if 7 in attributes:
        Dew_point_gain =myentropy-clac_entropy(data,7)
    if 8 in attributes:
        Solar_gain =myentropy- clac_entropy(data,8)
    if 9 in attributes:
        Rainfall_gain =myentropy- clac_entropy(data,9)
    if 10 in attributes:
        Snowfall_gain =myentropy- clac_entropy(data,10)
    if 11 in attributes:
        Seasons_gain =myentropy- SeasonsEntropy(data)
    if 12 in attributes:
        Holiday_gain =myentropy- HolidayEntropy(data)
    if 13 in attributes:
        Functioning_gain =myentropy- FunctioningEntropy(data)
    all_gains.append(Date_gain),all_gains.append(0),all_gains.append(Hour_gain),all_gains.append(Temperature_gain),all_gains.append(Humidity_gain)
    all_gains.append(WindSpeed_gain),all_gains.append(Visibility_gain),all_gains.append(Dew_point_gain),all_gains.append(Solar_gain)
    all_gains.append(Rainfall_gain),all_gains.append(Snowfall_gain),all_gains.append(Seasons_gain),all_gains.append(Holiday_gain),all_gains.append(Functioning_gain)
    for i in range(0,len(all_gains)):
        if all_gains[i] > max:
            max = all_gains[i]
            index = i
    return index,max


def towentropy(data,attribute):
    x10, x11, x20, x21= (0, 0, 0, 0)
    col= data[:, attribute]
    a1= data[col == 0]
    a2=data[col == 1]
    if len(a1) != 0:
        x10,x11 = clacul_entropy(a1)
    if len(a2) != 0:
        x20,x21 = clacul_entropy(a2)
    entropy = ((len(a1)/len(data)*(x10+x11))+(len(a2)/len(data)*(x20+x21)))
    return entropy



def clac_entropy(data, attribute):
    x10, x11, x20, x21, x30, x31= (0, 0, 0, 0, 0, 0)
    col= data[:,attribute]
    a1= data[col == 100]
    a2=data[col == 200]
    a3=data[col == 300]
    if len(a1) != 0:
        x10,x11 = clacul_entropy(a1)
    if len(a2) != 0:
        x20,x21 = clacul_entropy(a2)
    if len(a3) != 0:
        x30,x31 = clacul_entropy(a3)
    entropy = ((len(a1)/len(data)*(x10+x11))+(len(a2)/len(data)*(x20+x21))+(len(a3)/len(data)*(x30+x31)))
    return entropy

def FunctioningEntropy(data):
    x10, x11, x20, x21= (0, 0, 0, 0)
    fum_col= data[:,13]
    a1= data[fum_col == 'Yes']
    a2=data[fum_col == 'No']
    if len(a1) != 0:
        x10,x11 = clacul_entropy(a1)
    if len(a2) != 0:
        x20,x21 = clacul_entropy(a2)
    entropy = ((len(a1)/len(data)*(x10+x11))+(len(a2)/len(data)*(x20+x21)))
    return entropy

def HolidayEntropy(data):
    x10, x11, x20, x21= (0, 0, 0, 0)
    hol_col= data[:,12]
    a1= data[hol_col == 'Holiday']
    a2=data[hol_col == 'No Holiday']
    if len(a1) != 0:
        x10,x11 = clacul_entropy(a1)
    if len(a2) != 0:
        x20,x21 = clacul_entropy(a2)
    entropy = ((len(a1)/len(data)*(x10+x11))+(len(a2)/len(data)*(x20+x21)))
    return entropy


def SeasonsEntropy(data):
    x10, x11, x20, x21, x30, x31, x40, x41= (0, 0, 0, 0, 0, 0, 0, 0)
    ses_col= data[:,11]
    a1= data[ses_col == 'Winter']
    a2=data[ses_col == 'Spring']
    a3=data[ses_col== 'Summer']
    a4=data[ses_col == 'Autumn']
    if len(a1) != 0:
        x10,x11 = clacul_entropy(a1)
    if len(a2) != 0:
        x20,x21 = clacul_entropy(a2)
    if len(a3) != 0:
        x30,x31 = clacul_entropy(a3)
    if len(a4) != 0:
        x40,x41 = clacul_entropy(a4)
    entropy = ((len(a1)/len(data)*(x10+x11))+(len(a2)/len(data)*(x20+x21))+(len(a3)/len(data)*(x30+x31))+(len(a4)/len(data)*(x40+x41)))
    return entropy


def clacul_entropy(data):  # calculate the entropy for one data item
    busy_col = data[:,-1]
    a, counts = np.unique(busy_col, return_counts= True)
    if len(a) == 1:
        return [0,0]
    prob = counts / counts.sum()
    for i in prob:
        if i == 0:
            i =1
    entropy = prob*(-np.log2(prob))
    return entropy


def classify_data(data):
    busy_col = data[:,-1]
    uniqe_classes,counts_uniqe = np.unique(busy_col,return_counts=True)
    index= counts_uniqe.argmax()
    classification= (uniqe_classes[index])
    if classification == 0:
        return 'False'
    if classification ==1:
        return 'True'

def check_purity(data): #  check if all the tags are the same
    busy_col = data[:,-1]
    unique_classes = np.unique(busy_col)
    if len(unique_classes) ==1:
        return True
    else:
        return False

def train_test_splite(df, test_size): # split to train and test
    test_size = round(test_size*len(df))
    indices = df.index.tolist()
    test_indices= random.sample(population=indices, k= test_size)
    test_df = df.loc[test_indices]
    train_df = df.drop(test_indices)
    return train_df,test_df

def is_busy(row_input): # train in data, test on 1 given row
    data,df = start()
    train_df,test_df = train_test_splite(df,0)
    attributes = [0,2,3,4,5,6,7,8,9,10,11,12,13]
    tree= decision_tree(train_df,None,attributes, counter=0)
    tree.proning()
    new_row = set_one_Range(row_input)
    ans = find_ans(new_row, tree, True)
    print("the answere is :",ans)




def tree_error(k):
    data,df = start()
    left_data = df
    all_data_to_test = []
    all_test_indicees = []
    all_errors = []
    sum_error = 0
    n = k
    for i in range(0,k):
        new_data,left_data,test_indices = split_to_k(left_data,n)
        n = n-1
        all_data_to_test.append(new_data) # saving k data sets
        all_test_indicees.append(test_indices)
    for i in  range(len(all_data_to_test)):   # for each one of the data set - train on the rest, test the data
        test_df = all_data_to_test[i]
        train_df= df.drop(all_test_indicees[i])
        attributes = [0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
        tree = decision_tree(train_df, None, attributes, counter=0)
        tree.proning()
        error = test(test_df, tree)
        sum_error = sum_error + error
        all_errors.append(error)
    print("Error rate:",((sum_error) / len(all_errors)))

def split_to_k(df,k):  # spliting the data to test and train with k, it returns one data set in k/tot len , and the left data
    test_size = round(len(df)/k)
    indices = df.index.tolist()
    test_indices= random.sample(population=indices, k= test_size)
    new_data = df.loc[test_indices]
    left_data = df.drop(test_indices)
    return new_data,left_data,test_indices

def build_tree(ratio): # the main function for bilding a tree
    data, df = start()
    train_df,test_df = train_test_splite(df,(1-ratio)) # split the data to test and train acording to the rate
    attributes = [0,2,3,4,5,6,7,8,9,10,11,12,13]   # list of attributes
    tree= decision_tree(train_df,None,attributes, counter=0) # the lerning function
    tree.proning()
    pptree.print_tree(tree,childattr = 'children' ,horizontal=True)
    p_flase = test(test_df,tree) # test data
    print("Error rate: ", p_flase)




def start(): #reading the data and set ranges
    if __name__ == '__main__':
        df = pd.read_csv("SeoulBikeData.csv", encoding= 'unicode_escape')
        set_Ranges(df)
        data = df.values
        return data,df



def allOptions(node):  #help print function - getting back the ranges
    myans = ''
    if node.father.lable == 'Hour':
        if node.val == 0:
            myans = '= day'
        if node.val == 1:
            myans = '= nigth'
    elif node.father.lable == 'Functioning Day':
        myans = '= ' + node.val
    elif node.father.lable == 'Holiday':
        myans = '= ' + node.val
    elif node.father.lable == 'Seasons':
        myans = '= ' + node.val
    elif node.father.lable == 'Date':
        if node.val == 0:
            myans = '= weekday'
        if node.val == 1:
            myans = '= weekend'
    elif node.father.lable == 'Temperature(°C)':
        if node.val == 100:
            myans = '<= -10'
        if node.val == 200:
            myans = 'between {-10,10}'
        if node.val == 300:
            myans = '>=10'
    elif node.father.lable == 'Humidity(%)':
        if node.val == 100:
            myans = '<= 30'
        if node.val == 200:
            myans = 'between {30,60}'
        if node.val == 300:
            myans = '>=60'
    elif node.father.lable == 'Wind speed (m/s)':
        if node.val == 100:
            myans = '<= 2'
        if node.val == 200:
            myans = 'between {2,6}'
        if node.val == 300:
            myans = '>=6'
    elif node.father.lable == 'Visibility (10m)':
        if node.val == 100:
            myans = '<= 600'
        if node.val == 200:
            myans = 'between {600,1200}'
        if node.val == 300:
            myans = '>=1200'
    elif node.father.lable == 'Dew point temperature(°C)':
        if node.val == 100:
            myans = '<= -20'
        if node.val == 200:
            myans = 'between {-20,0}'
        if node.val == 300:
            myans = '>=0'
    elif node.father.lable == 'Solar Radiation (MJ/m2)':
        if node.val == 100:
            myans = '<= 1'
        if node.val == 200:
            myans = 'between {1,2.5}'
        if node.val == 300:
            myans = '>=2.5'
    elif node.father.lable == 'Rainfall(mm)':
        if node.val == 100:
            myans = '<= 15'
        if node.val == 200:
            myans = 'between {15,30}'
        if node.val == 300:
            myans = '>=30'
    elif node.father.lable == 'Snowfall (cm)':
        if node.val == 100:
            myans = '<= 3'
        if node.val == 200:
            myans = 'between {3,6}'
        if node.val == 300:
            myans = '>=6'
    return myans


def set_one_Range(row_input): # setting the data ranges for one row only
    new_row = row_input.copy()
    dt_ob = datetime.strptime(row_input[0],"%d/%m/%Y")
    if dt_ob.weekday() in(0,1,2,3):
        new_row[0] = 0
    elif dt_ob.weekday() in(4,5,6):
        new_row[0] = 1
    if row_input[1] in (6,7,8,9,10,11,12,13,14,15,16,17):
        new_row[1] = 0
    elif row_input[1] in (18,19,20,21,22,23,0,1,2,3,4,5):
        new_row[1] = 1
    if row_input[2] <=-10:
        new_row[2] =100
    elif row_input[2] >=-10 and  row_input[2] <= 10:
        new_row[2] = 200
    elif row_input[2] >= 10:
        new_row[2] = 300
    if row_input[3] <30:
        new_row[3] =100
    elif row_input[3] >=30 and row_input[3] < 60:
        new_row[3] = 200
    elif row_input[3] >= 60:
        new_row[3] = 300
    if row_input[4] <=2:
        new_row[4] =100
    elif row_input[4] >2 and row_input[4] < 6:
        new_row[4] = 200
    elif row_input[4] >= 6:
        new_row[4] = 300
    if row_input[5] <=600:
        new_row[5] =100
    elif row_input[5] > 600 and row_input[5]  < 1200:
        new_row[5] = 200
    elif row_input[5] >= 1200:
        new_row[5] = 300
    if row_input[6] <=-20:
        new_row[6] =100
    elif row_input[6] >-20 and row_input[6] <0:
        new_row[6] = 200
    elif row_input[6] >= 0:
        new_row[6] = 300
    if row_input[7] <=1:
        new_row[7] =100
    elif row_input[7] > 1 and row_input[7] <2.5:
        new_row[7] = 200
    elif row_input[7] >= 2.5:
        new_row[7] = 300
    if row_input[8] <=15:
        new_row[8] =100
    elif row_input[8] >15  and row_input[8] <30:
        new_row[8] = 200
    elif row_input[8] >= 30:
        new_row[8] = 300
    if row_input[9] <=3:
        new_row[9] =100
    elif row_input[9] > 3 and row_input[9] <6:
        new_row[9] = 200
    elif row_input[9] >= 6:
        new_row[9] = 300
    new_row[10]=row_input[10]
    new_row[11]=row_input[11]
    new_row[12]=row_input[12]
    return new_row


def set_Ranges(examples):  # setting the data ranges for pandas array
    examples['busy']= np.where(examples['Rented Bike Count']>=650,1,0)
    examples["Hour"] = examples["Hour"].replace([(6,7,8,9,10,11,12,13,14,15,16,17)], 0)
    examples["Hour"] = examples["Hour"].replace([(18,19,20,21,22,23,0,1,2,3,4,5)],1)
    examples['Date'] = pd.to_datetime(examples['Date'], format = "%d/%m/%Y").dt.weekday
    examples["Date"] = examples["Date"].replace([(0,1,2,3)], 0)
    examples["Date"] = examples["Date"].replace([(4,5,6)], 1)

    examples["Temperature(°C)"] = np.where(examples["Temperature(°C)"]<= (-10),100 , examples["Temperature(°C)"])
    examples["Temperature(°C)"] = np.where(examples["Temperature(°C)"].between(-10,10), 200, examples["Temperature(°C)"])
    examples["Temperature(°C)"] = np.where(examples["Temperature(°C)"]>=10, 300, examples["Temperature(°C)"])

    examples["Humidity(%)"] = np.where(examples["Humidity(%)"]<=30,100, examples["Humidity(%)"])
    examples["Humidity(%)"] = np.where(examples["Humidity(%)"].between(30,60),200, examples["Humidity(%)"])
    examples["Humidity(%)"] = np.where(examples["Humidity(%)"]>=60,300, examples["Humidity(%)"])

    examples["Wind speed (m/s)"] = np.where(examples["Wind speed (m/s)"]<=2,100, examples["Wind speed (m/s)"])
    examples["Wind speed (m/s)"] = np.where(examples["Wind speed (m/s)"].between(2,6),200, examples["Wind speed (m/s)"])
    examples["Wind speed (m/s)"] = np.where(examples["Wind speed (m/s)"]>=6,300, examples["Wind speed (m/s)"])

    examples["Visibility (10m)"] = np.where(examples["Visibility (10m)"].between(0,600),100, examples["Visibility (10m)"])
    examples["Visibility (10m)"] = np.where(examples["Visibility (10m)"].between(600,1200),200, examples["Visibility (10m)"])
    examples["Visibility (10m)"] = np.where(examples["Visibility (10m)"]>=1200,300, examples["Visibility (10m)"])

    examples["Dew point temperature(°C)"] = np.where(examples["Dew point temperature(°C)"]<= (-20), 100, examples["Dew point temperature(°C)"])
    examples["Dew point temperature(°C)"] = np.where(examples["Dew point temperature(°C)"].between(-20,0), 200, examples["Dew point temperature(°C)"])
    examples["Dew point temperature(°C)"] = np.where(examples["Dew point temperature(°C)"]>=0, 300, examples["Dew point temperature(°C)"])

    examples["Solar Radiation (MJ/m2)"] = np.where(examples["Solar Radiation (MJ/m2)"]<=1, 100, examples["Solar Radiation (MJ/m2)"])
    examples["Solar Radiation (MJ/m2)"] = np.where(examples["Solar Radiation (MJ/m2)"].between(1,2.5), 200, examples["Solar Radiation (MJ/m2)"])
    examples["Solar Radiation (MJ/m2)"] = np.where(examples["Solar Radiation (MJ/m2)"]>=2.5, 300, examples["Solar Radiation (MJ/m2)"])

    examples["Rainfall(mm)"] = np.where(examples["Rainfall(mm)"]<=15, 100, examples["Rainfall(mm)"])
    examples["Rainfall(mm)"] = np.where(examples["Rainfall(mm)"].between(15,30), 200, examples["Rainfall(mm)"])
    examples["Rainfall(mm)"] = np.where(examples["Rainfall(mm)"]>=30, 300, examples["Rainfall(mm)"])

    examples["Snowfall (cm)"] = np.where(examples["Snowfall (cm)"]<= 3, 100, examples["Snowfall (cm)"])
    examples["Snowfall (cm)"] = np.where(examples["Snowfall (cm)"].between(3,6), 200, examples["Snowfall (cm)"])
    examples["Snowfall (cm)"] = np.where(examples["Snowfall (cm)"]>=6, 300, examples["Snowfall (cm)"])

