#! /usr/bin/python
# www.github.com/andpol5/credit-classifier
#########################################################
# This script prreprocceses the data from german.data.txt
# and outputs the result into a new file

# python std
import csv
# ML libraries
from sklearn import preprocessing
import numpy as np

# This function takes a column of the raw matrix and finds unique values for encoding
def uniqueItems(column):
    list = np.unique(column)
    fixedList = []
    for item in list:
        fixeListItem = [item]
        fixedList.append(fixeListItem)
    return fixedList

def newWidth(column):
    return len(np.unique(column))

def encodeColumn(oldCol, encoder):
    newCol = []
    for c in oldCol:
        newCol.append(encoder.transform(c).toarray())
    return np.array(newCol)

# Binarize the column with trueVal becoming +1 and everything else -1
def binarizeColumn(oldCol, trueVal):
    return [1 if x==trueVal else -1 for x in oldCol]

# Read the raw data file into arrays
with open('german.data.txt') as rawDataFile:
    csvReader = csv.reader(rawDataFile, delimiter=' ', quotechar='|')
    rows = []
    for row in csvReader:
        cols = []
        for col in row:
            # Change the value here into a floating point number
            if col[0] == 'A':
                value = float(col[1:])
            else:
                value = float(col)
            cols.append(value)
        rows.append(cols)

rowCount = len(rows)
colCount = len(rows[0])

# read it into an ndarray
arr = np.array(rows)

# Data transformation by column:
# Use One-Hot encoding for the categorical features
# and use Gaussian normalization for numerical features
#   (translate feature to have zero mean and unit variance)

enc1 = preprocessing.OneHotEncoder() # 1 - Status of checking account
enc1.fit(uniqueItems(arr[:,0]))
# 2 - Duration in months
enc3 = preprocessing.OneHotEncoder() # 3 - Credit history
enc3.fit(uniqueItems(arr[:,2]))
enc4 = preprocessing.OneHotEncoder() # 4 - Purpose of loan
enc4.fit(uniqueItems(arr[:,3]))
# 5- Credit amount
enc6 = preprocessing.OneHotEncoder() # 6 - savings account/bonds
enc6.fit(uniqueItems(arr[:,5]))
enc7 = preprocessing.OneHotEncoder() # 7 - present employment since
enc7.fit(uniqueItems(arr[:,6]))
# 8 - Installment rate in percentage of disposable income
enc9 = preprocessing.OneHotEncoder() # 9 - Personal status and sex
enc9.fit(uniqueItems(arr[:,8]))
enc10 = preprocessing.OneHotEncoder() # 10 - Other debtors / guarantors
enc10.fit(uniqueItems(arr[:,9]))
# 11- Present residence since
enc12 = preprocessing.OneHotEncoder() # 12 - Property
enc12.fit(uniqueItems(arr[:,11]))
# 13 - Age in years
enc14 = preprocessing.OneHotEncoder() # 14 - Other installment plans
enc14.fit(uniqueItems(arr[:,13]))
enc15 = preprocessing.OneHotEncoder() # 15 - Housing
enc15.fit(uniqueItems(arr[:,14]))
# 16 - Number of existing credits at this bank
enc17 = preprocessing.OneHotEncoder() # 17 - Job
enc17.fit(uniqueItems(arr[:,16]))
# 18 - Number of people being liable to provide maintenance for
# 19 - Telephone (binary)
# 20 - Foreign worker (binary)
# 21 - Output: Good vs bad credit (binary)
enc21 = preprocessing.OneHotEncoder() # 21 - output
enc21.fit(uniqueItems(arr[:,20]))


# Create columns of new data
col1 = encodeColumn(arr[:,0], enc1)
col2 = preprocessing.scale(arr[:,1]) # numeric
col3 = encodeColumn(arr[:,2], enc3)
col4 = encodeColumn(arr[:,3], enc4)
col5 = preprocessing.scale(arr[:,4]) # numeric
col6 = encodeColumn(arr[:,5], enc6)
col7 = encodeColumn(arr[:,6], enc7)
col8 = preprocessing.scale(arr[:,7]) # numeric
col9 = encodeColumn(arr[:,8], enc9)
col10 = encodeColumn(arr[:,9], enc10)
col11 = preprocessing.scale(arr[:,10]) # numeric
col12 = encodeColumn(arr[:,11], enc12)
col13 = preprocessing.scale(arr[:,12]) # numeric
col14 = encodeColumn(arr[:,13], enc14)
col15 = encodeColumn(arr[:,14], enc15)
col16 = preprocessing.scale(arr[:,15]) # numeric
col17 = encodeColumn(arr[:,16], enc17)
col18 = preprocessing.scale(arr[:,17]) # numeric
col19 = binarizeColumn(arr[:,18], 192) # binary
col20 = binarizeColumn(arr[:,19], 201) # binary
col21 = encodeColumn(arr[:,20], enc21)

# the widths of the new columns
w1 = newWidth(arr[:,0])
w3 = newWidth(arr[:,2])
w4 = newWidth(arr[:,3])
w6 = newWidth(arr[:,5])
w7 = newWidth(arr[:,6])
w9 = newWidth(arr[:,8])
w10 = newWidth(arr[:,9])
w12 = newWidth(arr[:,11])
w14 = newWidth(arr[:,13])
w15 = newWidth(arr[:,14])
w17 = newWidth(arr[:,16])
w21 = newWidth(arr[:,20])

# Create a placeholder for new data
newData = np.zeros((1000, w1+w3+w4+w6+w7+w9+w10+w12+w14+w15+w17+w21+9))

# populate the matrix with the new columns
c = 0; # index of current column (relative to old data)
newData[:,c:c+w1] = col1.reshape((1000, w1)); c=c+w1                 # 1
newData[:, c] = col2; c = c + 1                                      # 2
newData[:, c:c+w3]=col3.reshape((1000, w3)); c=c+w3                  # 3
newData[:, c:c+w4]=col4.reshape((1000, w4)); c=c+w4                  # 4
newData[:, c] = col5; c=c + 1                                        # 5
newData[:, c:c+w6]=col6.reshape((1000, w6)); c=c+w6                 # 6                                                               # 6
newData[:, c:c+w7]=col7.reshape((1000, w7)); c=c+w7                  # 7
newData[:, c] = col8; c = c + 1                                      # 8
newData[:, c:c+w9]=col9.reshape((1000, w9)); c=c+w9                  # 9
newData[:, c:c+w10]=col10.reshape((1000, w10)); c=c+w10                 # 10
newData[:, c] = col11; c = c + 1                                      # 11
newData[:, c:c+w12]=col12.reshape((1000, w12)); c=c+w12             # 12
newData[:, c] = col13; c = c + 1                                      # 13
newData[:, c:c+w14]=col14.reshape((1000, w14)); c=c+w14             # 14
newData[:, c:c+w15]=col15.reshape((1000, w15)); c=c+w15             # 15
newData[:, c] = col16; c = c + 1                                      # 16
newData[:, c:c+w17]=col17.reshape((1000, w17)); c=c+w17                 # 17
newData[:, c] = col18; c = c + 1                                        # 18
newData[:, c] = col19; c = c + 1                                        # 19
newData[:, c] = col20; c = c + 1                                        # 20
newData[:, c:c+w21] = col21.reshape(1000, w21)                        # 21

# Save to csv file
np.savetxt("newData.csv", newData, delimiter=",")
