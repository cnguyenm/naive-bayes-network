
import pandas
import numpy as np
import json
import sys

# import local files
import preprocess

train_path = sys.argv[1]
test_path = sys.argv[2]

# ------------------------------------
# Learning / training
# ------------------------------------

# originally: 19 attrib + 1 label
df = pandas.read_csv(train_path)
df = df.fillna('None')
label = 'outdoorSeating'

# replace multi-value attrib
# now have 42 attrib + 1 label
df = preprocess.replace_multivalue_attrib(df)

# get class prior
n_label_true  = len(df.loc[df[label] == True])
n_label_false = len(df.index) - n_label_true  # fastest way to get total len
p_label_true  = n_label_true / len(df.index)
p_label_false = n_label_false / len(df.index)

# print(p_label_true)
# print(p_label_false)

# construct CPDs of all attrib
col_names = ['alcohol', 'delivery']

# db = {
#     "alcohol": {
#         "none": [
#             0.42392553435991726, # given: outdoor=True
#             0.6862088218872139   # given: outdoor=False
#         ],
#       ...
#       },
#     "delivery":{}
# }
db = {}


# calculate CPDs (Conditional probability distribution)
# of attrib, then add it to db
# ex: p(delivery=T|out=T), p(delivery=F|out=T), ...
def add_to_db(col_name):
    col_values = df[col_name].unique()
    k = len(col_values)
    db[col_name] = dict()  # init empty dict for that col

    for value in col_values:
        nomi = len(df[(df[col_name] == value) & (df[label] == True)]) + 1
        deno = n_label_true + k
        r = nomi / deno
        db[col_name][str(value)] = [r]  # 1st time, init list

    for value in col_values:
        nomi = len(df[(df[col_name] == value) & (df[label] == False)]) + 1
        deno = n_label_false + k
        r = nomi / deno
        db[col_name][str(value)].append(r)  # append to already list


# calculate CPDs for all atrib
for col in df.columns:
    if col == label:
        continue
    add_to_db(col)

# print("Finish learning")
# print(len(db))
# print(json.dumps(db, indent=4))

# ------------------------------------
# predict
# ------------------------------------


# test predict
# TODO: what if mutivalue_unique not have values
# missing columns
df2 = pandas.read_csv(test_path)
df2 = df2.fillna('None')
df2 = preprocess.replace_multivalue_attrib(df2)
zero_one_loss = 0
square_loss = 0

for index, row in df2.iterrows():
    p_output_true = p_label_true
    for col in df2.columns:
        if col == label:
            continue

        # p(label=True|X) = p(X1|label)p(X2|label)...p(label)
        # p(col=value|label)= db[col][value][label]
        prob = 0
        try:
            prob = db[col][str(row[col])][0]
        except KeyError:
            prob = 1 / (n_label_true + len(df[col].unique()) + 1)

        p_output_true *= prob

    p_output_false = p_label_false
    for col in df2.columns:
        if col == label:
            continue
        prob = 0
        try:
            prob = db[col][str(row[col])][1]
        except KeyError:
            prob = 1 / (n_label_false + len(df[col].unique()) + 1)

        p_output_false *= prob

    # normalize sum -> 1
    p_output_true = p_output_true / (p_output_true+p_output_false)
    p_output_false = 1 - p_output_true

    # zero-one loss
    predict = False
    if p_output_true > p_output_false:
        predict = True

    if predict != row[label]:
        zero_one_loss += 1

    # square loss
    p_correct = p_output_false
    if row[label] == True:
        p_correct = p_output_true

    square_loss += (1-p_correct)**2


N = len(df2.index)
print('ZERO-ONE LOSS=%.4f' % (zero_one_loss / N))
print('SQUARED LOSS=%.4f' % (square_loss/N))
















