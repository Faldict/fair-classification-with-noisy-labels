import os
import random
import numpy as np
import pandas as pd
import shap
import sklearn.preprocessing as preprocessing
from sklearn.preprocessing import LabelEncoder,StandardScaler

DATASET_DIR = 'dataset'

def german():
    df = pd.read_csv(f"{DATASET_DIR}/german_processed.csv")
    Y = df["GoodCustomer"].values
    Y = (Y + 1)/2
    Y = Y.astype(int)
    X = df.drop(labels=['GoodCustomer'], axis=1)
    X = pd.get_dummies(X, columns=['PurposeOfLoan'])
    
    A = df['Gender']
    return X, Y, A

def adult():
    """ Adult dataset.
    """
    X_raw, Y = shap.datasets.adult()
    A = X_raw["Sex"]
    X = X_raw.drop(labels=['Sex'],axis = 1)
    X = pd.get_dummies(X)

    sc = StandardScaler()
    X_scaled = sc.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

    le = LabelEncoder()
    Y = le.fit_transform(Y)
    return X_scaled, Y, A

def balanced_adult(ngroups=1):
    """ Balanced Adult dataset
    """
    df = pd.read_csv(f"{DATASET_DIR}/adult_gender_balanced_processed.csv")
    Y = df["IncomeOver50K"].values
    Y = (Y + 1)/2
    Y = Y.astype(int)
    X = df.drop(labels=['IncomeOver50K'], axis=1)

    if ngroups == 2:
        df['sensitive_attribute'] = df['Female'] + df['Immigrant'] * 2
    elif ngroups == 3:
        df['sensitive_attribute'] = df['Female'] + df['Immigrant'] * 2 + df['Married'] * 4
    else:
        df['sensitive_attribute'] = df['Female']
    
    A = df['sensitive_attribute']
    return X, Y, A

def compas_arrest_gender(ngroups=1):
    df = pd.read_csv(f"{DATASET_DIR}/compasa_gender_balanced_processed.csv")
    Y = df["arrest"].values
    Y = (Y+1)/2
    Y = Y.astype(int)
    X = df.drop(labels=['arrest'], axis=1)

    if ngroups == 2:
        df['sensitive_attribute'] = df['female'] + df['age_leq_21'] * 2
    else:
        df['sensitive_attribute'] = df['female']
    A = df['sensitive_attribute']
    return X, Y, A

def compas_arrest_race(ngroups=1):
    df = pd.read_csv(f"{DATASET_DIR}/compasa_race_balanced_processed.csv")
    Y = df["arrest"].values
    Y = (Y+1)/2
    Y = Y.astype(int)
    X = df.drop(labels=['arrest'], axis=1)

    if ngroups == 2:
        df['sensitive_attribute'] = df['black'] + df['female'] * 2
    else:
        df['sensitive_attribute'] = df['black']
    A = df['sensitive_attribute'].astype('int32')
    return X, Y, A

def compas_violent_gender(ngroups=1):
    df = pd.read_csv(f"{DATASET_DIR}/compasv_gender_balanced_processed.csv")
    Y = df["violent"].values
    Y = (Y+1)/2
    Y = Y.astype(int)
    X = df.drop(labels=['violent'], axis=1)

    if ngroups == 2:
        df['sensitive_attribute'] = df['female'] + df['age_leq_21'] * 2
    else:
        df['sensitive_attribute'] = df['female']
    A = df['sensitive_attribute']
    return X, Y, A

def compas_violent_race(ngroups=1):
    df = pd.read_csv(f"{DATASET_DIR}/compasv_race_balanced_processed.csv")
    Y = df["violent"].values
    Y = (Y+1)/2
    Y = Y.astype(int)
    X = df.drop(labels=['violent'], axis=1)

    if ngroups == 2:
        df['sensitive_attribute'] = df['black'] + df['female'] * 2
    else:
        df['sensitive_attribute'] = df['black']
    A = df['sensitive_attribute'].astype('int32')
    return X, Y, A

def compas():
    FEATURES_CLASSIFICATION = ["age_cat", "race", "sex", "priors_count", "c_charge_degree"] #features to be used for classification
    CONT_VARIABLES = ["priors_count"] # continuous features, will need to be handled separately from categorical features, categorical features will be encoded using one-hot
    CLASS_FEATURE = "two_year_recid" # the decision variable

    COMPAS_INPUT_FILE = DATASET_DIR + os.sep + "compas-scores-two-years.csv"

    # load the data and get some stats
    df = pd.read_csv(COMPAS_INPUT_FILE)
    df = df.dropna(subset=["days_b_screening_arrest"]) # dropping missing vals

    # convert to np array
    data = df.to_dict('list')
    for k in data.keys():
        data[k] = np.array(data[k])

    # These filters are the same as propublica (refer to https://github.com/propublica/compas-analysis)
    # If the charge date of a defendants Compas scored crime was not within 30 days from when the person was arrested, we assume that because of data quality reasons, that we do not have the right offense.
    idx = np.logical_and(data["days_b_screening_arrest"]<=30, data["days_b_screening_arrest"]>=-30)

    # We coded the recidivist flag -- is_recid -- to be -1 if we could not find a compas case at all.
    idx = np.logical_and(idx, data["is_recid"] != -1)

    # In a similar vein, ordinary traffic offenses -- those with a c_charge_degree of 'O' -- will not result in Jail time are removed (only two of them).
    idx = np.logical_and(idx, data["c_charge_degree"] != "O") # F: felony, M: misconduct

    # We filtered the underlying data from Broward county to include only those rows representing people who had either recidivated in two years, or had at least two years outside of a correctional facility.
    idx = np.logical_and(idx, data["score_text"] != "NA")

    # we will only consider blacks and whites for this analysis
    idx = np.logical_and(idx, np.logical_or(data["race"] == "African-American", data["race"] == "Caucasian"))

    # select the examples that satisfy this criteria
    for k in data.keys():
        data[k] = data[k][idx]

    # convert class label 0 to -1
    Y = data[CLASS_FEATURE]

    # print("\nNumber of people recidivating within two years")
    # print(pd.Series(y).value_counts())
    # print("\n")


    X = np.array([]).reshape(len(Y), 0) # empty array with num rows same as num examples, will hstack the features to it

    feature_names = []
    for attr in FEATURES_CLASSIFICATION:
        vals = data[attr]
        if attr in CONT_VARIABLES:
            vals = [float(v) for v in vals]
            vals = preprocessing.scale(vals) # 0 mean and 1 variance
            vals = np.reshape(vals, (len(Y), -1)) # convert from 1-d arr to a 2-d arr with one col

        else: # for binary categorical variables, the label binarizer uses just one var instead of two
            lb = preprocessing.LabelBinarizer()
            lb.fit(vals)
            vals = lb.transform(vals)


        # add to learnable features
        X = np.hstack((X, vals))

        if attr in CONT_VARIABLES: # continuous feature, just append the name
            feature_names.append(attr)
        else: # categorical features
            if vals.shape[1] == 1: # binary features that passed through lib binarizer
                feature_names.append(attr)
            else:
                for k in lb.classes_: # non-binary categorical features, need to add the names for each cat
                    feature_names.append(attr + "_" + str(k))

    X_raw = pd.DataFrame(X, columns=feature_names)
    A = X_raw["race"].astype(int)
    print(Y.shape)
    return X_raw, Y, A

def law(frac=1, scaler=True):
    LAW_FILE = f'{DATASET_DIR}/law_data_clean.csv'
    data = pd.read_csv(LAW_FILE)

    # Switch two columns to make the target label the last column.
    cols = data.columns.tolist()
    cols = cols[:9]+cols[11:]+cols[10:11]+cols[9:10]
    data = data[cols]

    data = data.loc[data['racetxt'].isin(['White', 'Black'])]

    # categorical fields
    category_col = ['racetxt']
    for col in category_col:
        b, c = np.unique(data[col], return_inverse=True)
        data[col] = c

    datamat = data.values
    datamat = datamat[datamat[:,9].argsort()]

    A = np.copy(datamat[:, 9])
    Y = datamat[:, -1]
    datamat = datamat[:, :-1]

    if scaler:
        scaler = StandardScaler()
        scaler.fit(datamat)
        datamat = scaler.transform(datamat)

    datamat[:, 9] = A
    X_raw = pd.DataFrame(datamat, columns=data.columns[:-1])
    A = X_raw['racetxt'].astype(int)
    # print(X_raw.head())
    return X_raw, Y.astype(int), A

def apnea_preproc():
    import wfdb
    import matplotlib.pyplot as plt
    import numpy as np
    from hrv.filters import quotient, moving_median
    from scipy import interpolate
    from tqdm import tqdm
    import os
    FS = 100.0

    # From https://github.com/rhenanbartels/hrv/blob/develop/hrv/classical.py
    def create_time_info(rri):
        rri_time = np.cumsum(rri) / 1000.0  # make it seconds
        return rri_time - rri_time[0]   # force it to start at zero

    def create_interp_time(rri, fs):
        time_rri = create_time_info(rri)
        return np.arange(0, time_rri[-1], 1 / float(fs))

    def interp_cubic_spline(rri, fs):
        time_rri = create_time_info(rri)
        time_rri_interp = create_interp_time(rri, fs)
        tck = interpolate.splrep(time_rri, rri, s=0)
        rri_interp = interpolate.splev(time_rri_interp, tck, der=0)
        return time_rri_interp, rri_interp

    def interp_cubic_spline_qrs(qrs_index, qrs_amp, fs):
        time_qrs = qrs_index / float(FS)
        time_qrs = time_qrs - time_qrs[0]
        time_qrs_interp = np.arange(0, time_qrs[-1], 1/float(fs))
        tck = interpolate.splrep(time_qrs, qrs_amp, s=0)
        qrs_interp = interpolate.splev(time_qrs_interp, tck, der=0)
        return time_qrs_interp, qrs_interp

    data_path = f'{DATASET_DIR}/apnea/'
    train_data_name = ['a01', 'a02', 'a03', 'a04', 'a05',
                'a06', 'a07', 'a08', 'a09', 'a10',
                'a11', 'a12', 'a13', 'a14', 'a15',
                'a16', 'a17', 'a18', 'a19',
                'b01', 'b02', 'b03', 'b04',
                'c01', 'c02', 'c03', 'c04', 'c05',
                'c06', 'c07', 'c08', 'c09',
                ]
    test_data_name = ['a20','b05','c10']
    age = [51, 38, 54, 52, 58,
        63, 44, 51, 52, 58,
        58, 52, 51, 51, 60,
        44, 40, 52, 55, 58,
        44, 53, 53, 42, 52,
        31, 37, 39, 41, 28,
        28, 30, 42, 37, 27]
    sex = [1, 1, 1, 1, 1,
        1, 1, 1, 1, 1,
        1, 1, 1, 1, 1,
        1, 1, 1, 1, 1,
        0, 1, 1, 1, 1,
        1, 1, 1, 0, 0,
        0, 0, 1, 1, 1]


    def get_qrs_amp(ecg, qrs):
        interval = int(FS * 0.250)
        qrs_amp = []
        for index in range(len(qrs)):
            curr_qrs = qrs[index]
            amp = np.max(ecg[curr_qrs-interval:curr_qrs+interval])
            qrs_amp.append(amp)

        return qrs_amp

    MARGIN = 10
    FS_INTP = 4
    MAX_HR = 300.0
    MIN_HR = 20.0
    MIN_RRI = 1.0 / (MAX_HR / 60.0) * 1000
    MAX_RRI = 1.0 / (MIN_HR / 60.0) * 1000
    train_input_array = []
    train_label_array = []

    for data_index in range(len(train_data_name)):
        print (train_data_name[data_index])
        win_num = len(wfdb.rdann(os.path.join(data_path,train_data_name[data_index]), 'apn').symbol)
        signals, fields = wfdb.rdsamp(os.path.join(data_path,train_data_name[data_index]))
        for index in tqdm(range(1, win_num)):
            samp_from = index * 60 * FS # 60 seconds
            samp_to = samp_from + 60 * FS  # 60 seconds

            qrs_ann = wfdb.rdann(data_path + train_data_name[data_index], 'qrs', sampfrom=samp_from - (MARGIN*100), sampto=samp_to + (MARGIN*100)).sample
            apn_ann = wfdb.rdann(data_path + train_data_name[data_index], 'apn', sampfrom=samp_from, sampto=samp_to-1).symbol

            qrs_amp = get_qrs_amp(signals, qrs_ann)

            rri = np.diff(qrs_ann)
            rri_ms = rri.astype('float') / FS * 1000.0
            try:
                rri_filt = moving_median(rri_ms)

                if len(rri_filt) > 5 and (np.min(rri_filt) >= MIN_RRI and np.max(rri_filt) <= MAX_RRI):
                    time_intp, rri_intp = interp_cubic_spline(rri_filt, FS_INTP)
                    qrs_time_intp, qrs_intp = interp_cubic_spline_qrs(qrs_ann, qrs_amp, FS_INTP)
                    rri_intp = rri_intp[(time_intp >= MARGIN) & (time_intp < (60+MARGIN))]
                    qrs_intp = qrs_intp[(qrs_time_intp >= MARGIN) & (qrs_time_intp < (60 + MARGIN))]
                    #time_intp = time_intp[(time_intp >= MARGIN) & (time_intp < (60+MARGIN))]

                    if len(rri_intp) != (FS_INTP * 60):
                        skip = 1
                    else:
                        skip = 0

                    if skip == 0:
                        rri_intp = rri_intp - np.mean(rri_intp)
                        qrs_intp = qrs_intp - np.mean(qrs_intp)
                        if apn_ann[0] == 'N': # Normal
                            label = 0.0
                        elif apn_ann[0] == 'A': # Apnea
                            label = 1.0
                        else:
                            label = 2.0

                        train_input_array.append([rri_intp, qrs_intp, age[data_index], sex[data_index]])
                        train_label_array.append(label)
            except:
                hrv_module_error = 1
    print(train_input_array[0])
    print(train_label_array[0])
    np.save('train_input.npy', train_input_array)
    np.save('train_label.npy', train_label_array)

    test_input_array = []
    test_label_array = []
    for data_index in range(len(test_data_name)):
        print (test_data_name[data_index])
        win_num = len(wfdb.rdann(os.path.join(data_path,test_data_name[data_index]), 'apn').symbol)
        signals, fields = wfdb.rdsamp(os.path.join(data_path,test_data_name[data_index]))
        for index in tqdm(range(1, win_num)):
            samp_from = index * 60 * FS # 60 seconds
            samp_to = samp_from + 60 * FS  # 60 seconds

            qrs_ann = wfdb.rdann(data_path + test_data_name[data_index], 'qrs', sampfrom=samp_from - (MARGIN*100), sampto=samp_to + (MARGIN*100)).sample
            apn_ann = wfdb.rdann(data_path + test_data_name[data_index], 'apn', sampfrom=samp_from, sampto=samp_to-1).symbol

            qrs_amp = get_qrs_amp(signals, qrs_ann)

            rri = np.diff(qrs_ann)
            rri_ms = rri.astype('float') / FS * 1000.0
            try:
                rri_filt = moving_median(rri_ms)

                if len(rri_filt) > 5 and (np.min(rri_filt) >= MIN_RRI and np.max(rri_filt) <= MAX_RRI):
                    time_intp, rri_intp = interp_cubic_spline(rri_filt, FS_INTP)
                    qrs_time_intp, qrs_intp = interp_cubic_spline_qrs(qrs_ann, qrs_amp, FS_INTP)
                    rri_intp = rri_intp[(time_intp >= MARGIN) & (time_intp < (60+MARGIN))]
                    qrs_intp = qrs_intp[(qrs_time_intp >= MARGIN) & (qrs_time_intp < (60 + MARGIN))]
                    #time_intp = time_intp[(time_intp >= MARGIN) & (time_intp < (60+MARGIN))]

                    if len(rri_intp) != (FS_INTP * 60):
                        skip = 1
                    else:
                        skip = 0

                    if skip == 0:
                        rri_intp = rri_intp - np.mean(rri_intp)
                        qrs_intp = qrs_intp - np.mean(qrs_intp)
                        if apn_ann[0] == 'N': # Normal
                            label = 0.0
                        elif apn_ann[0] == 'A': # Apnea
                            label = 1.0
                        else:
                            label = 2.0

                        test_input_array.append([rri_intp, qrs_intp, age[data_index], sex[data_index]])
                        test_label_array.append(label)
            except:
                hrv_module_error = 1
    np.save('test_input.npy', test_input_array)
    np.save('test_label.npy', test_label_array)

def apnea():
    X_train = np.load(f'{DATASET_DIR}/apnea/' + 'train_input.npy', allow_pickle=True)
    Y_train = np.load(f'{DATASET_DIR}/apnea/' + 'train_label.npy', allow_pickle=True)
    X_test = np.load(f'{DATASET_DIR}/apnea/' + 'test_input.npy', allow_pickle=True)
    Y_test = np.load(f'{DATASET_DIR}/apnea/' + 'test_label.npy', allow_pickle=True)

    X = np.concatenate((X_train, X_test), axis=0)
    rri = np.array(X[:, 0]).tolist()
    qrs = np.array(X[:, 1]).tolist()

    rri = np.array(rri)
    qrs = np.array(qrs)
    # print(rri.shape, qrs.shape)

    column_names = [f'rri{i}' for i in range(240)] + [f'qrs{i}' for i in range(240)] + ['age', 'sex']
    
    X_raw = pd.DataFrame(np.concatenate((rri, qrs, X[:, 2:]), axis=1), columns=column_names)
    Y = np.concatenate((Y_train, Y_test), axis=0).astype(int)

    A = X_raw['sex']

    return X_raw, Y, A


if __name__ == '__main__':
    apnea()