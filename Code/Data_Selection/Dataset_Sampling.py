import pandas as pd
from pathlib import Path
import numpy as np
from scipy.spatial import ConvexHull
from imblearn.over_sampling import SMOTE


RANDOM_STATES = [666, 22, 10, 25, 43, 143, 57]

def get_interior(db):
    interior = db[db['Ld'] != db['Lq']]
    interior.reset_index(inplace=True, drop=True)
    return interior

def get_exterior(db):
    exterior = db[db['Ld'] == db['Lq']]
    exterior.reset_index(inplace=True, drop=True)
    return exterior

def calc_ode_params(db):
    new_db = pd.DataFrame()
    new_db.loc[:,"p1"] = (db["UDC"]/2)/(db["Ld"]*db["In"]*1.5)
    new_db.loc[:,"p2"] = (db["p"]*db["Omegan"]*db["Lq"])/(db["Ld"])
    new_db.loc[:,"p3"] = -(db["Rs"])/(db["Ld"])
    new_db.loc[:,"p4"] = (db["UDC"]/2)/(db["Lq"]*db["In"]*1.5)
    new_db.loc[:,"p5"] = -(db["p"]*db["Omegan"]*db["Ld"])/(db["Lq"])
    new_db.loc[:,"p6"] = -(db["p"]*db["Omegan"]*db["Psip"])/(db["Lq"]*db["In"]*1.5)
    new_db.loc[:,"p7"] = -(db["Rs"])/(db["Lq"])
    return new_db

def calc_electrical_params(db, Rs, p, In):
    new_db = pd.DataFrame()
    new_db.loc[:,"Ld"] = -Rs/db["p3"]
    new_db.loc[:,"Lq"] = -Rs/db["p7"]
    new_db.loc[:,"Omegan"] = db["p2"]/p/new_db["Lq"]*new_db["Ld"]
    new_db.loc[:,"UDC"] = db["p1"]*new_db["Ld"]*In*2*1.5
    new_db.loc[:,"Psip"] = -db["p6"]/p/new_db["Omegan"]*new_db["Lq"]*In*1.5
    new_db.loc[:,"Rs"] = Rs
    new_db.loc[:,"p"] = p
    new_db.loc[:,"In"] = In
    return new_db

def downsampling(db, num_points):
    columns = db.columns
    points = db.to_numpy()
    mean = np.mean(points, axis=0)
    std = np.std(points, axis=0)
    selected_vectors = []
    positions = np.array([x for x in range(points.shape[0])])
    original_positions = []
    selected_vectors.append(points[0])
    original_positions.append(positions[0])
    positions = np.delete(positions, 0, axis=0)
    points = np.delete(points, 0, axis=0)
    for j in range(num_points-1):
        points_calc = (points - mean) / std
        distances = np.zeros(points.shape[0])
        for vector in selected_vectors:
            vector_calc = (vector - mean)/std
            distances = distances + ((vector_calc - points_calc)**2).sum(axis=1)
        position = np.argmax(distances)
        selected_vectors.append(points[position])
        original_positions.append(positions[position])
        points = np.delete(points, position, axis= 0)
        positions = np.delete(positions, position, axis= 0)
    new_db = pd.DataFrame(selected_vectors, columns=columns)
    return new_db, original_positions

def SMOTE_Sampling(minor, major, num_samples=75):
    minor_amount = len(minor.index)
    major_amount = len(major.index)
    minor_arr = np.zeros(minor_amount)
    major_arr = np.zeros(major_amount) +1
    minmaj = pd.concat([major, minor])
    minmaj.reset_index(inplace=True, drop=True)
    sampling_strategy = {
        0:num_samples,
    }
    classes = np.concatenate(( major_arr, minor_arr))
    smotenc = SMOTE(random_state=RANDOM_STATES[0], sampling_strategy=sampling_strategy, k_neighbors=5)
    new_minmaj, new_classes = smotenc.fit_resample(minmaj, classes)
    sPMSMs = new_minmaj.iloc[:-num_samples]
    IPMSMs = new_minmaj.iloc[-num_samples:]
    sPMSMs.reset_index(inplace=True, drop=True)
    IPMSMs.reset_index(inplace=True, drop=True)
    return IPMSMs, sPMSMs


code_path = Path(__file__).parent.absolute()
motor_db_path = code_path.parent.parent / "MotorDB"
motor_db = pd.read_excel(motor_db_path / "Complete.xlsx")
IPMSMs = get_interior(motor_db)
SPMSMs = get_exterior(motor_db)
IPMSMs_ode = calc_ode_params(IPMSMs)
SPMSMs_ode = calc_ode_params(SPMSMs)

###SPMSM Sampling

##Sample 75 SPMSMs
SPMSMs_ode_new, SPMSMs_ode_pos = downsampling(SPMSMs_ode, 35)
SPMSMs_new_el = SPMSMs.loc[SPMSMs_ode_pos]
SPMSMs_ode_new2 = SPMSMs_ode.drop(SPMSMs_ode_pos)
SPMSMs_ode_new2 = SPMSMs_ode_new2.sample(40, random_state=RANDOM_STATES[1])
SPMSMs_new_el2 = SPMSMs.drop(SPMSMs_ode_pos).loc[SPMSMs_ode_new2.index]
SPMSMs_ode = pd.concat([SPMSMs_ode_new,SPMSMs_ode_new2])
SPMSMs_ode_el = pd.concat([SPMSMs_new_el,SPMSMs_new_el2])
SPMSMs_ode.reset_index(drop=True,inplace=True)
SPMSMs_ode_el.reset_index(drop=True,inplace=True)

##Select 50 training and 25 test motors
hull = ConvexHull(SPMSMs_ode[['p1', 'p2','p3','p6']].to_numpy())
vertices = hull.vertices
SPMSMs_training1 = SPMSMs_ode.iloc[vertices]
SPMSMs_training1_el = SPMSMs_ode_el.iloc[vertices]
SPMSMs_ode.drop(vertices, inplace=True)
SPMSMs_ode_el.drop(vertices, inplace=True)
SPMSMs_training2 = SPMSMs_ode.sample(50-len(vertices), random_state=RANDOM_STATES[2])
SPMSMs_training2_el = SPMSMs_ode_el.loc[SPMSMs_training2.index]
SPMSMs_test = SPMSMs_ode.drop(SPMSMs_training2.index)
SPMSMs_test_el = SPMSMs_ode_el.drop(SPMSMs_training2.index)
SPMSMs_test.reset_index(inplace=True, drop=True)
SPMSMs_test_el.reset_index(inplace=True, drop=True)
SPMSMs_training = pd.concat([SPMSMs_training1, SPMSMs_training2])
SPMSMs_training_el = pd.concat([SPMSMs_training1_el, SPMSMs_training2_el])
SPMSMs_training.reset_index(inplace=True, drop=True)
SPMSMs_training_el.reset_index(inplace=True, drop=True)

###IPMSM Sampling

##Create 75 IPMSMs

#Source of in_hull():
#https://stackoverflow.com/questions/16750618/whats-an-efficient-way-to-find-if-a-point-lies-in-the-convex-hull-of-a-point-cl
from scipy.optimize import linprog
def in_hull(points, x):
    n_points = len(points)
    c = np.zeros(n_points)
    A = np.r_[points.T,np.ones((1,n_points))]
    b = np.r_[x, np.ones(1)]
    lp = linprog(c, A_eq=A, b_eq=b)
    return lp.success

hull = ConvexHull(IPMSMs_ode[['p1', 'p2','p3','p6', 'p7']].to_numpy())
num_smote_samples = 100
IPMSMs_ode_new, _ = SMOTE_Sampling(IPMSMs_ode[['p1', 'p2','p3','p6', 'p7']], SPMSMs_ode[['p1', 'p2','p3','p6', 'p7']], num_samples = num_smote_samples)
IPMSMs_ode_new_el = calc_electrical_params(IPMSMs_ode_new,Rs=1,p=4,In=5)
IPMSMs_ode_new = calc_ode_params(IPMSMs_ode_new_el)
for i in range((num_smote_samples-14)):
    if not in_hull(IPMSMs_ode[['p1', 'p2','p3','p6', 'p7']].to_numpy(), IPMSMs_ode_new[['p1', 'p2','p3','p6', 'p7']].loc[i+14].to_numpy()):
        IPMSMs_ode_new.drop(i+14, inplace=True)

##Select 50 training and 25 test motors
IPMSMs_training1 = IPMSMs_ode_new[:14]
IPMSMs_ode_new.drop(hull.vertices, inplace=True)
IPMSMs_training2 = IPMSMs_ode_new.sample(36, random_state=RANDOM_STATES[3])
IPMSMs_ode_new.drop(IPMSMs_training2.index, inplace=True)
IPMSMs_test = IPMSMs_ode_new.sample(25, random_state=RANDOM_STATE[4])
IPMSMs_training = pd.concat([IPMSMs_training1, IPMSMs_training2])
IPMSMs_training_el = pd.concat([IPMSMs, calc_electrical_params(IPMSMs_training2,Rs=1,p=4,In=5)])
IPMSMs_test_el = calc_electrical_params(IPMSMs_test,Rs=1,p=4,In=5)

###Create final training and test set
def save_to_excel( file, folder, name):
    file.to_excel(folder / name, index=False)

train = pd.concat([SPMSMs_training_el, IPMSMs_training_el]).sample(frac=1, random_state=RANDOM_STATES[5]).reset_index(drop=True)
test = pd.concat([SPMSMs_test_el, IPMSMs_test_el]).sample(frac=1, random_state=RANDOM_STATES[6]).reset_index(drop=True)
save_to_excel(train, motor_db_path, "Training.xlsx")
save_to_excel(test, motor_db_path, "Test.xlsx")
save_to_excel(calc_ode_params(train), motor_db_path, "ODETraining.xlsx")
save_to_excel(calc_ode_params(test), motor_db_path, "ODETest.xlsx")
