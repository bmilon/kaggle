

import pandas as pd 
import numpy as np
import pyreadr
from datetime import datetime
from dateutil import parser
import seaborn as sns 

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix,explained_variance_score
from sklearn import preprocessing
from sklearn import tree
import sklearn
from sklearn.ensemble import RandomForestClassifier
import graphviz
from matplotlib import pyplot

in_scope_countries = ['USA' , 'Germany' ]
basefolder= r"C:\Users\mbhattac\OneDrive - HERE Global B.V-\coding\speed limit QI model\\"
selections_2019= pd.read_excel(basefolder + 'Selections_2019.xlsx',sheet_name='Selections_2019')


# extract relevant rows 
relevant_routes = selections_2019[[ (a in in_scope_countries) for a in selections_2019.Country ]]

# temp reset will be removed later 
relevant_routes = selections_2019

# find rows which have a not null R.NAV
relevant_routes= relevant_routes[~relevant_routes['R.NAV'].isna()]



# TODO use the getMCW_SL get this data 
multi_mcw_sl = pyreadr.read_r(basefolder+ 'RoW18-Short58-MCW-SL.Rdata')
multi_mcw_sl=multi_mcw_sl['multi_mcw_sl']
multi_mcw_sl['link_length_meters'] = multi_mcw_sl.link_length_meters.astype(float)

#  TODO use the getRWT function to generate this data 
multi_RWT= pyreadr.read_r(basefolder+ "RoW18-Short59-RWT-results.Rdata") 
multi_RWT=multi_RWT['multi_RWT']




shorts = ['CHBUT','EZE','COLON','CURIT','BELEM','PTRLN','BAURU','RIO2BARBC','EDM','KOLK','VISA',
                            'THRPM','GURG2JAI','JAK','BAYA','BOGOR','PKBAR','JAMBI','JB','MALAC','BHARU','SPNG',
                            'GVMD','PUEBL','CHIHU','MER','LUIS','WELL','WHNNZ','PLMNZ','MNLA','CEBU','BCOR','CZEST',
                            'ZAKO','GDSK2KATO','PITR','NNOV', 'ADLER','LMONO','YGORY','UFA2CHELY','NOVROS','KLSY',
                            'TABK','ARAR','SING','JOH','PRET','TP','NBUR','PHU','LAMP','TRAT','HH','YNSHR','DUBAI',
                            'ALAIN','RAK2AJMAN'];


sel = relevant_routes.loc[[ (a.Short in shorts) for a in relevant_routes.itertuples() ]]     
sel2  = pd.DataFrame(sel[['Short','Week.NAV']])     
sel3= pd.read_excel(basefolder + 'Selections_2019.xlsx',sheet_name='Weeks')
sel4 = sel2.join(sel3.set_index('Week'),on='Week.NAV',how='inner')


multi_mcw_sl_dates = multi_mcw_sl.join(sel4.set_index('Short'),on='Short',how='inner')
multi_mcw_sl_dates['Monday_numeric']  = [datetime.strptime(a, '%Y-%m-%d') for a in multi_mcw_sl_dates.Monday]
# change it id there is a MM versus DD doubts 
multi_mcw_sl_dates['change_date_numeric']  = [parser.parse(a) for a in multi_mcw_sl_dates.change_date]


# definition of good needs to change RWT date and MCW date should be close 
__temp__ = [ (a.change_date_numeric <= a.Monday_numeric) for a in multi_mcw_sl_dates.itertuples()]

multi_mcw_sl_dates_good   =  multi_mcw_sl_dates.loc[__temp__]
multi_mcw_sl_dates_bad   =  multi_mcw_sl_dates.loc[ [not a for a in __temp__] ]


multi_model_df_full =  multi_RWT.join(multi_mcw_sl_dates_good.set_index('link_pvid'),on='LINK.ID',lsuffix='rwt',rsuffix='mcw', how='inner')
#multi_model_df_full.drop(columns=['I','LINK.ID','Monday','change_date','Countrymcw'],inplace=True) 

#multi_model_df_full.to_excel('multi_model_df_full.xlsx')

# find columns with large nos of Null values and drop them manually 
multi_model_df_full.isna().astype(int).sum()

# TODO split clusters , changedate and testid see if can use it 

# modeling all together 
y =  multi_model_df_full['C']
x =  multi_model_df_full.drop(columns=['version','speed_limit_date','Monday_numeric','change_date_numeric','attribute_value_old_numeric','attribute_value_new_numeric','change_date_numeric','attribute_value_old','attribute_value_new','description','dir_of_travel_source','divider_source','admin_l4_display_name','C','Countryrwt','admin_l2_display_name','admin_l3_display_name','change_date','testid','LINK.ID','Shortrwt','complete_name','STREET.NAME','geometry_source','cluster','link_id','work_execution_detail','Week.NAV', 'Monday', 'Monday_numeric'])

# create categorical variable s

newx=pd.DataFrame()

for columns in x.columns:
    if x.dtypes[columns]  in [ 'object','bool']:
        foo  =  pd.get_dummies(x[columns])
        newx = pd.concat([foo, newx], axis=1)
    else:
        newx[columns]  =  x[columns] 


x=newx

x_train, x_valid, y_train, y_valid = train_test_split(x, y, test_size=0.4)

# simple tree 
classifier_tree = tree.DecisionTreeClassifier(random_state=0)
classifier_tree = classifier_tree.fit(x_train,y_train) 
y_valid_predict = classifier_tree.predict(x_valid)
fpr, tpr, _ = sklearn.metrics.roc_curve(y_valid_predict ,y_valid)
sns.set()
pyplot.plot(fpr, tpr, marker='.', label='Decision Tree FPR TPR')
pyplot.xlabel('False Positive Rate')
pyplot.ylabel('True Positive Rate')
pyplot.legend()
precision, recall, _ = sklearn.metrics.precision_recall_curve(y_valid_predict ,y_valid)
pyplot.plot(precision, recall, marker='.', label='Decision Tree PR Curve')
pyplot.xlabel('precision')
pyplot.ylabel('recall')
pyplot.legend()















