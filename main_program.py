from exploracao_dados import exploracao_dados
from regressao_linear_simples import regressao_linear_simples
from regressao_linear_multipla import regressao_linear_multipla
from regressao_linear_multipla_statsmodel import regressao_linear_multipla_statsmodel
from regressao_polinomial import regressao_polinomial 
from regressao_svr import regressao_svr 
from sklearn.linear_model import LinearRegression
from  validacao_cruzada_svr import validacao_cruzada_svr
from avaliacao_modelo import avaliacao_modelo
from regressao_arvore_decisao import regressao_arvore_decisao 
from validacao_cruzada_arvore_decisao import validacao_cruzada_arvore_decisao
from regressao_random_forest import regressao_random_forest 
from validacao_cruzada_random_forest import validacao_cruzada_random_forest 
from regressao_xgboost import regressao_xgboost 
from validacao_cruzada_xgboost import validacao_cruzada_xgboost
from validacao_cruzada_lgbm import validacao_cruzada_lgbm 
from regressao_lgbm  import regressao_lgbm
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import seaborn as sns

"""
file = "housing.csv"

dataset =pd.read_csv(file, sep =',', encoding='utf-8') 
print(dataset.head(7))
"""

#exploracao_dados(dataset)
##odelo de regressao linear
#regressao_linear_simples(dataset)
#avaliacao_modelo(dataset)
#Modelo de regressao linear multipla



"""
#------------Regressão-------------
regressao_linear_multipla(dataset)
regressao_linear_multipla_statsmodel(dataset)
regressao_polinomial(dataset)
regressao_svr(dataset)
regressao_arvore_decisao(dataset)
regressao_random_forest(dataset)
regressao_xgboost(dataset)
regressao_lgbm(dataset)
#--------------------------------
#------------Validacao
validacao_cruzada_svr(dataset)
validacao_cruzada_arvore_decisao(dataset)
validacao_cruzada_random_forest(dataset)
validacao_cruzada_xgboost(dataset)
validacao_cruzada_lgbm(dataset)
validacao_cruzada_svr(dataset)
validacao_cruzada_arvore_decisao(dataset)
validacao_cruzada_random_forest(dataset)
validacao_cruzada_xgboost(dataset)
validacao_cruzada_lgbm(dataset)


df_validacao_svr = pd.read_csv('SVR_regressao.csv', sep =';', encoding='utf-8')
df_arvore_decisao = pd.read_csv('Arvore_decisao_regressao.csv', sep =';', encoding='utf-8')
df_cruzada_random_forest = pd.read_csv('Random_forest_regressao.csv', sep =';', encoding='utf-8')
df_xgboost = pd.read_csv('XGBRegressor.csv', sep =';', encoding='utf-8')
df_lgbm= pd.read_csv('LightGBM_regressao.csv', sep =';', encoding='utf-8')

import pandas as pd
df_classifier = pd.concat([df_validacao_svr,df_arvore_decisao,df_cruzada_random_forest,df_xgboost,df_lgbm], axis =1)
df_classifier.to_csv('df_regressor.csv',sep=';', encoding='utf-8', index =False)
print(df_classifier.head(10))

"""

file2 = 'df_regressor.csv'
df_classifier =pd.read_csv(file2, sep =';', encoding='utf-8')
   
print(df_classifier.columns)
from pylab import rcParams
rcParams['figure.figsize'] = 20, 15
sns.kdeplot(data =df_classifier, x = 'Arvore_Decisao', label ='Arvore_Decisao',  marker = 'o')
sns.kdeplot(data =df_classifier, x = 'LightGBM', label ='LightGBM', marker = 'v' )
sns.kdeplot(data =df_classifier, x = 'Random Forest', label ='Random Forest',  marker = '^')
sns.kdeplot(data =df_classifier, x = 'SVR', label ='SVR',marker = '<'  )
sns.kdeplot(data =df_classifier, x = 'XGBRegressor', label ='XGBRegressor.csv', marker ='x')
plt.legend(loc ='best')
plt.tight_layout()
plt.title('Regressor distribution')
plt.xlabel('Accuracy')
plt.savefig('distribution_regressor.jpg', dpi =300, format = 'jpg')
plt.show()  



