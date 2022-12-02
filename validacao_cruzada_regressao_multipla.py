def validacao_cruzada_regressao_multipla(var_independente, var_dependente):
      from sklearn.model_selection import KFold
      from sklearn.model_selection import cross_val_score
      from sklearn.linear_model import LinearRegression
      import matplotlib.pyplot as plt  
      modelo =LinearRegression()
      resultados_medios=[]
      
      for i in range(0,50):
          kfold = KFold(n_splits=10, shuffle=True, random_state=i)
          resulado = cross_val_score(modelo,var_independente,var_dependente, cv =kfold ) 
          resultados_medios.append(resulado.mean())
        
        
      import pandas as pd
      import seaborn as sns
      nome_modelo ='regressao_linear_multipla'
      dataframe = pd.DataFrame({nome_modelo:resultados_medios})
     # dataframe.to_csv('validacao_knn.csv',sep=';', encoding='utf-8')
      sns.kdeplot(data = dataframe,x= nome_modelo, label='Distribution Accuracy')
      plt.legend(loc ='best')
      plt.xlabel('Accuracy')
      plt.tight_layout()
      plt.title('Distribution regressao')
      plt.savefig('Distribution_regressao.jpg', dpi =300, format = 'jpg')
      plt.show()    
     

      
      
      
      return(0)