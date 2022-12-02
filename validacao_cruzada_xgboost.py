def validacao_cruzada_xgboost(df):
    
    from sklearn.model_selection import KFold
    from sklearn.model_selection import cross_validate, cross_val_score
    from sklearn.preprocessing import StandardScaler
    print('Validação Cruzada Xboost') 
               
    var_independente= df.iloc[:, 0:3].values
    var_dependente =df.iloc[:,3].values
    
    independe_scaler =StandardScaler()
    dependente_scaler =StandardScaler()
    
    var_independente_scaler =independe_scaler.fit_transform(var_independente)
    var_dependente_scaler = dependente_scaler.fit_transform(var_dependente.reshape(-1,1))
    
    resultados_medios =[]
    from xgboost import XGBRegressor 
    modelo =  XGBRegressor(n_estimators=100, max_depth =3, learning_rate =0.05, objective ="reg:squarederror")


       
  
    
    for i in range(0,30):
          kfold = KFold(n_splits=10, shuffle=True, random_state=i)
          resulado = cross_val_score(modelo,var_independente_scaler ,var_dependente_scaler.ravel(), cv =kfold )     
          resultados_medios.append(resulado.mean()) 
    
    
  
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    nome_modelo ='XGBRegressor'
    dataframe = pd.DataFrame({nome_modelo:resultados_medios})
    print(dataframe.describe())
    
    dataframe.to_csv('XGBRegressor.csv',sep=';', encoding='utf-8', index=False)
    sns.kdeplot(data = dataframe,x= nome_modelo, label='XGBRegressor')
    plt.legend(loc ='best')
    plt.tight_layout()
    plt.title('XGBRegressor_cross_validatio')
    plt.savefig('XGBRegressor_validation.jpg', dpi =300, format = 'jpg')
    plt.show() 
    
    return(0)   

