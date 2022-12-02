def regressao_random_forest(df): 
     var_independente = df.iloc[:,0:3].values
     var_dependente = df.iloc[:,3].values
    
    
     from sklearn.model_selection import train_test_split
     x_treino,x_teste,y_treino,y_teste =train_test_split( var_independente, var_dependente, train_size=0.3, random_state=0 )
    
     from sklearn.preprocessing import StandardScaler
     x_scaler =StandardScaler()
     y_scaler = StandardScaler()
    
     x_treino_scaler = x_scaler.fit_transform(x_treino) 
     x_teste_scaler = x_scaler.fit_transform(x_teste)
    
     y_treino_scaler = y_scaler.fit_transform(y_treino.reshape(-1,1)) 
     y_teste_scaler = y_scaler.fit_transform(y_teste.reshape(-1,1))
    
     from sklearn.model_selection import cross_val_score
     from sklearn.ensemble import RandomForestRegressor
     modelo_arvore = RandomForestRegressor(max_depth =6,random_state =10)
     modelo_arvore.fit(x_treino_scaler,y_treino_scaler.ravel())
     print('Score dados treino: {:.4f}'.format(modelo_arvore.score(x_treino_scaler,y_treino_scaler)))
     modelo_arvore.fit(x_teste_scaler,y_teste_scaler.ravel())
     print('Score dados teste: {:.4f}'.format(modelo_arvore.score(x_teste_scaler,y_teste_scaler)))
 
     previsores_scaler_teste = modelo_arvore.predict(x_teste_scaler)
     previsores_inverse_teste = y_scaler.inverse_transform( previsores_scaler_teste.reshape(-1,1))
    
     from metricas_erros import metricas_erros
     metricas_erros(y_teste, previsores_inverse_teste.ravel() )
     return(0)