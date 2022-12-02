def regressao_svr(df):
    
    var_independente = df.iloc[:,0:3]. values
    var_dependente  = df.iloc[:,3].values
    
    
    from sklearn.model_selection import train_test_split
    x_treino, x_teste, y_treino, y_teste = train_test_split(var_independente,var_dependente, test_size=0.3, random_state=0)
    
    #padronização de escalas
    from sklearn.preprocessing import StandardScaler
    x_scaler  = StandardScaler()
    y_scaler =  StandardScaler()
    
    x_treino_scaler =x_scaler.fit_transform(x_treino)
    x_teste_scaler  =x_scaler.fit_transform(x_teste)
    
    
    y_treino_scaler =y_scaler.fit_transform(y_treino.reshape(-1,1))
   
    y_teste_scaler = y_scaler.fit_transform(y_teste.reshape(-1,1)) 
    
    
    
    
    
    
    from sklearn.svm import SVR
    #kernel = rbf, linear, polinomial (poly)
    SVR_kernel =SVR(kernel='rbf')
    SVR_kernel.fit( x_treino_scaler,y_treino_scaler.ravel() )
    print('Coef. de determinação dados treino: {:.4f}'.format(  SVR_kernel.score(x_treino_scaler,y_treino_scaler)))
    print('Coef. de determinação dados teste: {:.4f}'.format(  SVR_kernel.score(x_teste_scaler,y_teste_scaler)))
    
    print('Métricas dados de teste:')
    from metricas_erros import metricas_erros
    y_previsao_treino_scaler = SVR_kernel.predict(x_treino_scaler)
    
    y_previsao_treino_inverse =  y_scaler.inverse_transform(y_previsao_treino_scaler.reshape(-1,1))
   
    
    metricas_erros(y_treino, y_previsao_treino_inverse)
    
    y_previsao_teste_scaler = SVR_kernel.predict(x_teste_scaler)
    y_previsao_teste_inverse =  y_scaler.inverse_transform(y_previsao_teste_scaler.reshape(-1,1))
    metricas_erros(y_teste,  y_previsao_teste_inverse )
    
    
    
    #Revertendo a transormação para voltar aos dados originais
    
    
    return(0)