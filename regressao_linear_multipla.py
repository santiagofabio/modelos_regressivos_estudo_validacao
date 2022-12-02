def regressao_linear_multipla(df):
     #Pressupostos 
     # 1 - Relação linear entre as varivaeis independentes e dependentes
     # 2 - Sem presença de outliers na analise dos residuos.
     # 3 - Passar no teste de homocedasticidde
     # 4 - Residuos distribuidos com média  0 e Variancia constante
     # 5 - Ausencia de multocolanirieadae e autocorrelacao
     
     from sklearn.linear_model import LinearRegression
     from sklearn.model_selection import train_test_split
     from metricas_erros import metricas_erros
     from validacao_cruzada_regressao_multipla import validacao_cruzada_regressao_multipla
     independente= df.iloc[:,0:3].values
     dependente =df.iloc[:,3]. values
     
     x_treino, x_teste, y_treino,y_teste =train_test_split(independente,dependente, test_size=0.3, random_state=0)
     
     regressao_multipla = LinearRegression()
     regressao_multipla.fit(x_treino,y_treino)
     print('Intercept {:.4f}'.format(regressao_multipla.intercept_))
     print('Coef RM {:.4f}'.format(regressao_multipla.coef_[0]))
     print('Coef LSTAT {:.4f}'.format(regressao_multipla.coef_[1]))
     print('Coef PTRATIO {:.4f}'.format(regressao_multipla.coef_[2]))
     
     
     print('Coef. de determinacao: {:.4f}'.format(regressao_multipla.score(x_treino,y_treino)))
     previsoes_teste = regressao_multipla.predict(x_teste)
     previsoes_treino = regressao_multipla.predict(x_treino)
     print('Métricas dados treinos:')
     metricas_erros(y_treino,previsoes_treino)
     print('Métricas dados teste:')
     metricas_erros(y_teste, previsoes_teste)
     
     validacao_cruzada_regressao_multipla(independente,dependente )
     

      
    
     
    
     return(0)