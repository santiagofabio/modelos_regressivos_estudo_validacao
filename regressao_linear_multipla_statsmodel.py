def regressao_linear_multipla_statsmodel(df):
    import statsmodels.formula.api as smf
    import statsmodels.stats.api as sms
    import matplotlib.pyplot as plt
    
    modelo_regressao =smf.ols('MEDV ~ RM + LSTAT + PTRATIO', data =df).fit()
    residuos = modelo_regressao.resid
    
    import scipy.stats as stats 
    alfa =0.05
    estatistica, valor_p = stats.shapiro(residuos)
    print('Valor_p {:.4f}'.format(valor_p))
    if valor_p>= alfa:
        print('Resíduos se distribuem normalmente')
    else:
        print('Não existe distribuição normal dos residuos.')
    
    
  
    import matplotlib.pyplot as plt
    stats.probplot(residuos, dist ='norm',plot=plt)
    plt.title('Normal Q-Q plot - Residuos')
    plt.savefig('Normal_QQ_plot_redisuos_stats.jpg', dpi =300, format ='jpg')
    plt.show()
    
    
    #Analise de homocedasticidade dos resíduos
    plt.scatter(y=residuos,x =modelo_regressao.predict(), color ='blue' )
    plt.hlines(y=0, xmin=0,xmax =800000, color ='green')
    plt.xlabel('Valores preditos')
    plt.ylabel('Residuos')
    plt.title('homocedasticidade dos resíduos -Statsmodel')
    plt.savefig('homocedasticidade_statsmodel.jpg', dpi =300, format ='jpg')
    plt.show() 
   
    
    # Teste de Breush-pagan (Homocedasticidade ou heteorcedasticidade)
    #H0:  existe homocedasticidade:  valor_p>0.05
    #H1: Não existe homocedasticidade: valor_p<=0.05
    
    from statsmodels.compat import lzip
    
    estatistica, valor_p, f, fp = sms.het_breuschpagan(modelo_regressao.resid, modelo_regressao.model.exog)
    
    if valor_p>=alfa:
        print("Os resíduos possuem homocedasticidade")
    else:
        print('Não existe homocedasticidade ')
    
    
    # Estudo dos resíduos.
    outliers = modelo_regressao.outlier_test()
    print( ) 
    print(f'Outliers max {outliers.max()}')
    print(f'Outliers min {outliers.min()}')
    
    #Analise de multicolinearidade.
    #multicolinearidade se r>0
    
    variaveis_independetes =df[['RM','LSTAT','PTRATIO']] 
    
    correlacoes = variaveis_independetes.corr(method ='pearson')
    print(correlacoes)
    
    
    # Analise dos Modelo 
    # Valor_p: para cada coeficente <0.05 (Estatisticamente significativos)
    # Adjusted-R (Exlicação do modelo através do dados)
    # Valor_p  da esttistcia F (Valia o delo regressão)
    
    print(modelo_regressao.summary())
    
    df['previsao']= modelo_regressao.fittedvalues
    import seaborn as sns 
    sns.lmplot(x='previsao', y='MEDV', data =df)
    plt.savefig('regressivo_statsmodel.jpg', dpi =300, format ='jpg')
    plt.show()
    
    return(0)