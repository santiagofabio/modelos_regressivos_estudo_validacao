def avaliacao_modelo(df):
     import numpy as np
     import pandas as pd
     import matplotlib.pyplot as plt
     import statsmodels.formula.api as smf
    
     x2 = df.iloc[:,1:2].values
     y = df.iloc[:,3].values
    
     regressao=smf.ols('y~x2',data=df).fit()
     residuos_regressao =regressao.resid
    
          #--------
     print('Teste Shapiro-Willk/ Normalidade')
     from scipy import stats
     alpha =0.05 # significancia
     estatistica, valor_p = stats.shapiro(residuos_regressao)
     if valor_p>=alpha:
         print(f'Residuos: Normalmente distribuido')
     else:
         print(f'Residuos: Não normalmente distribuido')
     
     import scipy.stats as stats
     """
     stats.probplot(residuos_regressao, dist ='norm', plot=plt)
     plt.title('Normal Q-Q plot - Résiduos')
     plt.savefig('Normal_residuos.jpg', dpi= 300, format ='jpg')
     plt.show()
     """
     # Analise de homocedasticidade dos residuos
     # residuos com variacao constant.
     plt.scatter(y= residuos_regressao, x= regressao.predict(), color ='blue')
     plt.hlines(y=0,xmin =0, xmax=700000, color ='red')
     plt.ylabel('Resíduos')
     plt.xlabel('Valores preditos')
     plt.title('Analise de homoscedasticidade')
     plt.savefig('homoscedasticidade.jpg', dpi =300, format ='jpg')
     plt.show()
     
     
     #Breusch–Pagan test
     alpha =0.05
     #H0 -> Existe homocedasticidade: valor_p>0.05
     #H1 -> Não existe homocedasticidade: valor_p<0.05
     
     
     """
     Heterocedasticidade:A variância dos erros será diferente para cada valor condicional
     Homocedasticidade: A variância dos erros e, condicionada aos valores das variáveis explanatórias, será
                          constante.
     """
     from statsmodels.compat import lzip
     import statsmodels.stats.api as sms
     
     estatistica,valor_p, f, fp = sms.het_breuschpagan(residuos_regressao, regressao.model.exog)
     if valor_p> alpha:
         print('Existe homocedasticidade dos residuos')
     else: 
         print('Residuos possuem Heterocedasticidade.')
         
     outliers = regressao.outlier_test()
     print(outliers.max())
     #print ('Outliers max: {:.4f}'.format(outliers.max())) 
     #print ('Outliers min: {:.4f}'.format(outliers.min() )) 
     
     coefs = pd.DataFrame(regressao.params)
     coefs.columns =['Coeficientes']
     print(coefs)
     
     plt.scatter(y=df.MEDV, x =df.LSTAT, color ='blue', s =80, alpha =0.9)
     x_plot =np.linspace(0,40)
     plt.plot(x_plot,x_plot*regressao.params[1]+regressao.params[0], color ='r')
     plt.title('Valor da casa')
     plt.xlabel('LSTAT')
     plt.savefig('regressao_LSTAT.jpg', dpi =300, format ='jpg')
     plt.show()
     
     
          
     return(0)
    
    