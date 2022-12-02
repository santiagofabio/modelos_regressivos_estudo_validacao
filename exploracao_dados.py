def exploracao_dados(dataset):
     import numpy as np
     import seaborn as sns 
     import matplotlib.pyplot as plt 
 
     # Verificação de valores faltantes.
     print(dataset.isnull().sum())
     #Identificação dos tipos de dados.
     print(dataset.dtypes)
     """
     #RM         float64
     #LSTAT      float64 
     #PTRATIO    float64
     #MEDV       float64
     """
     #Analise grafica de normalidade.
     plt.rcParams['figure.figsize'] = (10, 10)
     
     import scipy.stats as stats
     figure,( (ax1, ax2), (ax3,ax4)) = plt.subplots(2, 2)
     figure.suptitle('BoxPlot')
     sns.boxplot(data=dataset,x='RM', ax=ax1)
     sns.boxplot(data=dataset,x='LSTAT',ax =ax2)
     sns.boxplot(data=dataset,x='PTRATIO', ax= ax3)
     sns.boxplot(data=dataset,x='MEDV', ax= ax4)
     plt.savefig('boxplot.png',dpi =300, format='jpg')
          
     
     
     
     #mapeamento de correlaçoes
     import seaborn as sns 
     import matplotlib.pyplot as plt 
     sns.pairplot(dataset)
     plt.savefig('mapeamento_correlacoes.jpg',dpi=300, format='jpg')
  

     #Analise grafica de normalidade.
     plt.rcParams['figure.figsize'] = (10, 10)
     import scipy.stats as stats
     figure,( (ax1, ax2), (ax3,ax4)) = plt.subplots(2, 2)
     figure.suptitle('Nomral Q-Q Plot')
     stats.probplot(dataset.RM, dist='norm', plot=ax1 )
     ax1. set_title('RM') 
     
     stats.probplot(dataset.LSTAT, dist='norm', plot=ax2) 
     ax2.set_title('LSTAT') 
    
     stats.probplot(dataset.PTRATIO, dist='norm', plot= ax3)
     ax3.set_title('PTRATIO')
    
     stats.probplot(dataset.MEDV, dist='norm', plot= ax4)
     ax4.set_title('MEDV')
     
     plt.savefig('NormalQQ.png',dpi =300, format='jpg')
    
     #--------
     print('Teste Shapiro-Willk/ Normalidade')
     #Teste de Shapiro-Willk 
     nomes_atributos =['RM','LSTAT', 'PTRATIO','MEDV'] 
     alpha =0.05 # significancia
     for nome in nomes_atributos:
             estatistica, valor_p = stats.shapiro(dataset[nome])
             if valor_p>=alpha:
                 print(f'{nome}: Normalmente distribuido')
             else:
                  print(f'{nome}: Não normalmente distribuido')
            
     print('\n')            

     print('Teste de  Kolmogorov/Normalidade' )            
     #Teste Kolmogorov
     import statsmodels
     from statsmodels.stats.diagnostic import lilliefors 
     for nome in nomes_atributos:
             estatistica, valor_p = statsmodels.stats.diagnostic.lilliefors(dataset[nome])
             if valor_p>=alpha:
                 print(f'{nome}: Normalmente distribuido')
             else:
                 print(f'{nome}: Não normalmente distribuido')

     print('\n')  
     #Teste de Hipotese para correlaçao linear
     #alpha =0.05
     # H0 -> Não há uma correlação linear: valor_p>alpha 
     # Ha ->Existe uma correlação linear: valor_p>alpha
     #-----------------------------
    
     print('Teste de correlação linear para os dados')
     #Teste de Person(Paramétrico ->Distribuição Normal)
     coeficiente, valor_p =stats.pearsonr(dataset['MEDV'],dataset['RM']) 
     print('Teste de Person')
     if valor_p> alpha:
         print('Não há uma correlação linear. ')
     else:
         print('Existe uma correlação linear')
         print('Coef. de correlação:{:.4f}'.format(coeficiente))     

     # Teste de Spearmanr( Não Paramétrico >Distribuição Não Normal)
     coeficiente, valor_p =stats.spearmanr(dataset['MEDV'],dataset['RM'])   
     
     print('Teste de Speraman')
     if valor_p> alpha:
         print('Não há uma correlação linear. ')
     else:
         print('Existe uma correlação linear')
         print('Coef. de correlação:{:.4f}'.format(coeficiente))    

     # Teste de Kendall( Não Paramétrico >Distribuição Não Normal)
     coeficiente, valor_p =stats.kendalltau(dataset['MEDV'],dataset['RM']) 
     print('Teste de Kendall')
     if valor_p> alpha:
          print('Não há uma correlação linear. ')
     else:
          print('Existe uma correlação linear')
          print('Coef. de correlação:{:.4f}'.format(coeficiente))     
   

     
     correlacoes=  dataset.corr(method='spearman')
     plt.figure()
     sns.heatmap(correlacoes, annot =True)
     plt.savefig('heatmap.jpg', dpi =300,format ='jpg')
     plt.show()
     
     
     return(0)
     
     








 

