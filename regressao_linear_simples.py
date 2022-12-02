def regressao_linear_simples(dataset):
     from sklearn.linear_model import LinearRegression
     from sklearn.model_selection import train_test_split
     from metricas_erros import metricas_erros
     
     x1 = dataset.iloc[:, 1:2].values #Variavel previsora
     y =  dataset.iloc[:, 3].values # VAriavel alvo
     
     x_treino,x_teste,y_treino,y_teste =train_test_split(x1,y,test_size=0.3, random_state=10)
     
     print(f'x_treino.shape: {x_treino.shape}')
     print(f'y_treino.shape:{y_treino.shape}')
     print(f'x_teste.shape: {x_teste.shape}')
     print(f'y_teste.shape:{y_teste.shape}')
     
     from sklearn.linear_model import LinearRegression
     
     reg_linear1 = LinearRegression()
     reg_linear1.fit(x_treino,y_treino)
     
     #Intercept (Coeficiente linear)
     print(' Intercepto: {:.4f}'.format(reg_linear1.intercept_))
     #print('Coef:{:.4f}'.format(reg_linear1.coef_))
     
     #Coeficiente de determinação dados treinos
     print('Coef. de Determinação dados treino:{:.4f}'.format(reg_linear1.score(x_treino, y_treino)))
     print('Coef. de Determinação dados teste:{:.4f}'.format(reg_linear1.score(x_teste, y_teste)))
     
     
     previsoes_treino = reg_linear1.predict(x_treino)
    
     
     previsoes_teste = reg_linear1.predict(x_teste)
     
     print("Métricas dados de teste:")
     metricas_erros(y_teste, previsoes_teste)
     print('\n')
     
     print("Métricas dados de treino:")
     metricas_erros(y_treino, previsoes_treino)
     print('\n')
     import numpy as np
     import matplotlib.pyplot as plt 
  
     plt.scatter(y=y_treino,x= x_treino, color ='blue', s=10, alpha=0.9 )
     x_plot =np.linspace(5,35)
     plt.plot(x_plot, x_plot*reg_linear1.coef_+reg_linear1.intercept_, color ='r')
     plt.title('Regressão Linear Simples -Valores treinos ')
     plt.xlabel('LSTAT')
     plt.ylabel('Valor médio(R$)')
     plt.savefig('regressao_simples_2_treino.png',dpi =300, format='jpg')
     
    
     
     
     
     plt.scatter(y=y_teste,x= x_teste, color ='blue', s=10, alpha=0.9 )
     x_plot =np.linspace(5,35)
     plt.plot(x_plot, x_plot*reg_linear1.coef_+reg_linear1.intercept_, color ='r')
     plt.title('Regressão Linear Simples -Valores teste')
     plt.xlabel(' LSTAT')
     plt.ylabel('Valor médio(R$)')
     plt.savefig('regressao_simples_2_teste.png',dpi =300, format='jpg')
     
   
    

     
    
     
     
     return(0)
    