def  regressao_polinomial(df):
    
    
     x1 =df.iloc[:,0:1].values 
     y = df.iloc[:,3].values 
    
     from sklearn.model_selection import train_test_split
     x_treino, x_teste, y_treino, y_teste = train_test_split(x1,y, test_size=0.3, random_state=0)
     print(f'x_treino.shape:{x_treino.shape}')
     print(f'y_treino.shape:{y_treino.shape}')
     print(f'x_teste.shape:{x_teste.shape}')
     print(f'y_teste.shape:{y_teste.shape}')
     
     from sklearn.preprocessing import PolynomialFeatures
     from sklearn.linear_model import LinearRegression
     
     grau_polinomial = PolynomialFeatures(degree=2)
     
     x_poly = grau_polinomial.fit_transform(x_teste)
     # x_poly-> x^0| x^1 |x^2
     
     polinomial =LinearRegression()
     polinomial.fit(x_poly, y_teste)
     previsores_treino =polinomial.predict(x_poly)
     
     print(polinomial.coef_)
     print(polinomial.intercept_)
     
     import numpy as np 
     numeros = np.linspace(3,9.84,147)
     
     
     # Polinomio interpolador de grau 2 , dados de treino
     #valor  =1640107.0085836346 -568528.11104731*numeros+ 60092.59048475*numeros**2
     
     #Polinomio interpolador de grau 2 dados de teste
     valor = 1755018.5868680933 -591140.7292315*numeros + 60520.80800829*numeros**2


     
     import matplotlib.pyplot as plt
     plt.scatter(x_teste, y_teste, c= 'blue', label ='Dados de teste')
     plt.xlabel("Quantidade comodos")
     plt.ylabel("Valor da casa")
     plt.title("Previsao valor das casas dados de treino")
     plt.plot(numeros, valor, color ='green', label ='Regressao polinomial')
     plt.legend(loc='best')
     plt.savefig('Preisao_polinomial_g2_teste.jpg', dpi = 300, format ='jpg')
     plt.show()
     
     
     print('Coef. de determinação dados treino: {:.4f}'.format(polinomial.score(x_poly,y_teste)))

 
     

     return(0)