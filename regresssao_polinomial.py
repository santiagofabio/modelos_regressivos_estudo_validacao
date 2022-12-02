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
     
     x_poly = grau_polinomial.fit_transform(x_treino)
     # x_poly-> x^0| x^1 |x^2
     
     polinomial =LinearRegression()
     polinomial.fit(x_poly, y_treino)
     previsores_treino =polinomial.predict(x_poly)
     
     print(polinomial.coef_)
     print(polinomial.intercept_)
     


     return(0)