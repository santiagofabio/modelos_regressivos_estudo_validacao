def metricas_erros(y_real, y_previsto):
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    import numpy as np 
    
    print('Erro absoluto: {:.4f}'.format(abs(y_real-y_previsto).sum() ))
    print('Erro médio absoluto: {:.4f}'.format(mean_absolute_error(y_real,y_previsto)))
    print('Erro quadratico médio: {:.4f}'.format(np.sqrt(mean_squared_error(y_real,y_previsto))))
    
    
    
    return(0)