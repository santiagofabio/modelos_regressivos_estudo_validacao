## Algoritmos de regressão 
**Objetivo**: Está implementação visa aplicar e avaliar diferentes algoritmos de Machine Learning  para a determinação do preço medio de venda de um imóvel para a cidade de Boston,a partir da base de dados *BostonHousing*.
## Algoritmos regressivos aplicados
* Regressão linear simples
* Resessão linear multipla
* Regressão polinomial
* Support Vector Regression (SVR)
*  DecisionTreeRegressor
* RandomForestRegressor
* XGBoost Regression
* LGBMRegressor
### Etapas de desenvolvimento do projeto
1. Analise exploratoria de dados.
2. Aplicação da regressão linar simples
3. Aplicação da regressão linear multipla
4. Aplicação dos regressão polinomial 
5. Aplicação dos algritmos de regressão classivos.
6. Validação dos resulatdos obtido.

## 1. **Analise exploratoria de dados**
1.1 **Detecção de outliers**
![Boxplot](boxplot.png)
1.2 **Mapeamento de correlações**
![mapeamento_correlacoes](mapeamento_correlacoes.jpg)
1.3 **Teste de normalidade**
![Normal QQ](NormalQQ.png)
1.4 **Heatmap**
![heatmap](heatmap.jpg)

## 2. Regressão linear simples (RLS)
2.1  **RLS -Valores de teste - RMxR\$**
![RLS_teste](regressao_simples_1_teste.png)
2.2  **RLS -Valores de treino - RMxR\$**
![RLS_treino](regressao_simples_1_treino.png)
2.3  **RLS -Valores de teste - LSTATxR\$**
![RLS_teste2](regressao_simples_2_teste.png)
2.4  **RLS -Valores de treino - LSTATxR\$**
![RLS_treino2](regressao_simples_2_treino.png)
2.4  **Avaliação homoscedasticidade**
![homoscedasticidade](homoscedasticidade.jpg)

2.4  **RLS -Statsmodel**
![regressao_LSTAT](regressao_LSTAT.jpg)

## 3. Regressão Linear Multipla (RLM)
3.1 **Distribuição Acurracy**
![Distribution_regressao](Distribution_regressao.jpg)
## 3. Regressão Linear Multipla (Statsmodel)
3.1 **Normal_QQ_plot_redisuos_stats**
![Normal_QQ_plot_redisuos_stats](Normal_QQ_plot_redisuos_stats.jpg)
3.2 **Homocedasticidade -Statsmodel**
![homocedasticidade dos resíduos -Statsmodel](homocedasticidade_statsmodel.jpg)
3.3 **Regressivo-Statsmodel**
![Regressivo-Statsmodel](regressivo_statsmodel.jpg)

4 **Regressão polinomial (RP)**
4.1 **RP-teste**
![Preisao_polinomial_g2](Preisao_polinomial_g2_teste.jpg)
4.1 **RP-teste**
![Preisao_polinomial_g2_treino](Preisao_polinomial_g2_treino.jpg)


5 **Resultados Regressores**
![distribution_regressor](distribution_regressor.jpg)









