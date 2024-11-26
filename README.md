# Tarjeta de Modelo: Predicción del Precio de Bitcoin Basado en LSTM

## Descripción del Modelo
Este modelo LSTM está diseñado para predecir el precio de Bitcoin (BTC-USD) utilizando datos históricos de precios e indicadores técnicos. La arquitectura del modelo se basa en capas de memoria a largo corto plazo (*Long Short-Term Memory*, LSTM), que son efectivas para capturar patrones temporales en datos de series de tiempo.

## Características y Componentes Principales
1. **Preparación de Datos**
   - **Fuente de Datos**: Datos históricos de BTC (cierre, volumen, máximo y mínimo) obtenidos de Yahoo Finance.
   - **Indicadores Técnicos**: Incluye promedios móviles (7 y 30 días), RSI y MACD para capturar información adicional sobre tendencias y momentum del precio.
   - **Creación de Secuencias**: Se generan secuencias de datos de 60 días, permitiendo que el modelo aprenda de los patrones recientes del precio.

2. **Arquitectura del Modelo**
   - **Capas LSTM**: Dos capas LSTM (100 unidades cada una) para capturar dependencias secuenciales.
   - **Capas Dropout**: Regularización con *dropout* al 20% para prevenir el sobreajuste.
   - **Capas Densas**: Dos capas densas para generar una predicción única del precio.

3. **Configuración de Entrenamiento**
   - **Función de Pérdida**: Pérdida de Huber, elegida por su robustez ante valores atípicos, adecuada para datos financieros volátiles.
   - **Optimizador**: Optimizador Adam con una tasa de aprendizaje de 0.001 para ajustes eficientes de los pesos.
   - **Parada Temprana**: Monitoreo de la pérdida de validación con una paciencia de 10 épocas para evitar el sobreajuste.

## Métricas de Rendimiento
- **Métrica de Evaluación**: Error cuadrático medio de la raíz (*Root Mean Squared Error*, RMSE) en los conjuntos de entrenamiento y prueba para medir la precisión de las predicciones.
- **Resultados**:
  - **RMSE en Entrenamiento**: _X.XX_
  - **RMSE en Prueba**: _X.XX_

## Caso de Uso Ejemplar
Este modelo puede predecir el movimiento del precio de Bitcoin, proporcionando valor para operadores diarios o inversores interesados en identificar tendencias a corto plazo. También puede servir como base para la integración en sistemas más amplios de pronóstico financiero.

## Limitaciones
- **Sensibilidad a los Datos**: El rendimiento del modelo depende significativamente de la calidad y estabilidad de los datos de entrada.
- **Riesgos de Sobreajuste**: A pesar del uso de *dropout* y parada temprana, la naturaleza volátil de los precios puede llevar al sobreajuste.

## Mejoras Futuras
- **Indicadores Adicionales**: Incluir indicadores técnicos más avanzados para ofrecer un mejor contexto.
- **Ajuste de Hiperparámetros**: Experimentar con diferentes unidades LSTM, tasas de *dropout* y tasas de aprendizaje para optimizar el rendimiento.
