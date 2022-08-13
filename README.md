# DEEP NEURAL NETWORK MODEL FOR GE STOCK DEPLOY ON A MICROSERVICE ON FLASK
A deep learning model to predict the GE stock build in a Flask enviroment to deploy a Microservice to get the forecast of the next day return.


Explicación!! 

Esto es un proyecto end-to-end para la creación de un modelo de red neuronal profunda que busca predecir la rentabilidad del día siguiente para la acción de General Electrics (Elección: La primera que me acorde jajaja). Consiste de las siguientes partes.

#### PRIMERA PARTE CREACIÓN DEL MODELO.

- Obtención de la data a través de pandas datareader desde yahoo finance. 
- Preprocesamiento de la data. 
- Creación de variables de entrada del modelo (Features) Aquí utilice medias moviles la verdad ayudan bastante a percibir la volatidad de la rentabilidad diaria. 
- Creación del Modelo Multilayer Perceptron. El más facil de hacer y con excelentes resultados al ser entrenados con vectores de entrada con información pasada de la red en ventanas cortas de tiempo (entre 5 días hacia atrás, 10 días, máximo 30) siendo un vector de 3 dimensiones pero con un Flatten queda listo para la MLP. 

#### SEGUNDA PARTE PRUEBA DEL MODELO.

- Luego se realiza la predicción de estos modelos para los datos de prueba, se calcula el MSE como métrica para ver la eficiencia del modelo, con el 20% de la data. Esta data no esta vista por el modelo!! (Obviamente). 
- Gráfica de los resultados de prueba. (Verlos!! estan re piolas, para un modelo rapido)

### TERCERA PARTE: CREACIÓN DEL MICROSERVICIO EN FLASK. 

- Creación de aplicativo en flask. Aquí la idea era crear un Microservicio que aceptara las variables de entrada del modelo en un archivo .json como body con el fin de poder interacturar con este de manera más comoda con otros microservicios (Los que necesite negocio ajajaj). 

- Creación de firma automatica. Dentro de los archivos de preprocessing. Cree una firma que solo utilizara la fecha que quieres predecir y automaticamente esta calculara los campos de entrada del modelo con el fin de no estar calculandolos para poder tener un predicción ( Este modelo tmb podria ser usado manual y daría mucha utilidad). 

- Creación del metodo POST para obtener la respuesta del microservicio llamando al modelo, dandole las variables de entrada y obteniendo la predicción del modelo. Además se entregan las varibales de entradas calculadas automáticamente, 


Por lo tanto: El MS queda así: 

ruta: your-site(localhost:8080)/ge-prediction

body(firma)

```
 1. { "day-to-predict": "2022-05-18"}
 
```

Response: 

```
1. {
2.    "inputVariables": {
3.        "movingAvaregeOneMonthBack": 0,
4.        "movingAvaregeOneWeekBack": 0.0207,
5.        "returnOneDayBack": 0.0253,
6.        "returnOneMonthBack": -0.0003,
7.        "returnOneWeekBack": 0.045
8.    },
9.    "returnPredition": "0.41%"
10. }
 
```

Y voila, tenemos un microservicio que con solo entregarle el día a predecir nos da de resultado el retorno del día a predecir y las variables que utilizo como entrada!!




